import streamlit as st
import requests
import json
import warnings
import geopandas as gpd
import rasterio
import ee
import geemap.foliumap as geemap
from geopy.geocoders import Nominatim
import matplotlib.pyplot as plt
import psutil
import re
from pathlib import Path
import uuid

warnings.filterwarnings("ignore")

# --- Set Page Config ---
st.set_page_config(
    page_title="GIS Reasoning Agent (Mistral + GEE)",
    page_icon="🧠",
    layout="wide"
)

# --- Initialize Earth Engine ---
try:
    ee.Initialize(project='gdgadk')
    st.success("✅ GEE initialized.")
except Exception as e:
    st.error(f"❌ GEE Error: {e}")

# --- Map Processing Logic ---
def get_map(location, analysis):
    geolocator = Nominatim(user_agent="my-gis-agent-contact@example.com")
    geo = geolocator.geocode(location)
    if not geo:
        st.error("❌ Location not found.")
        return None

    lat, lon = geo.latitude, geo.longitude
    m = geemap.Map(center=[lat, lon], zoom=11)
    m.add_basemap("OpenTopoMap")
    m.add_marker(location=[lat, lon], popup=location)

    if analysis == "landsat":
        img = ee.ImageCollection('LANDSAT/LC08/C02/T1_L2').filterDate('2018', '2020').filter(ee.Filter.lt('CLOUD_COVER', 20)).median()
        vis = {'bands': ['SR_B4', 'SR_B3', 'SR_B2'], 'min': 0, 'max': 3000}
        m.addLayer(img, vis, "Landsat RGB")

    elif analysis == "ndvi":
        img = ee.ImageCollection('LANDSAT/LC08/C02/T1_L2').filterDate('2018', '2020').median()
        ndvi = img.normalizedDifference(['SR_B5', 'SR_B4']).rename('NDVI')
        vis = {'min': -1, 'max': 1, 'palette': ['blue', 'white', 'green']}
        m.addLayer(ndvi, vis, "NDVI")

    elif analysis == "elevation":
        elev = ee.Image('USGS/SRTMGL1_003')
        vis = {'min': 0, 'max': 4000, 'palette': ['blue', 'green', 'brown']}
        m.addLayer(elev, vis, "Elevation")

    elif analysis == "landcover":
        landcover = ee.Image("ESA/WorldCover/v100/2020")
        vis = {'bands': ['Map'], 'palette': ['#006400', '#ffbb22', '#ffff4c', '#f096ff', '#fa0000']}
        m.addLayer(landcover, vis, "Land Cover")

    elif analysis == "flood_risk":
        sentinel1 = ee.ImageCollection('COPERNICUS/S1_GRD') \
            .filterBounds(ee.Geometry.Point(geo.longitude, geo.latitude)) \
            .filterDate('2020-01-01', '2020-12-31') \
            .filter(ee.Filter.eq('instrumentMode', 'IW'))

        def get_band(img, preferred="VV", fallback="HH"):
            return ee.Image(ee.Algorithms.If(
                img.bandNames().contains(preferred),
                img.select(preferred),
                img.select(fallback)
            ))

        pre = sentinel1.filterDate('2020-01-01', '2020-06-01').mean()
        post = sentinel1.filterDate('2020-07-01', '2020-12-31').mean()

        pre_band = get_band(pre)
        post_band = get_band(post)

        flood = ee.Image(post_band).subtract(pre_band).gt(0.3)
        vis = {'palette': ['red', 'yellow', 'green'], 'min': 0, 'max': 1}
        m.addLayer(flood, vis, "Flood Risk")

        flood_hazard = ee.Image('JRC/GSW1_3/GlobalSurfaceWater').select('recurrence')
        hazard_vis = {'min': 0, 'max': 100, 'palette': ['#0000FF', '#00FF00']}
        m.addLayer(flood_hazard, hazard_vis, "Flood Hazard")

    elif analysis == "site_suitability":
        elev = ee.Image('USGS/SRTMGL1_003')
        ndvi = ee.ImageCollection('LANDSAT/LC08/C02/T1_L2').filterDate('2018', '2020').median().normalizedDifference(['SR_B5', 'SR_B4'])
        flood = ee.Image('JRC/GSW1_3/GlobalSurfaceWater').select('recurrence')
        suit = elev.lte(500).And(ndvi.gte(0.2)).And(flood.lte(30))
        vis = {'min': 0, 'max': 1, 'palette': ['red', 'yellow', 'green']}
        m.addLayer(suit, vis, "Site Suitability")

    elif analysis == "land_cover_monitoring":
        landsat = ee.ImageCollection('LANDSAT/LC08/C02/T1_L2').filterDate('2018', '2020').median()
        ndvi = landsat.normalizedDifference(['SR_B5', 'SR_B4']).rename('NDVI')
        training = ee.FeatureCollection([
            ee.Feature(ee.Geometry.Point([-120, 40]), {'landcover': 0}),
            ee.Feature(ee.Geometry.Point([15, 48]), {'landcover': 1}),
            ee.Feature(ee.Geometry.Point([100, 15]), {'landcover': 2}),
            ee.Feature(ee.Geometry.Point([-90, -10]), {'landcover': 3})
        ])
        training_data = landsat.addBands(ndvi).sampleRegions(collection=training, properties=['landcover'], scale=1000)
        classifier = ee.Classifier.smileRandomForest(10).train(training_data, 'landcover', ['SR_B2', 'SR_B3', 'SR_B4', 'SR_B5', 'NDVI'])
        classified = landsat.addBands(ndvi).classify(classifier)
        vis = {'min': 0, 'max': 3, 'palette': ['#006400', '#ffbb22', '#f096ff', '#fa0000']}
        m.addLayer(classified, vis, "LC Classification")

    return m

# --- Session State ---
if "messages" not in st.session_state:
    st.session_state.messages = []
if "ollama_ready" not in st.session_state:
    st.session_state.ollama_ready = False
if "location_name" not in st.session_state:
    st.session_state.location_name = "Mumbai, India"
if "analysis_type" not in st.session_state:
    st.session_state.analysis_type = "flood_risk"

# --- Mistral-Ollama Chain of Thought Assistant ---
class GISChatbot:
    def __init__(self):
        self.ollama_url = "http://localhost:11434/api/generate"
        self.model_name = "mistral"

    def check_ollama(self):
        try:
            return requests.get("http://localhost:11434").status_code == 200
        except requests.ConnectionError:
            return False

    def rag_prompt(self, query):
        context_docs = "QGIS, GDAL, GeoPandas, RasterIO, and Earth Engine APIs"
        system_prompt = """
You are a GIS expert. From the user query, generate:
1. Chain of Thought (CoT) Reasoning: Explain the steps to process the query, including parameter extraction and workflow generation.
2. JSON workflow using tools: GeoPandas, RasterIO, ee.Image, GDAL.
Return both in a structured format.
"""
        full_prompt = f"{context_docs}\n\n{system_prompt}\n\nUser: {query}"
        return self.call_model(full_prompt)

    def call_model(self, prompt):
        payload = {
            "model": self.model_name,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": 0.7, "num_predict": 512, "top_p": 0.9, "top_k": 50}
        }
        try:
            res = requests.post(self.ollama_url, json=payload)
            res.raise_for_status()
            result = json.loads(res.text)
            return result.get("response", "No response from model.")
        except requests.RequestException as e:
            return f"Error calling Mistral: {e}"

# --- Initialize Bot ---
if "chatbot" not in st.session_state:
    st.session_state.chatbot = GISChatbot()
if st.session_state.chatbot.check_ollama():
    st.session_state.ollama_ready = True
else:
    st.warning("Ollama is not running. Start with: `ollama run mistral`")

# --- Sidebar Configuration ---
with st.sidebar:
    st.header("⚙️ Configuration")
    st.text_input("Ollama Model", value="mistral", key="model_name")
    st.write(f"📀 RAM Available: {psutil.virtual_memory().available / 1e9:.2f} GB")
    st.divider()

    st.header("📍 Map Controls")
    location_name = st.text_input("Enter Location", value=st.session_state.location_name)
    analysis_options = [
        "landsat", "ndvi", "elevation", "landcover",
        "flood_risk", "site_suitability", "land_cover_monitoring"
    ]
    analysis_type = st.selectbox("Choose Analysis", analysis_options, index=analysis_options.index(st.session_state.analysis_type))
    run_map = st.button("🛱 Generate Map")

# --- Left Panel: LLM Chat ---
st.header("🧠 GIS Assistant (Mistral)")
for msg in st.session_state.messages:
    with st.chat_message(msg['role']):
        st.markdown(msg['content'])
        if msg['role'] == 'assistant' and 'map_html' in msg:
            st.components.v1.html(msg['map_html'], height=650, scrolling=True)
            with open("last_map.html", "rb") as f:
                st.download_button("📂 Download Map as HTML", f, file_name="map.html")

# --- Query Processing with Loading Button ---
query = st.chat_input("Ask geospatial question (e.g., 'Show flood risk for Mumbai')...")
if query:
    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.markdown(query)

    with st.chat_message("assistant"):
        # --- Loading Button for Mistral ---
        with st.spinner("Processing with Mistral..."):
            output = st.session_state.chatbot.rag_prompt(query)
            st.markdown("### Mistral Response")
            st.markdown(output)

        # --- Extract Location and Analysis Type ---
        query_lower = query.lower()
        analysis_keywords = {
            "landsat": "landsat",
            "ndvi": "ndvi",
            "elevation": "elevation",
            "land cover": "landcover",
            "flood": "flood_risk",
            "site suitability": "site_suitability",
            "land cover monitoring": "land_cover_monitoring"
        }

        matched_analysis = None
        for keyword, analysis in analysis_keywords.items():
            if keyword in query_lower:
                matched_analysis = analysis
                break

        location_match = re.search(r"(?:in|for|of|map of)\s+([a-zA-Z ,]+)", query_lower)
        extracted_location = location_match.group(1).strip() if location_match else None
        if not extracted_location and len(query_lower.split()) > 2:
            extracted_location = " ".join(query_lower.split()[-2:])

        # --- Generate Visualization if Parameters are Extracted ---
        if matched_analysis and extracted_location:
            st.session_state.location_name = extracted_location
            st.session_state.analysis_type = matched_analysis
            st.subheader(f"🌐 Visualizing: {matched_analysis.replace('_', ' ').title()} in {extracted_location.title()}")
            
            with st.spinner("Generating map visualization..."):
                result_map = get_map(extracted_location, matched_analysis)
                if result_map:
                    map_html = result_map._repr_html_()
                    Path("last_map.html").write_text(map_html, encoding="utf-8")
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": f"🗾️ Here's your map: **{matched_analysis.replace('_', ' ').title()} in {extracted_location.title()}**",
                        "map_html": map_html
                    })
                    st.components.v1.html(map_html, height=650, scrolling=True)
                    with open("last_map.html", "rb") as f:
                        st.download_button("📂 Download Map as HTML", f, file_name="map.html")
        else:
            st.warning("⚠️ Could not extract location and analysis from your query. Please use the sidebar or clarify your query (e.g., 'Show flood risk for Mumbai').")

# --- Execute Map (from Sidebar) ---
if run_map:
    st.subheader(f"🌐 Visualizing: {analysis_type.replace('_', ' ').title()} in {location_name.title()}")
    with st.spinner("Generating map visualization..."):
        result_map = get_map(location_name, analysis_type)
        if result_map:
            map_html = result_map._repr_html_()
            Path("last_map.html").write_text(map_html, encoding="utf-8")
            st.components.v1.html(map_html, height=650, scrolling=True)
            with open("last_map.html", "rb") as f:
                st.download_button("📂 Download Map as HTML", f, file_name="map.html")