
import ee
import folium
import geemap.foliumap as geemap
from flask import Flask, render_template, jsonify, request
import uuid
import logging

app = Flask(__name__)

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Initialize Earth Engine
try:
    ee.Initialize(project='gdgadk')
    logger.info("GEE initialized successfully for project gdgadk")
except Exception as e:
    logger.error(f"Error initializing GEE: {e}")

# Default map center (global view)
MAP_CENTER = [0, 0]  # Center of the world
MAP_ZOOM = 2  # Zoom level for global view

def create_map():
    """Create a base Folium map."""
    m = geemap.Map(center=MAP_CENTER, zoom=MAP_ZOOM)
    m.add_basemap('Esri.WorldImagery')
    return m

@app.route('/')
def index():
    """Render the main page with the map."""
    m = create_map()
    map_html = m._repr_html_()
    return render_template('index.html', map_html=map_html)

@app.route('/gee_data', methods=['POST'])
def gee_data():
    """Handle GEE data requests based on button actions."""
    action = request.form.get('action')
    logger.debug(f"Received action: {action}")
    try:
        m = create_map()
        if action == 'landsat':
            landsat = ee.ImageCollection('LANDSAT/LC08/C02/T1_L2') \
                      .filterDate('2015-01-01', '2020-12-31') \
                      .filter(ee.Filter.lt('CLOUD_COVER', 20)) \
                      .median()
            vis_params = {'bands': ['SR_B4', 'SR_B3', 'SR_B2'], 'min': 0, 'max': 0.3}
            m.addLayer(landsat, vis_params, 'Landsat 8')
        elif action == 'ndvi':
            landsat = ee.ImageCollection('LANDSAT/LC08/C02/T1_L2') \
                      .filterDate('2015-01-01', '2020-12-31') \
                      .filter(ee.Filter.lt('CLOUD_COVER', 20)) \
                      .median()
            ndvi = landsat.normalizedDifference(['SR_B5', 'SR_B4']).rename('NDVI')
            vis_params = {'min': -1, 'max': 1, 'palette': ['blue', 'white', 'green']}
            m.addLayer(ndvi, vis_params, 'NDVI')
        elif action == 'elevation':
            elevation = ee.Image('USGS/SRTMGL1_003')
            vis_params = {'min': 0, 'max': 4000, 'palette': ['blue', 'green', 'brown']}
            m.addLayer(elevation, vis_params, 'Elevation')
        elif action == 'landcover':
            landcover = ee.Image('ESA/WorldCover/v100/2020')
            vis_params = {'bands': ['Map'], 'palette': ['#006400', '#ffbb22', '#ffff4c', '#f096ff', '#fa0000']}
            m.addLayer(landcover, vis_params, 'Land Cover')
        elif action == 'flood_risk':
            sentinel1 = ee.ImageCollection('COPERNICUS/S1_GRD') \
                         .filterDate('2020-01-01', '2020-12-31') \
                         .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VV')) \
                         .filter(ee.Filter.eq('instrumentMode', 'IW'))
            pre_flood = sentinel1.filterDate('2020-01-01', '2020-06-01').mean()
            post_flood = sentinel1.filterDate('2020-07-01', '2020-12-31').mean()
            diff = post_flood.select('VV').subtract(pre_flood.select('VV')).gt(0.3)
            vis_params = {'palette': ['red', 'yellow', 'green'], 'min': 0, 'max': 1}
            m.addLayer(diff, vis_params, 'Flood Risk')
            flood_hazard = ee.Image('JRC/GSW1_3/GlobalSurfaceWater').select('recurrence')
            m.addLayer(flood_hazard, {'min': 0, 'max': 100, 'palette': ['#0000FF', '#00FF00']}, 'Flood Hazard')
        elif action == 'site_suitability':
            elevation = ee.Image('USGS/SRTMGL1_003')
            landsat = ee.ImageCollection('LANDSAT/LC08/C02/T1_L2') \
                      .filterDate('2015-01-01', '2020-12-31') \
                      .filterBounds(ee.Geometry.Rectangle([77, 12, 78, 13])) \
                      .filter(ee.Filter.lt('CLOUD_COVER', 20)) \
                      .median()
            ndvi = landsat.normalizedDifference(['SR_B5', 'SR_B4']).rename('NDVI')
            flood_hazard = ee.Image('JRC/GSW1_3/GlobalSurfaceWater').select('recurrence')
            # Ensure data validity with mask
            elevation_mask = elevation.lte(500).rename('elevation_suit')
            ndvi_mask = ndvi.gt(0.2).rename('ndvi_suit')
            flood_mask = flood_hazard.lte(50).rename('flood_suit')
            suitability = elevation_mask.addBands(ndvi_mask).addBands(flood_mask).reduce('sum').multiply(100).divide(3)
            vis_params = {'min': 0, 'max': 100, 'palette': ['red', 'yellow', 'green']}
            m.addLayer(suitability, vis_params, 'Site Suitability')

        elif action == 'land_cover_monitoring':
            # Global land cover classification
            landsat = ee.ImageCollection('LANDSAT/LC08/C02/T1_L2') \
                      .filterDate('2015-01-01', '2020-12-31') \
                      .filter(ee.Filter.lt('CLOUD_COVER', 20))
            median_image = landsat.median()
            def addIndices(image):
                ndvi = image.normalizedDifference(['SR_B5', 'SR_B4']).rename('NDVI')
                return image.addBands(ndvi)
            processed_image = addIndices(median_image)
            # Placeholder global training data (replace with actual data)
            training_regions = ee.FeatureCollection([
                ee.Feature(ee.Geometry.Point([-120, 40]), {'landcover': 0}),  # Forest (California)
                ee.Feature(ee.Geometry.Point([15, 48]), {'landcover': 1}),    # Cropland (France)
                ee.Feature(ee.Geometry.Point([100, 15]), {'landcover': 2}),   # Urban (Thailand)
                ee.Feature(ee.Geometry.Point([-90, -10]), {'landcover': 3}),  # Rainforest (Amazon)
            ])
            training_data = processed_image.sampleRegions(
                collection=training_regions, properties=['landcover'], scale=1000
            )
            classifier = ee.Classifier.smileRandomForest(10).train(
                features=training_data, classProperty='landcover', inputProperties=['SR_B2', 'SR_B3', 'SR_B4', 'SR_B5', 'NDVI']
            )
            classified = processed_image.classify(classifier)
            vis_params = {'min': 0, 'max': 3, 'palette': ['#006400', '#ffbb22', '#f096ff', '#fa0000']}
            m.addLayer(classified, vis_params, 'Land Cover Classification')
        map_id = str(uuid.uuid4())
        map_html = m._repr_html_()
        logger.debug(f"Successfully generated map HTML for action: {action}")
        return jsonify({'map_html': map_html, 'map_id': map_id})
    except AttributeError as e:
        logger.error(f"Attribute error in gee_data for action {action}: {e}")
        return jsonify({'error': str(e)}), 500
    except Exception as e:
        logger.error(f"Error in gee_data for action {action}: {e}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)