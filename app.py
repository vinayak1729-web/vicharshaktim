import streamlit as st
import requests
import json
import warnings
warnings.filterwarnings("ignore")

# Set page config
st.set_page_config(
    page_title="Mistral GIS Assistant with Ollama",
    page_icon="🗺️",
    layout="wide"
)

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []
if "ollama_ready" not in st.session_state:
    st.session_state.ollama_ready = False

class MistralOllamaChatbot:
    def __init__(self):
        self.ollama_url = "http://localhost:11434/api/generate"
        self.model_name = "mistral"
    
    def check_ollama(self):
        """Check if Ollama server is running"""
        try:
            response = requests.get("http://localhost:11434")
            if response.status_code == 200:
                return True
            return False
        except requests.ConnectionError:
            return False
    
    def generate_response(self, user_input, max_length=512, temperature=0.7):
        """Generate response using Ollama's API"""
        try:
            # Flood workflow system prompt
            system_prompt = """You are a helpful AI assistant specialized in GIS and geospatial analysis, with a focus on flood management in India. You can assist with:
- Geographic Information Systems (GIS) operations
- Spatial data analysis
- OpenStreetMap data processing
- QGIS, GDAL, and other geospatial tools
- Map creation and visualization
- Flood management workflows, including:
  1. Monitoring and early warning systems (river levels, rainfall, satellite imagery)
  2. Risk assessment and flood mapping
  3. Emergency response planning (evacuation, shelters, supplies)
  4. Training and capacity building
  5. Rapid deployment of resources
  6. Communication and coordination
  7. Post-flood recovery and rehabilitation
  8. Review and improvement of flood response

Provide clear, actionable responses tailored to geospatial and flood management tasks."""
            
            # Prepare prompt
            prompt = f"{system_prompt}\n\nUser: {user_input}"
            
            # Ollama API payload
            payload = {
                "model": self.model_name,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": temperature,
                    "num_predict": max_length,
                    "top_p": 0.9,
                    "top_k": 50
                }
            }
            
            # Send request to Ollama
            response = requests.post(self.ollama_url, json=payload)
            response.raise_for_status()
            
            # Parse response
            result = json.loads(response.text)
            return result.get("response", "Error: No response from model").strip()
            
        except Exception as e:
            return f"Error generating response: {str(e)}"

# Initialize chatbot
if "chatbot" not in st.session_state:
    st.session_state.chatbot = MistralOllamaChatbot()

# Check Ollama server status
if st.session_state.chatbot.check_ollama():
    st.session_state.ollama_ready = True
else:
    st.error("Ollama server not running. Please start Ollama with 'ollama run mistral'.")

# Main UI
st.title("🗺️ Mistral GIS Assistant with Ollama")
st.subheader("Your AI-powered geospatial and flood management companion")

# Sidebar for configuration
with st.sidebar:
    st.header("⚙️ Configuration")
    
    # Model name input
    model_name = st.text_input(
        "Ollama Model Name",
        value="mistral",
        help="Name of the model loaded in Ollama (e.g., mistral)"
    )
    st.session_state.chatbot.model_name = model_name
    
    # Model status
    if st.session_state.ollama_ready:
        st.success("✅ Ollama Ready")
    else:
        st.warning("⏳ Ollama Not Running")
    
    st.divider()
    
    # Generation parameters
    st.header("🎛️ Generation Settings")
    temperature = st.slider("Temperature", 0.1, 1.0, 0.7, 0.1)
    max_length = st.slider("Max Response Length", 128, 1024, 512, 64)
    
    st.divider()
    
    # System info
    st.header("💻 System Info")
    st.write(f"**Device:** CPU")
    import psutil
    st.write(f"**Available RAM:** {psutil.virtual_memory().available / 1024**3:.1f} GB")

# Main chat interface
col1, col2 = st.columns([3, 1])

with col1:
    # Display chat messages
    chat_container = st.container()
    with chat_container:
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.write(message["content"])
    
    # Chat input
    if prompt := st.chat_input("Ask about GIS, flood management, or spatial analysis..."):
        if not st.session_state.ollama_ready:
            st.error("Ollama server not running. Please start Ollama first!")
        else:
            # Add user message
            st.session_state.messages.append({"role": "user", "content": prompt})
            
            # Display user message
            with st.chat_message("user"):
                st.write(prompt)
            
            # Generate and display assistant response
            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    response = st.session_state.chatbot.generate_response(
                        prompt, 
                        max_length=max_length, 
                        temperature=temperature
                    )
                st.write(response)
            
            # Add assistant message
            st.session_state.messages.append({"role": "assistant", "content": response})

with col2:
    st.header("🛠️ Quick Actions")
    
    if st.button("🗑️ Clear Chat"):
        st.session_state.messages = []
        st.rerun()
    
    if st.button("💾 Save Chat"):
        if st.session_state.messages:
            chat_text = "\n\n".join([
                f"**{msg['role'].title()}:** {msg['content']}" 
                for msg in st.session_state.messages
            ])
            st.download_button(
                "📥 Download Chat",
                chat_text,
                "mistral_chat.txt",
                "text/plain"
            )
    
    st.divider()
    
    st.header("📋 Sample Questions")
    sample_questions = [
        "How do I load OSM data in Python for flood mapping?",
        "Explain GDAL raster operations for flood risk assessment",
        "What is spatial indexing and its role in flood management?",
        "How to create a flood buffer in PostGIS?",
        "How to integrate ISRO satellite data for flood monitoring?"
    ]
    
    for question in sample_questions:
        if st.button(f"💬 {question}", key=question):
            if st.session_state.ollama_ready:
                # Append user message
                st.session_state.messages.append({"role": "user", "content": question})
                
                # Generate and display assistant response
                with st.chat_message("assistant"):
                    with st.spinner("Thinking..."):
                        response = st.session_state.chatbot.generate_response(question, max_length=max_length, temperature=temperature)
                    st.write(response)
                
                # Append assistant message
                st.session_state.messages.append({"role": "assistant", "content": response})

# Footer
st.divider()
st.markdown("""
**Next Steps:**
1. Ensure Ollama is running (`ollama run mistral`)
2. Start chatting about GIS and flood management
3. Integrate OSM data processing or ISRO satellite data in the next version

**Data Available:**
- Model: `mistral` (via Ollama)
- OSM Data: `india-latest.osm.pbf`

**Note:** Ensure at least 8GB RAM for Mistral via Ollama. For faster CPU performance, Ollama optimizes inference automatically.
""")