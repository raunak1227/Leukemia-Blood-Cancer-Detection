import streamlit as st
import numpy as np
import cv2
from tensorflow.keras.models import load_model
from PIL import Image
import time

# ============================================================================
# PAGE CONFIGURATION
# ============================================================================

st.set_page_config(
    page_title="Leukemia Detection AI",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================================
# CUSTOM CSS FOR BEAUTIFUL UI
# ============================================================================

st.markdown("""
    <style>
    /* Main background gradient */
    .stApp {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    }
    
    /* Header styling */
    .main-header {
        background: linear-gradient(120deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 20px;
        text-align: center;
        box-shadow: 0 10px 30px rgba(0,0,0,0.3);
        margin-bottom: 2rem;
        animation: fadeIn 1s ease-in;
    }
    
    .main-header h1 {
        color: white;
        font-size: 3rem;
        font-weight: 800;
        margin: 0;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
    }
    
    .main-header p {
        color: #f0f0f0;
        font-size: 1.3rem;
        margin-top: 0.5rem;
    }
    
    /* Card styling */
    .card {
        background: white;
        padding: 2rem;
        border-radius: 20px;
        box-shadow: 0 10px 40px rgba(0,0,0,0.2);
        margin: 1rem 0;
        transition: transform 0.3s ease, box-shadow 0.3s ease;
    }
    
    .card:hover {
        transform: translateY(-5px);
        box-shadow: 0 15px 50px rgba(0,0,0,0.3);
    }
    
    /* Result cards with gradient backgrounds */
    .result-card-success {
        background: linear-gradient(135deg, #a8edea 0%, #fed6e3 100%);
        padding: 2rem;
        border-radius: 20px;
        border-left: 5px solid #10b981;
        box-shadow: 0 8px 25px rgba(16, 185, 129, 0.3);
        margin: 1rem 0;
        animation: slideInRight 0.5s ease;
    }
    
    .result-card-danger {
        background: linear-gradient(135deg, #ffecd2 0%, #fcb69f 100%);
        padding: 2rem;
        border-radius: 20px;
        border-left: 5px solid #ef4444;
        box-shadow: 0 8px 25px rgba(239, 68, 68, 0.3);
        margin: 1rem 0;
        animation: slideInRight 0.5s ease;
    }
    
    /* Animated buttons */
    .stButton>button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        padding: 0.8rem 2rem;
        border-radius: 50px;
        font-size: 1.1rem;
        font-weight: 600;
        cursor: pointer;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.4);
    }
    
    .stButton>button:hover {
        transform: translateY(-3px);
        box-shadow: 0 8px 25px rgba(102, 126, 234, 0.6);
    }
    
    /* Progress bar styling */
    .stProgress > div > div > div {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
    }
    
    /* Sidebar styling */
    .css-1d391kg {
        background: linear-gradient(180deg, #667eea 0%, #764ba2 100%);
    }
    
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #667eea 0%, #764ba2 100%);
    }
    
    section[data-testid="stSidebar"] .element-container {
        color: white;
    }
    
    /* Info boxes */
    .info-box {
        background: linear-gradient(135deg, #e0c3fc 0%, #8ec5fc 100%);
        padding: 1.5rem;
        border-radius: 15px;
        border-left: 4px solid #667eea;
        margin: 1rem 0;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.2);
    }
    
    /* Metric cards */
    .metric-card {
        background: white;
        padding: 1.5rem;
        border-radius: 15px;
        text-align: center;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
        margin: 0.5rem;
        transition: transform 0.3s ease;
    }
    
    .metric-card:hover {
        transform: scale(1.05);
    }
    
    .metric-value {
        font-size: 2.5rem;
        font-weight: 800;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }
    
    .metric-label {
        color: #64748b;
        font-size: 1rem;
        margin-top: 0.5rem;
        font-weight: 500;
    }
    
    /* Animations */
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(-20px); }
        to { opacity: 1; transform: translateY(0); }
    }
    
    @keyframes slideInRight {
        from { opacity: 0; transform: translateX(50px); }
        to { opacity: 1; transform: translateX(0); }
    }
    
    @keyframes pulse {
        0%, 100% { transform: scale(1); }
        50% { transform: scale(1.05); }
    }
    
    /* Upload section styling */
    .uploadedFile {
        border-radius: 15px;
        border: 2px dashed #667eea;
        padding: 2rem;
    }
    
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    
    /* Custom scrollbar */
    ::-webkit-scrollbar {
        width: 10px;
    }
    
    ::-webkit-scrollbar-track {
        background: #f1f1f1;
    }
    
    ::-webkit-scrollbar-thumb {
        background: linear-gradient(180deg, #667eea 0%, #764ba2 100%);
        border-radius: 10px;
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background: #764ba2;
    }
    </style>
""", unsafe_allow_html=True)

# ============================================================================
# LOAD MODELS
# ============================================================================

@st.cache_resource(show_spinner=False)
def load_models():
    """Load models with caching"""
    try:
        binary_model = load_model("binary.keras")
        binary_status = "✅ Loaded"
    except:
        binary_model = None
        binary_status = "❌ Not Found"
    
    try:
        multiclass_model = load_model("multiscale_cnn_lstm_final.keras")
        multiclass_status = "✅ Loaded"
    except:
        multiclass_model = None
        multiclass_status = "❌ Not Found"
    
    return binary_model, multiclass_model, binary_status, multiclass_status

binary_model, multiclass_model, bin_status, multi_status = load_models()

# ============================================================================
# CONFIGURATION
# ============================================================================

IMG_SIZE = (128, 128)
STAGES = ["Benign", "Early", "Pre", "Pro"]

STAGE_DESCRIPTIONS = {
    "Benign": "🟢 Less aggressive form of leukemia with better prognosis",
    "Early": "🟡 Initial phase - Early intervention recommended",
    "Pre": "🟠 Intermediate phase - Close monitoring required",
    "Pro": "🔴 Advanced phase - Immediate treatment necessary"
}

# ============================================================================
# PREDICTION FUNCTIONS
# ============================================================================

def preprocess_image(image):
    """Preprocess uploaded image"""
    img = image.convert("RGB")
    img = np.array(img)
    img = cv2.resize(img, IMG_SIZE)
    img = img / 255.0
    img = np.expand_dims(img, axis=0)
    return img

def detect_leukemia(image):
    """Two-stage detection"""
    img_array = preprocess_image(image)
    
    # Stage 1: Binary
    binary_pred = binary_model.predict(img_array, verbose=0)[0][0]
    
    if binary_pred >= 0.5:
        return {
            'status': 'normal',
            'confidence': binary_pred * 100,
            'stage': None,
            'stage_confidence': None,
            'probabilities': None
        }
    else:
        # Stage 2: Multiclass
        multiclass_pred = multiclass_model.predict(img_array, verbose=0)[0]
        stage_idx = np.argmax(multiclass_pred)
        
        return {
            'status': 'leukemia',
            'confidence': (1 - binary_pred) * 100,
            'stage': STAGES[stage_idx],
            'stage_confidence': multiclass_pred[stage_idx] * 100,
            'probabilities': {stage: prob * 100 for stage, prob in zip(STAGES, multiclass_pred)}
        }

# ============================================================================
# SIDEBAR
# ============================================================================

with st.sidebar:
    st.markdown("""
        <div style='text-align: center; padding: 1rem;'>
            <h1 style='color: white; font-size: 2rem;'>🔬</h1>
            <h2 style='color: white; font-size: 1.5rem; margin: 0;'>Leukemia AI</h2>
            <p style='color: #e0e0e0; font-size: 0.9rem;'>Powered by Deep Learning</p>
        </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    st.markdown("""
        <div style='color: white; padding: 1rem;'>
            <h3 style='color: white;'>📋 System Info</h3>
        </div>
    """, unsafe_allow_html=True)
    
    st.markdown(f"""
        <div style='background: rgba(255,255,255,0.1); padding: 1rem; border-radius: 10px; margin: 0.5rem 0;'>
            <p style='color: white; margin: 0;'><strong>Binary Model:</strong></p>
            <p style='color: #e0e0e0; margin: 0.5rem 0 0 0;'>{bin_status}</p>
        </div>
    """, unsafe_allow_html=True)
    
    st.markdown(f"""
        <div style='background: rgba(255,255,255,0.1); padding: 1rem; border-radius: 10px; margin: 0.5rem 0;'>
            <p style='color: white; margin: 0;'><strong>Multi-Class Model:</strong></p>
            <p style='color: #e0e0e0; margin: 0.5rem 0 0 0;'>{multi_status}</p>
        </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    st.markdown("""
        <div style='color: white; padding: 1rem;'>
            <h3 style='color: white;'>ℹ️ How It Works</h3>
            <ol style='color: #e0e0e0; line-height: 1.8;'>
                <li>Upload blood cell image</li>
                <li>AI analyzes the image</li>
                <li>Get instant results</li>
                <li>View confidence scores</li>
            </ol>
        </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    st.markdown("""
        <div style='background: rgba(255,255,255,0.1); padding: 1rem; border-radius: 10px; text-align: center;'>
            <p style='color: #e0e0e0; margin: 0.5rem 0 0 0; font-size: 0.75rem;'>Not a substitute for professional medical diagnosis</p>
        </div>
    """, unsafe_allow_html=True)

# ============================================================================
# MAIN CONTENT
# ============================================================================

# Header
st.markdown("""
    <div class='main-header'>
        <h1>🔬 Leukemia Detection System</h1>
        <p>AI-Powered Blood Cancer Detection & Classification</p>
    </div>
""", unsafe_allow_html=True)

# Check models
if not binary_model or not multiclass_model:
    st.error("⚠️ **Models not loaded!** Please ensure both model files are present.")
    st.stop()

# Main content area
col1, col2 = st.columns([1, 1], gap="large")

with col1:
    st.markdown("""
        <div class='card'>
            <h2 style='color: #1e3a8a; margin-top: 0;'>📤 Upload Blood Cell Image</h2>
        </div>
    """, unsafe_allow_html=True)
    
    uploaded_file = st.file_uploader(
        "Choose an image file",
        type=['png', 'jpg', 'jpeg'],
        help="Upload a microscopic blood cell image for analysis",
        label_visibility="collapsed"
    )
    
    if uploaded_file:
        image = Image.open(uploaded_file)
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.image(image, caption="📸 Uploaded Image", use_column_width=True)
        st.markdown("</div>", unsafe_allow_html=True)
        
        # Analyze button
        if st.button("🔍 Analyze Image", use_container_width=True):
            with col2:
                # Progress animation
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                status_text.text("🔄 Preprocessing image...")
                progress_bar.progress(25)
                time.sleep(0.3)
                
                status_text.text("🧠 Running AI models...")
                progress_bar.progress(50)
                time.sleep(0.5)
                
                # Get prediction
                result = detect_leukemia(image)
                
                status_text.text("📊 Analyzing results...")
                progress_bar.progress(75)
                time.sleep(0.3)
                
                progress_bar.progress(100)
                status_text.text("✅ Analysis complete!")
                time.sleep(0.5)
                
                progress_bar.empty()
                status_text.empty()
                
                # Display results
                if result['status'] == 'normal':
                    st.markdown("""
                        <div class='result-card-success'>
                            <h2 style='color: #10b981; margin: 0; font-size: 2rem;'>✅ Normal Blood Cells</h2>
                            <p style='color: #065f46; font-size: 1.2rem; margin-top: 0.5rem;'>No leukemia detected</p>
                        </div>
                    """, unsafe_allow_html=True)
                    
                    # Confidence metric
                    st.markdown(f"""
                        <div class='metric-card'>
                            <div class='metric-value'>{result['confidence']:.1f}%</div>
                            <div class='metric-label'>Detection Confidence</div>
                        </div>
                    """, unsafe_allow_html=True)
                    
                    st.success("🎉 Great news! The blood sample appears to be normal.")
                
                else:
                    st.markdown("""
                        <div class='result-card-danger'>
                            <h2 style='color: #ef4444; margin: 0; font-size: 2rem;'>⚠️ Leukemia Detected</h2>
                            <p style='color: #991b1b; font-size: 1.2rem; margin-top: 0.5rem;'>Further analysis recommended</p>
                        </div>
                    """, unsafe_allow_html=True)
                    
                    # Metrics row
                    met_col1, met_col2 = st.columns(2)
                    with met_col1:
                        st.markdown(f"""
                            <div class='metric-card'>
                                <div class='metric-value'>{result['confidence']:.1f}%</div>
                                <div class='metric-label'>Detection Confidence</div>
                            </div>
                        """, unsafe_allow_html=True)
                    
                    with met_col2:
                        st.markdown(f"""
                            <div class='metric-card'>
                                <div class='metric-value'>{result['stage']}</div>
                                <div class='metric-label'>Detected Stage</div>
                            </div>
                        """, unsafe_allow_html=True)
                    
                    # Stage info
                    st.markdown("<br>", unsafe_allow_html=True)
                    st.info(STAGE_DESCRIPTIONS[result['stage']])
                    
                    # Stage confidence
                    st.markdown(f"""
                        <div class='info-box'>
                            <h3 style='margin: 0; color: #1e3a8a;'>🎯 Stage Confidence</h3>
                            <p style='font-size: 1.5rem; font-weight: 700; margin: 0.5rem 0; color: #667eea;'>
                                {result['stage_confidence']:.1f}%
                            </p>
                        </div>
                    """, unsafe_allow_html=True)
                    
                    # Probability bars
                    st.markdown("<br>", unsafe_allow_html=True)
                    st.markdown("### 📊 Stage Probabilities")
                    
                    for stage, prob in result['probabilities'].items():
                        is_predicted = stage == result['stage']
                        color = "#ef4444" if is_predicted else "#3b82f6"
                        
                        st.markdown(f"""
                            <div style='margin: 1rem 0;'>
                                <div style='display: flex; justify-content: space-between; margin-bottom: 0.5rem;'>
                                    <span style='font-weight: {"700" if is_predicted else "500"}; color: {color};'>
                                        {"👉 " if is_predicted else ""}{stage}
                                    </span>
                                    <span style='font-weight: 600; color: {color};'>{prob:.1f}%</span>
                                </div>
                                <div style='background: #e2e8f0; border-radius: 10px; height: 20px; overflow: hidden;'>
                                    <div style='background: {color}; width: {prob}%; height: 100%; border-radius: 10px; transition: width 0.5s ease;'></div>
                                </div>
                            </div>
                        """, unsafe_allow_html=True)
                    
                    st.warning("⚠️ **Important:** Consult a medical professional for proper diagnosis and treatment.")

with col2:
    if not uploaded_file:
        st.markdown("""
            <div class='card' style='text-align: center; padding: 3rem;'>
                <h2 style='color: #667eea;'>👈 Upload an image to begin</h2>
                <p style='color: #64748b; font-size: 1.1rem; margin-top: 1rem;'>
                    Select a blood cell microscopic image from your device
                </p>
                <div style='margin-top: 2rem;'>
                    <img src='https://img.icons8.com/clouds/200/000000/upload.png' width='150'/>
                </div>
            </div>
        """, unsafe_allow_html=True)

# Footer
st.markdown("<br><br>", unsafe_allow_html=True)
st.markdown("""
    <div style='text-align: center; padding: 2rem; background: white; border-radius: 20px; box-shadow: 0 4px 15px rgba(0,0,0,0.1);'>
        <p style='color: #64748b; margin: 0; font-size: 0.9rem;'>
            🔬 <strong>Leukemia Detection System</strong> • Powered by Multi-Scale CNN-LSTM
        </p>
        <p style='color: #94a3b8; margin: 0.5rem 0 0 0; font-size: 0.8rem;'>
            Final Year Project • 2024 
        </p>
    </div>
""", unsafe_allow_html=True)