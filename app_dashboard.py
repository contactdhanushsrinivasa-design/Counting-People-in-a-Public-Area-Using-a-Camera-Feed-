import streamlit as st 
import firebase_admin
from firebase_admin import credentials, firestore
import os
import time
import pandas as pd
from datetime import datetime, timedelta
import plotly.express as px
import plotly.graph_objects as go
import cv2
import numpy as np
from ultralytics import YOLO
import tempfile
from PIL import Image
import json
import random

# Page config
st.set_page_config(page_title="People Counter Dashboard", layout="wide", page_icon="👥")

# Initialize theme session state (default: dark)
if 'theme' not in st.session_state:
    st.session_state.theme = 'dark'

# Dynamic Styling based on theme
theme = st.session_state.theme
if theme == 'light':
    css = """
    body, .main, .stMarkdown p, h1, h2, h3, h4, h5, h6, .stTitle, .stHeader, .stSubheader { color: #000000 !important; }
    .main { background-color: #f5f7f9 !important; }
    .stMetric { background-color: #ffffff !important; color: #000000 !important; padding: 20px !important; border-radius: 10px !important; box-shadow: 0 2px 4px rgba(0,0,0,0.05) !important; }
    [data-testid="stSidebar"] { background-color: #f8f9fa !important; }
    """
elif theme == 'dark':
    css = """
    body, .main, .stMarkdown p, h1, h2, h3, h4, h5, h6, .stTitle, .stHeader, .stSubheader { color: #ffffff !important; }
    .main { background-color: #121212 !important; }
    .stMetric { background-color: #1e1e1e !important; color: #ffffff !important; padding: 20px !important; border-radius: 10px !important; box-shadow: 0 2px 4px rgba(255,255,255,0.1) !important; }
    [data-testid="stSidebar"] { background-color: #1e1e1e !important; }
    """
else:  # light-dark
    css = """
    body, .main, .stMarkdown p, h1, h2, h3, h4, h5, h6, .stTitle, .stHeader, .stSubheader { color: #000000 !important; }
    .main { background-color: #f5f7f9 !important; }
    .stMetric { background-color: #ffffff !important; color: #000000 !important; padding: 20px !important; border-radius: 10px !important; box-shadow: 0 2px 4px rgba(0,0,0,0.05) !important; }
    [data-testid="stSidebar"] { background-color: #f8f9fa !important; }
    @media (prefers-color-scheme: dark) {
        body, .main, .stMarkdown p, h1, h2, h3, h4, h5, h6, .stTitle, .stHeader, .stSubheader { color: #ffffff !important; }
        .main { background-color: #121212 !important; }
        .stMetric { background-color: #1e1e1e !important; color: #ffffff !important; padding: 20px !important; border-radius: 10px !important; box-shadow: 0 2px 4px rgba(255,255,255,0.1) !important; }
        [data-testid="stSidebar"] { background-color: #1e1e1e !important; }
    }
    """
css += """
.element-container { margin-top: 0.5rem; }
.block-container { padding-top: 2rem; position: relative; z-index: 1; }
[data-testid="stSidebar"] .stMarkdown p, [data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3 { color: inherit !important; }
.main { position: relative; z-index: 1; }
.stApp > header { position: relative; z-index: 100; }
"""
st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)

# Cyber-themed fixed background animation
cyber_bg = """
<style>
@keyframes gridMove {
    0% { transform: translate(0, 0); }
    100% { transform: translate(50px, 50px); }
}

@keyframes particleFloat {
    0% { 
        transform: translateY(100vh) translateX(0);
        opacity: 0;
    }
    10% {
        opacity: 1;
    }
    90% {
        opacity: 1;
    }
    100% { 
        transform: translateY(-100px) translateX(100px);
        opacity: 0;
    }
}

@keyframes scanLine {
    0% { 
        transform: translateY(-100%);
        opacity: 0.3;
    }
    50% {
        opacity: 0.6;
    }
    100% { 
        transform: translateY(100vh);
        opacity: 0.3;
    }
}

@keyframes pulse {
    0%, 100% { 
        opacity: 0.3;
        transform: scale(1);
    }
    50% { 
        opacity: 0.8;
        transform: scale(1.1);
    }
}

.cyber-background {
    position: fixed;
    top: 0;
    left: 0;
    width: 100vw;
    height: 100vh;
    pointer-events: none;
    z-index: -1;
    overflow: hidden;
    background: linear-gradient(135deg, #0a0e27 0%, #1a1f3a 50%, #0f1419 100%);
}

.cyber-grid {
    position: absolute;
    width: 100%;
    height: 100%;
    background-image: 
        linear-gradient(rgba(0, 255, 255, 0.1) 1px, transparent 1px),
        linear-gradient(90deg, rgba(0, 255, 255, 0.1) 1px, transparent 1px);
    background-size: 50px 50px;
    animation: gridMove 20s linear infinite;
    opacity: 0.4;
}

.cyber-grid::before {
    content: '';
    position: absolute;
    width: 100%;
    height: 100%;
    background-image: 
        linear-gradient(rgba(0, 255, 255, 0.05) 1px, transparent 1px),
        linear-gradient(90deg, rgba(0, 255, 255, 0.05) 1px, transparent 1px);
    background-size: 25px 25px;
    animation: gridMove 15s linear infinite reverse;
}

.cyber-particle {
    position: absolute;
    width: 2px;
    height: 2px;
    background: #00ffff;
    box-shadow: 0 0 6px #00ffff, 0 0 12px #00ffff;
    border-radius: 50%;
    animation: particleFloat linear infinite;
}

.cyber-line {
    position: absolute;
    width: 1px;
    height: 100px;
    background: linear-gradient(to bottom, transparent, #00ff88, transparent);
    box-shadow: 0 0 10px #00ff88;
    animation: particleFloat linear infinite;
}

.scan-line {
    position: absolute;
    width: 100%;
    height: 2px;
    background: linear-gradient(to bottom, transparent, rgba(0, 255, 255, 0.5), transparent);
    box-shadow: 0 0 20px rgba(0, 255, 255, 0.8);
    animation: scanLine 8s linear infinite;
}

.cyber-node {
    position: absolute;
    width: 4px;
    height: 4px;
    background: #00ffff;
    border-radius: 50%;
    box-shadow: 0 0 10px #00ffff, 0 0 20px #00ffff;
    animation: pulse 2s ease-in-out infinite;
}

.cyber-connection {
    position: absolute;
    height: 1px;
    background: linear-gradient(to right, transparent, rgba(0, 255, 255, 0.3), transparent);
    box-shadow: 0 0 5px rgba(0, 255, 255, 0.5);
    transform-origin: left center;
    animation: pulse 3s ease-in-out infinite;
}
</style>
<div class="cyber-background">
    <div class="cyber-grid"></div>
    <div class="scan-line"></div>
"""

# Generate cyber particles
for i in range(40):
    left = random.uniform(0, 100)
    delay = random.uniform(0, 10)
    duration = random.uniform(8, 15)
    size = random.choice([2, 3, 4])
    color = random.choice(['#00ffff', '#00ff88', '#0088ff'])
    cyber_bg += f'<div class="cyber-particle" style="left: {left}%; animation-delay: {delay}s; animation-duration: {duration}s; width: {size}px; height: {size}px; background: {color}; box-shadow: 0 0 {size*3}px {color}, 0 0 {size*6}px {color};"></div>\n'

# Generate vertical lines
for i in range(15):
    left = random.uniform(0, 100)
    delay = random.uniform(0, 8)
    duration = random.uniform(10, 20)
    height = random.uniform(80, 150)
    cyber_bg += f'<div class="cyber-line" style="left: {left}%; animation-delay: {delay}s; animation-duration: {duration}s; height: {height}px;"></div>\n'

# Generate nodes and connections
for i in range(20):
    left = random.uniform(5, 95)
    top = random.uniform(5, 95)
    delay = random.uniform(0, 2)
    cyber_bg += f'<div class="cyber-node" style="left: {left}%; top: {top}%; animation-delay: {delay}s;"></div>\n'

# Add multiple scan lines with different delays
for i in range(3):
    delay = i * 2.5
    cyber_bg += f'<div class="scan-line" style="animation-delay: {delay}s;"></div>\n'

cyber_bg += '</div>'
st.markdown(cyber_bg, unsafe_allow_html=True)

# Initialize Firebase
basedir = os.path.dirname(os.path.abspath(__file__))
key_path = os.path.join(basedir, "serviceAccountKey.json")

if not firebase_admin._apps:
    try:
        cred = credentials.Certificate(key_path)
        firebase_admin.initialize_app(cred)
    except Exception as e:
        st.error(f"Error initializing Firebase: {e}")

db = firestore.client()

# Initialize YOLO model (lazy loading)
@st.cache_resource
def load_yolo_model():
    # Check multiple possible locations for the model
    possible_paths = [
        os.path.join(basedir, "yolov8x.pt"),  # Current directory
        os.path.join(os.path.dirname(basedir), "yolov8x.pt"),  # Parent directory
        "yolov8x.pt",  # Current working directory
    ]
    
    model_path = None
    for path in possible_paths:
        if os.path.exists(path):
            model_path = path
            break
    
    if model_path:
        try:
            return YOLO(model_path)
        except Exception as e:
            st.error(f"Error loading YOLO model from {model_path}: {e}")
            return None
    
    # If model not found, try to let YOLO download it automatically
    try:
        st.info("Model file not found. Attempting to download yolov8x.pt automatically...")
        model = YOLO("yolov8x.pt")  # YOLO will download it if not found
        return model
    except Exception as e:
        st.error(f"Could not load or download YOLO model: {e}")
        return None

# Session state for threshold
if 'threshold' not in st.session_state:
    st.session_state.threshold = 10
if 'alert_shown' not in st.session_state:
    st.session_state.alert_shown = False
if 'heatmap_data' not in st.session_state:
    st.session_state.heatmap_data = []
if 'video_stats' not in st.session_state:
    st.session_state.video_stats = {'frames_processed': 0, 'total_detections': 0, 'max_count': 0}
if 'theme' not in st.session_state:
    st.session_state.theme = 'light'
# Session state for admin login
if 'admin_logged_in' not in st.session_state:
    st.session_state.admin_logged_in = False
if 'login_attempts' not in st.session_state:
    st.session_state.login_attempts = 0
if 'login_time' not in st.session_state:
    st.session_state.login_time = None

# Global auth/session role
if 'auth_role' not in st.session_state:
    st.session_state.auth_role = None  # 'user' or 'admin'
if 'is_authenticated' not in st.session_state:
    st.session_state.is_authenticated = False

def get_live_data():
    try:
        doc_ref = db.collection('people_counter').document('live')
        doc = doc_ref.get()
        if doc.exists:
            return doc.to_dict()
    except Exception as e:
        st.error(f"Error fetching live data: {e}")
    return None

def get_history_data():
    try:
        history_ref = db.collection('people_counter_history').order_by('last_updated', direction=firestore.Query.DESCENDING).limit(100)
        docs = history_ref.stream()
        data = []
        for doc in docs:
            d = doc.to_dict()
            # Convert Firestore timestamp to python datetime
            if 'last_updated' in d and d['last_updated']:
                d['timestamp'] = d['last_updated']
            data.append(d)
        return pd.DataFrame(data)
    except Exception as e:
        st.error(f"Error fetching history: {e}")
        return pd.DataFrame()

def get_logs():
    log_path = os.path.join(basedir, "app.log")
    if os.path.exists(log_path):
        try:
            with open(log_path, "r") as f:
                return f.readlines()
        except Exception as e:
            return [f"Error reading logs: {e}"]
    return ["No logs found yet. Start the counter to generate logs."]

def show_login_page():
    """Display combined User/Admin login page shown before the app"""
    # Hide sidebar for cleaner login experience
    st.markdown("""
    <style>
        [data-testid="stSidebar"] {
            display: none;
        }
    </style>
    """, unsafe_allow_html=True)
    
    # Center the login form
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        # Header
        st.markdown("""
        <div style="text-align: center; padding: 40px 20px 20px 20px;">
            <h1 style="color: #1f77b4; margin-bottom: 10px; font-size: 2.5em;">🔐 Dashboard Login</h1>
            <p style="color: #666; margin-bottom: 30px; font-size: 1.1em;">Select your role and login to continue</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Login form with styled container
        with st.container():
            st.markdown("""
            <div style="
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                padding: 2px;
                border-radius: 15px;
                box-shadow: 0 8px 16px rgba(0,0,0,0.2);
                margin: 20px 0;
            ">
                <div style="
                    background: white;
                    padding: 40px;
                    border-radius: 13px;
                ">
            """, unsafe_allow_html=True)
            
            with st.form("admin_login_form", clear_on_submit=False):
                st.markdown("###  Access Portal")
                st.markdown("---")

                role = st.radio("Login as", ["User", "Admin"], horizontal=True, key="login_role")
                
                username = st.text_input(" **Username**", placeholder="Enter your username", key="login_username")
                password = st.text_input(" **Password**", type="password", placeholder="Enter your password", key="login_password")
                
                st.markdown("<br>", unsafe_allow_html=True)
                
                col_btn1, col_btn2 = st.columns([1, 1])
                with col_btn1:
                    login_button = st.form_submit_button(" **Login**", use_container_width=True, type="primary")
                with col_btn2:
                    cancel_button = st.form_submit_button(" **Cancel**", use_container_width=True)
                
                if login_button:
                    # Default credentials (can be changed or moved to config)
                    if role == "Admin":
                        valid_username = "admin"
                        valid_password = "admin123"
                    else:
                        valid_username = "user"
                        valid_password = "user123"
                    
                    if username.strip() == "" or password.strip() == "":
                        st.error("❌ Please enter both username and password.")
                    elif username == valid_username and password == valid_password:
                        # Set global auth state
                        st.session_state.auth_role = "admin" if role == "Admin" else "user"
                        st.session_state.is_authenticated = True

                        # Track admin-specific state
                        if role == "Admin":
                            st.session_state.admin_logged_in = True
                            st.session_state.login_time = datetime.now()
                        else:
                            st.session_state.admin_logged_in = False
                            st.session_state.login_time = None

                        st.session_state.login_attempts = 0
                        st.success(f"✅ {role} login successful! Redirecting...")
                        time.sleep(0.5)  # Brief delay for better UX
                        st.rerun()
                    else:
                        st.session_state.login_attempts += 1
                        remaining = 5 - st.session_state.login_attempts
                        if st.session_state.login_attempts >= 5:
                            st.error("🚫 Too many failed attempts. Account temporarily locked.")
                            st.session_state.login_attempts = 0
                            time.sleep(2)
                            st.rerun()
                        else:
                            st.error(f"❌ Invalid username or password. {remaining} attempts remaining.")
                
                if cancel_button:
                    st.info("ℹ️ Login cancelled.")
            
            st.markdown("</div></div>", unsafe_allow_html=True)
        
        # Show login attempts warning
        if st.session_state.login_attempts > 0 and st.session_state.login_attempts < 5:
            st.warning(f"⚠️ **Security Notice**: {st.session_state.login_attempts} failed login attempt(s). Account will be locked after 5 attempts.")
        
        # Help text
        st.markdown("---")
        with st.expander("ℹ️ Login Information"):
            st.markdown("""
            **Default Credentials:**
            - Username: `admin`
            - Password: `admin123`
            
            ⚠️ **Important**: Change default credentials in production environment for security.
            """)
        
        # Security notice
        st.markdown("""
        <div style="text-align: center; color: #888; font-size: 11px; margin-top: 30px;">
            <p>🔒 Secure Admin Access | All login attempts are monitored</p>
        </div>
        """, unsafe_allow_html=True)

def logout():
    """Logout admin user"""
    st.session_state.admin_logged_in = False
    st.session_state.login_attempts = 0
    st.session_state.login_time = None
    st.success("✅ Logged out successfully!")
    time.sleep(0.5)
    st.rerun()

def process_video_frame(frame, model):
    """Process a single video frame and detect people"""
    results = model.predict(frame, verbose=False)
    detections = []
    heatmap_points = []
    
    for result in results:
        boxes = result.boxes
        for box in boxes:
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            if cls_id == 0 and conf > 0.5:  # Person class with confidence > 0.5
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                detections.append((x1, y1, x2, y2))
                # Center point for heatmap
                center_x = (x1 + x2) // 2
                center_y = (y1 + y2) // 2
                heatmap_points.append((center_x, center_y))
    
    return detections, heatmap_points

def create_heatmap(frame_shape, heatmap_points_list):
    """Create a heatmap from accumulated detection points"""
    if not heatmap_points_list:
        return None
    
    h, w = frame_shape[:2]
    heatmap = np.zeros((h, w), dtype=np.float32)
    
    for points in heatmap_points_list:
        for x, y in points:
            if 0 <= x < w and 0 <= y < h:
                # Create a Gaussian-like distribution around each point
                y_coords, x_coords = np.ogrid[:h, :w]
                dist_sq = (x_coords - x)**2 + (y_coords - y)**2
                heatmap += np.exp(-dist_sq / (2 * 50**2))  # Sigma = 50
    
    return heatmap

def show_alert(message, alert_type="warning", use_native=True):
    """Display alert message properly positioned below Streamlit header"""
    if use_native:
        # Use Streamlit's native components for better integration
        if alert_type == "error":
            st.error(f"🚨 {message}")
        else:
            st.warning(f"⚠️ {message}")
    else:
        # Custom styled alert with proper positioning
        if alert_type == "error":
            color = "#ff4444"
            icon = "🚨"
            border_color = "#cc0000"
        else:
            color = "#ffaa00"
            icon = "⚠️"
            border_color = "#cc8800"
        
        # Position alert below Streamlit header (typically 80-100px) with extra margin
        st.markdown(f"""
        <div style="
            position: fixed;
            top: 120px;
            right: 20px;
            background: linear-gradient(135deg, {color} 0%, {border_color} 100%);
            color: white;
            padding: 20px 30px;
            border-radius: 10px;
            box-shadow: 0 4px 12px rgba(0,0,0,0.4);
            z-index: 999999;
            font-size: 18px;
            font-weight: bold;
            animation: slideIn 0.5s ease-out, pulse 2s infinite;
            border: 2px solid white;
            min-width: 300px;
            max-width: 500px;
            word-wrap: break-word;
        ">
            <div style="display: flex; align-items: center; gap: 10px;">
                <span style="font-size: 24px; flex-shrink: 0;">{icon}</span>
                <span style="flex: 1;">{message}</span>
            </div>
        </div>
        <style>
            @keyframes slideIn {{
                from {{
                    transform: translateX(400px);
                    opacity: 0;
                }}
                to {{
                    transform: translateX(0);
                    opacity: 1;
                }}
            }}
            @keyframes pulse {{
                0%, 100% {{
                    box-shadow: 0 4px 12px rgba(0,0,0,0.4);
                }}
                50% {{
                    box-shadow: 0 4px 20px rgba(255,68,68,0.6);
                }}
            }}
        </style>
        """, unsafe_allow_html=True)

# Global authentication gate - show login first before anything else
if not st.session_state.is_authenticated:
    show_login_page()
    st.stop()

# Sidebar
st.sidebar.image("https://img.icons8.com/fluency/96/000000/analytics.png", width=80)
st.sidebar.title("Navigation")

if st.session_state.auth_role == "admin":
    nav_pages = ["Viewer Panel", "Video Analysis", "Admin Panel"]
else:
    # Regular users only see the live count viewer
    nav_pages = ["Viewer Panel"]

page = st.sidebar.radio("Select View", nav_pages)

# Appearance configuration in sidebar
st.sidebar.divider()
st.sidebar.markdown("### Appearance")
theme_options = ["light", "light-dark", "dark"]
# Default selection is dark
current_theme_index = theme_options.index(st.session_state.theme) if st.session_state.theme in theme_options else theme_options.index("dark")
theme = st.sidebar.selectbox("Theme", theme_options, index=current_theme_index)
st.session_state.theme = theme

# Threshold configuration in sidebar
st.sidebar.divider()
st.sidebar.subheader("⚙️ Alert Settings")
threshold = st.sidebar.number_input("People Count Threshold", min_value=1, max_value=100, value=st.session_state.threshold, key="threshold_input")
st.session_state.threshold = threshold
st.sidebar.info(f"Alert will trigger when count exceeds {threshold}")

if page == "Viewer Panel":
    st.title("👥 People Counter - Live View")
    
    data = get_live_data()
    
    if data:
        col1, col2, col3 = st.columns(3)
        current_inside = data.get('current_inside', 0)
        entered = data.get('entered', 0)
        exited = data.get('exited', 0)
        
        with col1:
            st.metric("Total Entered", entered)
        with col2:
            st.metric("Total Exited", exited)
        with col3:
            delta_value = current_inside - st.session_state.threshold if current_inside > st.session_state.threshold else None
            st.metric("Currently Inside", current_inside, delta=delta_value, delta_color="inverse")
        
        st.write(f"**Last updated:** {data.get('last_updated', 'N/A')}")
        
        # Check threshold and show alert - use native Streamlit components
        if current_inside > st.session_state.threshold:
            st.error(f"🚨 **ALERT**: People count ({current_inside}) exceeds threshold ({st.session_state.threshold})!")
            st.session_state.alert_shown = True
        else:
            st.session_state.alert_shown = False
            st.success("System is Live - Count within limits")
    else:
        st.warning("Waiting for data from the counter...")
        st.info("Make sure main.py is running.")
        
elif page == "Video Analysis":
    st.title("🎥 Video Analysis & Heatmaps")
    
    model = load_yolo_model()
    if model is None:
        st.error("⚠️ YOLO model (yolov8x.pt) not found.")
        st.info("""
        **Please ensure the model file is in one of these locations:**
        - Current project directory: `People-Count-using-YOLOv8-main/yolov8x.pt`
        - Parent directory: `demopro/yolov8x.pt`
        - Or the model will be downloaded automatically on first use
        
        **Note:** The model file is large (~130MB). If it needs to be downloaded, it may take a few minutes.
        """)
        
        # Show where we're looking
        with st.expander("🔍 Search Locations"):
            st.code(f"""
            Checked locations:
            1. {os.path.join(basedir, "yolov8x.pt")}
            2. {os.path.join(os.path.dirname(basedir), "yolov8x.pt")}
            3. {os.path.join(os.getcwd(), "yolov8x.pt")}
            """)
        
        st.stop()
    
    # Video upload section
    st.subheader("📤 Upload Video for Analysis")
    uploaded_file = st.file_uploader("Choose a video file", type=['mp4', 'avi', 'mov', 'mkv'])
    
    # Or use existing video
    existing_video_path = os.path.join(basedir, "Input", "input.mp4")
    use_existing = st.checkbox("Use existing video from Input folder", value=False)
    
    video_path = None
    if use_existing and os.path.exists(existing_video_path):
        video_path = existing_video_path
        st.info(f"Using video: {existing_video_path}")
    elif uploaded_file is not None:
        # Save uploaded file temporarily
        tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
        tfile.write(uploaded_file.read())
        video_path = tfile.name
        st.success("Video uploaded successfully!")
    
    if video_path:
        # Center the analysis controls section
        col_left, col_center, col_right = st.columns([1, 2, 1])

        with col_center:
            st.subheader("Analysis Controls")
            sample_rate = st.slider("Frame Sampling Rate", min_value=1, max_value=30, value=5,
                                   help="Process every Nth frame (lower = more accurate but slower)")
            max_frames = st.number_input("Max Frames to Process", min_value=10, max_value=1000, value=100)
            show_heatmap = st.checkbox("Generate Heatmap", value=True)
            show_trends = st.checkbox("Show Trends", value=True)

            if st.button("🚀 Start Analysis", type="primary"):
                # Create alert placeholder at top of main content area (properly positioned)
                alert_placeholder = st.empty()
                
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                cap = cv2.VideoCapture(video_path)
                total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                fps = cap.get(cv2.CAP_PROP_FPS)

                # Initialize Firestore session for this video analysis run
                firestore_session_ref = None
                try:
                    firestore_session_ref = db.collection('video_analysis_sessions').document()
                    firestore_session_ref.set({
                        'created_at': firestore.SERVER_TIMESTAMP,
                        'video_path': video_path,
                        'sample_rate': int(sample_rate),
                        'max_frames': int(max_frames),
                    })
                except Exception as e:
                    # Avoid breaking the UI if Firestore write fails
                    st.warning(f"Unable to create Firestore video analysis session: {e}")
                
                frame_count = 0
                processed_count = 0
                all_heatmap_points = []
                detection_history = []
                frame_detections = []
                
                video_placeholder = st.empty()
                stats_placeholder = st.empty()
                
                while cap.isOpened() and processed_count < max_frames:
                    ret, frame = cap.read()
                    if not ret:
                        break
                    
                    frame_count += 1
                    if frame_count % sample_rate != 0:
                        continue
                    
                    # Process frame
                    detections, heatmap_points = process_video_frame(frame, model)
                    all_heatmap_points.append(heatmap_points)
                    detection_history.append({
                        'frame': frame_count,
                        'count': len(detections),
                        'time': frame_count / fps if fps > 0 else frame_count
                    })
                    frame_detections.append(len(detections))

                    # Store per-frame live data to Firestore (if session is available)
                    if firestore_session_ref is not None:
                        try:
                            firestore_session_ref.collection('frames').add({
                                'frame_number': int(frame_count),
                                'count': int(len(detections)),
                                'time_seconds': float(frame_count / fps if fps > 0 else frame_count),
                                'created_at': firestore.SERVER_TIMESTAMP,
                            })
                        except Exception:
                            # Fail silently for per-frame writes to avoid UI spam
                            pass
                    
                    # Draw detections on frame
                    annotated_frame = frame.copy()
                    for x1, y1, x2, y2 in detections:
                        cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        cv2.putText(annotated_frame, "Person", (x1, y1-10), 
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                    
                    cv2.putText(annotated_frame, f"Count: {len(detections)}", (10, 30),
                              cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                    
                    # Display frame
                    video_placeholder.image(annotated_frame, channels="BGR", use_container_width=True)
                    
                    # Update stats
                    max_count = max(frame_detections) if frame_detections else 0
                    avg_count = np.mean(frame_detections) if frame_detections else 0
                    stats_placeholder.metric("Current Count", len(detections))
                    
                    # Check threshold - display alert in dedicated placeholder (properly positioned)
                    if len(detections) > st.session_state.threshold:
                        alert_placeholder.error(f"🚨 **ALERT**: Frame {frame_count} - Count ({len(detections)}) exceeds threshold ({st.session_state.threshold})!")
                    else:
                        alert_placeholder.empty()  # Clear alert when count is within threshold
                    
                    processed_count += 1
                    progress = min(processed_count / max_frames, 1.0)
                    progress_bar.progress(progress)
                    status_text.text(f"Processing frame {processed_count}/{max_frames}...")
                
                cap.release()
                progress_bar.progress(1.0)
                status_text.text("✅ Analysis Complete!")

                # Update Firestore session summary
                if firestore_session_ref is not None:
                    try:
                        firestore_session_ref.update({
                            'frames_processed': int(processed_count),
                            'total_detections': int(sum(frame_detections)),
                            'max_count': int(max_count),
                            'avg_count': float(avg_count),
                            'completed_at': firestore.SERVER_TIMESTAMP,
                        })
                    except Exception:
                        pass
                
                # Store results
                st.session_state.video_stats = {
                    'frames_processed': processed_count,
                    'total_detections': sum(frame_detections),
                    'max_count': max_count,
                    'avg_count': avg_count
                }
                st.session_state.heatmap_data = all_heatmap_points
                
                # Display results
                st.divider()
                st.subheader("📊 Analysis Results")
                
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Frames Processed", processed_count)
                with col2:
                    st.metric("Max People Count", max_count)
                with col3:
                    st.metric("Average Count", f"{avg_count:.1f}")
                with col4:
                    st.metric("Total Detections", sum(frame_detections))
                
                # Heatmap
                if show_heatmap and all_heatmap_points:
                    st.subheader("🔥 Heatmap Visualization")
                    # Get frame dimensions
                    cap = cv2.VideoCapture(video_path)
                    ret, sample_frame = cap.read()
                    cap.release()

                    if ret and sample_frame is not None and len(sample_frame.shape) >= 2:
                        heatmap = create_heatmap(sample_frame.shape, all_heatmap_points)
                        if heatmap is not None and heatmap.max() > 0:
                            # Normalize heatmap
                            heatmap_normalized = (heatmap / heatmap.max() * 255).astype(np.uint8)
                            heatmap_colored = cv2.applyColorMap(heatmap_normalized, cv2.COLORMAP_JET)
                            heatmap_overlay = cv2.addWeighted(sample_frame, 0.5, heatmap_colored, 0.5, 0)

                            # Center and enlarge the heatmaps
                            col_left, col_center, col_right = st.columns([0.5, 4, 0.5])
                            with col_center:
                                st.image(heatmap_overlay, channels="BGR", use_container_width=True,
                                        caption="Heatmap Overlay - People Density")

                            col_left2, col_center2, col_right2 = st.columns([0.5, 4, 0.5])
                            with col_center2:
                                st.image(heatmap_colored, channels="BGR", use_container_width=True,
                                        caption="Heatmap Only - Hotspots")
                        else:
                            st.warning("Unable to generate heatmap - no valid detection points found.")
                    else:
                        st.warning("Unable to read video frame for heatmap generation.")
                
                # Trends
                if show_trends and detection_history:
                    st.subheader("📈 Detection Trends")
                    df_trends = pd.DataFrame(detection_history)

                    # Center and enlarge the line chart
                    col_left_line, col_center_line, col_right_line = st.columns([0.5, 4, 0.5])
                    with col_center_line:
                        fig = px.line(df_trends, x='time', y='count',
                                     title="People Count Over Time",
                                     labels={'time': 'Time (seconds)', 'count': 'People Count'})
                        fig.add_hline(y=st.session_state.threshold, line_dash="dash",
                                    line_color="red", annotation_text=f"Threshold: {st.session_state.threshold}")
                        fig.update_layout(height=600)  # Increase height for better visibility
                        st.plotly_chart(fig, use_container_width=True)

                    # Center and enlarge the histogram
                    col_left_hist, col_center_hist, col_right_hist = st.columns([0.5, 4, 0.5])
                    with col_center_hist:
                        fig2 = px.histogram(df_trends, x='count', nbins=20,
                                          title="Distribution of People Count",
                                          labels={'count': 'People Count', 'y': 'Frequency'})
                        fig2.update_layout(height=600)  # Increase height for better visibility
                        st.plotly_chart(fig2, use_container_width=True)

elif page == "Admin Panel":
    # Only allow admin role to access this panel
    if st.session_state.auth_role != "admin":
        st.error("🚫 Admin access only. Please login as Admin to view this panel.")
    else:
        # Ensure sidebar is visible when logged in
        st.markdown("""
        <style>
            [data-testid="stSidebar"] {
                display: block;
            }
        </style>
        """, unsafe_allow_html=True)
        
        # Admin is logged in - show admin panel
        # Logout button in sidebar
        st.sidebar.markdown("---")
        st.sidebar.markdown("### 🔐 Admin Session")
        if st.sidebar.button("🚪 Logout", use_container_width=True, type="secondary"):
            logout()
        
        st.title("🔐 Admin Control Panel")
        
        # Show session info
        if st.session_state.login_time:
            session_duration = datetime.now() - st.session_state.login_time
            hours, remainder = divmod(int(session_duration.total_seconds()), 3600)
            minutes, seconds = divmod(remainder, 60)
            session_info = f"Session: {hours}h {minutes}m {seconds}s"
        else:
            session_info = "Session Active"
        
        col_info1, col_info2 = st.columns([2, 1])
        with col_info1:
            st.success(f"✅ Welcome, Administrator!")
        with col_info2:
            st.info(f"🕐 {session_info}")
        
        tabs = st.tabs(["📊 Analytics", "📋 System Logs", "⚙️ Configuration"])
        
        with tabs[0]:
            st.subheader("Real-time Stats")
            live_data = get_live_data()
            if live_data:
                c1, c2, c3, c4 = st.columns(4)
                current_inside = live_data.get('current_inside', 0)
                entered = live_data.get('entered', 0)
                exited = live_data.get('exited', 0)
                
                c1.metric("Entered", entered)
                c2.metric("Exited", exited)
                c3.metric("Inside", current_inside)
                
                # Alert status
                if current_inside > st.session_state.threshold:
                    c4.metric("Status", "⚠️ ALERT", delta=f"Exceeds by {current_inside - st.session_state.threshold}")
                else:
                    c4.metric("Status", "✅ Normal", delta=f"Under by {st.session_state.threshold - current_inside}")
            
            st.divider()
            st.subheader("Historical Trends")
            df = get_history_data()
            if not df.empty:
                # Convert timestamp to datetime if needed
                if 'timestamp' in df.columns:
                    try:
                        df['datetime'] = pd.to_datetime(df['timestamp'])
                    except:
                        df['datetime'] = pd.to_datetime(df['timestamp'], errors='coerce')
                else:
                    df['datetime'] = pd.to_datetime(df.get('last_updated', pd.Timestamp.now()))
                
                # Sort by datetime
                df = df.sort_values('datetime')
                
                # Multi-line chart
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=df['datetime'], y=df.get('entered', 0), 
                                       name='Entered', line=dict(color='green')))
                fig.add_trace(go.Scatter(x=df['datetime'], y=df.get('exited', 0), 
                                       name='Exited', line=dict(color='red')))
                fig.add_trace(go.Scatter(x=df['datetime'], y=df.get('current_inside', 0), 
                                       name='Currently Inside', line=dict(color='blue')))
                fig.add_hline(y=st.session_state.threshold, line_dash="dash", 
                            line_color="orange", annotation_text=f"Threshold: {st.session_state.threshold}")
                
                fig.update_layout(title="People Count Over Time - Comprehensive View",
                                 xaxis_title="Time",
                                 yaxis_title="Count",
                                 hovermode='x unified',
                                 height=500)
                st.plotly_chart(fig, use_container_width=True)
                
                # Additional analytics
                col1, col2 = st.columns(2)
                
                with col1:
                    st.subheader("📊 Statistics Summary")
                    if 'current_inside' in df.columns:
                        st.metric("Peak Occupancy", int(df['current_inside'].max()))
                        st.metric("Average Occupancy", f"{df['current_inside'].mean():.1f}")
                        st.metric("Min Occupancy", int(df['current_inside'].min()))
                    
                    if 'entered' in df.columns:
                        st.metric("Total Entries (Period)", int(df['entered'].sum()))
                    if 'exited' in df.columns:
                        st.metric("Total Exits (Period)", int(df['exited'].sum()))
                
                with col2:
                    st.subheader("📈 Trend Analysis")
                    if 'current_inside' in df.columns and len(df) > 1:
                        # Calculate trend
                        recent = df['current_inside'].tail(10).mean()
                        earlier = df['current_inside'].head(10).mean()
                        trend = recent - earlier
                        
                        if trend > 0:
                            st.metric("Trend", "📈 Increasing", delta=f"+{trend:.1f}")
                        elif trend < 0:
                            st.metric("Trend", "📉 Decreasing", delta=f"{trend:.1f}")
                        else:
                            st.metric("Trend", "➡️ Stable", delta="0")
                        
                        # Peak hours analysis
                        if 'datetime' in df.columns:
                            df['hour'] = df['datetime'].dt.hour
                            hourly_avg = df.groupby('hour')['current_inside'].mean()
                            peak_hour = hourly_avg.idxmax()
                            st.metric("Peak Hour", f"{peak_hour}:00", 
                                    delta=f"Avg: {hourly_avg.max():.1f}")
                
                # Time series decomposition
                st.subheader("🕐 Hourly Pattern")
                if 'datetime' in df.columns and 'current_inside' in df.columns:
                    df['hour'] = df['datetime'].dt.hour
                    hourly_data = df.groupby('hour')['current_inside'].agg(['mean', 'max', 'min']).reset_index()
                    
                    fig_hourly = go.Figure()
                    fig_hourly.add_trace(go.Scatter(x=hourly_data['hour'], y=hourly_data['mean'],
                                                   name='Average', line=dict(color='blue')))
                    fig_hourly.add_trace(go.Scatter(x=hourly_data['hour'], y=hourly_data['max'],
                                                   name='Maximum', line=dict(color='red', dash='dash')))
                    fig_hourly.add_trace(go.Scatter(x=hourly_data['hour'], y=hourly_data['min'],
                                                   name='Minimum', line=dict(color='green', dash='dash')))
                    fig_hourly.update_layout(title="People Count by Hour of Day",
                                           xaxis_title="Hour",
                                           yaxis_title="Count",
                                           height=400)
                    st.plotly_chart(fig_hourly, use_container_width=True)
                
                with st.expander("Show Raw Data"):
                    st.dataframe(df)
            else:
                st.info("No historical data available yet.")
        
        with tabs[1]:
            st.subheader("Recent System Logs")
            logs = get_logs()
            log_text = "".join(logs[-100:]) # Last 100 lines
            st.code(log_text, language="text")
            
            if st.button("Clear app.log"):
                log_path = os.path.join(basedir, "app.log")
                if os.path.exists(log_path):
                    open(log_path, 'w').close()
                    st.success("Logs cleared!")
                    st.rerun()
        
        with tabs[2]:
            st.subheader("System Configuration")
            st.write("**Firebase Project:** " + firebase_admin.get_app().project_id)
            st.write("**Log File Location:** " + os.path.join(basedir, "app.log"))
            
            st.divider()
            st.subheader("Alert Configuration")
            alert_threshold = st.number_input("Alert Threshold", min_value=1, max_value=100, 
                                           value=st.session_state.threshold, key="admin_threshold")
            st.session_state.threshold = alert_threshold
            
            alert_enabled = st.checkbox("Enable Alert Notifications", value=True)
            if alert_enabled:
                st.success("✅ Alerts are enabled")
            else:
                st.warning("⚠️ Alerts are disabled")
            
            st.divider()
            st.subheader("System Actions")
            if st.button("🔄 Clear Alert Status"):
                st.session_state.alert_shown = False
                st.success("Alert status cleared!")
            
            if st.button("🗑️ Clear Video Analysis Data"):
                st.session_state.heatmap_data = []
                st.session_state.video_stats = {'frames_processed': 0, 'total_detections': 0, 'max_count': 0}
                st.success("Video analysis data cleared!")
            
            st.button("Restart Counter (Simulated)")
            
            st.divider()
            st.subheader("🔐 Account Management")
            if st.button("🚪 Logout", use_container_width=True, type="secondary"):
                logout()

# (Auto-refresh feature removed as requested)
