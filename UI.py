import streamlit as st
import os
from datetime import datetime
import explainer
from PIL import Image

# 1. Load the image using PIL
icon = Image.open("C:/Users/Vicky/XAI project/Icon.png")

# Page config
st.set_page_config(
    page_title="HeartCare AI Diagnostic Portal",
    page_icon=icon,
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Clear Streamlit cache on page load
@st.cache_data
def clear_cache_on_load():
    return True

clear_cache_on_load()

# --- CLINICAL MEDICAL THEME CSS ---
st.markdown("""
<style>
    /* --- IMPORT GOOGLE SERIF FONT --- */
    @import url('https://fonts.googleapis.com/css2?family=Source+Serif+4:ital,opsz,wght@0,8..60,200..900;1,8..60,200..900&display=swap');

    /* 1. UNIVERSAL FONT OVERRIDE */
    /* This forces the font onto every element including buttons, inputs, and labels */
    html, body, [class*="st-"], div, span, p, label, button, input, textarea, select {
        font-family: 'Source Serif 4', serif !important;
    }

    /* Global App Background */
    .stApp { 
        background-color: #F4F7F6; /* Sterile Linen */
    }
    
    /* Hide Default Streamlit Elements */
    #MainMenu, footer, header { visibility: hidden; }
    
    /* Main container padding */
    .block-container { 
        padding-top: 1rem; 
        padding-bottom: 10rem; 
        max-width: 1200px; 
    }
    
    /* Clinical Headers (Teal to Shadow Teal Gradient) */
    .main-header {
        background: linear-gradient(135deg, #008080 0%, #4F7C7C 100%);
        padding: 1.5rem 2rem;
        border-radius: 12px;
        color: white;
        margin-bottom: 0px;
        box-shadow: 0 4px 10px rgba(0,128,128,0.1);
        display: flex;
        align-items: center;
    }
    
    /* Landing Page Section Headers */
    .section-header {
        background: #ffffff;
        padding: 1.5rem;
        border-radius: 12px 12px 0 0;
        border-top: 5px solid #008080; /* Deep Teal */
        border-bottom: 1px solid #CBD1C3; /* Fossil Green */
    }
    
    /* Landing Page Content Area */
    .section-body {
        background: white;
        padding: 2rem;
        border-radius: 0 0 12px 12px;
        box-shadow: 0 2px 10px rgba(0,0,0,0.03);
        margin-bottom: 1rem;
    }
    
    /* Chat Messages */
    .chat-message {
        padding: 1rem 1.5rem;
        border-radius: 12px;
        margin-bottom: 1rem;
        line-height: 1.6;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    }
    
    .bot-message {
        background-color: #ffffff;
        color: #2c3e50;
        border-left: 5px solid #2FCED6; /* Arctic Cyan Accent */
        margin-right: 15%;
    }
    
    .user-message {
        background: #4F7C7C; /* Shadow Teal */
        color: white;
        margin-left: 15%;
    }
    
    .patient-user-message {
        background: #008080; /* Deep Teal */
        color: white;
        margin-left: 15%;
    }

    /* Reports & Metrics Styles */
    .report-box {
        background-color: #ffffff;
        padding: 1.5rem;
        border-radius: 8px;
        border: 1px solid #CBD1C3; /* Fossil Green */
        font-family: 'Courier New', Courier, monospace !important; /* Keep for technical medical findings */
        white-space: pre-wrap;
        margin-top: 10px;
        color: #2c3e50;
    }
    
    /* Updated Severity Badges (Pastel Palette) */
    .severity-badge {
        padding: 0.5rem 1rem;
        border-radius: 20px;
        font-weight: bold;
        text-transform: uppercase;
        font-size: 0.85rem;
        display: inline-block;
    }
    .severity-high { background-color: #D6A2A2; color: #721c24; border: 1px solid #c98a8a; } /* Dust Rose */
    .severity-medium { background-color: #DDBB8E; color: #856404; border: 1px solid #ccaa7a; } /* Harvest Gold */
    .severity-low { background-color: #B7D7D0; color: #155724; border: 1px solid #a3c4bd; } /* Frost Mint */

    /* Buttons */
    .stButton>button {
        background-color: #008080; /* Deep Teal */
        color: white;
        border-radius: 6px;
        padding: 0.5rem 1rem;
        font-weight: 600;
        border: none;
        height: auto;
        min-height: 45px;
        transition: background-color 0.2s;
    }
    .stButton>button:hover {
        background-color: #4F7C7C; /* Shadow Teal */
    }
    
    /* Consistent Off-White Input Fields */
    div[data-testid="stTextInput"] input, 
    div[data-testid="stNumberInput"] input {
        background-color: #E2E8E7 !important; /* Hospital Blue-Grey */
        color: #2c3e50 !important;
        border: 1px solid #CBD1C3 !important; /* Fossil Green border */
        border-radius: 8px !important;
    }
    
    /* Static Footer */
    .footer {
        position: fixed;
        bottom: 0;
        left: 0;
        width: 100%;
        background-color: white;
        color: #4F7C7C; /* Shadow Teal */
        text-align: center;
        padding: 1.2rem;
        border-top: 1px solid #CBD1C3; /* Fossil Green */
        font-size: 0.75rem;
        z-index: 999;
    }
    
</style>
""", unsafe_allow_html=True)

# --- INITIALIZE SESSION STATE ---
for key in ['authenticated', 'user_type', 'messages', 'current_step', 'patient_data', 'show_history', 'selected_patient_folder', 'uploaded_images', 'show_about']:
    if key not in st.session_state:
        if key in ['authenticated', 'show_history', 'show_about']: st.session_state[key] = False
        elif key in ['messages', 'uploaded_images']: st.session_state[key] = []
        elif key == 'current_step': st.session_state[key] = 0
        elif key == 'patient_data': st.session_state[key] = {}
        else: st.session_state[key] = None

FIELDS = [
    {'key': 'patient_name', 'label': 'Patient Name', 'type': 'text'},
    {'key': 'age', 'label': 'Age', 'type': 'number'},
    {'key': 'sex', 'label': 'Sex (1=male, 0=female)', 'type': 'number'},
    {'key': 'chest pain type', 'label': 'Chest Pain Type (0-3)', 'type': 'number'},
    {'key': 'resting bp s', 'label': 'Resting Blood Pressure', 'type': 'number'},
    {'key': 'cholesterol', 'label': 'Cholesterol (mg/dl)', 'type': 'number'},
    {'key': 'fasting blood sugar', 'label': 'Fasting Blood Sugar (1 if >120 mg/dl, else 0)', 'type': 'number'},
    {'key': 'resting ecg', 'label': 'Resting ECG Results (0-2)', 'type': 'number'},
    {'key': 'max heart rate', 'label': 'Maximum Heart Rate Achieved', 'type': 'number'},
    {'key': 'exercise angina', 'label': 'Exercise Induced Angina (1=yes, 0=no)', 'type': 'number'},
    {'key': 'oldpeak', 'label': 'ST Depression (oldpeak)', 'type': 'number'},
    {'key': 'ST slope', 'label': 'Slope of Peak Exercise ST (0-2)', 'type': 'number'},
    {'key': 'dicom_images', 'label': 'Upload DICOM Images (.dcm files)', 'type': 'file'}
]

# --- HELPER FUNCTIONS ---
def call_explainer(patient_data, user_type):
    try:
        patient_name = patient_data.pop('patient_name')
        patient_image_folder = patient_data.pop('dicom_folder_path')
        text_input = patient_data
        
        results = explainer.generate_reports_for_new_patient(
            patient_name=patient_name,
            text_input=text_input,
            patient_image_folder=patient_image_folder,
            patient_id=None
        )
        
        patient_id = patient_name.replace(" ", "_")
        patient_folder = os.path.join("saved_reports", patient_id)
        
        return {
            'doctorReportPath': os.path.join(patient_folder, "doctor_report.txt"),
            'patientReportPath': os.path.join(patient_folder, "patient_report.txt"),
            'textShapPath': results.get('text_shap_path'),
            'cnnShapPaths': results.get('cnn_shap_paths', []),
            'results': results
        }
    except Exception as e:
        st.error(f"Clinical Explainer Error: {e}")
        raise e

def save_uploaded_files(uploaded_files):
    if not uploaded_files: return None
    temp_folder = f"temp_dicom_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    os.makedirs(temp_folder, exist_ok=True)
    for uploaded_file in uploaded_files:
        with open(os.path.join(temp_folder, uploaded_file.name), "wb") as f:
            f.write(uploaded_file.getbuffer())
    return temp_folder

def reset_conversation():
    st.session_state.messages = []
    st.session_state.current_step = 0
    st.session_state.patient_data = {}
    st.session_state.uploaded_images = []
    st.session_state.show_history = False
    st.session_state.selected_patient_folder = None
    msg = 'Clinical Intake System Active. Doctor, please provide the patient name.' if st.session_state.user_type == 'doctor' else "Hello. I'll guide you through your heart health assessment. What is your name?"
    st.session_state.messages.append({'type': 'bot', 'content': msg})

def process_input(user_input=None, uploaded_files=None):
    current_field = FIELDS[st.session_state.current_step]
    if current_field['type'] == 'file':
        if uploaded_files:
            st.session_state.messages.append({'type': 'user', 'content': f"Uploaded {len(uploaded_files)} DICOM file(s)"})
            st.session_state.patient_data['dicom_folder_path'] = save_uploaded_files(uploaded_files)
            st.session_state.uploaded_images = uploaded_files
        else:
            st.warning("Clinical imaging required.")
            return
    else:
        if not user_input or not user_input.strip(): return
        st.session_state.messages.append({'type': 'user', 'content': user_input})
        st.session_state.patient_data[current_field['key']] = user_input
    
    if st.session_state.current_step < len(FIELDS) - 1:
        st.session_state.current_step += 1
        next_field = FIELDS[st.session_state.current_step]
        st.session_state.messages.append({'type': 'bot', 'content': f"Please provide: **{next_field['label']}**"})
    else:
        st.session_state.messages.append({'type': 'bot', 'content': 'Processing diagnostic biomarkers and cardiac imaging...'})
        try:
            result = call_explainer(st.session_state.patient_data.copy(), st.session_state.user_type)
            st.session_state.messages.append({'type': 'bot', 'content': 'Clinical Analysis Complete.', 'result': result})
        except Exception as e:
            st.session_state.messages.append({'type': 'bot', 'content': f'System Error: {str(e)}'})
        st.session_state.current_step, st.session_state.patient_data, st.session_state.uploaded_images = 0, {}, []

def logout():
    for key in list(st.session_state.keys()): del st.session_state[key]
    st.rerun()

def get_patient_folders():
    base_folder = "saved_reports"
    if not os.path.exists(base_folder): return []
    folders = []
    for item in os.listdir(base_folder):
        item_path = os.path.join(base_folder, item)
        if os.path.isdir(item_path):
            folders.append({
                'name': item.replace("_", " "), 'folder': item,
                'date': datetime.fromtimestamp(os.path.getmtime(item_path)).strftime("%Y-%m-%d %H:%M"),
                'path': item_path
            })
    return sorted(folders, key=lambda x: x['date'], reverse=True)

def handle_enter_key():
    user_input = st.session_state.get(f"user_input_field_{st.session_state.current_step}", "")
    if user_input and user_input.strip():
        process_input(user_input=user_input)
        # REMOVED st.rerun() here to fix the no-op warning
        # Streamlit automatically reruns after this callback finishes

# --- MAIN UI ROUTING ---

# 1. ABOUT SECTION OVERLAY
if st.session_state.show_about:
        st.markdown("""
        ### Project Overview
        HeartCare AI is an advanced demonstration of **Explainable Artificial Intelligence (XAI)** in cardiology. 
        It combines tabular clinical data analysis with Deep Learning (CNNs) for medical imaging to provide:
        
        * **Risk Prediction:** Calculating cardiovascular risk probability.
        * **Visual Explanations:** Using SHAP (SHapley Additive exPlanations) to highlight critical regions in cardiac images and key factors in patient history.
        * **Role-Based Reporting:** Generating tailored reports for both clinicians (technical) and patients (simplified).
        
        **Developed By:** Harsh Sharma, V. Vijay Kumar, B. Yashwanth, and P. Sai Kumar \n
        **Technology Stack:** Python, Streamlit, TensorFlow, SHAP.
        """)
        if st.button("Close"):
            st.session_state.show_about = False
            st.rerun()

if not st.session_state.authenticated:
    # --- LANDING PAGE ---
    
    # Utility Bar (Top Right)
    util_col1, util_col2 = st.columns([6, 1])
    with util_col2:
        if st.button("About", type="secondary", width="stretch"):
            st.session_state.show_about = not st.session_state.show_about
            st.rerun()

    st.markdown('<div class="main-header" style="justify-content:center; text-align:center;"><h1>HeartCare AI Portal</h1><p style="margin-left:15px; opacity:0.9;">Explainable Clinical Decision Support System</p></div>', unsafe_allow_html=True)
    st.write("") 
    
    col1, col2 = st.columns(2, gap="medium")
    
    with col1:
        st.markdown('<div class="section-header"><h3>Clinician Access</h3><p style="margin:0; color:#666;">Review diagnostics & records</p></div>', unsafe_allow_html=True)
        with st.container():
            st.markdown('<div class="section-body">', unsafe_allow_html=True)
            password = st.text_input("Clinical ID Password", type="password", key="doctor_pass")
            st.write("")
            if st.button("Enter Dashboard", key="doctor_login", width="stretch"):
                if password == "doctor123":
                    st.session_state.authenticated, st.session_state.user_type = True, 'doctor'
                    reset_conversation(); st.rerun()
                else: st.error("Invalid Credentials")
            st.markdown('</div>', unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="section-header"><h3>Patient Intake</h3><p style="margin:0; color:#666;">Start heart health assessment</p></div>', unsafe_allow_html=True)
        with st.container():
            st.markdown('<div class="section-body">', unsafe_allow_html=True)
            st.write("") 
            st.write("") 
            if st.button("Start Intake", key="patient_login", width="stretch"):
                st.session_state.authenticated, st.session_state.user_type = True, 'patient'
                reset_conversation(); st.rerun()
            st.write("") 
            st.markdown('</div>', unsafe_allow_html=True)

else:
    # --- LOGGED IN DASHBOARD ---
    
    # Header & Buttons Layout
    header_col, btn_col = st.columns([3, 2], gap="medium")
    
    with header_col:
        portal_icon = "" if st.session_state.user_type == 'doctor' else ""
        portal_name = "Clinician Dashboard" if st.session_state.user_type == 'doctor' else "Patient Intake"
        st.markdown(f"""
        <div class="main-header">
            <h2 style="margin:0; font-size:1.8rem;">{portal_icon} {portal_name}</h2>
        </div>
        """, unsafe_allow_html=True)
    
    with btn_col:
        st.write("") 
        
        # 4 Columns for buttons: About | Archives/Home | Reset | Exit
        if st.session_state.user_type == 'doctor':
            b0, b1, b2, b3 = st.columns(4, gap="small")
            with b0:
                if st.button("About", type="secondary", width="stretch"):
                    st.session_state.show_about = not st.session_state.show_about; st.rerun()
            with b1:
                 if st.button("Archives", width="stretch"):
                    st.session_state.show_history = not st.session_state.show_history; st.rerun()
            with b2:
                if st.button("Reset", width="stretch"): reset_conversation(); st.rerun()
            with b3:
                if st.button("Exit", width="stretch"): logout()
        else:
            b0, b1, b2, b3 = st.columns(4, gap="small")
            with b0:
                if st.button("About", type="secondary", width="stretch"):
                    st.session_state.show_about = not st.session_state.show_about; st.rerun()
            with b1:
                if st.button("Home", width="stretch"): reset_conversation(); st.rerun()
            with b2:
                if st.button("Reset", width="stretch"): reset_conversation(); st.rerun()
            with b3:
                if st.button("Exit", width="stretch"): logout()

    # --- ARCHIVES VIEW ---
    if st.session_state.show_history and st.session_state.user_type == 'doctor':
        st.markdown("<br>", unsafe_allow_html=True)
        if st.session_state.selected_patient_folder:
            p_folder = st.session_state.selected_patient_folder
            col1, col2 = st.columns([5, 1])
            col1.markdown(f"### Patient: {p_folder['name']}")
            if col2.button("Back", width="stretch"): st.session_state.selected_patient_folder = None; st.rerun()
            
            st.divider()
            rep_path = os.path.join(p_folder['path'], "doctor_report.txt")
            if os.path.exists(rep_path):
                with open(rep_path, 'r', encoding='utf-8') as f: content = f.read()
                st.markdown("#### Clinical Report")
                st.markdown(f'<div class="report-box">{content}</div>', unsafe_allow_html=True)
                
                # DOWNLOAD BUTTON FOR ARCHIVED REPORT
                col_a, col_b = st.columns([1, 4])
                with col_a:
                    st.download_button("Download", content, os.path.basename(rep_path), width="stretch")
            
            for sub, title in [("text_shap", "Feature Importance"), ("cnn_shap", "Image Heatmaps")]:
                folder = os.path.join(p_folder['path'], sub)
                if os.path.exists(folder):
                    imgs = [f for f in os.listdir(folder) if f.endswith('.png')]
                    if imgs:
                        st.markdown(f"#### {title}")
                        for j in range(0, len(imgs), 3):
                            cols = st.columns(3)
                            for idx, img in enumerate(imgs[j:j+3]):
                                with cols[idx]: st.image(os.path.join(folder, img), width="stretch")
        else:
            st.markdown("### Longitudinal Archives")
            patient_folders = get_patient_folders()
            search = st.text_input("🔍 Filter by Name")
            if search: patient_folders = [p for p in patient_folders if search.lower() in p['name'].lower()]
            
            if not patient_folders: st.info("No records found.")
            for idx, p in enumerate(patient_folders):
                c1, c2, c3 = st.columns([3, 2, 1])
                c1.markdown(f"**{p['name']}**")
                c2.markdown(f"{p['date']}")
                if c3.button("View", key=f"v_{idx}", width="stretch"):
                    st.session_state.selected_patient_folder = p; st.rerun()
                st.divider()

    # --- CHAT INTERFACE ---
    else:
        st.markdown("<br>", unsafe_allow_html=True)
        
        for msg in st.session_state.messages:
            if msg['type'] == 'user':
                m_class = 'patient-user-message' if st.session_state.user_type == 'patient' else 'user-message'
                st.markdown(f'<div class="chat-message {m_class}">{msg["content"]}</div>', unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="chat-message bot-message">{msg["content"]}</div>', unsafe_allow_html=True)
                
                if 'result' in msg:
                    r = msg['result']
                    rd = r.get('results', {})
                    if rd:
                        st.markdown("<div style='margin: 10px 0;'></div>", unsafe_allow_html=True)
                        sev = rd.get('severity', 'N/A')
                        s_class = f"severity-{sev.lower()}"
                        
                        c1, c2 = st.columns(2)
                        c1.markdown(f'<div style="text-align:center; padding:10px;"><span class="severity-badge {s_class}"> Severity: {sev}</span></div>', unsafe_allow_html=True)
                        c2.markdown(f'<div style="text-align:center; background:#fff; padding:10px; border-radius:10px; border:1px solid #ddd;">Risk Score: <b>{rd.get("text_pred", "N/A")}</b></div>', unsafe_allow_html=True)
                        
                        if rd.get('cnn_classes'):
                            st.info(f"**Detected Pathologies:** {', '.join(rd.get('cnn_classes', []))}")
                    
                    rep_path = r.get('doctorReportPath') if st.session_state.user_type == 'doctor' else r.get('patientReportPath')
                    if rep_path and os.path.exists(rep_path):
                        with open(rep_path, 'r', encoding='utf-8') as f: content = f.read()
                        st.markdown(f"#### Findings")
                        st.markdown(f'<div class="report-box">{content}</div>', unsafe_allow_html=True)
                        col_a, col_b = st.columns([1, 4])
                        with col_a:
                            st.download_button("Download", content, os.path.basename(rep_path), width="stretch")

                    if st.session_state.user_type == 'doctor':
                        ts = r.get('textShapPath')
                        if ts and os.path.exists(ts):
                            st.markdown("#### Diagnostic Feature Importance")
                            st.image(ts, width="stretch")
                        
                        cs = r.get('cnnShapPaths', [])
                        if cs:
                            st.markdown("#### Cardiac Imaging Analysis")
                            for i in range(0, len(cs), 3):
                                cols = st.columns(3)
                                for idx, pth in enumerate(cs[i:i+3]):
                                    if os.path.exists(pth):
                                        with cols[idx]: st.image(pth, width="stretch", caption=f"Activation Map {i+idx+1}")

        # --- INPUT AREA ---
        st.divider()
        curr_field = FIELDS[st.session_state.current_step] if st.session_state.current_step < len(FIELDS) else None
        
        if curr_field and curr_field['type'] == 'file':
            st.markdown("#### Upload Imaging Data")
            files = st.file_uploader("Drop clinical images", type=['dcm'], accept_multiple_files=True, key="dicom_uploader")
            if files:
                if st.button(" Process Clinical Data", width="stretch"):
                    process_input(uploaded_files=files); st.rerun()
        else:
            st.text_input("Response:", key=f"user_input_field_{st.session_state.current_step}", placeholder="Type here...", label_visibility="collapsed", on_change=handle_enter_key)

# --- FIXED FOOTER (OUTSIDE ALL IF/ELSE) ---
st.markdown("""
<div class="footer">
    <p><strong>EDUCATIONAL DISCLOSURE:</strong> This application is a prototype developed exclusively for research and educational purposes. 
    It leverages experimental Explainable AI (XAI) algorithms to demonstrate potential clinical decision support workflows. 
    <strong>This system is not a certified medical device and has not been cleared by the FDA or any regulatory body.</strong> 
    The predictions, severity scores, and visual explanations provided herein should not be used as a substitute for professional medical judgment, diagnosis, or treatment.</p>
    <p style="opacity: 0.6; margin-top: 5px;">© 2025 HeartCare AI Research Group. All rights reserved.</p>
</div>
""", unsafe_allow_html=True)