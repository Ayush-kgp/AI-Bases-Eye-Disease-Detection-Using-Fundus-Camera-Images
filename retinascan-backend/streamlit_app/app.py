"""RetinaScan AI — Streamlit Web Application

Interactive clinical AI interface for 10-class ocular disease detection,
dual-backbone ensemble agreement, and Grad-CAM visual explainability.
"""

import io
import os
from pathlib import Path
import httpx
from PIL import Image
import streamlit as st

# Backend API Configuration
BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000")

st.set_page_config(
    page_title="RetinaScan AI — Clinical Fundus Classifier",
    page_icon="👁️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1E293B;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #64748B;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background-color: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-radius: 10px;
        padding: 1rem;
        margin-bottom: 1rem;
    }
    .badge-agree {
        background-color: #DCFCE7;
        color: #166534;
        padding: 4px 10px;
        border-radius: 12px;
        font-weight: 600;
        font-size: 0.85rem;
    }
    .badge-disagree {
        background-color: #FEF3C7;
        color: #92400E;
        padding: 4px 10px;
        border-radius: 12px;
        font-weight: 600;
        font-size: 0.85rem;
    }
    .badge-warning {
        background-color: #FEE2E2;
        color: #991B1B;
        padding: 6px 12px;
        border-radius: 8px;
        font-weight: 600;
        font-size: 0.88rem;
    }
</style>
""", unsafe_allow_html=True)


def check_backend_health():
    try:
        r = httpx.get(f"{BACKEND_URL}/health", timeout=3.0)
        return r.status_code == 200, r.json() if r.status_code == 200 else {}
    except Exception:
        return False, {}


def get_model_info():
    try:
        r = httpx.get(f"{BACKEND_URL}/model/info", timeout=3.0)
        return r.json() if r.status_code == 200 else {}
    except Exception:
        return {}


# Sidebar Controls
with st.sidebar:
    st.markdown("### 👁️ RetinaScan AI")
    st.markdown("**Dual-Model Ensemble (ONNX)**")
    
    is_healthy, health_data = check_backend_health()
    if is_healthy:
        st.success("🟢 Backend API: Online (200 OK)")
    else:
        st.error(f"🔴 Backend API: Offline at {BACKEND_URL}")
        st.caption("Start backend with: `uvicorn app.main:app --port 8000`")

    st.divider()
    st.markdown("#### 🔬 Evaluation Benchmarks")
    info = get_model_info()
    if info:
        st.markdown(f"**Combination Strategy:** `{info.get('combination_method')}`")
        st.markdown(f"**Ensemble Macro-F1:** `{info.get('ensemble_test_macro_f1', 0.705):.3f}`")
        solo = info.get("solo_test_macro_f1", {})
        st.markdown(f"- EfficientNet-B0: `{solo.get('efficientnet', 0.625):.3f}`")
        st.markdown(f"- MobileNet-V2: `{solo.get('mobilenet', 0.615):.3f}`")
        
        with st.expander("Low-Support Classes (<25 samples)"):
            for cls_name in info.get("low_support_classes", []):
                st.caption(f"⚠️ {cls_name}")

    st.divider()
    st.markdown("#### 📚 Diagnostic Classes (10)")
    classes_list = [
        "Diabetic Retinopathy", "Glaucoma", "Healthy", "Myopia",
        "Macular Scar", "Retinitis Pigmentosa", "Disc Edema",
        "Retinal Detachment", "CSCR", "Pterygium"
    ]
    for c in classes_list:
        st.caption(f"• {c}")


# Main Content Area
st.markdown('<div class="main-header">RetinaScan AI Diagnostic Assistant</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">AI-Assisted Retinal Disease Detection with ONNX Runtime Ensemble & Grad-CAM Visual Explainability</div>', unsafe_allow_html=True)

# Sample Images Finder
samples_dir = Path(__file__).resolve().parent.parent.parent / "training-artifacts" / "test-images"
sample_options = {}
if samples_dir.exists():
    for disease_folder in sorted(samples_dir.iterdir()):
        if disease_folder.is_dir():
            for f in list(disease_folder.glob("*.jpg"))[:2] + list(disease_folder.glob("*.png"))[:2]:
                sample_options[f"{disease_folder.name} — {f.name}"] = str(f)

col_input, col_action = st.columns([2, 1])

with col_input:
    input_source = st.radio("Input Source", ["Upload Image", "Load Sample Test Image"], horizontal=True)

selected_image_bytes = None
selected_pil_image = None

if input_source == "Upload Image":
    uploaded_file = st.file_uploader(
        "Upload Retinal Fundus Photograph (JPEG / PNG / WEBP)",
        type=["jpg", "jpeg", "png", "webp"],
    )
    if uploaded_file is not None:
        selected_image_bytes = uploaded_file.getvalue()
        selected_pil_image = Image.open(io.BytesIO(selected_image_bytes))
else:
    if sample_options:
        chosen_sample = st.selectbox("Select sample test image:", list(sample_options.keys()))
        sample_path = sample_options[chosen_sample]
        with open(sample_path, "rb") as f:
            selected_image_bytes = f.read()
        selected_pil_image = Image.open(io.BytesIO(selected_image_bytes))
    else:
        st.info("No sample images found in training-artifacts/test-images.")


if selected_pil_image is not None and selected_image_bytes is not None:
    st.divider()
    
    col_left, col_right = st.columns([1, 1])
    
    with col_left:
        st.markdown("#### 📷 Input Fundus Image")
        st.image(selected_pil_image, use_container_width=True, caption=f"Resolution: {selected_pil_image.size[0]}x{selected_pil_image.size[1]}px")
        
        analyze_button = st.button("🚀 Analyze & Generate Heatmap", type="primary", use_container_width=True)

    with col_right:
        if analyze_button:
            if not is_healthy:
                st.error("Cannot connect to backend server. Please ensure FastAPI is running on port 8000.")
            else:
                with st.spinner("Executing dual-backbone inference and Grad-CAM hooks..."):
                    try:
                        files = {"file": ("fundus.jpg", selected_image_bytes, "image/jpeg")}
                        response = httpx.post(f"{BACKEND_URL}/predict", files=files, timeout=30.0)
                        
                        if response.status_code == 200:
                            data = response.json()
                            pred = data["prediction"]
                            agreement = data["model_agreement"]
                            flag = data["reliability_flag"]
                            expl = data["explainability"]
                            meta = data["meta"]
                            
                            st.markdown("#### 🩺 Diagnostic Result")
                            
                            # Top condition & confidence
                            st.markdown(f"### **{pred['disease']}**")
                            conf_pct = pred['confidence'] * 100
                            st.progress(pred['confidence'], text=f"Confidence: {conf_pct:.1f}%")
                            
                            # Model Agreement Indicator
                            st.markdown("##### Backbone Consensus")
                            agree_badge = (
                                '<span class="badge-agree">✓ Dual Model Agreement</span>'
                                if agreement["agree"]
                                else '<span class="badge-disagree">⚠️ Disagreement Between Models</span>'
                            )
                            st.markdown(agree_badge, unsafe_allow_html=True)
                            st.caption(
                                f"• EfficientNet-B0: **{agreement['efficientnet_prediction']}**  \n"
                                f"• MobileNet-V2: **{agreement['mobilenet_prediction']}**"
                            )
                            
                            # Reliability Warning if low support
                            if flag["is_low_support_class"]:
                                st.markdown(
                                    f'<div class="badge-warning">⚠️ {flag["note"]}</div>',
                                    unsafe_allow_html=True,
                                )
                            
                            st.caption(f"⚡ Latency: {meta['inference_time_ms']} ms | Strategy: `{pred['combination_method']}`")
                            
                            # Display Grad-CAM Heatmap
                            st.divider()
                            st.markdown("#### 🔥 Grad-CAM Attention Heatmap")
                            overlay_b64 = expl["gradcam_overlay_base64"]
                            if overlay_b64.startswith("data:image/png;base64,"):
                                st.image(overlay_b64, use_container_width=True, caption="Visual explanation highlights regions influencing prediction")
                            
                            # Full probability distribution chart
                            st.divider()
                            st.markdown("#### 📊 Probability Distribution (All 10 Classes)")
                            probs = pred["class_probabilities"]
                            sorted_probs = dict(sorted(probs.items(), key=lambda item: item[1], reverse=True))
                            st.bar_chart(sorted_probs)
                            
                        else:
                            st.error(f"Prediction Error ({response.status_code}): {response.text}")
                    except Exception as e:
                        st.error(f"Inference failed: {str(e)}")
        else:
            st.info("👈 Click **'Analyze & Generate Heatmap'** to run classification and explainability.")

else:
    st.info("👆 Please upload an eye fundus image or pick a sample test image above to begin.")

# Medical disclaimer footer
st.divider()
st.caption("⚕️ **Medical Disclaimer:** Research prototype for academic and technical evaluation. Not certified for clinical diagnosis or medical decision-making.")
