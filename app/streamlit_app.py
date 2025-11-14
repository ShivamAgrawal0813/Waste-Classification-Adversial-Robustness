"""
Streamlit web app for Waste Classification with Adversarial Robustness Demo.
"""

import streamlit as st
import torch
import torch.nn as nn
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import json

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import (
    CLEAN_MODEL_PATH, ADV_MODEL_PATH, METRICS_FILE, OUTPUTS_DIR,
    EPS_FGSM, EPS_PGD, PGD_STEPS_EVAL, CLASS_NAMES, IDX_TO_CLASS
)
from src.inference import (
    load_image, preprocess_image, predict, generate_adversarial_examples,
    tensor_to_image, visualize_perturbation, load_model_for_inference
)
from src.utils import get_device, load_metrics


def _safe_st_image(path_or_obj, caption=None, use_column_width=False):
    """Try to safely open a file path with PIL and display via Streamlit.

    Accepts either a path (str) or a PIL/Image/numpy object and returns True on success.
    """
    try:
        if isinstance(path_or_obj, str):
            if not os.path.exists(path_or_obj):
                return False
            img = Image.open(path_or_obj).convert('RGB')
            st.image(img, caption=caption, use_column_width=use_column_width)
            return True
        else:
            # Assume it's an image array or PIL Image
            st.image(path_or_obj, caption=caption, use_column_width=use_column_width)
            return True
    except Exception as e:
        st.warning(f"Could not display image {caption or path_or_obj}: {e}")
        return False


# Page configuration
st.set_page_config(
    page_title="Waste Classification & Adversarial Robustness",
    page_icon="🗑️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-container {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    </style>
""", unsafe_allow_html=True)

# Title
st.markdown('<p class="main-header">🗑️ Waste Classification & Adversarial Robustness</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">ResNet + FGSM + PGD + Adversarial Training Demo</p>', unsafe_allow_html=True)

# Sidebar
st.sidebar.header("⚙️ Configuration")

# Model selection
model_type = st.sidebar.selectbox(
    "Select Model",
    ["Clean Model", "Adversarial Model"],
    help="Choose between clean (standard) or adversarial (robust) model"
)

# File upload
uploaded_file = st.sidebar.file_uploader(
    "Upload Image",
    type=['jpg', 'jpeg', 'png'],
    help="Upload a waste classification image (JPEG/PNG)"
)

# Run button
run_classification = st.sidebar.button(
    "🚀 Run Classification & Attacks",
    type="primary"
)

# About section
st.sidebar.markdown("---")
st.sidebar.markdown("### 📖 About")
st.sidebar.markdown("""
This demo showcases:
- **Waste Classification**: 6 classes (cardboard, glass, metal, paper, plastic, trash)
- **Adversarial Attacks**: FGSM and PGD attacks (automatically generated)
- **Adversarial Defense**: Robust model trained with PGD adversarial training

**Attack Parameters** (fixed in code):
- FGSM: ε = 8/255
- PGD: ε = 8/255, steps = 20
""")

# Initialize session state
if 'model' not in st.session_state:
    st.session_state.model = None
if 'device' not in st.session_state:
    st.session_state.device = get_device()


@st.cache_resource
def load_model_cached(model_path: str, device: torch.device):
    """Load model with caching."""
    if os.path.exists(model_path):
        return load_model_for_inference(model_path, device)
    return None


# Load model based on selection
model_path = CLEAN_MODEL_PATH if model_type == "Clean Model" else ADV_MODEL_PATH
model_name = "Clean" if model_type == "Clean Model" else "Adversarial"

if os.path.exists(model_path):
    st.session_state.model = load_model_cached(model_path, st.session_state.device)
    st.sidebar.success(f"✓ {model_name} model loaded")
else:
    st.sidebar.error(f"❌ Model not found: {model_path}")
    st.sidebar.info("Please train the model first using `python src/train.py` or `python src/adv_train.py`")

# Main content
if run_classification and uploaded_file is not None:
    try:
        # Load and preprocess image
        image = load_image(uploaded_file)
        image_tensor = preprocess_image(image)
        
        # Get clean prediction
        clean_pred = predict(st.session_state.model, image_tensor, st.session_state.device, top_k=3)
        
        # Display original image and prediction
        st.header("📸 Original Image & Prediction")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.image(image, caption="Original Image")
        
        with col2:
            st.subheader("Clean Prediction")
            st.markdown(f"**Class:** {clean_pred['predicted_class']}")
            st.markdown(f"**Confidence:** {clean_pred['confidence']:.2%}")
            
            st.markdown("**Top 3 Predictions:**")
            for i, pred in enumerate(clean_pred['predictions']):
                st.progress(pred['confidence'], text=f"{i+1}. {pred['class']}: {pred['confidence']:.2%}")
        
        # Generate adversarial examples
        st.header("⚔️ Adversarial Attacks")
        st.info(f"**Attack Parameters (fixed in code):** FGSM ε={EPS_FGSM:.4f}, PGD ε={EPS_PGD:.4f}, steps={PGD_STEPS_EVAL}")
        
        predicted_label = clean_pred['predictions'][0]['class_idx']
        
        with st.spinner("Generating adversarial examples..."):
            adv_results = generate_adversarial_examples(
                st.session_state.model,
                image_tensor,
                predicted_label,
                st.session_state.device
            )
        
        # Display adversarial examples
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.subheader("🔵 FGSM Attack")
            fgsm_img = tensor_to_image(adv_results['fgsm_image'])
            st.image(fgsm_img, caption="FGSM Adversarial Image")
            st.markdown(f"**Predicted:** {adv_results['fgsm_prediction']['predicted_class']}")
            st.markdown(f"**Confidence:** {adv_results['fgsm_prediction']['confidence']:.2%}")
            
            # Perturbation visualization
            if st.checkbox("Show FGSM Perturbation", key="fgsm_pert"):
                fgsm_pert = visualize_perturbation(image_tensor, adv_results['fgsm_image'])
                st.image(fgsm_pert, caption="FGSM Perturbation (amplified)")
        
        with col2:
            st.subheader("🔴 PGD Attack")
            pgd_img = tensor_to_image(adv_results['pgd_image'])
            st.image(pgd_img, caption="PGD Adversarial Image")
            st.markdown(f"**Predicted:** {adv_results['pgd_prediction']['predicted_class']}")
            st.markdown(f"**Confidence:** {adv_results['pgd_prediction']['confidence']:.2%}")
            
            # Perturbation visualization
            if st.checkbox("Show PGD Perturbation", key="pgd_pert"):
                pgd_pert = visualize_perturbation(image_tensor, adv_results['pgd_image'])
                st.image(pgd_pert, caption="PGD Perturbation (amplified)")
        
        with col3:
            st.subheader("📊 Comparison")
            st.markdown("**Prediction Changes:**")
            
            results_comparison = {
                "Clean": clean_pred['predicted_class'],
                "FGSM": adv_results['fgsm_prediction']['predicted_class'],
                "PGD": adv_results['pgd_prediction']['predicted_class']
            }
            
            for attack_type, pred_class in results_comparison.items():
                status = "✅" if pred_class == clean_pred['predicted_class'] else "❌"
                st.markdown(f"{status} **{attack_type}:** {pred_class}")
            
            # Confidence comparison
            st.markdown("**Confidence Scores:**")
            st.markdown(f"Clean: {clean_pred['confidence']:.2%}")
            st.markdown(f"FGSM: {adv_results['fgsm_prediction']['confidence']:.2%}")
            st.markdown(f"PGD: {adv_results['pgd_prediction']['confidence']:.2%}")
        
        # Metrics section
        st.header("📈 Model Performance Metrics")
        
        # Load metrics if available (robustly)
        try:
            metrics = load_metrics(METRICS_FILE)
        except Exception as e:
            st.warning(f"Failed to load metrics file: {e}")
            metrics = {}

        # Fallback: try metrics_from_notebook.json in outputs/
        if not metrics:
            fallback = os.path.join(OUTPUTS_DIR, 'metrics_from_notebook.json')
            if os.path.exists(fallback):
                try:
                    with open(fallback, 'r', encoding='utf-8') as f:
                        metrics = json.load(f)
                    st.info(f"Loaded metrics from: {os.path.basename(fallback)}")
                except Exception:
                    metrics = {}

        if metrics:
            # Support multiple possible key names: 'clean'/'adv' or 'clean_model'/'adversarial_model'
            preferred_keys = {
                'Clean Model': ['clean_model', 'clean'],
                'Adversarial Model': ['adversarial_model', 'adv', 'adv_model']
            }
            candidates = preferred_keys.get(model_type, [])
            model_key = None
            for c in candidates:
                if c in metrics:
                    model_key = c
                    break

            # If still not found, try to detect common names
            if model_key is None:
                if 'clean' in metrics and model_type == 'Clean Model':
                    model_key = 'clean'
                elif 'adv' in metrics and model_type != 'Clean Model':
                    model_key = 'adv'

            if model_key and model_key in metrics:
                model_metrics = metrics[model_key]

                # Display accuracy metrics
                col1, col2, col3 = st.columns(3)

                with col1:
                    st.metric("Clean Accuracy", f"{model_metrics.get('clean_accuracy', 0):.2f}%")
                with col2:
                    st.metric("FGSM Robust Accuracy", f"{model_metrics.get('fgsm_accuracy', 0):.2f}%")
                with col3:
                    st.metric("PGD Robust Accuracy", f"{model_metrics.get('pgd_accuracy', 0):.2f}%")

                # If detailed eps-vs-accuracy arrays exist, plot them; otherwise try saved images
                if 'epsilons' in model_metrics and 'fgsm_accuracies' in model_metrics and 'pgd_accuracies' in model_metrics:
                    st.subheader("Accuracy vs Epsilon")
                    fig, ax = plt.subplots(figsize=(10, 6))
                    epsilons = model_metrics['epsilons']
                    clean_acc = model_metrics.get('clean_accuracy', 0)
                    fgsm_accs = model_metrics['fgsm_accuracies']
                    pgd_accs = model_metrics['pgd_accuracies']
                    ax.plot([0], [clean_acc], 'o-', label='Clean', linewidth=2, markersize=8)
                    ax.plot(epsilons, fgsm_accs, 's-', label='FGSM', linewidth=2, markersize=6)
                    ax.plot(epsilons, pgd_accs, '^-', label='PGD', linewidth=2, markersize=6)
                    ax.set_xlabel('Epsilon (ε)', fontsize=12)
                    ax.set_ylabel('Accuracy (%)', fontsize=12)
                    ax.set_title('Model Accuracy vs Adversarial Perturbation Strength', fontsize=14)
                    ax.legend(fontsize=11)
                    ax.grid(True, alpha=0.3)
                    st.pyplot(fig)
                else:
                    # Try saved plot images
                    acc_plot = os.path.join(OUTPUTS_DIR, 'accuracy_vs_eps.png')
                    acc_clean_plot = os.path.join(OUTPUTS_DIR, 'accuracy_vs_eps_clean.png')
                    if os.path.exists(acc_plot):
                        st.subheader('Accuracy vs Epsilon (saved)')
                        _safe_st_image(acc_plot, caption='Accuracy vs Epsilon (saved)')
                    if os.path.exists(acc_clean_plot):
                        st.subheader('Accuracy vs Epsilon (clean saved)')
                        _safe_st_image(acc_clean_plot, caption='Accuracy vs Epsilon (clean saved)')

                # Confusion matrices (use saved images if present)
                st.subheader("Confusion Matrices")
                col1, col2 = st.columns(2)
                with col1:
                    confusion_clean_path = os.path.join(OUTPUTS_DIR, 'confusion_clean.png')
                    if model_type != 'Clean Model':
                        confusion_clean_path = os.path.join(OUTPUTS_DIR, 'confusion_adv.png')
                    if os.path.exists(confusion_clean_path):
                        _safe_st_image(confusion_clean_path, caption="Clean Confusion Matrix")
                    else:
                        # Try JSON confusion matrix
                        cm = model_metrics.get('confusion_matrix')
                        if cm:
                            st.text('Confusion matrix available in metrics JSON (not visualized)')

                with col2:
                    confusion_pgd_path = os.path.join(OUTPUTS_DIR, 'confusion_clean_pgd.png')
                    if model_type != 'Clean Model':
                        confusion_pgd_path = os.path.join(OUTPUTS_DIR, 'confusion_adv_pgd.png')
                    if os.path.exists(confusion_pgd_path):
                        _safe_st_image(confusion_pgd_path, caption="PGD Attack Confusion Matrix")
            else:
                st.warning(f"Metrics not found for {model_name} model. Please run evaluation first.")
        else:
            st.warning("Metrics file not found or invalid. Please run evaluation using `python src/eval.py`")
    
    except Exception as e:
        st.error(f"Error during classification: {str(e)}")
        st.exception(e)

elif run_classification and uploaded_file is None:
    st.warning("⚠️ Please upload an image first!")

else:
    # Welcome message
    st.info("""
    👋 **Welcome to the Waste Classification & Adversarial Robustness Demo!**
    
    **Instructions:**
    1. Select a model (Clean or Adversarial) from the sidebar
    2. Upload an image (JPEG/PNG) of waste material
    3. Click "Run Classification & Attacks" to see:
       - Clean prediction on the original image
       - FGSM adversarial attack results
       - PGD adversarial attack results
       - Model performance metrics and visualizations
    
    **Note:** All attack parameters (epsilon, steps) are fixed in the code and cannot be changed from the UI.
    This ensures reproducible results and demonstrates the robustness of adversarially trained models.
    """)
    
    # Display sample images if available
    sample_dir = os.path.join(OUTPUTS_DIR, "sample_predictions")
    if os.path.exists(sample_dir):
        sample_files = [f for f in os.listdir(sample_dir) if f.endswith('.png')]
        if sample_files:
            st.subheader("📷 Sample Predictions")
            cols = st.columns(min(5, len(sample_files)))
            for i, sample_file in enumerate(sample_files[:5]):
                with cols[i]:
                    _safe_st_image(os.path.join(sample_dir, sample_file))

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666;'>
    <p>Waste Classification & Adversarial Robustness Demo</p>
    <p>Built with PyTorch, Streamlit, and ResNet</p>
</div>
""", unsafe_allow_html=True)

