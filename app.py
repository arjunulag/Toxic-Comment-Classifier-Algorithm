"""
Toxic Comment Classifier - Streamlit Web App
=============================================
A beautiful web interface to test comments for toxicity.
"""

import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# ============================================================================
# Page Configuration
# ============================================================================

st.set_page_config(
    page_title="Toxic Comment Classifier",
    page_icon="🛡️",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# ============================================================================
# Custom CSS for Beautiful UI
# ============================================================================

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;700&family=Outfit:wght@300;400;600;700&display=swap');
    
    .stApp {
        background: linear-gradient(135deg, #0f0f23 0%, #1a1a2e 50%, #16213e 100%);
    }
    
    .main-header {
        font-family: 'Outfit', sans-serif;
        font-size: 3rem;
        font-weight: 700;
        background: linear-gradient(90deg, #00d4ff, #7c3aed, #f472b6);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 0.5rem;
    }
    
    .sub-header {
        font-family: 'Outfit', sans-serif;
        color: #a0aec0;
        text-align: center;
        font-size: 1.1rem;
        margin-bottom: 2rem;
    }
    
    .result-box {
        padding: 2rem;
        border-radius: 16px;
        margin: 1.5rem 0;
        font-family: 'Outfit', sans-serif;
    }
    
    .toxic {
        background: linear-gradient(135deg, #dc2626 0%, #991b1b 100%);
        border: 2px solid #ef4444;
        box-shadow: 0 0 30px rgba(239, 68, 68, 0.3);
    }
    
    .non-toxic {
        background: linear-gradient(135deg, #059669 0%, #047857 100%);
        border: 2px solid #10b981;
        box-shadow: 0 0 30px rgba(16, 185, 129, 0.3);
    }
    
    .result-label {
        font-size: 1.8rem;
        font-weight: 700;
        color: white;
        margin-bottom: 0.5rem;
    }
    
    .confidence-text {
        font-size: 1.1rem;
        color: rgba(255, 255, 255, 0.9);
    }
    
    .stTextArea textarea {
        font-family: 'JetBrains Mono', monospace;
        font-size: 1rem;
        background: #1e1e2e;
        border: 2px solid #3b3b5c;
        border-radius: 12px;
        color: #e2e8f0;
    }
    
    .stTextArea textarea:focus {
        border-color: #7c3aed;
        box-shadow: 0 0 20px rgba(124, 58, 237, 0.3);
    }
    
    .stButton > button {
        font-family: 'Outfit', sans-serif;
        font-weight: 600;
        font-size: 1.1rem;
        background: linear-gradient(90deg, #7c3aed, #6366f1);
        color: white;
        border: none;
        border-radius: 12px;
        padding: 0.75rem 2rem;
        width: 100%;
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 10px 30px rgba(124, 58, 237, 0.4);
    }
    
    .stats-container {
        display: flex;
        justify-content: center;
        gap: 2rem;
        margin-top: 1rem;
    }
    
    .stat-box {
        background: rgba(255, 255, 255, 0.05);
        padding: 1rem 1.5rem;
        border-radius: 12px;
        text-align: center;
        border: 1px solid rgba(255, 255, 255, 0.1);
    }
    
    .stat-value {
        font-family: 'JetBrains Mono', monospace;
        font-size: 1.5rem;
        font-weight: 700;
        color: #00d4ff;
    }
    
    .stat-label {
        font-family: 'Outfit', sans-serif;
        font-size: 0.85rem;
        color: #a0aec0;
        margin-top: 0.25rem;
    }
    
    .example-btn {
        background: rgba(255, 255, 255, 0.05);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 8px;
        padding: 0.5rem 1rem;
        color: #a0aec0;
        font-size: 0.85rem;
        cursor: pointer;
        transition: all 0.2s;
    }
    
    .example-btn:hover {
        background: rgba(255, 255, 255, 0.1);
        color: white;
    }
    
    .footer {
        text-align: center;
        color: #4a5568;
        font-size: 0.85rem;
        margin-top: 3rem;
        padding-top: 2rem;
        border-top: 1px solid rgba(255, 255, 255, 0.05);
    }
</style>
""", unsafe_allow_html=True)


# ============================================================================
# Load Model (Cached)
# ============================================================================

@st.cache_resource
def load_model():
    """Load the trained model (cached for performance)."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    try:
        tokenizer = AutoTokenizer.from_pretrained('toxic_classifier_model')
        model = AutoModelForSequenceClassification.from_pretrained('toxic_classifier_model')
        model = model.to(device)
        model.eval()
        return tokenizer, model, device, None
    except Exception as e:
        return None, None, None, str(e)


def predict_toxicity(text, tokenizer, model, device):
    """Predict toxicity of a comment."""
    encoding = tokenizer(
        text,
        add_special_tokens=True,
        max_length=256,
        padding='max_length',
        truncation=True,
        return_attention_mask=True,
        return_tensors='pt'
    )
    
    input_ids = encoding['input_ids'].to(device)
    attention_mask = encoding['attention_mask'].to(device)
    
    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        probabilities = torch.softmax(outputs.logits, dim=1)
        prediction = torch.argmax(outputs.logits, dim=1).item()
    
    return {
        'is_toxic': prediction == 1,
        'confidence': probabilities[0][prediction].item(),
        'toxic_prob': probabilities[0][1].item(),
        'non_toxic_prob': probabilities[0][0].item()
    }


# ============================================================================
# Main App
# ============================================================================

def main():
    # Header
    st.markdown('<h1 class="main-header">🛡️ Toxic Comment Classifier</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Powered by Transfer Learning with TinyBERT</p>', unsafe_allow_html=True)
    
    # Load model
    tokenizer, model, device, error = load_model()
    
    if error:
        st.error(f"""
        ⚠️ **Model not found!** 
        
        Please train the model first by running:
        ```
        python train.py
        ```
        
        Error: {error}
        """)
        return
    
    # Success indicator
    st.success(f"✓ Model loaded successfully (Device: {device})")
    
    # Text input
    st.markdown("### 💬 Enter a comment to analyze")
    
    comment = st.text_area(
        label="Comment",
        placeholder="Type or paste a comment here...",
        height=120,
        label_visibility="collapsed"
    )
    
    # Example buttons
    st.markdown("**Try an example:**")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("😊 Friendly", key="ex1"):
            st.session_state.example = "Great article! I really enjoyed reading this."
    with col2:
        if st.button("😠 Rude", key="ex2"):
            st.session_state.example = "You're an idiot, nobody cares about your opinion."
    with col3:
        if st.button("🤔 Neutral", key="ex3"):
            st.session_state.example = "I disagree with this point of view."
    
    # Handle example selection
    if 'example' in st.session_state:
        comment = st.session_state.example
        del st.session_state.example
        st.rerun()
    
    # Analyze button
    st.markdown("")  # Spacing
    analyze_clicked = st.button("🔍 Analyze Comment", type="primary", use_container_width=True)
    
    # Prediction
    if analyze_clicked and comment.strip():
        with st.spinner("Analyzing..."):
            result = predict_toxicity(comment, tokenizer, model, device)
        
        # Display result
        if result['is_toxic']:
            st.markdown(f"""
            <div class="result-box toxic">
                <div class="result-label">🚨 TOXIC</div>
                <div class="confidence-text">
                    This comment appears to be toxic or harmful.
                </div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="result-box non-toxic">
                <div class="result-label">✅ NON-TOXIC</div>
                <div class="confidence-text">
                    This comment appears to be safe and respectful.
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        # Stats
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Confidence", f"{result['confidence']*100:.1f}%")
        with col2:
            st.metric("Toxicity Score", f"{result['toxic_prob']*100:.1f}%")
        
        # Probability bar
        st.markdown("### 📊 Probability Distribution")
        st.progress(result['toxic_prob'])
        st.caption(f"← Non-Toxic ({result['non_toxic_prob']*100:.1f}%) | Toxic ({result['toxic_prob']*100:.1f}%) →")
    
    elif analyze_clicked:
        st.warning("Please enter a comment to analyze.")
    
    # Footer
    st.markdown("""
    <div class="footer">
        Built with Streamlit & HuggingFace Transformers<br>
        Model: TinyBERT fine-tuned on toxic comments dataset
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
