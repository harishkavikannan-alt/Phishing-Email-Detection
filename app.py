"""
Phishing Email Detection - Streamlit Web Application
====================================================
An interactive, beginner-friendly web interface to classify emails as
Phishing or Safe using Scikit-Learn (TF-IDF + Logistic Regression).

Author: Antigravity Pair Programmer
Project: Phishing Email Detection Model
"""

import os
import joblib
import pandas as pd
import streamlit as st

# -----------------------------------------------------------------------------
# 1. Streamlit Page Configuration
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Phishing Email Detection Model",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for polished, modern aesthetics
st.markdown("""
<style>
    /* Main container styling */
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
    /* Metric Card Styling */
    .metric-card {
        background-color: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-radius: 10px;
        padding: 16px;
        text-align: center;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    }
    .metric-label {
        font-size: 0.85rem;
        font-weight: 600;
        color: #64748B;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    .metric-value {
        font-size: 1.7rem;
        font-weight: 800;
        color: #0F172A;
        margin-top: 4px;
    }
    /* Result Badges */
    .phishing-banner {
        background: linear-gradient(135deg, #FEF2F2 0%, #FEE2E2 100%);
        border: 2px solid #EF4444;
        border-radius: 12px;
        padding: 24px;
        color: #991B1B;
        margin-top: 15px;
        margin-bottom: 15px;
    }
    .safe-banner {
        background: linear-gradient(135deg, #F0FDF4 0%, #DCFCE7 100%);
        border: 2px solid #22C55E;
        border-radius: 12px;
        padding: 24px;
        color: #166534;
        margin-top: 15px;
        margin-bottom: 15px;
    }
    /* Keyword chip */
    .keyword-chip {
        display: inline-block;
        background-color: #FEE2E2;
        color: #B91C1C;
        border-radius: 9999px;
        padding: 4px 12px;
        font-size: 0.85rem;
        font-weight: 600;
        margin: 4px;
        border: 1px solid #FCA5A5;
    }
    /* Disclaimer Box */
    .disclaimer-box {
        background-color: #FFFBEB;
        border-left: 4px solid #F59E0B;
        padding: 12px 16px;
        border-radius: 4px;
        font-size: 0.88rem;
        color: #92400E;
        margin: 12px 0;
    }
</style>
""", unsafe_allow_html=True)

MODEL_FILE = "phishing_model.pkl"

# -----------------------------------------------------------------------------
# 2. Model Loading Helper
# -----------------------------------------------------------------------------
@st.cache_resource
def load_trained_model():
    """Loads the serialized model payload from disk."""
    if not os.path.exists(MODEL_FILE):
        return None
    try:
        data = joblib.load(MODEL_FILE)
        return data
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

model_data = load_trained_model()

# -----------------------------------------------------------------------------
# 3. Preset Test Samples
# -----------------------------------------------------------------------------
SAMPLE_TEMPLATES = {
    "-- Select an example email to test --": "",
    "🚨 Phishing: Urgent Bank Account Lock": (
        "URGENT: Your Chase bank account has been locked due to unauthorized login attempts. "
        "Verify your personal credentials immediately at our secure link to restore access before 24 hours: "
        "http://secure-chase-account-verify.net"
    ),
    "🚨 Phishing: PayPal Account Suspension": (
        "Notice: Your PayPal account has been restricted due to suspicious transactions. "
        "To lift this limitation, update your credit card details and identity information right away."
    ),
    "🚨 Phishing: IT Mailbox Migration": (
        "URGENT IT NOTICE: We are migrating email servers tonight. You must log in here with your "
        "corporate username and password to prevent permanent mailbox deletion."
    ),
    "🚨 Phishing: Unclaimed Bitcoin Lottery": (
        "Congratulations! You have been selected to receive 0.50 Bitcoin in our international lottery giveaway. "
        "Click here to submit your private wallet address and claim your funds now."
    ),
    "🛡️ Safe: Project Progress Report": (
        "Hey team, here is the weekly project progress report. We have successfully completed Sprint 14 "
        "and merged the latest feature branches. Please review the updated milestones before tomorrow's sync."
    ),
    "🛡️ Safe: Casual Lunch Invite": (
        "Hey everyone, a few of us are heading out for lunch at the new taco spot down the street around 12:30. "
        "Let me know if you would like to join us!"
    ),
    "🛡️ Safe: Dental Appointment Reminder": (
        "Reminder: Your upcoming routine dental cleaning appointment is scheduled for Tuesday, "
        "November 12th at 10:00 AM at Greenfield Dental Clinic. Call us if you need to reschedule."
    ),
    "🛡️ Safe: GitHub Pull Request Merge": (
        "GitHub notification: Pull request #42 has been successfully reviewed and merged into the main "
        "branch by Sarah. All 18 automated test suites passed successfully."
    )
}

COMMON_PHISHING_KEYWORDS = [
    "urgent", "immediately", "locked", "verify", "suspended", "restriction",
    "password", "credentials", "credit card", "bank", "unauthorized", "wire transfer",
    "lottery", "unclaimed", "winner", "bitcoin", "crypto", "irs", "refund",
    "click here", "overdue", "invoice", "payroll", "action required"
]

# -----------------------------------------------------------------------------
# 4. Sidebar: Navigation, Metrics & Crucial Disclaimer
# -----------------------------------------------------------------------------
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/shield.png", width=64)
    st.title("Phishing Detector")
    st.caption("TF-IDF + Logistic Regression")
    
    st.markdown("---")
    st.subheader("📊 Model Evaluation Metrics")
    
    if model_data and "metrics" in model_data:
        m = model_data["metrics"]
        col_m1, col_m2 = st.columns(2)
        with col_m1:
            st.metric(label="Accuracy", value=f"{m['accuracy'] * 100:.1f}%")
            st.metric(label="Recall", value=f"{m['recall'] * 100:.1f}%")
        with col_m2:
            st.metric(label="Precision", value=f"{m['precision'] * 100:.1f}%")
            st.metric(label="F1-Score", value=f"{m['f1_score'] * 100:.1f}%")
            
        with st.expander("🔍 View Confusion Matrix", expanded=False):
            cm = m.get("confusion_matrix", [[0, 0], [0, 0]])
            cm_df = pd.DataFrame(
                cm,
                index=["Actual Safe", "Actual Phishing"],
                columns=["Pred Safe", "Pred Phishing"]
            )
            st.dataframe(cm_df, use_container_width=True)
            st.caption(f"Tested on {m.get('test_samples', 0)} unseen samples.")
    else:
        st.warning("Model file not found or metrics unavailable. Please run `python train_model.py` first.")

    st.markdown("---")
    st.subheader("⚠️ Important Disclaimer")
    st.markdown(
        """
        <div class="disclaimer-box">
            <strong>Educational Demonstration Only:</strong><br>
            The accuracy and metrics displayed above are evaluated on a curated sample dataset (50 emails).
            <br><br>
            <strong>These results DO NOT prove real-world accuracy.</strong>
            Sophisticated production attacks use evasive techniques, zero-day spoofing, and advanced evasion
            that require multi-layered enterprise defense (SPF/DKIM/DMARC headers, URL scanners, and massive threat intelligence).
        </div>
        """,
        unsafe_allow_html=True
    )
    
    st.markdown("---")
    st.markdown("**How It Works:**")
    st.markdown("1. **TF-IDF**: Extracts word & bigram importance.")
    st.markdown("2. **Logistic Regression**: Computes phishing probability.")

# -----------------------------------------------------------------------------
# 5. Main UI Layout
# -----------------------------------------------------------------------------
st.markdown('<div class="main-header">🛡️ Phishing Email Detection Model</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-header">'
    'Classify suspicious messages in real-time using natural language processing (NLP) and machine learning.'
    '</div>',
    unsafe_allow_html=True
)

if not model_data:
    st.error(
        "⚠️ **Trained model (`phishing_model.pkl`) not found!**\n\n"
        "Please train the model first by opening your terminal and running:\n\n"
        "```bash\n"
        "python train_model.py\n"
        "```\n\n"
        "After training, refresh this page to begin analyzing emails."
    )
    st.stop()

# Preset Selector
selected_sample = st.selectbox(
    "💡 Choose a quick example or write your own below:",
    options=list(SAMPLE_TEMPLATES.keys()),
    index=0
)

# Text Area prefilled if template selected
default_text = SAMPLE_TEMPLATES[selected_sample] if selected_sample in SAMPLE_TEMPLATES else ""
email_input = st.text_area(
    "✉️ Email Content (Subject + Body):",
    value=default_text,
    height=180,
    placeholder="Paste the email subject and body text here to analyze..."
)

col_btn1, col_btn2 = st.columns([1, 5])
with col_btn1:
    analyze_clicked = st.button("🔍 Analyze Email", type="primary", use_container_width=True)
with col_btn2:
    if st.button("🔄 Clear Text", use_container_width=False):
        st.rerun()

# -----------------------------------------------------------------------------
# 6. Classification & Prediction Display
# -----------------------------------------------------------------------------
if analyze_clicked:
    if not email_input.strip():
        st.warning("Please enter or paste some email text before analyzing.")
    else:
        pipeline = model_data["pipeline"]
        classes = list(pipeline.classes_)

        # Predict class and probability
        prediction = pipeline.predict([email_input])[0]
        probabilities = pipeline.predict_proba([email_input])[0]

        # Map probabilities to classes
        prob_dict = {cls: prob for cls, prob in zip(classes, probabilities)}
        phishing_prob = prob_dict.get("Phishing", 0.0) * 100
        safe_prob = prob_dict.get("Safe", 0.0) * 100

        # Scan for detected trigger keywords
        lower_input = email_input.lower()
        found_keywords = [kw for kw in COMMON_PHISHING_KEYWORDS if kw in lower_input]

        st.markdown("---")
        st.subheader("Analysis Results")

        if prediction == "Phishing":
            st.markdown(
                f"""
                <div class="phishing-banner">
                    <h2 style="margin: 0; color: #991B1B;">🚨 PHISHING EMAIL DETECTED</h2>
                    <p style="font-size: 1.15rem; margin-top: 8px; margin-bottom: 0;">
                        This email exhibits characteristics and wording commonly found in phishing or social engineering attacks.
                    </p>
                    <p style="font-size: 1rem; margin-top: 6px; font-weight: 600;">
                        Phishing Confidence: {phishing_prob:.1f}%
                    </p>
                </div>
                """,
                unsafe_allow_html=True
            )
        else:
            st.markdown(
                f"""
                <div class="safe-banner">
                    <h2 style="margin: 0; color: #166534;">🛡️ SAFE EMAIL DETECTED</h2>
                    <p style="font-size: 1.15rem; margin-top: 8px; margin-bottom: 0;">
                        This email appears legitimate based on vocabulary patterns. Always verify sender address and links independently.
                    </p>
                    <p style="font-size: 1rem; margin-top: 6px; font-weight: 600;">
                        Safe Confidence: {safe_prob:.1f}%
                    </p>
                </div>
                """,
                unsafe_allow_html=True
            )

        # Confidence Metrics Display
        col_res1, col_res2 = st.columns(2)
        with col_res1:
            st.markdown(f"**Phishing Probability:** `{phishing_prob:.2f}%`")
            st.progress(int(phishing_prob))
        with col_res2:
            st.markdown(f"**Safe Probability:** `{safe_prob:.2f}%`")
            st.progress(int(safe_prob))

        # Highlight detected suspicious keywords
        st.markdown("#### 🔎 Suspicious Keywords & Triggers")
        if found_keywords:
            st.write("The model and heuristic scanner identified the following trigger words:")
            chips_html = "".join([f'<span class="keyword-chip">{kw}</span>' for kw in found_keywords])
            st.markdown(chips_html, unsafe_allow_html=True)
        else:
            st.info("No standard high-risk keywords detected in the message.")

        # Educational Guidance
        with st.expander("💡 Red Flags to Check in Real-World Emails", expanded=True):
            st.markdown("""
            Even when automated models classify an email, always verify:
            * **Sender Email Address**: Does the sender domain match the official company domain (e.g. `@support.paypal.com` vs `@paypal-security-alert.xyz`)?
            * **Artificial Urgency**: Are they pressuring you to act within minutes/hours under threat of account loss?
            * **Hidden Hyperlinks**: Hover over links without clicking to see their actual destination URL.
            * **Generic Greetings**: Generic greetings like "Dear Customer" instead of your actual name.
            * **Requests for Sensitive Data**: Legitimate institutions will rarely ask for passwords or PINs via email.
            """)
