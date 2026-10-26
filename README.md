# 🛡️ Phishing Email Detection Model

A beginner-friendly Machine Learning project that classifies emails as either **Phishing** or **Safe** using Natural Language Processing (NLP) with **TF-IDF** and **Logistic Regression** in Python and Scikit-Learn. It includes a training and evaluation script and an interactive web interface powered by **Streamlit**.

---

## 📁 Project Structure

```
Phishing-Email-Detection/
│
├── emails.csv            # Sample dataset containing labeled emails (Phishing / Safe)
├── train_model.py        # Model training, evaluation, metrics display & export
├── app.py                # Interactive Streamlit web user interface
├── requirements.txt      # Python dependencies
└── README.md             # Project documentation & setup guide
```

---

## ⚙️ How It Works (The ML Pipeline)

1. **Dataset (`emails.csv`)**:
   Contains curated email samples labeled as `Phishing` or `Safe`.

2. **Feature Extraction (`TF-IDF Vectorizer`)**:
   * Converts raw email text into numerical feature vectors.
   * Uses both individual words (unigrams) and word pairs (bigrams), such as *"verify account"*, *"urgent notice"*, and *"unclaimed prize"*.
   * Down-weights ubiquitous common words via Inverse Document Frequency.

3. **Classification (`Logistic Regression`)**:
   * Learns a linear decision boundary separating phishing messages from legitimate communications.
   * Outputs both a predicted category (`Phishing` vs `Safe`) and a calibrated probability score.

4. **Model Serialization**:
   * Uses `joblib` to bundle the vectorizer, trained classifier, and test evaluation metrics into `phishing_model.pkl`.

---

## 📊 Evaluation Metrics Explained

When running `train_model.py`, the model computes:

* **Accuracy**: The overall percentage of correctly classified emails (both Phishing and Safe).
* **Precision**: When the model flags an email as *Phishing*, how often is it actually phishing? High precision means fewer false alarms.
* **Recall (Sensitivity)**: Out of all the actual malicious *Phishing* emails, how many did the model catch? High recall means fewer phishing emails slip through undetected.
* **F1-Score**: The harmonic mean of Precision and Recall, representing the balance between both metrics.
* **Confusion Matrix**: A 2x2 grid showing:
  * **True Negatives (TN)**: Actual Safe predicted as Safe
  * **False Positives (FP)**: Actual Safe wrongly flagged as Phishing
  * **False Negatives (FN)**: Actual Phishing missed and flagged as Safe
  * **True Positives (TP)**: Actual Phishing correctly caught as Phishing

---

## 🚀 Setup & Execution Instructions

Follow these step-by-step instructions in your terminal or command prompt:

### Step 1: Open Terminal in Project Folder
Ensure your terminal is located in the `Phishing-Email-Detection` directory:
```bash
cd d:/Phishing-Email-Detection
```

### Step 2: (Recommended) Create & Activate a Virtual Environment
* **On Windows (PowerShell):**
  ```powershell
  python -m venv venv
  .\venv\Scripts\Activate.ps1
  ```
* **On Windows (Command Prompt):**
  ```cmd
  python -m venv venv
  venv\Scripts\activate.bat
  ```
* **On macOS / Linux:**
  ```bash
  python3 -m venv venv
  source venv/bin/activate
  ```

### Step 3: Install Required Dependencies
```bash
pip install -r requirements.txt
```

### Step 4: Train and Evaluate the Model
Run the training script to train the model, print the evaluation metrics, and export `phishing_model.pkl`:
```bash
python train_model.py
```

### Step 5: Launch the Streamlit Web Application
Run the interactive web dashboard:
```bash
streamlit run app.py
```
After running this command, your browser will automatically open at `http://localhost:8501`.

---

## ⚠️ CRITICAL DISCLAIMER: Educational Sample Dataset vs. Real-World Accuracy

> [!WARNING]
> **Sample dataset results DO NOT prove real-world accuracy or readiness for enterprise defense.**

1. **Dataset Scope**:
   * This project uses an educational sample dataset (`emails.csv` with 50 samples) designed to teach fundamental machine learning and NLP concepts in a simple, reproducible manner.
   * While the model may achieve high accuracy or F1-scores on this test split, this reflects performance on a small, controlled sample—**not** real-world email streams.

2. **Real-World Phishing Complexity**:
   * **Advanced Evasion**: Attackers frequently use typosquatting, zero-font techniques, homoglyphs, and image-based/QR-code phishing (*quishing*) to bypass keyword-based classifiers.
   * **Generative AI & Spear Phishing**: Modern attacks are often personalized, well-written, and devoid of traditional grammatical errors or blunt urgency cues.
   * **Header & Domain Authentication**: Production email security gateways (e.g., Google Workspace, Microsoft Defender) rely heavily on technical headers:
     * **SPF** (Sender Policy Framework)
     * **DKIM** (DomainKeys Identified Mail)
     * **DMARC** (Domain-based Message Authentication, Reporting, and Conformance)
     * Real-time domain age and reputation scoring
     * Dynamic sandboxing of attachments and redirect-chain URL analysis

3. **Production Best Practices**:
   * For real-world deployments, models must be trained on continuously updated threat feeds with hundreds of thousands or millions of emails, combined with multi-layered technical security controls.

---

## 📝 Technologies Used
* **Python 3.8+**
* **Scikit-Learn** (TF-IDF Vectorizer, Logistic Regression, Pipeline, Model Evaluation)
* **Pandas & NumPy** (Dataset manipulation)
* **Joblib** (Model serialization)
* **Streamlit** (Interactive web application)
