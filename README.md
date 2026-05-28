# 🛡️ Comment Toxicity Classifier Layout (LSTM & Bi-RNN)

An end-to-end deep learning framework built to detect multi-label comment toxicity using TensorFlow/Keras and deployed as an interactive batch-processing web application with Streamlit. 

This system automatically analyzes online text records and flags probability metrics across 6 distinct categories simultaneously: `toxic`, `severe_toxic`, `obscene`, `threat`, `insult`, and `identity_hate`.

---

## 📂 Project Structure

* **`CommentToxicity.ipynb`**: The foundational pipeline notebook where text cleanup data engineering occurs, tokenizers are registered, and both standard LSTM and Bidirectional Recurrent Neural Networks (Bi-RNN) are compiled, structured, trained, and exported.
* **`toxicity_app.py`**: The production-ready Streamlit frontend script. It loads the pre-trained assets, provides an interactive UI for dragging-and-dropping massive test datasets (.csv), handles out-of-vocabulary anomalies dynamically, and exports rounded percentage prediction spreadsheets.

---

## 🛠️ Installation & Dependency Environment Setup

Before executing the notebook or firing up the web interface, ensure you install the required computational packages. 

```bash
pip install streamlit tensorflow pandas numpy
