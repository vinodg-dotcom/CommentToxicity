import streamlit as st
import numpy as np
import tensorflow as tf
import pickle
import re
import pandas as pd
from tensorflow.keras.preprocessing.sequence import pad_sequences

@st.cache_resource
def load_assets():
    with open('tokenizer.pkl', 'rb') as f:
        tokenizer = pickle.load(f)
    # Run through below two models    
    #model = tf.keras.models.load_model('lstm_toxic_model.h5')
    model = tf.keras.models.load_model('birnn_toxic_model.h5')
    return tokenizer, model

tokenizer, model = load_assets()
labels = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']

st.title("🛡️ Toxicity Classifier Dashboard")
st.markdown("Evaluate comments using your trained LSTM/Bi-RNN framework. Choose between real-time or batch analysis below.")

# Create tabs for a clean user interface
tab1, tab2 = st.tabs(["💬 Real-time Prediction", "📁 CSV-file dataset prediction"])

# --- TAB 1: REAL-TIME SINGLE COMMENT PREDICTION ---
with tab1:
    st.header("Real-time Comment Analysis")
    st.markdown("Enter any comment below to receive real-time toxicity analysis.")
    
    user_comment = st.text_area("Type your comment here:", placeholder="e.g., Have a great day!")
    
    if st.button("Analyze Comment", key="realtime_btn"):
        if user_comment.strip() == "":
            st.warning("Please enter a valid comment before processing.")
        else:
            with st.spinner("Analyzing comment..."):
                # Clean the single input comment
                cleaned_comment = re.sub('[^a-zA-Z0-9 ]', '', user_comment.lower())
                
                # Convert to numeric sequences
                seqs = tokenizer.texts_to_sequences([cleaned_comment])
                
                # Apply the same token limit filter (< 10000)
                inner_cleaned_list = []    
                for token in seqs[0]:        
                    if token < 10000:
                        inner_cleaned_list.append(token)
                    else:
                        inner_cleaned_list.append(1)            
                
                # Pad sequences to matching maxlen
                padded = pad_sequences([inner_cleaned_list], maxlen=100, padding='post', truncating='post')
                
                # Run prediction
                raw_predictions = model.predict(padded)[0]
                
                # Display results
                st.subheader("Analysis Results")
                
                # Tabular format
                results_df = pd.DataFrame({
                    'Toxicity Category': labels,
                    'Probability (%)': [np.round(p * 100, 2) for p in raw_predictions]
                })
                st.dataframe(results_df, use_container_width=True)
                
                # Visual bar chart representation
                st.bar_chart(results_df.set_index('Toxicity Category'))


# --- TAB 2: CSV File Processing ---
with tab2:
    st.header("Batch Comment Toxicity Classifier")
    st.markdown("Upload a testing or validation dataset (.csv) to evaluate multiple records simultaneously.")

    uploaded_file = st.file_uploader("Choose your test CSV file", type=['csv'])

    if uploaded_file is not None:
        test_df = pd.read_csv(uploaded_file)
        
        st.success("File uploaded successfully!")
        st.write("Preview of your data:", test_df.head())
        
        if 'comment_text' not in test_df.columns:
            st.error("Error: The CSV file must contain a column named exactly 'comment_text'.")
        else:
            if st.button("Process File and Generate Predictions", key="batch_btn"):
                with st.spinner("Running batch predictions through LSTM/Bi-RNN..."):
                    
                    cleaned_comments = test_df['comment_text'].astype(str).str.lower().str.replace('[^a-zA-Z0-9 ]', '', regex=True)
                    
                    # Convert to numeric sequences
                    seqs = tokenizer.texts_to_sequences(cleaned_comments)
                    
                    cleaned_seqs = []
                    for text_list in seqs:    
                        inner_cleaned_list = []    
                        for token in text_list:        
                            if token < 10000:
                                inner_cleaned_list.append(token)
                            else:
                                inner_cleaned_list.append(1)            
                        cleaned_seqs.append(inner_cleaned_list)
                    
                    padded = pad_sequences(cleaned_seqs, maxlen=100, padding='post', truncating='post')
                    
                    # Run the dataset through the model
                    raw_predictions = model.predict(padded)
                    
                    # Append predictions back to the DataFrame as percentages
                    for idx, label in enumerate(labels):
                        test_df[f'{label}_prob_%'] = np.round(raw_predictions[:, idx] * 100, 2)
                    
                    st.write("Predictions Completed! Preview of results:")
                    st.write(test_df.head())
                    
                    csv_output = test_df.to_csv(index=False).encode('utf-8')
                    
                    st.download_button(
                        label="📥 Download Classified Predictions CSV",
                        data=csv_output,
                        file_name="toxic_predictions_output.csv",
                        mime="text/csv"
                    )