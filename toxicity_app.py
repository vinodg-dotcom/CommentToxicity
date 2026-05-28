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

st.title("🛡️ Batch Comment Toxicity Classifier")
st.markdown("Upload a testing or validation dataset (.csv) to evaluate multiple records simultaneously using your trained LSTM/Bi-RNN framework.")

uploaded_file = st.file_uploader("Choose your test CSV file", type=['csv'])

if uploaded_file is not None:
    test_df = pd.read_csv(uploaded_file)
    
    st.success("File uploaded successfully!")
    st.write("Preview of your data:", test_df.head())
    
    
    if 'comment_text' not in test_df.columns:
        st.error("Error: The CSV file must contain a column named exactly 'comment_text'.")
    else:
        # processing button
        if st.button("Process File and Generate Predictions"):
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