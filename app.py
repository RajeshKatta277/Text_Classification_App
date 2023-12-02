from flask import Flask, render_template, request
import joblib
import numpy as np
import requests
from io import BytesIO
import re
import PyPDF2
from nltk.tokenize import word_tokenize
import nltk
nltk.download('stopwords')
nltk.download('punkt')
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

def preprocess_text(text):
    stop_words = set(stopwords.words('english'))
    stemmer = PorterStemmer()
    text = text.lower()
    tokens = word_tokenize(text)
    tokens = [stemmer.stem(token) for token in tokens if token.isalnum() and token not in stop_words]
    preprocessed_text = ' '.join(tokens)
    return preprocessed_text


def extract_and_preprocess_text(url):
    try:
        response = requests.get(url)
        with BytesIO(response.content) as f:
            pdf_reader = PyPDF2.PdfReader(f)
            text = ''
            for page in pdf_reader.pages:
                text += page.extract_text()
        return preprocess_text(text)
    except Exception as e:
        print(f"Error processing URL: {e}")
        return "Error"    


def predict_product_type(pdf_text):
    # Transform the text using the loaded vectorizer
    text_vectorized = vectorizer.transform([pdf_text])
    
    # Make predictions using the loaded model
    predictions = model.predict(text_vectorized)
    probabilities = model.predict_proba(text_vectorized)
    
    return np.round(predictions[0],3), np.round(probabilities[0],3)
    
    
# Load the RandomForest model and Bag of Words vectorizer
model = joblib.load('rf_model_bg.pkl')
vectorizer = joblib.load('bg_vectorizer_main.pkl')


app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/extract', methods=['POST'])
def extract_data():
    pdf_url = request.form['pdf_url']
    pdf_text = extract_and_preprocess_text(pdf_url)
    return render_template('result.html', text=pdf_text)

@app.route('/product type', methods=['POST'])
def get_label():
    pdf_url = request.form['pdf_url']
    pdf_text = extract_and_preprocess_text(pdf_url)
    label, probabilities = predict_product_type(pdf_text)
    label_map={1:"Lighting",2:"Non-Lighting"}
    label=label_map[label]
    return render_template('label.html', label=label, probabilities=probabilities)

if __name__ == '__main__':
    app.run(debug=True)

