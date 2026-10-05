# Sentiment Analysis Using Deep Learning

A Deep Learning-based Twitter Sentiment Analysis project that classifies tweets into four sentiment categories: **Positive, Negative, Neutral, and Irrelevant**.

The project uses **Natural Language Processing (NLP)**, **Word Embeddings**, and a **Simple RNN** built with TensorFlow/Keras. A Streamlit web application is also provided for real-time sentiment prediction.

---

## 📌 Project Overview

Social media platforms contain a huge amount of text data. Understanding the sentiment behind these texts can help identify whether users are expressing positive, negative, neutral, or irrelevant opinions.

In this project, a Twitter sentiment classification model is trained using a dataset containing tweets and their corresponding sentiment labels.

The complete pipeline includes:

- Text preprocessing
- Label encoding
- Tokenization
- Sequence padding
- Word embeddings
- Simple RNN
- Model training
- Early stopping
- Model evaluation
- Streamlit deployment

---

## 🛠️ Technologies Used

- Python
- TensorFlow / Keras
- NumPy
- Pandas
- Scikit-learn
- Matplotlib
- Seaborn
- Streamlit
- NLP
- Simple RNN

---

## 📂 Project Structure

```text
Sentiment_Analysis_Using_DeepLearing/
│
├── notebooks/
│   └── sentiment_analysis.ipynb
│
├── model/
│   ├── sentiment_model.keras
│   ├── tokenizer.pkl
│   └── label_encode.pkl
│
├── app.py
├── requirements.txt
├── README.md
└── .gitignore


The project uses the Twitter Sentiment Analysis dataset from Kaggle.

The dataset contains tweets belonging to four sentiment categories:

Positive
Negative
Neutral
Irrelevant

The raw dataset is not included in this repository.

🔄 Project Workflow

Twitter Dataset
       ↓
Data Cleaning
       ↓
Train-Test Split
       ↓
Label Encoding
       ↓
Tokenization
       ↓
Padding
       ↓
Embedding Layer
       ↓
Simple RNN
       ↓
Dense + Softmax
       ↓
Model Training
       ↓
Evaluation
       ↓
Save Model
       ↓
Streamlit Application

🧹 Data Preprocessing

The tweets were cleaned before training the model.

The preprocessing steps include:

Converting text to lowercase
Removing URLs
Removing Twitter mentions
Removing special characters
Removing extra spaces
Handling missing tweets
Removing duplicate records where required


Label Encoding
Tokenization
Sequence Padding

Model Architecture

Input
  ↓
Embedding Layer
  ↓
Simple RNN
  ↓
Dropout
  ↓
Dense Layer
  ↓
Softmax


Model Performance: The model achieved approximately 79.25% accuracy on the test dataset.

Run the Application: streamlit run app.py

The main libraries used in this project are:
tensorflow
numpy
pandas
scikit-learn
matplotlib
seaborn
streamlit