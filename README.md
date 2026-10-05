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
- Natural Language Processing (NLP)
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
```

---

## 📊 Dataset

The project uses the **Twitter Entity Sentiment Analysis** dataset from Kaggle.

The dataset contains tweets belonging to four sentiment categories:

- Positive
- Negative
- Neutral
- Irrelevant

The raw dataset is not included in this repository.

---

## 🔄 Project Workflow

```text
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
Sequence Padding
       ↓
Embedding Layer
       ↓
Simple RNN
       ↓
Dropout
       ↓
Dense + Softmax
       ↓
Model Training
       ↓
Early Stopping
       ↓
Evaluation
       ↓
Save Model
       ↓
Streamlit Application
```

---

## 🧹 Data Preprocessing

The tweets were cleaned before training the model.

The preprocessing steps include:

- Converting text to lowercase
- Removing URLs
- Removing Twitter mentions
- Removing special characters
- Removing extra spaces
- Handling missing tweets
- Removing duplicate records where required

Example:

```text
Original:
"I LOVE this product!!! https://example.com"

After Cleaning:
"i love this product"
```

---

## 🔢 Label Encoding

Since machine learning models work with numerical values, sentiment labels were converted into numerical values using `LabelEncoder`.

The four sentiment classes were converted into integer labels for model training.

The `LabelEncoder` was also saved so that the numerical prediction can be converted back to the original sentiment label during inference.

---

## 🔤 Tokenization

The cleaned tweets were converted into sequences of integer IDs using the Keras `Tokenizer`.

An `<OOV>` token was used to handle words that were not present in the training vocabulary.

```python
tokenizer = Tokenizer(
    num_words=20000,
    oov_token="<OOV>"
)
```

The tokenizer was fitted only on the training data.

---

## 📏 Sequence Padding

Tweets have different lengths, but the neural network requires a fixed input size.

Therefore, the sequences were padded to a fixed maximum length of **70**.

```python
pad_sequences(
    sequences,
    maxlen=70,
    padding="post",
    truncating="post"
)
```

Shorter tweets are padded with zeros, while longer tweets are truncated.

---

## 🧠 Model Architecture

The project uses the following Deep Learning architecture:

```text
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
```

### Model Configuration

| Parameter | Value |
|---|---|
| Vocabulary Size | 20,000 |
| Embedding Dimension | 128 |
| RNN Units | 64 |
| RNN Dropout | 0.2 |
| Dropout | 0.5 |
| Output Classes | 4 |
| Output Activation | Softmax |
| Maximum Sequence Length | 70 |

The Softmax layer produces probabilities for the four sentiment classes.

---

## ⚙️ Model Compilation

The model was compiled using the Adam optimizer and Sparse Categorical Crossentropy loss.

```python
model.compile(
    optimizer=Adam(learning_rate=5e-4),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)
```

`Sparse Categorical Crossentropy` was used because the target labels are represented as integer values.

---

## 🚀 Model Training

The model was trained using:

- Batch Size: 64
- Maximum Epochs: 15
- Validation Split: 10%
- Early Stopping

Early stopping was used to prevent unnecessary training and reduce overfitting.

```python
EarlyStopping(
    monitor="val_loss",
    patience=2,
    restore_best_weights=True
)
```

---

## 📈 Model Performance

The model achieved approximately **79.25% accuracy** on the test dataset.

The model was evaluated using:

- Accuracy
- Precision
- Recall
- F1-score
- Confusion Matrix

---

## 💾 Model Saving

The trained model and preprocessing objects were saved for future predictions.

```text
model/
├── sentiment_model.keras
├── tokenizer.pkl
└── label_encode.pkl
```

### Saved Files

#### `sentiment_model.keras`

Contains the trained neural network architecture and learned weights.

#### `tokenizer.pkl`

Stores the tokenizer and vocabulary mapping used during training.

#### `label_encode.pkl`

Stores the mapping between numerical labels and sentiment names.

Keeping these files ensures that the same preprocessing used during training can be used during prediction.

---

## 🌐 Streamlit Application

A Streamlit web application was created to allow users to enter a tweet and receive a sentiment prediction.

The application performs the following steps:

```text
User enters tweet
       ↓
Text Cleaning
       ↓
Tokenization
       ↓
Padding
       ↓
Model Prediction
       ↓
Highest Probability Class
       ↓
Sentiment
```

### Run the Application

First, install the required dependencies:

```bash
pip install -r requirements.txt
```

Then run the Streamlit application:

```bash
streamlit run app.py
```

---

## 📦 Requirements

The main libraries used in this project are:

```text
tensorflow
numpy
pandas
scikit-learn
matplotlib
seaborn
streamlit
```

---

## 🔮 Future Improvements

The model can be improved further by:

- Replacing Simple RNN with LSTM or GRU
- Using Bidirectional LSTM
- Hyperparameter tuning
- Using pretrained word embeddings
- Experimenting with different sequence lengths
- Using Transformer-based models such as BERT
- Improving handling of long-term dependencies

---

## 🎯 Key Learning Outcomes

Through this project, the following concepts were implemented:

- Natural Language Processing
- Text preprocessing
- Label encoding
- Tokenization
- OOV handling
- Sequence padding
- Word embeddings
- Recurrent Neural Networks
- Multi-class classification
- Softmax activation
- Cross-entropy loss
- Adam optimizer
- Early stopping
- Model evaluation
- Model serialization
- Streamlit deployment

---

## ⭐ Conclusion

This project demonstrates an end-to-end Deep Learning pipeline for Twitter sentiment classification, starting from text preprocessing and tokenization to model training, evaluation, and deployment using Streamlit.

The trained model can classify tweets into **Positive, Negative, Neutral, and Irrelevant** sentiment categories with approximately **79.25% test accuracy**.

---

## 👩‍💻 Author

**Abhinav Kumar**

M.Tech – Computer Science and Engineering  
National Institute of Technology, Manipur