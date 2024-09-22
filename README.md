# Email Spam Classification

## Overview
This project is an implementation of a machine learning model to classify emails as either "Spam" or "Not Spam." It uses natural language processing (NLP) techniques and machine learning algorithms to build a predictive model.

## Features
- **Data preprocessing**: Clean and preprocess the dataset.
- **Vectorization**: Use of TF-IDF (Term Frequency-Inverse Document Frequency) vectorizer to transform the email text into a numerical form.
- **Model building**: Support Vector Machine (SVM) for classification.
- **Model evaluation**: Performance metrics including accuracy, precision, recall, F1 score, and ROC-AUC.

## Dependencies
The project requires the following libraries:
- `pandas`: For data manipulation.
- `sklearn`: For model building, training, and evaluation.
  - `train_test_split`, `GridSearchCV` for data splitting and hyperparameter tuning.
  - `TfidfVectorizer` for text vectorization.
  - `SVC` for the Support Vector Machine classifier.
  - `LabelEncoder` for encoding target labels.
  - Evaluation metrics like `accuracy_score`, `precision_score`, `recall_score`, `f1_score`, and `confusion_matrix`.

Install the necessary libraries by running:
```bash
pip install pandas scikit-learn
```

## Dataset
The dataset used in this project is a CSV file containing email data. It includes:
- `text`: The email content.
- `label`: The classification label, where "spam" indicates a spam email, and "ham" indicates a legitimate email.

The dataset is loaded using:
```python
df = pd.read_csv("path_to_dataset.csv", encoding='ISO-8859-1')
```

## Model Building Steps
1. **Data Preprocessing**: 
   - Clean the dataset by handling missing values and encoding the target labels.
2. **Text Vectorization**: 
   - Use the TF-IDF vectorizer to transform email text into numerical format.
3. **Model Training**: 
   - Apply SVM for classification, optimizing with GridSearchCV for the best hyperparameters.
4. **Evaluation**: 
   - Evaluate the model using metrics like accuracy, precision, recall, F1 score, and ROC-AUC.

## How to Run
1. Clone this repository.
2. Install the dependencies listed above.
3. Run the Jupyter notebook to preprocess the data, train the model, and evaluate its performance.

## Performance Metrics
The model is evaluated based on several metrics, including:
- **Accuracy**: Measures the overall correctness of the model.
- **Precision**: The number of correctly predicted spam emails out of all emails classified as spam.
- **Recall**: The number of correctly predicted spam emails out of all actual spam emails.
- **F1 Score**: The harmonic mean of precision and recall.
- **ROC-AUC**: Measures the trade-off between true positive rate and false positive rate.
