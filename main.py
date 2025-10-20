from src.data_processing import load_data, saveData
from src.modeling import train_model, evaluate_model
from sklearn.model_selection import train_test_split
import pandas as pd

train_path = "C:/Users/livmo/OneDrive - Trinity College Dublin/LoanPrediction/LoanPrediction/data/train.csv"
test_path = "C:/Users/livmo/OneDrive - Trinity College Dublin/LoanPrediction/LoanPrediction/data/test.csv"

# Load & clean data
# train data
df_train = load_data(train_path)
saveData(df_train, "df_train")

# Test Data
df_test = load_data(test_path)
saveData(df_test, "df_test")


# # Features & target
# X = df_train .drop('Loan_Status', axis=1)
# y = df_train ['Loan_Status']

# # Train/test split
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# # Train & evaluate
# model = train_model(X_train, y_train)
# accuracy, roc = evaluate_model(model, X_test, y_test)
# print("Accuracy:", accuracy, "ROC-AUC:", roc)
