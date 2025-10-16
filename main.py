from src.data_processing import load_data, fill_missing_values
from src.modeling import train_model, evaluate_model
from sklearn.model_selection import train_test_split

train_path = "C:/Users/livmo/OneDrive - Trinity College Dublin/LoanPrediction/LoanPrediction/train.csv"
test_path = "C:/Users/livmo/OneDrive - Trinity College Dublin/LoanPrediction/LoanPrediction/test.csv"

# Load & clean data
df_train = load_data(train_path)
df_train = fill_missing_values(df_train)

# Features & target
X = df_train .drop('Loan_Status', axis=1)
y = df_train ['Loan_Status']

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train & evaluate
model = train_model(X_train, y_train)
accuracy, roc = evaluate_model(model, X_test, y_test)
print("Accuracy:", accuracy, "ROC-AUC:", roc)

# Train
# print(train_df.head())
# print(train_df.info())
# print(train_df.describe())
print(df_train.isna().sum())
#print(train_df.duplicated().sum()) # == 0 , no duplicated 

# Test 
# print(test_df.head())
# print(test_df.info())
# print(test_df.describe())
# print(df_test.isna().sum())
# print(test_df.duplicated().sum()) # == 0 , no duplicated 

# df = pd.read_csv('dataset.csv')
# df.head()
# df.info()
# df.describe()
# df.isna().sum()
