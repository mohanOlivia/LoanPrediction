import pandas as pd


def load_data(path):
    return pd.read_csv(path)


def fill_missing_values(df):

    # Categorical    
    for col in ["Gender", "Married", "Self_Employed"]:
        df[col] = df[col].fillna(df[col].mode()[0], inplace=True)


    # Numerical 
    for col in ["LoanAmount", "Loan_Amount_Term"]:
        df[col] = df[col].fillna(df[col].median(), inplace=True)

    # Credit History - Boolean [Take mode of all booleans]
    df['Credit_History'].fillna(df['Credit_History'].mode()[0], inplace=True)

    return df
