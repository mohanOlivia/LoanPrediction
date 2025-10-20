import pandas as pd


def load_data(path):
    return pd.read_csv(path)


def fill_missing_values(df):

    # Categorical    
    for col in ["Gender", "Married", "Self_Employed"]:
        df[col] = df[col].fillna(df[col].mode()[0], inplace=True)


    # Numerical 
    for col in ["LoanAmount", "Loan_Amount_Term", "ApplicantIncome", "CoapplicantIncome"]:
        if col in df.columns:
            df[col] = df[col].fillna(df[col].median())

    # Credit History - Boolean [Take mode of all booleans]
    df['Credit_History'].fillna(df['Credit_History'].mode()[0], inplace=True)


    return df

def encode_categorical_variables(df):
    return pd.get_dummies(df, columns=['Gender', 'Married', 'Education', 'Self_Employed', 'Property_Area'], drop_first=True)

def saveData(df, name):
    df = mend_columns(df)
    df.to_csv("data/processed/"+name+"_clean.csv", index=False)


def mend_columns(df):

    columns_drop = ['Loan_ID', 'Education_Not Graduate',  'Property_Area_Semiurban', 'Property_Area_Urban']
    
    df = fill_missing_values(df)
    df = encode_categorical_variables(df)

    # 3. Combine Property_Area columns if needed
    if 'Property_Area' not in df.columns:
        if 'Property_Area_Urban' in df.columns:
            df['Property_Area'] = df['Property_Area_Urban'].fillna(0)
    
    df = df.drop(columns = [col for col in columns_drop if col in df.columns])


    # 4. Encode categorical variables-
    # Binary mappings
    binary_map = {'Yes': 1, 'No': 0, 'Male': 1, 'Female': 0, 'Y': 1, 'N' : 0 }
    for col in ['Married', 'Self_Employed', 'Gender', 'Loan_Status']:
        if col in df.columns:
            df[col] = df[col].map(binary_map)
    
    # Dependents: convert '3+' to 3 and fill missing with 0
    if 'Dependents' in df.columns:
        df['Dependents'] = df['Dependents'].replace('3+', 3).astype(float)
        df['Dependents'] = df['Dependents'].fillna(0)

    
    # Education: binary encoding Graduate=1, Not Graduate=0
    if 'Education' in df.columns:
        df['Education'] = df['Education'].map({'Graduate': 1, 'Not Graduate': 0})
    
    # Property_Area: map Urban=1, Semiurban/Rural=0
    if 'Property_Area' in df.columns:
        df['Property_Area'] = df['Property_Area'].map(lambda x: 1 if x == 1 else 0)
    
    return df