import pandas as pd


def load_data(path):
    return pd.read_csv(path)


def fill_missing_values(df):

    # Categorical    
    for col in ["Gender", "Married", "Self_Employed"]:
        if col in df.columns:
            mode_val = df[col].mode()
            df[col] = df[col].fillna(mode_val.iloc[0])
            # Also replace None with mode
            df[col] = df[col].replace({None: mode_val.iloc[0]})
        else:
            # If no mode exists, use a sensible default
            if col == 'Gender':
                df[col] = df[col].fillna('Male')
            elif col in ['Married', 'Self_Employed', 'Loan_Status']:
                df[col] = df[col].fillna('No')


    # Numerical 
    for col in ["LoanAmount", "Loan_Amount_Term", "ApplicantIncome", "CoapplicantIncome"]:
        if col in df.columns:
            df[col] = df[col].fillna(df[col].median())

    # Credit History - Boolean [Take mode of all booleans]
    df['Credit_History'].fillna(df['Credit_History'].mode()[0], inplace=True)


    return df

# def encode_categorical_variables(df):
#    return pd.get_dummies(df, columns=['Gender', 'Married', 'Education', 'Self_Employed', 'Property_Area'])

def saveData(df, name):
    df = mend_columns(df)
    df.to_csv("data/processed/"+name+"_clean.csv", index=False)


def mend_columns(df):

    columns_drop = ['Loan_ID']
    
    df = fill_missing_values(df)
    
    df = df.drop(columns = [col for col in columns_drop if col in df.columns])


    binary_cols = ['Married', 'Self_Employed', 'Gender', 'Loan_Status']
    
    # 4. Encode categorical variables
    # Expanded binary mappings to handle case variations
    binary_map = {
        'Yes': 1, 'No': 0, 'Male': 1, 'Female': 0, 'Y': 1, 'N': 0,
        'yes': 1, 'no': 0, 'male': 1, 'female': 0, 'y': 1, 'n': 0
    }
    
    for col in binary_cols:
        if col in df.columns:
            df[col] = df[col].map(binary_map)
            # Check what happened after mapping
            print(f"After mapping - {col}: {df[col].unique()}")
            print(f"NaN count in {col}: {df[col].isna().sum()}")
    
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