def encode_data(df):
    X = df[["CreditScore", "Age", "HasCrCard", "IsActiveMember", "NumOfProducts", "Tenure"]]
    X_ = pd.get_dummies(df["Geography"], drop_first=True)
    X = pd.concat([X, X_["Germany"]], axis=1)
    X_ = pd.get_dummies(df["Gender"], drop_first=True)
    X = pd.concat([X, X_], axis=1)
    y = df[target_col]
    return X, y