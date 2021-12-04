from sklearn.model_selection import cross_val_score
from sklearn.metrics import confusion_matrix, roc_curve, auc

def scores(model, X=X, y=y, cv=5):
    # CV Scores
    print(" Scores ")
    print("---------------------")
    precision_ = np.mean(cross_val_score(model, X, y, cv=cv, scoring="precision"))
    print(f"CV-Precision: {precision_:.2f}")
    f1_ = np.mean(cross_val_score(model, X, y, cv=cv, scoring="f1"))
    print(f"CV-F1: {f1_:.2f}")
    recall_ = np.mean(cross_val_score(model, X, y, cv=cv, scoring="recall"))
    print(f"CV-Recall: {recall_:.2f}")
    roc_auc_ = np.mean(cross_val_score(model, X, y, cv=cv, scoring="roc_auc"))
    print(f"CV-ROC AUC: {roc_auc_:.2f}")
    # Confusion Matrix
    model.fit(X, y)
    probs = model.predict_proba(X)
    y_pred = [0 if x < 0.5 else 1 for x in probs[:, 1]]
    confusion_matrix_ = confusion_matrix(y, y_pred)
    print(f"Confusion Matrix:\n{confusion_matrix_}")
    # ROC Curve
    fpr, tpr, threshold = roc_curve(y, probs[:, 1])
    roc_auc = auc(fpr, tpr)
    fig = plt.subplots(figsize=(6, 6))
    plt.title("ROC Curve", fontsize=15)
    plt.plot(fpr, tpr, "b", label="AUC = %0.2f" % roc_auc)
    plt.legend(loc="lower right")
    plt.plot([0, 1], [0, 1], "r--")
    plt.xlim([-0.01, 1.01])
    plt.ylim([-0.01, 1.01])
    plt.ylabel("True Positive Rate", fontsize=12)
    plt.xlabel("False Positive Rate", fontsize=12)
    plt.show()

def data_pipeline(df):
    X = df[["CreditScore", "Age", "HasCrCard", "IsActiveMember", "NumOfProducts", "Tenure"]]
    X_ = pd.get_dummies(df["Geography"], drop_first=True)
    X = pd.concat([X, X_["Germany"]], axis=1)
    X_ = pd.get_dummies(df["Gender"], drop_first=True)
    X = pd.concat([X, X_], axis=1)
    y = df[target_col]
    return X, y
