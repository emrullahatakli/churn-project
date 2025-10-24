import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, cross_validate

df = pd.read_csv("/Users/emrullahatakli/Desktop/kod/end-to-end/churn-project/data/telco_churn.csv")

df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

df["TotalCharges"].fillna(0, inplace=True)

def grab_col_names(dataframe, cat_th=5, car_th=15):
    cat_cols = [col for col in dataframe.columns if dataframe[col].dtypes == "O"]
    num_but_cat = [
        col
        for col in dataframe.columns
        if dataframe[col].nunique() < cat_th and dataframe[col].dtypes != "O"
    ]
    cat_but_car = [
        col
        for col in dataframe.columns
        if dataframe[col].nunique() > car_th and dataframe[col].dtypes == "O"
    ]

    cat_cols = cat_cols + num_but_cat
    cat_cols = [col for col in cat_cols if col not in cat_but_car]

    num_cols = [col for col in dataframe.columns if dataframe[col].dtypes != "O"]
    num_cols = [col for col in num_cols if col not in num_but_cat]

    print(f"Observation: {dataframe.shape[0]}")
    print(f"Variables: {dataframe.shape[1]}")
    print(f"cat_cols: {len(cat_cols)}")
    print(f"num_cols: {len(num_cols)}")
    print(f"cat_but_car: {len(cat_but_car)}")
    print(f"num_but_cat: {len(num_but_cat)}")

    return cat_cols, cat_but_car, num_cols, num_but_cat

df["SeniorCitizen"] = df["SeniorCitizen"].map({0: "No", 1: "Yes"})

df.drop("customerID", axis=1, inplace=True)

cat_cols, cat_but_car, num_cols, num_but_cat = grab_col_names(df)

df["NEW_HighRisk_Customer"] = ((df["tenure"] < 12) & (df["Contract"] == "Month-to-month") & (df["Partner"] == "No")).astype(int)

cat_cols, cat_but_car, num_cols, num_but_cat = grab_col_names(df)

df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})

df_dummies = pd.get_dummies(df, drop_first=True, dtype=int)

corr_with_churn = df_dummies.corr()["Churn"].sort_values(ascending=False)

churn_correlations = df_dummies.corr()["Churn"]
features_to_drop = churn_correlations[(churn_correlations >= -0.1) & (churn_correlations <= 0.1)].index
df_dummies = df_dummies.drop(columns=features_to_drop)

X = df_dummies.drop("Churn", axis=1)
y = df_dummies["Churn"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train_scaled, y_train)

best_params_rf = {
    "n_estimators": 20,
    "max_depth": 5,
    "min_samples_split": 5,
    "min_samples_leaf": 4,
    "criterion": "entropy",
    "class_weight": "balanced",
}

final_rf_model = RandomForestClassifier(**best_params_rf, random_state=42, n_jobs=-1)
final_rf_model.fit(X_train_resampled, y_train_resampled)

y_final_rf_pred = final_rf_model.predict(X_test_scaled)
y_final_rf_pred_proba = final_rf_model.predict_proba(X_test_scaled)[:, 1]

final_rf_auc_score = roc_auc_score(y_test, y_final_rf_pred_proba)

print("Random Forest Test Seti Performansı")
print(classification_report(y_test, y_final_rf_pred))
print(f"Test AUC Skor: {final_rf_auc_score:.4f}\n")

final_xgb_model_precision = precision_score(y_test, y_final_rf_pred)
final_xgb_model_recall = recall_score(y_test, y_final_rf_pred)
final_xgb_model_f1 = f1_score(y_test, y_final_rf_pred)

print("Metrikler")
print(f"Precision: {final_xgb_model_precision:.4f}")
print(f"Recall: {final_xgb_model_recall:.4f}")
print(f"F1 Score: {final_xgb_model_f1:.4f}\n")

cv_results_rf = cross_validate(
    final_rf_model,
    X_train_resampled,
    y_train_resampled,
    cv=5,
    scoring="roc_auc",
    return_train_score=True,
)

print(f"Eğitim Skoru: {cv_results_rf['train_score'].mean()}")
print(f"Validasyon Skoru: {cv_results_rf['test_score'].mean()}")
