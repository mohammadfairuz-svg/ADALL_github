"""
Employee Attrition Prediction — full worked pipeline
Mirrors the sample deck structure: clean -> feature engineer -> DT/RF/XGB stage1 & stage2 -> evaluate
Business framing: AI-driven workforce restructuring & reskilling
"""
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, matthews_corrcoef, roc_auc_score,
                             confusion_matrix, roc_curve)
from xgboost import XGBClassifier

RND = 42
df = pd.read_csv("attrition.csv")
print("RAW SHAPE:", df.shape)

# ------------------------------------------------------------------
# 1. TARGET
# ------------------------------------------------------------------
df["target"] = (df["Attrition"] == "Yes").astype(int)   # 1 = left (minority)
print("Attrition rate:", round(df["target"].mean()*100, 1), "%")

# ------------------------------------------------------------------
# 2. DROP JUNK  (mirrors sample: near-constant + identity noise)
# ------------------------------------------------------------------
constant_cols = [c for c in df.columns if df[c].nunique() == 1]   # EmployeeCount, StandardHours, Over18
id_cols = ["EmployeeNumber"]
print("Near-constant dropped:", constant_cols)
print("Identity dropped:", id_cols)
df = df.drop(columns=constant_cols + id_cols + ["Attrition"])

# ------------------------------------------------------------------
# 3. AI-EXPOSURE FEATURE  (literature-informed proxy by JobRole)
#    Higher = more exposed to AI/automation displacement.
#    Grounded loosely in Frey-Osborne / WEF Future of Jobs style rankings.
#    THIS IS A DEFENSIBLE PROXY, NOT GROUND TRUTH — stated on the slide.
# ------------------------------------------------------------------
ai_exposure = {
    "Sales Representative": 0.85,
    "Laboratory Technician": 0.75,
    "Research Scientist": 0.55,
    "Sales Executive": 0.65,
    "Human Resources": 0.60,
    "Manufacturing Director": 0.45,
    "Healthcare Representative": 0.50,
    "Manager": 0.30,
    "Research Director": 0.25,
}
df["AIExposure"] = df["JobRole"].map(ai_exposure)
print("\nAI Exposure by role assigned. Range:", df["AIExposure"].min(), "-", df["AIExposure"].max())

# ------------------------------------------------------------------
# 4. FEATURE ENGINEERING
# ------------------------------------------------------------------
# 4a. Tenure ratio: how much of career spent here (loyalty vs stagnation signal)
df["TenureRatio"] = df["YearsAtCompany"] / (df["TotalWorkingYears"] + 1)
# 4b. Income per level (under/over-paid for their level)
df["IncomePerLevel"] = df["MonthlyIncome"] / df["JobLevel"]
# 4c. Promotion stagnation flag
df["StagnantFlag"] = (df["YearsSinceLastPromotion"] >= 5).astype(int)
# 4d. OverTime -> binary
df["OverTime"] = (df["OverTime"] == "Yes").astype(int)

# ------------------------------------------------------------------
# 5. ENCODING  (ordinal for ordered, integer-code for nominal)
# ------------------------------------------------------------------
travel_map = {"Non-Travel": 0, "Travel_Rarely": 1, "Travel_Frequently": 2}
df["BusinessTravel"] = df["BusinessTravel"].map(travel_map)

nominal = ["Department", "EducationField", "Gender", "JobRole", "MaritalStatus"]
for c in nominal:
    df[c] = df[c].astype("category").cat.codes

print("\nFinal shape after FE:", df.shape)
print("Final columns:", list(df.columns))
assert df.select_dtypes(include="object").shape[1] == 0, "still object cols!"

# ------------------------------------------------------------------
# 6. SPLIT  (stratified 80/20, same as sample)
# ------------------------------------------------------------------
X = df.drop(columns=["target"])
y = df["target"]
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.20,
                                          stratify=y, random_state=RND)
print("\nTrain:", X_tr.shape, "Test:", X_te.shape)

# median imputation (dataset is clean but keep pipeline honest)
med = X_tr.median()
X_tr = X_tr.fillna(med); X_te = X_te.fillna(med)

# scale_pos_weight for stage 2
spw = (y_tr == 0).sum() / (y_tr == 1).sum()
print("scale_pos_weight:", round(spw, 2))

# ------------------------------------------------------------------
# 7. MODEL ZOO — Stage 1 (default) and Stage 2 (weighted)
# ------------------------------------------------------------------
def evaluate(name, model, X_te, y_te):
    p = model.predict(X_te)
    proba = model.predict_proba(X_te)[:, 1]
    return {
        "Model": name,
        "Accuracy": accuracy_score(y_te, p),
        "Precision": precision_score(y_te, p, zero_division=0),
        "Recall": recall_score(y_te, p),
        "F1": f1_score(y_te, p),
        "MCC": matthews_corrcoef(y_te, p),
        "AUC": roc_auc_score(y_te, proba),
    }

results = []

# --- Stage 1: baselines ---
dt1 = DecisionTreeClassifier(max_depth=4, random_state=RND).fit(X_tr, y_tr)
results.append({"Cycle": "Stage 1", **evaluate("DT Baseline", dt1, X_te, y_te)})

rf1 = RandomForestClassifier(n_estimators=300, random_state=RND, n_jobs=-1).fit(X_tr, y_tr)
results.append({"Cycle": "Stage 1", **evaluate("RF Default", rf1, X_te, y_te)})

xgb1 = XGBClassifier(n_estimators=300, max_depth=4, learning_rate=0.1,
                     eval_metric="logloss", random_state=RND).fit(X_tr, y_tr)
results.append({"Cycle": "Stage 1", **evaluate("XGB Default", xgb1, X_te, y_te)})

# --- Stage 2: class-weighted / imbalance-aware ---
dt2 = DecisionTreeClassifier(max_depth=6, class_weight="balanced",
                             random_state=RND).fit(X_tr, y_tr)
results.append({"Cycle": "Stage 2 - Weighted", **evaluate("DT Weighted", dt2, X_te, y_te)})

rf2 = RandomForestClassifier(n_estimators=400, class_weight="balanced",
                             max_depth=10, random_state=RND, n_jobs=-1).fit(X_tr, y_tr)
results.append({"Cycle": "Stage 2 - Weighted", **evaluate("RF Weighted", rf2, X_te, y_te)})

xgb2 = XGBClassifier(n_estimators=400, max_depth=6, learning_rate=0.08,
                     scale_pos_weight=spw, subsample=0.8, colsample_bytree=0.8,
                     eval_metric="logloss", random_state=RND).fit(X_tr, y_tr)
results.append({"Cycle": "Stage 2 - Weighted", **evaluate("XGB Weighted", xgb2, X_te, y_te)})

res_df = pd.DataFrame(results)[["Cycle","Model","Accuracy","Precision","Recall","F1","MCC","AUC"]]
pd.set_option("display.width", 200, "display.float_format", lambda v: f"{v:.4f}")
print("\n================ RESULTS ================")
print(res_df.to_string(index=False))

# Save artifacts for plotting
import joblib
joblib.dump({"xgb2": xgb2, "X_tr": X_tr, "X_te": X_te, "y_te": y_te,
             "res_df": res_df, "feat_names": list(X.columns)}, "artifacts.joblib")
res_df.to_csv("results.csv", index=False)
print("\nSaved artifacts.joblib + results.csv")
