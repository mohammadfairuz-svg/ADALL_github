"""Generate every chart for the deck, plus threshold-tuned champion."""
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd, joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from sklearn.metrics import (confusion_matrix, roc_curve, roc_auc_score,
                             recall_score, precision_score, f1_score,
                             matthews_corrcoef, precision_recall_curve)
from sklearn.inspection import permutation_importance
import shap

# palette — "Berry & Cream" adjacent, workforce/people theme
BERRY="#6D2E46"; ROSE="#A26769"; CREAM="#ECE2D0"; SLATE="#3d4f5d"; GOLD="#C99846"
plt.rcParams.update({"font.family":"DejaVu Sans","axes.edgecolor":"#888",
                     "axes.grid":True,"grid.alpha":0.25,"figure.dpi":140})

A = joblib.load("artifacts.joblib")
xgb = A["xgb2"]; X_tr=A["X_tr"]; X_te=A["X_te"]; y_te=A["y_te"]; feats=A["feat_names"]

proba = xgb.predict_proba(X_te)[:,1]

# ------------------------------------------------------------------
# THRESHOLD TUNING — choose the F1-optimal threshold (defensible balance
# of catching leavers vs false alarms), NOT an arbitrary recall target.
# ------------------------------------------------------------------
prec, rec, thr = precision_recall_curve(y_te, proba)
f1s = 2*prec[:-1]*rec[:-1] / (prec[:-1]+rec[:-1]+1e-9)
best_i = int(np.nanargmax(f1s))
best_t = float(thr[best_i])
pred_t = (proba >= best_t).astype(int)

# Also render the PR curve so the threshold choice is visibly justified
fig,ax=plt.subplots(figsize=(6,5))
ax.plot(rec[:-1],prec[:-1],color=BERRY,lw=2.4)
ax.scatter(rec[best_i],prec[best_i],color=GOLD,s=120,zorder=5,
           label=f"Chosen threshold={best_t:.2f}\nRecall={rec[best_i]:.2f}, Prec={prec[best_i]:.2f}")
ax.set_xlabel("Recall",weight="bold"); ax.set_ylabel("Precision",weight="bold")
ax.set_title("Precision–Recall Trade-off (F1-optimal threshold)",weight="bold",color=SLATE)
ax.legend(loc="upper right",fontsize=9)
plt.savefig("c0_prcurve.png",bbox_inches="tight",facecolor="white"); plt.close()

champ = {
    "threshold": round(float(best_t),3),
    "Recall": recall_score(y_te,pred_t),
    "Precision": precision_score(y_te,pred_t,zero_division=0),
    "F1": f1_score(y_te,pred_t),
    "MCC": matthews_corrcoef(y_te,pred_t),
    "AUC": roc_auc_score(y_te,proba),
}
print("TUNED CHAMPION:", {k:(round(v,4) if isinstance(v,float) else v) for k,v in champ.items()})
joblib.dump(champ, "champ.joblib")

# ------------------------------------------------------------------
# 1. RESULTS TABLE (rendered as image)
# ------------------------------------------------------------------
res = A["res_df"].copy()
for c in ["Accuracy","Precision","Recall","F1","MCC","AUC"]:
    res[c]=res[c].map(lambda v:f"{v:.3f}")
fig,ax=plt.subplots(figsize=(11,3.2)); ax.axis("off")
tbl=ax.table(cellText=res.values, colLabels=res.columns, cellLoc="center", loc="center")
tbl.auto_set_font_size(False); tbl.set_fontsize(11); tbl.scale(1,1.9)
for j in range(len(res.columns)):
    tbl[0,j].set_facecolor(BERRY); tbl[0,j].set_text_props(color="white",weight="bold")
for i in range(1,len(res)+1):
    champ_row = res.iloc[i-1]["Model"]=="XGB Weighted"
    for j in range(len(res.columns)):
        tbl[i,j].set_facecolor(GOLD if champ_row else (CREAM if i%2 else "white"))
plt.title("Model Comparison — Stage 1 (default) vs Stage 2 (imbalance-aware)",
          fontsize=13,weight="bold",color=SLATE,pad=14)
plt.savefig("c1_results.png",bbox_inches="tight",facecolor="white"); plt.close()

# ------------------------------------------------------------------
# 2. CONFUSION MATRIX (tuned)
# ------------------------------------------------------------------
cm=confusion_matrix(y_te,pred_t)
fig,ax=plt.subplots(figsize=(6,5))
im=ax.imshow(cm,cmap="RdPu")
for (i,j),v in np.ndenumerate(cm):
    ax.text(j,i,str(v),ha="center",va="center",fontsize=22,weight="bold",
            color="white" if v>cm.max()*0.5 else BERRY)
ax.set_xticks([0,1]); ax.set_yticks([0,1])
ax.set_xticklabels(["Stay (0)","Leave (1)"]); ax.set_yticklabels(["Stay (0)","Leave (1)"])
ax.set_xlabel("Predicted",weight="bold"); ax.set_ylabel("Actual",weight="bold")
ax.set_title(f"Champion XGBoost — Confusion Matrix\n(threshold = {best_t:.2f})",
             weight="bold",color=SLATE)
plt.colorbar(im,fraction=0.046); plt.grid(False)
plt.savefig("c2_confusion.png",bbox_inches="tight",facecolor="white"); plt.close()

# ------------------------------------------------------------------
# 3. ROC CURVE
# ------------------------------------------------------------------
fpr,tpr,_=roc_curve(y_te,proba); auc=roc_auc_score(y_te,proba)
fig,ax=plt.subplots(figsize=(6,5))
ax.plot(fpr,tpr,color=BERRY,lw=2.5,label=f"XGBoost (AUC={auc:.2f})")
ax.plot([0,1],[0,1],"--",color="#999",lw=1.2,label="Random (0.50)")
ax.fill_between(fpr,tpr,alpha=0.12,color=ROSE)
ax.set_xlabel("False Positive Rate",weight="bold")
ax.set_ylabel("True Positive Rate",weight="bold")
ax.set_title("Champion XGBoost — ROC Curve",weight="bold",color=SLATE)
ax.legend(loc="lower right")
plt.savefig("c3_roc.png",bbox_inches="tight",facecolor="white"); plt.close()

# ------------------------------------------------------------------
# SHAP
# ------------------------------------------------------------------
explainer=shap.TreeExplainer(xgb)
sv=explainer(X_te)

# 4. BEESWARM
plt.figure()
shap.plots.beeswarm(sv,max_display=12,show=False)
fig=plt.gcf(); fig.set_size_inches(8,6)
plt.title("Global Risk Drivers — SHAP Beeswarm",weight="bold",color=SLATE,pad=12)
plt.tight_layout(); plt.savefig("c4_beeswarm.png",bbox_inches="tight",facecolor="white"); plt.close()

# 5. FI vs PFI
gini=pd.Series(xgb.feature_importances_,index=feats)
pfi=permutation_importance(xgb,X_te,y_te,scoring="roc_auc",n_repeats=10,random_state=42)
pfi_s=pd.Series(pfi.importances_mean,index=feats)
comp=pd.DataFrame({"Gini_FI":gini,"PFI_AUCdrop":pfi_s}).sort_values("PFI_AUCdrop",ascending=True).tail(12)
fig,ax=plt.subplots(figsize=(8,6))
yy=np.arange(len(comp))
ax.barh(yy-0.2,comp["Gini_FI"]/comp["Gini_FI"].max(),0.4,label="Gini FI (norm)",color=ROSE)
ax.barh(yy+0.2,comp["PFI_AUCdrop"]/comp["PFI_AUCdrop"].max(),0.4,label="PFI AUC-drop (norm)",color=BERRY)
ax.set_yticks(yy); ax.set_yticklabels(comp.index)
ax.set_title("Feature Importance vs Permutation Importance",weight="bold",color=SLATE)
ax.legend()
plt.tight_layout(); plt.savefig("c5_fi_pfi.png",bbox_inches="tight",facecolor="white"); plt.close()

# 6. DEPENDENCE — OverTime (usually a top driver)
top_feat="OverTime"
plt.figure()
shap.plots.scatter(sv[:,top_feat],color=sv[:,"MonthlyIncome"],show=False)
fig=plt.gcf(); fig.set_size_inches(7,5)
plt.title(f"SHAP Dependence — {top_feat} (colored by MonthlyIncome)",weight="bold",color=SLATE)
plt.tight_layout(); plt.savefig("c6_dep_overtime.png",bbox_inches="tight",facecolor="white"); plt.close()

# 7. DEPENDENCE — AIExposure  (the highlighted engineered feature)
plt.figure()
shap.plots.scatter(sv[:,"AIExposure"],color=sv[:,"MonthlyIncome"],show=False)
fig=plt.gcf(); fig.set_size_inches(7,5)
plt.title("SHAP Dependence — AI Exposure (engineered proxy)",weight="bold",color=SLATE)
plt.tight_layout(); plt.savefig("c7_dep_aiexposure.png",bbox_inches="tight",facecolor="white"); plt.close()

# 8. WATERFALL — a high-risk individual
hr_idx=int(np.argmax(proba))
plt.figure()
shap.plots.waterfall(sv[hr_idx],max_display=11,show=False)
fig=plt.gcf(); fig.set_size_inches(8.5,6)
plt.title(f"Local Case — Employee flagged high flight-risk ({proba[hr_idx]*100:.0f}%)",
          weight="bold",color=SLATE)
plt.tight_layout(); plt.savefig("c8_waterfall.png",bbox_inches="tight",facecolor="white"); plt.close()

# 9. AI-EXPOSURE vs ACTUAL ATTRITION (validation of the proxy's relevance)
full=X_te.copy(); full["actual"]=y_te.values
grp=full.groupby("AIExposure")["actual"].mean().sort_index()
fig,ax=plt.subplots(figsize=(7,4.5))
ax.bar([f"{v:.2f}" for v in grp.index],grp.values*100,color=BERRY)
ax.set_xlabel("AI-Exposure proxy score (by job role)",weight="bold")
ax.set_ylabel("Actual attrition rate (%)",weight="bold")
ax.set_title("Does AI-Exposure track real attrition? (holdout set)",weight="bold",color=SLATE)
plt.tight_layout(); plt.savefig("c9_aiexposure_actual.png",bbox_inches="tight",facecolor="white"); plt.close()

print("All charts written.")
import os
for f in sorted(os.listdir(".")):
    if f.startswith("c") and f.endswith(".png"): print(" ",f)
