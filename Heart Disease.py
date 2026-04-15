"""
Heart Disease Diagnostic Analysis
- Multiple ML models with comparison
- Hyperparameter tuning via GridSearchCV
- Rich, publication-quality visualizations
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    roc_auc_score, roc_curve, precision_recall_curve, average_precision_score
)
import warnings
warnings.filterwarnings('ignore')

# ─── Aesthetics ───────────────────────────────────────────────────────────────
PALETTE  = {"bg": "#0f1117", "card": "#1a1d27", "accent": "#e05c5c",
            "green": "#4ecca3", "blue": "#3a86ff", "text": "#e8eaf0", "muted": "#6b7280"}
sns.set_theme(style="dark", rc={
    "figure.facecolor": PALETTE["bg"], "axes.facecolor": PALETTE["card"],
    "axes.edgecolor": PALETTE["muted"], "axes.labelcolor": PALETTE["text"],
    "xtick.color": PALETTE["muted"], "ytick.color": PALETTE["muted"],
    "text.color": PALETTE["text"], "grid.color": "#2a2d3a", "grid.linestyle": "--",
    "grid.alpha": 0.5,
})
plt.rcParams["font.family"] = "DejaVu Sans"


# ─── 1. Load & Explore ────────────────────────────────────────────────────────
df = pd.read_csv("D:\dissertation\heart.csv")
print("Shape:", df.shape)
print(df.head())
print("\nMissing values:\n", df.isnull().sum())
print("\nClass balance:\n", df["target"].value_counts(normalize=True).round(3))


# ─── 2. EDA Dashboard ─────────────────────────────────────────────────────────
fig = plt.figure(figsize=(20, 16), facecolor=PALETTE["bg"])
fig.suptitle("Heart Disease — Exploratory Data Analysis", fontsize=22,
             fontweight="bold", color=PALETTE["text"], y=0.98)
gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.45, wspace=0.35)

# 2a. Target distribution
ax0 = fig.add_subplot(gs[0, 0])
counts = df["target"].value_counts()
bars = ax0.bar(["No Disease", "Heart Disease"], counts.values,
               color=[PALETTE["blue"], PALETTE["accent"]], width=0.5, edgecolor="none")
for bar, val in zip(bars, counts.values):
    ax0.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5,
             str(val), ha="center", color=PALETTE["text"], fontsize=11)
ax0.set_title("Target Distribution", fontsize=13, color=PALETTE["text"], pad=10)
ax0.set_ylabel("Count")

# 2b. Age distribution by target
ax1 = fig.add_subplot(gs[0, 1])
for t, col, lbl in [(0, PALETTE["blue"], "No Disease"), (1, PALETTE["accent"], "Heart Disease")]:
    ax1.hist(df[df["target"] == t]["age"], bins=15, alpha=0.7, color=col, label=lbl, edgecolor="none")
ax1.set_title("Age Distribution", fontsize=13, color=PALETTE["text"], pad=10)
ax1.set_xlabel("Age"); ax1.set_ylabel("Count")
ax1.legend(fontsize=9)

# 2c. Max Heart Rate vs Age scatter
ax2 = fig.add_subplot(gs[0, 2])
scatter = ax2.scatter(df["age"], df["thalach"], c=df["target"],
                      cmap="RdBu", alpha=0.6, s=30, edgecolors="none")
ax2.set_title("Age vs Max Heart Rate", fontsize=13, color=PALETTE["text"], pad=10)
ax2.set_xlabel("Age"); ax2.set_ylabel("Max Heart Rate")
plt.colorbar(scatter, ax=ax2, label="Target")

# 2d. Correlation heatmap
ax3 = fig.add_subplot(gs[1, :2])
corr = df.corr()
mask = np.triu(np.ones_like(corr, dtype=bool))
cmap = sns.diverging_palette(220, 10, as_cmap=True)
sns.heatmap(corr, mask=mask, ax=ax3, cmap=cmap, center=0,
            annot=True, fmt=".2f", annot_kws={"size": 7},
            linewidths=0.3, linecolor=PALETTE["bg"],
            cbar_kws={"shrink": 0.8})
ax3.set_title("Feature Correlation Matrix", fontsize=13, color=PALETTE["text"], pad=10)

# 2e. Chest pain type vs target
ax4 = fig.add_subplot(gs[1, 2])
cp_target = df.groupby(["cp", "target"]).size().unstack(fill_value=0)
cp_target.plot(kind="bar", ax=ax4, color=[PALETTE["blue"], PALETTE["accent"]],
               edgecolor="none", width=0.65)
ax4.set_title("Chest Pain Type vs Target", fontsize=13, color=PALETTE["text"], pad=10)
ax4.set_xlabel("Chest Pain Type"); ax4.set_ylabel("Count")
ax4.set_xticklabels(ax4.get_xticklabels(), rotation=0)
ax4.legend(["No Disease", "Heart Disease"], fontsize=9)

# 2f. Cholesterol boxplot
ax5 = fig.add_subplot(gs[2, 0])
data_groups = [df[df["target"] == t]["chol"].values for t in [0, 1]]
bp = ax5.boxplot(data_groups, patch_artist=True, widths=0.4,
                 medianprops=dict(color=PALETTE["text"], linewidth=2))
for patch, color in zip(bp["boxes"], [PALETTE["blue"], PALETTE["accent"]]):
    patch.set_facecolor(color); patch.set_alpha(0.75)
ax5.set_xticks([1, 2]); ax5.set_xticklabels(["No Disease", "Heart Disease"])
ax5.set_title("Cholesterol by Target", fontsize=13, color=PALETTE["text"], pad=10)
ax5.set_ylabel("Serum Cholesterol (mg/dl)")

# 2g. Thalassemia type vs target
ax6 = fig.add_subplot(gs[2, 1])
thal_target = df.groupby(["thal", "target"]).size().unstack(fill_value=0)
thal_target.plot(kind="bar", ax=ax6, color=[PALETTE["blue"], PALETTE["accent"]],
                 edgecolor="none", width=0.65)
ax6.set_title("Thalassemia Type vs Target", fontsize=13, color=PALETTE["text"], pad=10)
ax6.set_xlabel("Thal Type"); ax6.set_ylabel("Count")
ax6.set_xticklabels(ax6.get_xticklabels(), rotation=0)
ax6.legend(["No Disease", "Heart Disease"], fontsize=9)

# 2h. Exercise-induced angina
ax7 = fig.add_subplot(gs[2, 2])
exang_target = df.groupby(["exang", "target"]).size().unstack(fill_value=0)
exang_target.plot(kind="bar", ax=ax7, color=[PALETTE["blue"], PALETTE["accent"]],
                  edgecolor="none", width=0.5)
ax7.set_title("Exercise Angina vs Target", fontsize=13, color=PALETTE["text"], pad=10)
ax7.set_xlabel("Exercise Angina (0=No, 1=Yes)"); ax7.set_ylabel("Count")
ax7.set_xticklabels(ax7.get_xticklabels(), rotation=0)
ax7.legend(["No Disease", "Heart Disease"], fontsize=9)

plt.savefig("eda_dashboard.png", dpi=150, bbox_inches="tight", facecolor=PALETTE["bg"])
plt.show()
print("EDA dashboard saved.")


# ─── 3. Preprocessing ─────────────────────────────────────────────────────────
X = df.drop(columns=["target"])
y = df["target"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s  = scaler.transform(X_test)


# ─── 4. Model Training & Hyperparameter Tuning ────────────────────────────────
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Logistic Regression
lr_params = {"C": [0.01, 0.1, 1, 10, 100], "solver": ["lbfgs", "liblinear"]}
lr = GridSearchCV(LogisticRegression(max_iter=1000), lr_params, cv=cv, scoring="roc_auc", n_jobs=-1)
lr.fit(X_train_s, y_train)
print(f"\n[LR]  Best params: {lr.best_params_}  |  CV AUC: {lr.best_score_:.4f}")

# Random Forest
rf_params = {"n_estimators": [100, 200], "max_depth": [None, 5, 10],
             "min_samples_split": [2, 5], "max_features": ["sqrt", "log2"]}
rf = GridSearchCV(RandomForestClassifier(random_state=42), rf_params, cv=cv,
                  scoring="roc_auc", n_jobs=-1)
rf.fit(X_train_s, y_train)
print(f"[RF]  Best params: {rf.best_params_}  |  CV AUC: {rf.best_score_:.4f}")

# Gradient Boosting
gb_params = {"n_estimators": [100, 200], "learning_rate": [0.05, 0.1],
             "max_depth": [3, 5], "subsample": [0.8, 1.0]}
gb = GridSearchCV(GradientBoostingClassifier(random_state=42), gb_params, cv=cv,
                  scoring="roc_auc", n_jobs=-1)
gb.fit(X_train_s, y_train)
print(f"[GB]  Best params: {gb.best_params_}  |  CV AUC: {gb.best_score_:.4f}")

# SVM
svm_params = {"C": [0.1, 1, 10], "kernel": ["rbf", "linear"], "gamma": ["scale", "auto"]}
svm = GridSearchCV(SVC(probability=True, random_state=42), svm_params, cv=cv,
                   scoring="roc_auc", n_jobs=-1)
svm.fit(X_train_s, y_train)
print(f"[SVM] Best params: {svm.best_params_}  |  CV AUC: {svm.best_score_:.4f}")

# Soft Voting Ensemble
ensemble = VotingClassifier(
    estimators=[("lr", lr.best_estimator_), ("rf", rf.best_estimator_),
                ("gb", gb.best_estimator_), ("svm", svm.best_estimator_)],
    voting="soft"
)
ensemble.fit(X_train_s, y_train)


# ─── 5. Evaluate All Models ───────────────────────────────────────────────────
models = {"Logistic Regression": lr.best_estimator_,
          "Random Forest":       rf.best_estimator_,
          "Gradient Boosting":   gb.best_estimator_,
          "SVM":                 svm.best_estimator_,
          "Ensemble":            ensemble}

results = {}
for name, m in models.items():
    y_pred  = m.predict(X_test_s)
    y_proba = m.predict_proba(X_test_s)[:, 1]
    results[name] = {
        "accuracy":  accuracy_score(y_test, y_pred),
        "auc":       roc_auc_score(y_test, y_proba),
        "y_pred":    y_pred,
        "y_proba":   y_proba,
        "cv_scores": cross_val_score(m, X_train_s, y_train, cv=cv, scoring="accuracy"),
    }
    print(f"\n{'='*50}\n{name}")
    print(classification_report(y_test, y_pred, target_names=["No Disease", "Heart Disease"]))


# ─── 6. Model Comparison Dashboard ───────────────────────────────────────────
fig2, axes = plt.subplots(2, 3, figsize=(20, 12), facecolor=PALETTE["bg"])
fig2.suptitle("Model Comparison & Evaluation", fontsize=22,
              fontweight="bold", color=PALETTE["text"], y=0.98)
colors_bar = [PALETTE["blue"], PALETTE["green"], "#f4a261", "#a78bfa", PALETTE["accent"]]

# 6a. Accuracy comparison
ax = axes[0, 0]
names  = list(results.keys())
accs   = [results[n]["accuracy"] for n in names]
aucs   = [results[n]["auc"] for n in names]
x      = np.arange(len(names))
width  = 0.35
bars1 = ax.bar(x - width/2, accs, width, label="Accuracy", color=colors_bar, alpha=0.85, edgecolor="none")
bars2 = ax.bar(x + width/2, aucs, width, label="AUC-ROC", color=colors_bar, alpha=0.45, edgecolor="none")
for b in list(bars1) + list(bars2):
    ax.text(b.get_x() + b.get_width()/2, b.get_height() + 0.005,
            f"{b.get_height():.3f}", ha="center", fontsize=7.5, color=PALETTE["text"])
ax.set_xticks(x); ax.set_xticklabels(names, rotation=20, ha="right", fontsize=9)
ax.set_ylim(0.75, 1.02); ax.set_title("Accuracy & AUC-ROC", fontsize=13, color=PALETTE["text"])
ax.legend(fontsize=9)

# 6b. ROC curves
ax = axes[0, 1]
for (name, res), col in zip(results.items(), colors_bar):
    fpr, tpr, _ = roc_curve(y_test, res["y_proba"])
    ax.plot(fpr, tpr, color=col, lw=2, label=f"{name} ({res['auc']:.3f})")
ax.plot([0,1],[0,1], "--", color=PALETTE["muted"], lw=1)
ax.set_title("ROC Curves", fontsize=13, color=PALETTE["text"])
ax.set_xlabel("False Positive Rate"); ax.set_ylabel("True Positive Rate")
ax.legend(fontsize=8)

# 6c. Precision-Recall curves
ax = axes[0, 2]
for (name, res), col in zip(results.items(), colors_bar):
    prec, rec, _ = precision_recall_curve(y_test, res["y_proba"])
    ap = average_precision_score(y_test, res["y_proba"])
    ax.plot(rec, prec, color=col, lw=2, label=f"{name} (AP={ap:.3f})")
ax.set_title("Precision-Recall Curves", fontsize=13, color=PALETTE["text"])
ax.set_xlabel("Recall"); ax.set_ylabel("Precision")
ax.legend(fontsize=8)

# 6d–6f. Confusion matrices for top 3 models
top3 = sorted(results.items(), key=lambda x: x[1]["auc"], reverse=True)[:3]
for idx, (name, res) in enumerate(top3):
    ax = axes[1, idx]
    cm = confusion_matrix(y_test, res["y_pred"])
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
                linewidths=1, linecolor=PALETTE["bg"],
                cbar=False, annot_kws={"size": 14, "weight": "bold"})
    ax.set_xlabel("Predicted", fontsize=10); ax.set_ylabel("Actual", fontsize=10)
    ax.set_xticklabels(["No Disease", "Heart Disease"], fontsize=9)
    ax.set_yticklabels(["No Disease", "Heart Disease"], fontsize=9, rotation=0)
    ax.set_title(f"Confusion Matrix — {name}", fontsize=12, color=PALETTE["text"])

plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.savefig("model_comparison.png", dpi=150, bbox_inches="tight", facecolor=PALETTE["bg"])
plt.show()
print("Model comparison dashboard saved.")


# ─── 7. Feature Importance (Best Model) ───────────────────────────────────────
best_name = max(results, key=lambda n: results[n]["auc"])
print(f"\nBest model by AUC: {best_name}")
best_model = models[best_name]

fig3, axes3 = plt.subplots(1, 2, figsize=(16, 6), facecolor=PALETTE["bg"])
fig3.suptitle(f"Feature Importance — {best_name}", fontsize=18,
              fontweight="bold", color=PALETTE["text"])

# RF / GB feature importances
ax = axes3[0]
if hasattr(best_model, "feature_importances_"):
    importances = best_model.feature_importances_
else:
    # Fall back to RF
    importances = rf.best_estimator_.feature_importances_
    ax.set_title("Feature Importance (Random Forest)", fontsize=13, color=PALETTE["text"])

feat_imp = pd.Series(importances, index=X.columns).sort_values()
colors_imp = [PALETTE["accent"] if v > feat_imp.median() else PALETTE["blue"] for v in feat_imp]
feat_imp.plot(kind="barh", ax=ax, color=colors_imp, edgecolor="none")
ax.set_title("Feature Importances", fontsize=13, color=PALETTE["text"])
ax.set_xlabel("Importance Score")

# LR coefficients
ax2 = axes3[1]
coef = pd.Series(np.abs(lr.best_estimator_.coef_[0]), index=X.columns).sort_values()
colors_coef = [PALETTE["accent"] if v > coef.median() else PALETTE["green"] for v in coef]
coef.plot(kind="barh", ax=ax2, color=colors_coef, edgecolor="none")
ax2.set_title("Logistic Regression |Coefficients|", fontsize=13, color=PALETTE["text"])
ax2.set_xlabel("|Coefficient|")

plt.tight_layout()
plt.savefig("feature_importance.png", dpi=150, bbox_inches="tight", facecolor=PALETTE["bg"])
plt.show()
print("Feature importance chart saved.")


# ─── 8. Cross-Validation Box Plot ─────────────────────────────────────────────
fig4, ax4 = plt.subplots(figsize=(12, 5), facecolor=PALETTE["bg"])
cv_data = [results[n]["cv_scores"] for n in names]
bp = ax4.boxplot(cv_data, patch_artist=True, widths=0.5,
                 medianprops=dict(color="white", linewidth=2.5))
for patch, col in zip(bp["boxes"], colors_bar):
    patch.set_facecolor(col); patch.set_alpha(0.7)
ax4.set_xticks(range(1, len(names)+1))
ax4.set_xticklabels(names, rotation=15, ha="right")
ax4.set_title("5-Fold Cross-Validation Accuracy Distribution", fontsize=15,
              color=PALETTE["text"], pad=12)
ax4.set_ylabel("Accuracy")
ax4.set_ylim(0.7, 1.05)
plt.tight_layout()
plt.savefig("cv_boxplot.png", dpi=150, bbox_inches="tight", facecolor=PALETTE["bg"])
plt.show()
print("CV boxplot saved.")