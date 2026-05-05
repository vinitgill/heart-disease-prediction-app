# =============================================================
# AI-Powered Heart Disease Prediction System
# Training Pipeline  -  train.py
# =============================================================
# Usage:  python train.py
# Outputs: best_model.pkl  scaler.pkl  features.pkl
#          roc_data.pkl    feature_importance.pkl
#          model_comparison.png  roc_curves.png
#          confusion_matrix.png  feature_importance.png
# =============================================================

import pandas as pd
import numpy as np
import pickle
import warnings
warnings.filterwarnings("ignore")

import matplotlib
matplotlib.use("Agg")   # non-interactive backend (safe for servers)
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, roc_auc_score,
    roc_curve, confusion_matrix, ConfusionMatrixDisplay,
)
from xgboost import XGBClassifier

# ── Dark-theme plotting style ──────────────────────────────
plt.rcParams.update({
    "figure.facecolor": "#0d1117",
    "axes.facecolor":   "#161b22",
    "axes.edgecolor":   "#30363d",
    "axes.labelcolor":  "#c9d1d9",
    "xtick.color":      "#8b949e",
    "ytick.color":      "#8b949e",
    "text.color":       "#c9d1d9",
    "grid.color":       "#21262d",
    "grid.linewidth":   0.8,
    "font.family":      "DejaVu Sans",
})

COLORS = ["#58a6ff", "#3fb950", "#f78166", "#d2a8ff"]

# =============================================================
# 1. Load & Validate Dataset
# =============================================================
print("=" * 60)
print("  Heart Disease Prediction -- Training Pipeline")
print("=" * 60)

df = pd.read_csv("data.csv")
df.dropna(inplace=True)

assert "target" in df.columns, "Dataset must have a 'target' column."

feature_cols = [c for c in df.columns if c != "target"]
X = df[feature_cols]
y = df["target"]

print(f"\n[OK] Dataset loaded: {df.shape[0]} rows x {df.shape[1]} columns")
print(f"[OK] Features ({len(feature_cols)}): {feature_cols}")
print(f"[OK] Target distribution:\n{y.value_counts().to_string()}")

# =============================================================
# 2. Train/Test Split + Scaling
# =============================================================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc  = scaler.transform(X_test)

print(f"\n[OK] Train size: {X_train_sc.shape[0]}  |  Test size: {X_test_sc.shape[0]}")

# =============================================================
# 3. Define Models
# =============================================================
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
    "Random Forest":       RandomForestClassifier(n_estimators=200, random_state=42),
    "XGBoost":             XGBClassifier(
                               n_estimators=200,
                               learning_rate=0.05,
                               random_state=42,
                               eval_metric="logloss",
                               verbosity=0,
                           ),
}

# =============================================================
# 4. Train, Evaluate, Track Best
# =============================================================
results   = {}
roc_data  = {}
best_name = None
best_auc  = -1.0

print("\n-- Model Results " + "-" * 44)
for name, model in models.items():
    model.fit(X_train_sc, y_train)
    y_pred  = model.predict(X_test_sc)
    y_proba = model.predict_proba(X_test_sc)[:, 1]

    acc = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_proba)
    fpr, tpr, _ = roc_curve(y_test, y_proba)

    results[name]  = {"accuracy": acc, "roc_auc": auc, "model": model}
    roc_data[name] = {"fpr": fpr, "tpr": tpr, "auc": auc}

    marker = " <-- BEST" if auc > best_auc else ""
    print(f"  {name:<22}  Accuracy={acc:.4f}  ROC-AUC={auc:.4f}{marker}")

    if auc > best_auc:
        best_auc  = auc
        best_name = name

print(f"\n[BEST] {best_name}  (ROC-AUC = {best_auc:.4f})")
best_model = results[best_name]["model"]

# =============================================================
# 5. Save Artifacts
# =============================================================
with open("best_model.pkl", "wb") as f:
    pickle.dump(best_model, f)
with open("scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)
with open("features.pkl", "wb") as f:
    pickle.dump(feature_cols, f)
with open("roc_data.pkl", "wb") as f:
    pickle.dump(roc_data, f)

# Feature importance (tree-based models only)
fi_dict = {}
for name, info in results.items():
    m = info["model"]
    if hasattr(m, "feature_importances_"):
        fi_dict[name] = dict(zip(feature_cols, m.feature_importances_))
if fi_dict:
    with open("feature_importance.pkl", "wb") as f:
        pickle.dump(fi_dict, f)

print("[OK] Saved: best_model.pkl  scaler.pkl  features.pkl")
print("[OK] Saved: roc_data.pkl  feature_importance.pkl")

# =============================================================
# 6. Visualisations
# =============================================================

# 6a -- Model Comparison Bar Chart (Accuracy + ROC-AUC)
names  = list(results.keys())
accs   = [results[n]["accuracy"] for n in names]
aucs_v = [results[n]["roc_auc"]  for n in names]
x_pos  = np.arange(len(names))
w      = 0.35

fig, ax = plt.subplots(figsize=(9, 5))
b1 = ax.bar(x_pos - w / 2, accs,   w, color=COLORS[0], label="Accuracy", zorder=3)
b2 = ax.bar(x_pos + w / 2, aucs_v, w, color=COLORS[1], label="ROC-AUC",  zorder=3)

for bar in list(b1) + list(b2):
    h = bar.get_height()
    ax.text(
        bar.get_x() + bar.get_width() / 2, h + 0.005,
        f"{h:.3f}", ha="center", va="bottom", fontsize=8, color="#c9d1d9",
    )

ax.set_xticks(x_pos)
ax.set_xticklabels(names, fontsize=10)
ax.set_ylim(0.5, 1.05)
ax.set_title("Model Comparison -- Accuracy & ROC-AUC", fontsize=13, pad=12, color="#e6edf3")
ax.legend(framealpha=0.2)
ax.grid(axis="y", zorder=0)
plt.tight_layout()
plt.savefig("model_comparison.png", dpi=150)
plt.close()
print("[OK] Saved: model_comparison.png")

# 6b -- ROC Curves
fig, ax = plt.subplots(figsize=(7, 6))
for i, (name, data) in enumerate(roc_data.items()):
    ax.plot(
        data["fpr"], data["tpr"],
        lw=2, color=COLORS[i],
        label=f"{name} (AUC={data['auc']:.3f})",
    )
ax.plot([0, 1], [0, 1], "w--", lw=1, alpha=0.4)
ax.set_xlabel("False Positive Rate")
ax.set_ylabel("True Positive Rate")
ax.set_title("ROC Curves -- All Models", fontsize=13, pad=12, color="#e6edf3")
ax.legend(framealpha=0.2)
ax.grid(zorder=0)
plt.tight_layout()
plt.savefig("roc_curves.png", dpi=150)
plt.close()
print("[OK] Saved: roc_curves.png")

# 6c -- Confusion Matrix (best model)
y_pred_best = best_model.predict(X_test_sc)
cm = confusion_matrix(y_test, y_pred_best)
fig, ax = plt.subplots(figsize=(5, 4))
disp = ConfusionMatrixDisplay(cm, display_labels=["No Disease", "Disease"])
disp.plot(ax=ax, colorbar=False, cmap="Blues")
ax.set_title(f"Confusion Matrix -- {best_name}", fontsize=12, pad=10, color="#e6edf3")
plt.tight_layout()
plt.savefig("confusion_matrix.png", dpi=150)
plt.close()
print("[OK] Saved: confusion_matrix.png")

# 6d -- Feature Importance (best tree model)
if best_name in fi_dict:
    fi = fi_dict[best_name]
    sorted_fi = dict(sorted(fi.items(), key=lambda item: item[1], reverse=True))
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh(list(sorted_fi.keys()), list(sorted_fi.values()),
            color=COLORS[3], zorder=3)
    ax.invert_yaxis()
    ax.set_xlabel("Importance Score")
    ax.set_title(f"Feature Importance -- {best_name}", fontsize=13, pad=12, color="#e6edf3")
    ax.grid(axis="x", zorder=0)
    plt.tight_layout()
    plt.savefig("feature_importance.png", dpi=150)
    plt.close()
    print("[OK] Saved: feature_importance.png")

print("\n" + "=" * 60)
print("  Training Complete! All artifacts saved.")
print("=" * 60)
print("  Run the app:  streamlit run app.py")
print("=" * 60)
