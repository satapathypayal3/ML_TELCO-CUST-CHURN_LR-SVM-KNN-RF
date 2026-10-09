"""
Customer churn: compare logistic regression, SVM, KNN and random forest.

"Churn" means a customer leaving the company. We use the Telco Customer Churn
dataset from Kaggle (about 7,000 telecom customers) and ask four different
models to predict who will leave, then compare them fairly.

Before running:
  1. Download the dataset from Kaggle (search "Telco Customer Churn").
  2. Put Telco-Customer-Churn.csv in the same folder as this file.
  3. pip install pandas scikit-learn matplotlib

Run:  python churn_models.py
It prints a comparison table and saves four pictures next to this file.
"""
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, roc_curve, confusion_matrix)

DATA_FILE = "Telco-Customer-Churn.csv"

# ---------------------------------------------------------------
# 1. Load and clean the data
# ---------------------------------------------------------------
df = pd.read_csv(DATA_FILE)
print(f"Loaded {len(df):,} customers and {df.shape[1]} columns.")

# TotalCharges is stored as text and is blank for brand-new customers.
# Turn it into a number; blanks become 0 because those customers have paid nothing yet.
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce").fillna(0)

# The customer ID is just a label. It cannot help predict anything.
df = df.drop(columns="customerID")

# The answer we want to predict: 1 = customer left, 0 = customer stayed
y = (df["Churn"] == "Yes").astype(int)
print(f"Customers who left: {y.sum():,} of {len(y):,} ({y.mean():.1%})\n")

# Picture 1: a first look at the data, before any model
by_contract = df.groupby("Contract")["Churn"].apply(lambda s: (s == "Yes").mean() * 100)
by_contract = by_contract.reindex(["Month-to-month", "One year", "Two year"])
plt.figure(figsize=(7, 4.5))
bars = plt.bar(by_contract.index, by_contract.values, color="tab:blue")
plt.bar_label(bars, fmt="%.0f%%")
plt.ylabel("Customers who left (%)")
plt.title("Churn by contract type")
plt.tight_layout()
plt.savefig("1_churn_by_contract.png", dpi=150)

# Models need numbers, not words. Turn each text column into 0/1 columns
# (for example Contract becomes "Contract_One year" and "Contract_Two year").
X = pd.get_dummies(df.drop(columns="Churn"), drop_first=True).astype(float)

# ---------------------------------------------------------------
# 2. Split: 80% to learn from, 20% hidden away as a final exam
# ---------------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)   # stratify keeps the same churn % in both parts

# ---------------------------------------------------------------
# 3. The four models
#    StandardScaler puts every column on the same scale. Logistic regression,
#    SVM and KNN need this; random forest does not.
# ---------------------------------------------------------------
models = {
    "Logistic regression": make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000)),
    "SVM":                 make_pipeline(StandardScaler(), SVC()),
    "KNN":                 make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=15)),
    "Random forest":       RandomForestClassifier(n_estimators=300, random_state=42),
}


def churn_scores(model, data):
    """A 'how likely to leave' score for each customer, used for the ROC curve."""
    if hasattr(model, "predict_proba"):
        return model.predict_proba(data)[:, 1]
    return model.decision_function(data)


# ---------------------------------------------------------------
# 4. Train, cross-validate and test each model
# ---------------------------------------------------------------
rows, matrices, curves = [], {}, {}
for name, model in models.items():
    # Cross-validation: 5 practice exams using only the training data
    cv = cross_val_score(model, X_train, y_train, cv=5, scoring="roc_auc")

    # Final exam: learn from all the training data, predict the hidden 20%
    model.fit(X_train, y_train)
    predicted = model.predict(X_test)
    scores = churn_scores(model, X_test)

    rows.append({
        "Model": name,
        "Accuracy": accuracy_score(y_test, predicted),
        "Precision": precision_score(y_test, predicted),
        "Recall": recall_score(y_test, predicted),
        "F1": f1_score(y_test, predicted),
        "ROC AUC": roc_auc_score(y_test, scores),
        "CV AUC": cv.mean(),
        "CV spread": cv.std(),
    })
    matrices[name] = confusion_matrix(y_test, predicted)
    curves[name] = roc_curve(y_test, scores)

results = pd.DataFrame(rows).set_index("Model")
baseline = 1 - y_test.mean()
print(f"Baseline: always guessing 'stays' is right {baseline:.1%} of the time.\n")
print(results.round(3).to_string())
results.round(4).to_csv("results_table.csv")

# ---------------------------------------------------------------
# 5. Pictures
# ---------------------------------------------------------------
# Picture 2: confusion matrices (right and wrong guesses, counted)
fig, axes = plt.subplots(1, 4, figsize=(15, 4.4), layout="constrained")
for ax, (name, m) in zip(axes, matrices.items()):
    ax.imshow(m, cmap="Blues")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{m[i, j]:,}", ha="center", va="center", fontsize=14,
                    color="white" if m[i, j] > m.max() / 2 else "black")
    ax.set_xticks([0, 1], ["Predicted\nstays", "Predicted\nleaves"])
    ax.set_yticks([0, 1], ["Really\nstayed", "Really\nleft"])
    ax.set_title(name)
fig.suptitle("Confusion matrices on the hidden test customers")
fig.savefig("2_confusion_matrices.png", dpi=150)

# Picture 3: ROC curves
plt.figure(figsize=(7, 6))
for name, (fpr, tpr, _) in curves.items():
    plt.plot(fpr, tpr, linewidth=2, label=f"{name} (AUC {results.loc[name, 'ROC AUC']:.3f})")
plt.plot([0, 1], [0, 1], "k--", linewidth=1, label="random guessing (AUC 0.5)")
plt.xlabel("False alarms: stayers wrongly flagged (share)")
plt.ylabel("Leavers correctly caught (share)")
plt.title("ROC curves")
plt.legend(loc="lower right")
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("3_roc_curves.png", dpi=150)

# Picture 4: the scores side by side
ax = results[["Accuracy", "Precision", "Recall", "F1", "ROC AUC"]].T.plot(
    kind="bar", figsize=(10, 5), rot=0, width=0.8)
ax.axhline(baseline, color="black", linestyle="--", linewidth=1, label="accuracy of always guessing 'stays'")
ax.set_ylim(0, 1)
ax.set_ylabel("Score (higher is better)")
ax.set_title("Model comparison on the hidden test customers")
ax.legend(loc="upper center", ncol=5, fontsize=8)
plt.tight_layout()
plt.savefig("4_model_comparison.png", dpi=150)

print("\nSaved: 1_churn_by_contract.png, 2_confusion_matrices.png, "
      "3_roc_curves.png, 4_model_comparison.png, results_table.csv")
plt.show()
