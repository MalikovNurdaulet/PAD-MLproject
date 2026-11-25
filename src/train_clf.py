
import json
from pathlib import Path
import joblib
import pandas as pd
from sklearn.compose import make_column_transformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, f1_score, precision_score,recall_score, roc_auc_score)
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier

ROOT = Path(__file__).resolve().parents[1]
params = json.loads(
    (ROOT / "params_clf.json").read_text(encoding="utf-8")
)

# Данные и целевая переменная
df = pd.read_csv(ROOT / "data" / "clean.csv")

df["at_risk"] = (df["G3"] < 10).astype(int)

X = df.drop(columns=["G3", "at_risk"])
y = df["at_risk"]

num_cols = X.select_dtypes(include="number").columns.tolist()
cat_cols = X.select_dtypes(exclude="number").columns.tolist()

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=params["test_size"],
    random_state=params["random_state"],
    stratify=y
)

prep = make_column_transformer(
    (
        make_pipeline(
            SimpleImputer(strategy="median"),
            StandardScaler()
        ),
        num_cols
    ),
    (
        make_pipeline(
            SimpleImputer(strategy="most_frequent"),
            OneHotEncoder(handle_unknown="ignore")
        ),
        cat_cols
    )
)

classifiers = {
    "logreg": LogisticRegression(
        C=params["C"],
        class_weight=params["class_weight"],
        max_iter=2000
    ),
    "tree": DecisionTreeClassifier(
        max_depth=4,
        class_weight=params["class_weight"],
        random_state=params["random_state"]
    )
}

model = make_pipeline(
    prep,
    classifiers[params["model"]]
)

# Кросс-валидация
cv_f1 = cross_val_score(
    model, X_train, y_train,
    cv=5, scoring="f1"
).mean()

cv_auc = cross_val_score(
    model, X_train, y_train,
    cv=5, scoring="roc_auc"
).mean()

model.fit(X_train, y_train)

proba = model.predict_proba(X_test)[:, 1]
pred = (proba >= params["threshold"]).astype(int)

metrics = {
    "cv_f1": round(float(cv_f1), 4),
    "cv_roc_auc": round(float(cv_auc), 4),
    "test_accuracy": round(float(accuracy_score(y_test, pred)), 4),
    "test_precision": round(float(precision_score(y_test, pred, zero_division=0)), 4),
    "test_recall": round(float(recall_score(y_test, pred)), 4),
    "test_f1": round(float(f1_score(y_test, pred, zero_division=0)), 4),
    "test_roc_auc": round(float(roc_auc_score(y_test, proba)), 4)
}

(ROOT / "models").mkdir(exist_ok=True)
joblib.dump(
    model,
    ROOT / "models" / "model_clf.joblib"
)
(ROOT / "metrics_clf.json").write_text(
    json.dumps(metrics, indent=2) + "\n",
    encoding="utf-8"
)
print(params["model"], metrics)
