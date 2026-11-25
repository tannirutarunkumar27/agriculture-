import sys
from pathlib import Path

import pandas as pd
from sklearn.tree import DecisionTreeClassifier

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.evaluation import evaluate_model

train_df = pd.read_csv(PROJECT_ROOT / "data" / "processed" / "train.csv")
test_df = pd.read_csv(PROJECT_ROOT / "data" / "processed" / "test.csv")

X_train = train_df.drop("label", axis=1)
y_train = train_df["label"]

X_test = test_df.drop("label", axis=1)
y_test = test_df["label"]

model = DecisionTreeClassifier(random_state=42)

model.fit(X_train, y_train)

predictions = model.predict(X_test)

metrics = evaluate_model(
    "decision_tree",
    y_test,
    predictions
)

print(metrics)