import os
import numpy as np
from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import precision_score, recall_score

from models.preprocess import preprocessCsv

# Retirando variáveis com variância = 0, gerando variáveis dummy com k-1 features por variável.
path = os.path.join(os.path.dirname(__file__), "..", "..", "HR-Employee.csv")
(X, y) = preprocessCsv(path)

# Regressão logística treinada com 10-fold Cross Validation
clf = LogisticRegressionCV(cv=10, random_state=42).fit(X, y)
res = clf.predict(X)

# Checando performance contra baseline
baseline = np.zeros(len(y))
print("\nAccuracy scores")
print(f"Baseline (all 0s) accuracy: {np.mean(baseline == y) * 100:.2f}")
print(f"Logistic Regression accuracy: {np.mean(res == y) * 100:.2f}")

# Medindo precision/recall
y_score = clf.decision_function(X)
base_precision = precision_score(y, baseline, zero_division=0)
precision = precision_score(y, res)
base_recall = recall_score(y, baseline)
recall = recall_score(y, res)
print("\nPrecision/recall scores")
print(
    f"Baseline (all 0s) precision/recall score: {base_precision:.2f}/{base_recall:.2f}"
)
print(f"Logistic Regression precision/recall score: {precision:.2f}/{recall:.2f}")
print("\n")
