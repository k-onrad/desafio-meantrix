import os
import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import precision_score, recall_score

from models.preprocess import preprocessCsv

# Retirando variáveis com variância = 0, gerando variáveis dummy com k-1 features por variável.
path = os.path.join(os.path.dirname(__file__), "..", "..", "HR-Employee.csv")
(X, y) = preprocessCsv(path)

# Support Vector Machine com 10-fold Cross Validation, GridSearch busca por parâmetros ótimos
svc = SVC(kernel="rbf")
C_range = np.logspace(-10, 0, 10)
param_grid = dict(C=C_range)
clf = GridSearchCV(estimator=svc, param_grid=param_grid, n_jobs=-1, cv=10)
res = clf.fit(X, y).predict(X)

# Checando performance contra baseline
baseline = np.zeros(len(y))
print("\nAccuracy scores")
print(f"Baseline (all 0s) accuracy: {np.mean(baseline == y) * 100:.2f}")
print(f"SVM accuracy: {np.mean(res == y) * 100:.2f}")

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
print(f"SVM precision/recall score: {precision:.2f}/{recall:.2f}")
print("\n")
