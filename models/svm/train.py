import os
import numpy as np
from sklearn.preprocessing import RobustScaler
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import classification_report
from models.preprocess import preprocessCsv

# Retirando variáveis com variância = 0, gerando variáveis dummy com k-1 features por variável.
path = os.path.join(os.path.dirname(__file__), "..", "..", "HR-Employee.csv")
(X, y) = preprocessCsv(path, RobustScaler)

# Dividindo em training e test set.
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Preparando pipeline com Principal Component Analysis e Support Vector Machine Classifier.
pipe = Pipeline(steps=[("pca", PCA()), ("svm", SVC())])

# Preparando parâmetros a serem otimizados.
# N-Components = número de componentes a serem mantidos no PCA. Caso None, usar as features sem alteração.
# C = peso dado ao termo de custo da função a ser otimizada.
# C alto -> + variância, - bias. C baixo -> + bias, - variância.
# Gamma = inverso do raio de influência de um support vector
# Gamma alto -> + bias, - variância. Gamma baixo -> + variância, - bias.
# Kernels testados: Radial Basis Function, Sigmóide e Polinomial.
param_grid = {
    "pca__n_components": [None] + [int(X.shape[1] - i) for i in [*range(1, 10)]],
    "svm__C": np.logspace(-2, 3, 11),
    "svm__gamma": np.logspace(-5, 0, 11),
    "svm__kernel": ("rbf", "sigmoid", "poly"),
}

# Definindo performance por F1 = média harmônica entre precision e recall.
# Caso maior importância fosse dada para a identificação de verdadeiros positivos, poderíamos usar score="recall".
score = "f1"
print(f"\nOtimizando parâmetros em relação ao score {score}\n")

# Buscando parâmetros ótimos via brute force, com Support Vector Classifier como algoritmo de aprendizado.
# CV = 10 define que o training set deve ser dividido 10 vezes entre training e cross validation (k-Fold CV).
# Por default, a função utiliza k-Fold CV estratificado, mantendo proporção entre classes de outcomes (no caso, 0/1).
clf = GridSearchCV(
    estimator=pipe,
    param_grid=param_grid,
    n_jobs=-1,
    cv=10,
    scoring=f"{score}",
    verbose=1,
)
clf.fit(X_train, y_train)

print(f"\nMelhores parâmetros encontrados no training set: {clf.best_params_}")
print(f"Melhor score encontrado no training set: {clf.best_score_:.02f}")

print("\nScores no test set:")
print(classification_report(y_test, clf.predict(X_test), zero_division=0))
print()
