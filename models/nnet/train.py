from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report


def fitMLP(X_train, X_test, y_train, y_test, score):
    """
    Cria uma pipeline que estima y_train com a utilização de PCA e Neural Network.
    Os parâmetros ótimos são procurados via GridSearchCV, que aplica 10-fold Cross Validation.
    A exatidão, precisão e recall do algoritmo em y_test são printadas ao final da execução.
    """

    # Preparando pipeline com Principal Component Analysis e Multi-layer Perceptron Classifier.
    pipe = Pipeline(
        steps=[("pca", PCA()), ("mlp", MLPClassifier(max_iter=10000, solver="lbfgs"))]
    )

    param_grid = {
        "pca__n_components": [None]
        + [int(X_test.shape[1] - i) for i in [*range(1, 10)]],
        "mlp__activation": ["logistic", "tanh", "relu"],
        "mlp__hidden_layer_sizes": [(i,) for i in [*range(5, 25)]],
    }

    # Buscando parâmetros ótimos via brute force, com Neural Network como algoritmo de aprendizado.
    # CV = 10 define que o training set deve ser dividido 10 vezes entre training e cross validation (k-Fold CV).
    # Por default, a função utiliza k-Fold CV estratificado, mantendo proporção entre classes de outcomes (no caso, 0/1).
    print("Iniciando treinamento da Neural Network")
    print(f"Otimizando parâmetros em relação ao score {score}\n")
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
