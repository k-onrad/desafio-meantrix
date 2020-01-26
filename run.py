import os
from sklearn.preprocessing import StandardScaler, PowerTransformer
from sklearn.model_selection import train_test_split
from sklearn.externals import joblib

from models.preprocess import preprocessCsv
from models.logistic.train import fitSkLogistic, fitCustomLogistic
from models.svm.train import fitSVM
from models.plot import plotAll

# Retirando variáveis com variância = 0, gerando variáveis dummy com k-1 features por variável.
path = os.path.join(os.path.dirname(__file__), "data", "HR-Employee.csv")
(X, y) = preprocessCsv(path)


def fitAll(X, y):
    """
    Executa todos os algoritmos em sequência, printando sua performance no dataset de teste.
    O dataset é divido em 80% para treinamento e 20% para teste.
    O score utilizado para otimizar parâmetros nos algoritmos é a estatística F1, média harmônica entre precisão e recall.
    Retorna os classificadores treinados em X_train.
    """

    # Centraliza e padroniza X (retira média, divide por std).
    # X = StandardScaler().fit_transform(X)

    # Podemos usar PowerTransform para aplicar a transformação Yeo-Johnson (aceita valores negativos) ou
    # Box-Cox (não aceita valores negativos) para diminuir a assimetria de variáveis, além de centralizar e padronizar.
    X = PowerTransformer().fit_transform(X)

    # Dividindo em training e test set de maneira estratificada (balanceada entre outcomes possíveis).
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Accuracy otimiza os modelos em respeito à exatidão. Como os outcomes 0/1 são assimétricos, tal score pode não ser muito informativo.
    # score = "accuracy"

    # O score F1 calcula a média harmônica entre precisão e recall,
    # e portanto dá igual peso à identificação de verdadeiros positivos e verdadeiros negativos.
    score = "f1"

    # Caso maior importância for dada para a identificação de verdadeiros positivos, podemos usar score="recall".
    # score = "recall"

    # Regressor logístico implementado em numpy
    clear()
    customLogClassifier = fitCustomLogistic(X_train, X_test, y_train, y_test, score)
    input("Pressione [ENTER] para continuar...")

    # Regressor logístico da library sklearn
    clear()
    skLogClassifier = fitSkLogistic(X_train, X_test, y_train, y_test, score)
    input("Pressione [ENTER] para continuar...")

    # Support Vector Classifier da library sklearn
    clear()
    svmClassifier = fitSVM(X_train, X_test, y_train, y_test, score)
    input("Pressione [ENTER] para continuar...")

    # Plotamos curvas de Receiver Operating Characteristic (ROC),
    # que representam a proporção entre verdadeiros e falsos positivos,
    # para os regressores da library sklearn.
    classifiers = {
        "cLog": customLogClassifier,
        "skLog": skLogClassifier,
        "svmC": svmClassifier,
    }
    plotAll(X_test, y_test, classifiers)

    return classifiers


def clear():
    os.system("cls" if os.name == "nt" else "clear")


if __name__ == "__main__":
    # Recebe resultados e cataloga em dict.
    classifiers = fitAll(X, y)

    # joblib salva os classifiers em arquivos .joblib em data/trained models.
    # Os classifiers podem ser recuperados com joblib.load()
    for name, clf in classifiers.items():
        _ = joblib.dump(
            clf,
            os.path.join(
                os.path.dirname(__file__), "data", "trained_models", f"{name}.joblib"
            ),
            compress=True,
        )
