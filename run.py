import os
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

from models.preprocess import preprocessCsv
from models.logistic.train import fitSkLogistic, fitCustomLogistic
from models.svm.train import fitSVM
from models.nnet.train import fitMLP

# Retirando variáveis com variância = 0, gerando variáveis dummy com k-1 features por variável.
path = os.path.join(os.path.dirname(__file__), "data", "HR-Employee.csv")
(X, y) = preprocessCsv(path)


def runAll(X, y, scaler=StandardScaler):
    """
    Executa todos os algoritmos em sequência, printando sua performance no dataset de teste.
    O dataset é divido em 80% para treinamento e 20% para teste.
    O score utilizado para otimizar parâmetros nos algoritmos é a estatística F1, média harmônica entre precisão e recall.
    """

    # Por default, centraliza e padroniza X (retira média, divide por std).
    X = scaler().fit_transform(X)

    # Dividindo em training e test set.
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Caso maior importância fosse dada para a identificação de verdadeiros positivos, poderíamos usar score="recall".
    score = "f1"

    # Regressor logístico implementado em numpy
    clear()
    fitCustomLogistic(X_train, X_test, y_train, y_test, score)
    input("Pressione [ENTER] para continuar...\n")

    # Regressor logístico da library sklearn
    clear()
    fitSkLogistic(X_train, X_test, y_train, y_test, score)
    input("Pressione [ENTER] para continuar...\n")

    # Multi-layer Perceptron Classifier da library sklearn
    clear()
    fitMLP(X_train, X_test, y_train, y_test, score)
    input("Pressione [ENTER] to continuar...\n")

    # Support Vector Classifier da library sklearn
    clear()
    fitSVM(X_train, X_test, y_train, y_test, score)


def clear():
    os.system("cls" if os.name == "nt" else "clear")


if __name__ == "__main__":
    runAll(X, y)
