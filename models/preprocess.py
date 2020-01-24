import pandas as pd
from sklearn.preprocessing import StandardScaler


def preprocessCsv(path, scaler=StandardScaler):
    """
    Recebe o path do arquivo csv a ser pré-processado.
    Retorna tuple com numpy arrays de features e de variável target.
    """

    # Buscando dados no diretório acima
    data = pd.read_csv(path)

    # Gerando dummy variables, drop_first evita colinearidade perfeita
    data = pd.get_dummies(data, drop_first=True)

    # Removendo variáveis com variância = 0
    mask = data.var() == 0
    cols = data.loc[:, mask].columns
    data = data.drop(columns=cols)

    # Separando features e target variable.
    y = data["Attrition_Yes"]
    X = data.drop(columns=["Attrition_Yes"])

    # Standardizando as features com standard scaling para melhor comportamento.
    X = scaler().fit_transform(X)

    return (X, y)
