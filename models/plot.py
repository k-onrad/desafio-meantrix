import random
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
import matplotlib

# To use matplotlib in virtual environment
matplotlib.use("TkAgg")


def plotAll(X_test, y_test, classifiers):
    colors = ["b", "g", "r", "c", "m", "y", "w"]
    for name, clf in classifiers.items():
        # Pular Regressão Logística customizada. (predict_proba precisa ser implementado conforme specs sklearn)
        if name == "cLog":
            continue
        y_pred = clf.predict_proba(X_test)[:, 1]
        clfFpr, clfTpr, _ = roc_curve(y_test, y_pred)
        clfRocAuc = auc(clfFpr, clfTpr)
        plt.plot(
            clfFpr,
            clfTpr,
            color=random.choice(colors),
            linestyle="-",
            label=f"{name} (AUC = {clfRocAuc:0.2f})",
            lw=2,
        )

    plt.plot([0, 1], [0, 1], linestyle="--", lw=2, color="k", label="Sorte")

    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curves of Model")
    plt.legend(loc="lower right")
    plt.show()
