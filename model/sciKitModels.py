import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.mixture import GaussianMixture
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier


def getAllModels():
    list_model_param = [
        getLogisticRegressionModell(),
        getGaussianMixtureModell(),
        getSvcModell(),
        getDecisionTreeModell(),
        getRandeomForestModell(),
    ]

    list_model, list_params = np.split(np.array(list_model_param), 2, axis=1)
    return list_model.flatten(), list_params.flatten()


def getLogisticRegressionModell():
    log_reg = LogisticRegression()
    log_reg_params = {
        "C": list(np.arange(0.001, 10, 0.1)),
        "penalty": ["l2"],  # , "elasticnet"
    }
    return log_reg, log_reg_params


def getGaussianMixtureModell():
    gmm = GaussianMixture(n_components=2)
    gmm_params = {
        "init_params": ["kmeans", "k-means++", "random"],
        "covariance_type": ["tied", "diag", "full"],
    }
    return gmm, gmm_params


def getSvcModell():
    svm = SVC(probability=True)
    svm_params = {
        "C": list(np.arange(0.0005, 0.01, 0.005)),
        "degree": list(range(2, 5, 1)),
        "kernel": ["linear", "rbf"],
    }
    return svm, svm_params


def getDecisionTreeModell():
    dt = DecisionTreeClassifier()
    dt_params = {
        "criterion": ["gini", "entropy"],
        "max_depth": list(range(4, 20, 2)),
        "min_samples_split": list(range(2, 30)),
    }
    return dt, dt_params


def getRandeomForestModell():
    rf = RandomForestClassifier(
        max_depth=10, criterion="entropy", class_weight="balanced"
    )
    rf_params = {"n_estimators": list(range(1, 100))}
    return rf, rf_params
