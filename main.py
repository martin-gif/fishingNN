import sys

import keras
import pandas as pd
import numpy as np

from model.simple.LogisticeRegression import get_LogisticRegression_model
from src.pythonSQL.v1.database_writer import fishingDataLoader
from src.pythonSQL.v1.database_loader import (
    get_mean_var,
    get_dataset_generator,
)
from src.pythonSQL.v2.database_loader_v2 import get_dataset_v2
from model.nn.shipType import gen_compiled_ship_type_classifier_model
from model.nn.trajectoryPredict import gen_compiled_LSTM_model
from src.utils import transform_dataset_dict_to_list_with_X_Y
import tensorflow as tf

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import balanced_accuracy_score, confusion_matrix, f1_score
from sklearn.linear_model import LogisticRegression
from sklearn.mixture import GaussianMixture
from scipy.stats import multivariate_normal as mvn
from sklearn.svm import SVC, LinearSVC
from sklearn.tree import plot_tree, DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import OneClassSVM


def train():
    # # get Dataset and preprocess it
    # points_per_trip = 5  # How many datapoints should be per on trip
    # trips_per_batch = 4  # How many batches should be used per iteration
    # batches_per_class = 500
    #
    # dataset = get_dataset_v2(batch_size=50)
    #
    # dataset = dataset.map(
    #     lambda x: transform_dataset_dict_to_list_with_X_Y(
    #         dictonary=x, lable="Prediction"
    #     )
    # )
    #
    # single_dataset_batch = next(dataset.as_numpy_iterator())
    # input_dim = single_dataset_batch[0].shape[1]
    # output_dim = single_dataset_batch[1].shape[1]
    #
    # # print(next(iter(dataset)), "\n\n\n")
    # callback = keras.callbacks.EarlyStopping(monitor="loss", patience=3, min_delta=0.01)
    #
    # # Model 1: used to classify ship type based on a trip
    # ship_type_classifier = gen_compiled_ship_type_classifier_model(
    #     input_dim=input_dim, output_dim=output_dim
    # )
    # # ship_type_classifier.fit(x=dataset, epochs=20, verbose=1)
    #
    # # Model 2: used to reconstruct missing AIS Data based on a trip
    # reproduction_lstm = gen_compiled_LSTM_model()
    # # reproduction_lstm.fit(x=x_2, epochs=20, callbacks=[callback])

    dataset = get_dataset_v2(batch_size=1000)

    dataset = dataset.map(
        lambda x: transform_dataset_dict_to_list_with_X_Y(
            dictonary=x, lable="Prediction"
        )
    )

    x_data, y_data = next(dataset.as_numpy_iterator())
    y_data = np.argmax(y_data, axis=-1)

    x_train, x_test, y_train, y_test = train_test_split(
        x_data, y_data, test_size=0.2, shuffle=True, random_state=1
    )
    models = []
    params = []

    log_reg = LogisticRegression(penalty="l2")
    log_reg_params = {"C": list(np.arange(0.001, 10, 0.1))}
    models.append(log_reg)
    params.append(log_reg_params)

    gmm = GaussianMixture(n_components=2)
    gmm_params = {
        "init_params": ["kmeans", "k-means++", "random"],
        "covariance_type": ["tied", "diag", "full"],
    }
    models.append(gmm)
    params.append(gmm_params)

    svm = SVC(probability=True)
    svm_params = {
        "C": list(np.arange(0.0005, 0.01, 0.005)),
        "degree": list(range(2, 5, 1)),
        "kernel": ["linear", "rbf"],
    }
    models.append(svm)
    params.append(svm_params)

    dt = DecisionTreeClassifier()
    dt_params = {
        "criterion": ["gini", "entropy"],
        "max_depth": list(range(4, 20, 2)),
        "min_samples_split": list(range(2, 30)),
    }
    models.append(dt)
    params.append(dt_params)

    # not trained every time because its to Time consuming
    rf = RandomForestClassifier(
        max_depth=10, criterion="entropy", class_weight="balanced"
    )
    rf_params = {"n_estimators": list(range(1, 100))}
    models.append(rf)
    params.append(rf_params)

    for model, param in zip(models, params):
        clf = GridSearchCV(estimator=model, param_grid=param, scoring="f1", refit=True)
        clf.fit(X=x_train, y=y_train)
        best_model_pred = clf.predict(x_test)

        print("Model: ", model.__class__.__name__)
        print("Parameter:", clf.best_params_)
        print("F1 score:", round(f1_score(y_test, best_model_pred), 3))
        print(
            "balanced test Acc:",
            round(balanced_accuracy_score(y_test, best_model_pred), 3),
        )
        print("confusion Matrix:\n", confusion_matrix(y_test, best_model_pred))

        print()

    # model_list = [ship_type_classifier, reproduction_lstm]
    # return model_list


if __name__ == "__main__":
    train()
