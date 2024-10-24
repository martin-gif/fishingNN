import sys
import keras
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from src.pythonSQL.v3.data_loader_v3 import get_data
from model.nn.shipType import gen_compiled_ship_type_classifier_model
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import balanced_accuracy_score, confusion_matrix, f1_score
from imblearn.over_sampling import SMOTE
import model.sciKitModels as sciKitModels


def train(use_smote: bool = False, df_column_suffix: str = "") -> pd.DataFrame:
    # Get data from pythonSQL V3
    dataset = get_data()

    # Prepare Data
    df_y = dataset.pop("response")
    df_x = dataset

    # Smote Oversampling
    if use_smote:
        smote = SMOTE()
        df_x, df_y = smote.fit_resample(df_x, df_y)

    scaler = StandardScaler()
    df_x = scaler.fit_transform(df_x)
    df_on_hot_y = pd.get_dummies(df_y)

    test_size = 0.4
    random_state = 1

    # Split Data into Train and Test
    df_train_x, df_test_x = train_test_split(
        df_x, test_size=test_size, random_state=random_state
    )
    df_train_y, df_test_y = train_test_split(
        df_y, test_size=test_size, random_state=random_state
    )
    df_train_one_hot_y, df_test_one_hot_y = train_test_split(
        df_on_hot_y, test_size=test_size, random_state=random_state
    )

    input_dim = df_train_x.shape[1]
    output_dim = df_train_one_hot_y.shape[1]

    # Dataframe wich stores Test results
    df_results = pd.DataFrame()

    # Create Callbacks for Keras Model here
    callback_early_stop = keras.callbacks.EarlyStopping(
        monitor="val_accuracy", patience=5, min_delta=0.01, start_from_epoch=10
    )

    # Model 1: used to classify ship type based on a trip
    ship_type_classifier = gen_compiled_ship_type_classifier_model(
        input_dim=input_dim, output_dim=output_dim
    )
    ship_type_classifier.fit(
        x=df_train_x,
        y=df_train_one_hot_y,
        epochs=50,
        batch_size=1,
        shuffle=True,
        verbose=1,
        validation_split=0.2,
        callbacks=[callback_early_stop],
    )

    # Score the ship type classifier
    dict_network_score = {}
    dict_network_score["model Name"] = "Ship Type Classifier"
    network_prediction_y = ship_type_classifier.predict(df_test_x).argmax(axis=1)
    network_balanced_acc = balanced_accuracy_score(
        y_true=df_test_y, y_pred=network_prediction_y
    )
    print("Network Balanced Acc:", network_balanced_acc)
    dict_network_score[f"Balanced Accuracy {df_column_suffix}"] = round(
        network_balanced_acc, 3
    )

    network_f1 = f1_score(df_test_y, network_prediction_y)
    print("Network F1 Score:", network_f1)
    dict_network_score[f"F1 Score {df_column_suffix}"] = round(network_f1, 3)

    df_tmp = pd.DataFrame([dict_network_score])
    df_results = pd.concat([df_results, df_tmp], ignore_index=True)
    # print(ship_type_classifier.summary())

    # Model 2: used to reconstruct missing AIS Data based on a trip
    # reproduction_lstm = gen_compiled_LSTM_model()
    # reproduction_lstm.fit(x=x_2, epochs=20, callbacks=[callback])

    # get SciKit Models and their parameters
    # modify the model Parameters in the model/sciKitModels.py file
    models, params = sciKitModels.getAllModels()

    for model, param in zip(models, params):
        clf = GridSearchCV(estimator=model, param_grid=param, scoring="f1", refit=True)
        clf.fit(X=df_train_x, y=df_train_y)
        best_model_pred = clf.predict(df_test_x)
        tmp_dict = {}

        model_name = model.__class__.__name__
        print("Model: ", model_name)
        tmp_dict["model Name"] = model_name

        best_parameter = clf.best_params_
        print("Parameter:", best_parameter)
        tmp_dict[f"best parameter {df_column_suffix}"] = best_parameter

        f_one = round(f1_score(df_test_y, best_model_pred), 3)
        print("F1 score:", f_one)
        tmp_dict[f"F1 Score {df_column_suffix}"] = f_one

        balanced_acc = round(balanced_accuracy_score(df_test_y, best_model_pred), 3)
        print("balanced test Acc:", balanced_acc)
        tmp_dict[f"Balanced Accuracy {df_column_suffix}"] = balanced_acc

        print(
            "confusion Matrix:\n",
            confusion_matrix(df_test_y, best_model_pred),
        )
        print()

        df_tmp = pd.DataFrame([tmp_dict])
        df_results = pd.concat([df_results, df_tmp], ignore_index=True)

    return df_results

    # model_list = [ship_type_classifier, reproduction_lstm]
    # return model_list


if __name__ == "__main__":
    df_no_smote = train()
    df_smote = train(use_smote=True, df_column_suffix="SMOTE")

    df_results = pd.merge(df_no_smote, df_smote, on="model Name")
    df_results.to_csv("figures/results.csv")
