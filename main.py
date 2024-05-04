import os.path
import sys
from typing import List

import keras.utils
import numpy as np
import pandas as pd
import tensorflow as tf
from keras.utils import FeatureSpace
from sqlalchemy import create_engine
from dataloader import fishingDataLoader
from dataset_read import get_tf_sql_dataset_all_typs, get_mean_var
from sklearn.metrics import confusion_matrix

from model.nn.shipType import gen_compiled_ship_type_classifier_model
from model.nn.trajectoryPredict import gen_compiled_LSTM_model


ENGINE = create_engine(url="sqlite:///data/data.db", echo=False)


def parse_sql_data(*vals):
    features = {
        "timestamp": tf.convert_to_tensor(vals[0]),
        "distance_from_shore": tf.convert_to_tensor(vals[1]),
        "distance_from_port": tf.convert_to_tensor(vals[2]),
        "speed": tf.convert_to_tensor(vals[3]),
        "course": tf.convert_to_tensor(vals[4]),
        "lat": tf.convert_to_tensor(vals[5]),
        "lon": tf.convert_to_tensor(vals[6]),
        "is_fishing": tf.convert_to_tensor(vals[7]),
        "lable": tf.convert_to_tensor(vals[8]),
    }
    # lable = tf.convert_to_tensor(vals[8])
    # lable = tf.one_hot(indices=lable, depth=7)
    return features


def get_FeatureSpace(dataset: tf.data.Dataset) -> keras.utils.FeatureSpace:
    path = "data/featurespace.keras"
    if os.path.isfile(path):
        reloaded_feature_space = keras.models.load_model(path, compile=True)
        # print((reloaded_feature_space.get_inputs()))
        reloaded_feature_space.get_inputs()  # I don't know why but this is need for the loaded model to work
        return reloaded_feature_space
    else:
        features = FeatureSpace(
            features={
                "timestamp": FeatureSpace.float_rescaled(scale=1.0 / 1480032000),
                "distance_from_shore": FeatureSpace.float_rescaled(
                    scale=1.0 / 4430996.5
                ),
                "distance_from_port": FeatureSpace.float_rescaled(
                    scale=1.0 / 12452204.0
                ),
                "speed": FeatureSpace.float_rescaled(scale=1.0 / 103),
                "course": FeatureSpace.float_rescaled(scale=1.0 / 511),
                "lat": FeatureSpace.float_rescaled(scale=1.0 / 360, offset=0.5),
                "lon": FeatureSpace.float_rescaled(scale=1.0 / 360, offset=0.5),
                "is_fishing": FeatureSpace.float_rescaled(scale=1.0 / -1),
                "lable": FeatureSpace.integer_categorical(),
            },
            output_mode="dict",
        )
        features.adapt(dataset)
        features.save(path)
        # print(features)
        return features


def print_conf_matrix(data: tf.data.Dataset, model: keras.Model):
    for feature, lable in data:
        # print(feature)
        pred = model.predict([feature])
        lab = lable.numpy()
        pred = tf.argmax(pred, axis=-1).numpy()
        lab = tf.argmax(lab, axis=-1).numpy()
        print(confusion_matrix(y_true=lab, y_pred=pred))
        break


def train():
    # get Dataset and preprocess it
    points_per_trip = 5
    trips_per_batch = 4
    batches_per_class = 500

    total_points_per_class = points_per_trip * trips_per_batch * batches_per_class
    NUM_CLASSES = 6

    dataset = get_tf_sql_dataset_all_typs(limit_each_class=total_points_per_class)
    dataset = dataset.map(parse_sql_data)

    # Get adapted feature Space
    feature_space = get_FeatureSpace(dataset=dataset)
    dataset = dataset.map(lambda x: (feature_space(x)))  # normalize features

    dataset = dataset.batch(points_per_trip)
    dataset = dataset.shuffle(buffer_size=total_points_per_class * NUM_CLASSES, seed=42)
    dataset = dataset.batch(trips_per_batch)

    print(next(iter(dataset)))

    # generate models
    model_list = [
        gen_compiled_ship_type_classifier_model()
        # gen_compiled_LSTM_model()
    ]

    callback = keras.callbacks.EarlyStopping(monitor="loss", patience=3)

    for model in model_list:
        model.fit(x=dataset, epochs=20, verbose=1, callbacks=[callback])

    return model_list, dataset


def gen_mean_var_Dataframe():
    path = "data/mean_var_df.pkl"
    if os.path.isfile(path):
        df = pd.read_pickle(filepath_or_buffer=path)
    else:
        df = get_mean_var(
            [
                "data.timestamp",
                "data.distance_from_shore",
                "data.distance_from_port",
                "data.speed",
                "data.course",
                "data.lat",
                "data.lon",
                "data.is_fishing",
                "ship_type.id",
            ],
            ENGINE,
        )
    df.to_pickle(path=path)
    return df


def get_mean_var_list():
    df = gen_mean_var_Dataframe()
    mean_list = df["mean"].to_list()
    var_list = df["var"].to_list()

    return mean_list, var_list


if __name__ == "__main__":
    loader = fishingDataLoader(input_engine=ENGINE)
    loader.gen_SQL_db()
    train()
