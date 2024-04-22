import os.path
import sys

import keras.utils
import pandas as pd
import tensorflow as tf
from sqlalchemy import create_engine
from dataloader import fishingDataLoader
from dataset_read import get_tf_sql_dataset_all_typs, get_mean_var
from model.nn.shipType import gen_compiled_ship_type_classifier_model
from sklearn.metrics import confusion_matrix


ENGINE = create_engine(url="sqlite:///data/data.db", echo=False)


def parse_sql_data(*vals):
    mean_var_df = gen_mean_var_Dataframe()
    # print(mean_var_df)
    mean_tensor = tf.convert_to_tensor(mean_var_df["mean"])
    var_tensor = tf.convert_to_tensor(mean_var_df["var"])

    # features = {
    #     "timestamp": tf.convert_to_tensor(vals[1]),
    #     "distance_from_shore": tf.convert_to_tensor(vals[2]),
    #     "distance_from_port": tf.convert_to_tensor(vals[3]),
    #     "speed": tf.convert_to_tensor(vals[4]),
    #     "course": tf.convert_to_tensor(vals[5]),
    #     "lat": tf.convert_to_tensor(vals[6]),
    #     "lon": tf.convert_to_tensor(vals[7]),
    #     "is_fishing": tf.convert_to_tensor(vals[8]),
    # }
    features = tf.convert_to_tensor(vals[:8])
    # features = tf.nn.batch_normalization(
    #     x=features, mean=mean_tensor, variance=var_tensor, offset=None, scale=None
    # )
    features = tf.transpose(features)
    lable = tf.one_hot(vals[8], depth=6)
    return (features, lable)


def train():
    # get Dataset and preprocess it
    raw_data = get_tf_sql_dataset_all_typs(limit_each_class=10000)
    dataset = raw_data.shuffle(buffer_size=1000, seed=42)
    dataset = dataset.batch(2)
    dataset = dataset.map(parse_sql_data)

    # print(next(iter(dataset)))
    ds_training, ds_val = keras.utils.split_dataset(dataset, left_size=0.8)

    # generate models
    ship_type_classifier = gen_compiled_ship_type_classifier_model()

    for feature, lable in dataset:
        pred = ship_type_classifier.predict([feature])
        lab = lable.numpy()
        print(tf.argmax(pred))
        print(lab)
        # print(confusion_matrix(y_true=lab, y_pred=pred))
        break

    # train models
    # ship_type_classifier.fit(
    #     x=ds_training, epochs=10, validation_data=ds_val, verbose=2
    # )

    for feature, lable in dataset:
        pred = ship_type_classifier.predict([feature])
        lab = lable.numpy()
        print(confusion_matrix(y_true=lab, y_pred=pred))
        break


def gen_mean_var_Dataframe():
    path = "data/mean_var_df.pkl"
    if os.path.isfile(path):
        df = pd.read_pickle(filepath_or_buffer=path)
    else:
        df = get_mean_var(
            [
                "data.timestamp",
                "data.timestamp",
                "data.distance_from_shore",
                "data.distance_from_port",
                "data.speed",
                "data.course",
                "data.lat",
                "data.lon",
            ],
            ENGINE,
        )
    df.to_pickle(path=path)
    return df


if __name__ == "__main__":
    loader = fishingDataLoader(input_engine=ENGINE)
    loader.gen_SQL_db()
    train()
