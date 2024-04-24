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
    features = tf.transpose(features)

    lable = tf.convert_to_tensor(vals[8])
    print(vals[8])
    # lable = tf.transpose(lable)
    lable = tf.one_hot(lable, depth=7)
    return (features, lable)


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
    points_per_class = 3000
    NUM_CLASSES = 6
    raw_data = get_tf_sql_dataset_all_typs(limit_each_class=points_per_class)
    dataset = raw_data.shuffle(buffer_size=points_per_class * NUM_CLASSES, seed=42)
    dataset = dataset.batch(10)
    dataset = dataset.map(parse_sql_data)

    # print(next(iter(dataset)))
    ds_training, ds_val = keras.utils.split_dataset(dataset, left_size=0.8)

    # generate models
    model_list = [gen_compiled_ship_type_classifier_model()]

    for model in model_list:
        print_conf_matrix(data=dataset, model=model)

        # train models
        model.fit(x=ds_training, epochs=20, validation_data=ds_val, verbose=2)

        print_conf_matrix(data=dataset, model=model)


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
