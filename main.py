import os.path

import keras.utils
import pandas as pd
from sqlalchemy import create_engine
from dataloader import fishingDataLoader
from dataset_read import (
    get_mean_var,
    get_dataset_generator,
)

from model.nn.shipType import gen_compiled_ship_type_classifier_model
from model.nn.trajectoryPredict import gen_compiled_LSTM_model
from src.utils import remove_entry_and_to_list, transform_dataset_dict_to_list_with_X_Y

ENGINE = create_engine(url="sqlite:///data/data.db", echo=False)


def train():
    # get Dataset and preprocess it
    points_per_trip = 5
    trips_per_batch = 4
    batches_per_class = 500

    dataset = get_dataset_generator(points_per_trip, trips_per_batch, batches_per_class)

    # print(next(iter(dataset)), "\n\n\n")
    callback = keras.callbacks.EarlyStopping(monitor="loss", patience=3, min_delta=0.01)

    # Model 1: used to classify ship type based on a trip
    ship_type_classifier = gen_compiled_ship_type_classifier_model()
    x = dataset.map(
        lambda x: transform_dataset_dict_to_list_with_X_Y(
            dictonary=x, lable="lable", reduce=False
        )
    )
    # x = x.prefetch(tf.data.AUTOTUNE)
    # ship_type_classifier.fit(x=x, epochs=20, verbose=1, callbacks=[callback])

    # Model 2: used to reconstruct missing AIS Data based on a trip
    reproduction_lstm = gen_compiled_LSTM_model()
    x_2 = dataset.map(
        lambda x: remove_entry_and_to_list(x, key_list=["lon", "lat"], exclude=False)
    )
    x_2 = x_2.map(lambda x: (x[:, : points_per_trip - 1], x[:, points_per_trip - 1 :]))
    # print(next(iter(x_2))[1])
    reproduction_lstm.fit(x=x_2, epochs=20, callbacks=[callback])

    model_list = [ship_type_classifier, reproduction_lstm]

    return model_list, x


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
