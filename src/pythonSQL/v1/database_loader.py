import os
from typing import Type, Optional

import tensorflow as tf
from tensorflow.python.data.experimental.ops.readers import SqlDatasetV2
from tensorflow.python.data.ops.dataset_ops import DatasetV2
import pandas as pd
from sqlalchemy import create_engine

from src.utils import get_FeatureSpace, parse_sql_data


def get_dataset_generator(
    points_per_trip: int = 5, trips_per_batch: int = 4, batches_per_class: int = 500
) -> DatasetV2:
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

    return dataset


def get_tf_sql_dataset_all_typs(limit_each_class: int = 1000) -> SqlDatasetV2:
    result_dataset = None

    for id in range(7):
        if id == 3:  # use because id 3 is dataset with unknown ids
            continue
        data = get_tf_sql_dataset_by_shipType_id(
            ship_type_id=id, limit=limit_each_class
        )
        if result_dataset is None:
            result_dataset = data
        else:
            result_dataset = result_dataset.concatenate(dataset=data)

    # print("length: ", result_dataset.reduce(0, lambda x, _: x + 1).numpy())
    return result_dataset


def get_tf_sql_dataset_by_shipType_id(
    ship_type_id: int, limit: int = 10000
) -> SqlDatasetV2:
    dataset = tf.data.experimental.SqlDataset(
        driver_name="sqlite",
        data_source_name="data/Anonymized_AIS_training_data.db",
        query=f"""
                SELECT Anonymized_AIS_training_data.timestamp, Anonymized_AIS_training_data.distance_from_shore, Anonymized_AIS_training_data.distance_from_port, Anonymized_AIS_training_data.speed, Anonymized_AIS_training_data.course, Anonymized_AIS_training_data.lat, Anonymized_AIS_training_data.lon, Anonymized_AIS_training_data.is_fishing, ship_type.id
                FROM Anonymized_AIS_training_data
                JOIN trip ON Anonymized_AIS_training_data.tripId = trip.id
                JOIN ship_type ON trip.ship_type_id = ship_type.id
                WHERE ship_type.id = {ship_type_id}
                LIMIT {limit};
            """,
        output_types=(
            tf.double,
            tf.double,
            tf.double,
            tf.double,
            tf.double,
            tf.double,
            tf.double,
            tf.double,
            tf.int8,
        ),
    )
    return dataset


def get_trip_by_id(id: int = 0):
    trip = pd.read_sql(
        sql=f"""
            SELECT *
            from Anonymized_AIS_training_data
            WHERE date.tripId = {id}
        """,
        con=create_engine(
            "Anonymized_AIS_training_data/Anonymized_AIS_training_data.db"
        ),
    )

    return trip


def get_mean_var(columns: list[str], con=None) -> pd.DataFrame:
    if con is None:
        raise ValueError("con darf nicht None sein")
    df = pd.DataFrame()
    for column_table in columns:
        table, name = column_table.split(sep=".")

        querry = f"""SELECT  avg({column_table}) as mean, SUM(({column_table}-(SELECT AVG({column_table}) FROM {table}))*
                    ({column_table}-(SELECT AVG({column_table}) FROM {table})) ) / (COUNT({column_table})-1) AS var
                    FROM {table}
                """
        tmp = pd.read_sql(sql=querry, con=con)
        tmp["name"] = name
        df = pd.concat([df, tmp], ignore_index=True)

    return df


# todo: Maybe needs to be removed in future
ENGINE = create_engine(
    url="sqlite:///Anonymized_AIS_training_data/Anonymized_AIS_training_data.db",
    echo=False,
)  # Database dosen't exists anymore. code still for future updates


def gen_mean_var_Dataframe():
    path = "saves/mean_var_df.pkl"
    if os.path.isfile(path):
        df = pd.read_pickle(filepath_or_buffer=path)
    else:
        df = get_mean_var(
            [
                "Anonymized_AIS_training_data.timestamp",
                "Anonymized_AIS_training_data.distance_from_shore",
                "Anonymized_AIS_training_data.distance_from_port",
                "Anonymized_AIS_training_data.speed",
                "Anonymized_AIS_training_data.course",
                "Anonymized_AIS_training_data.lat",
                "Anonymized_AIS_training_data.lon",
                "Anonymized_AIS_training_data.is_fishing",
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
