from typing import Type, Optional

import tensorflow as tf
from tensorflow.python.data.experimental.ops.readers import SqlDatasetV2
from tensorflow.python.data.ops.dataset_ops import DatasetV2
import pandas as pd


def get_tf_sql_dataset_all_typs(limit_each_class: int = 1000) -> SqlDatasetV2:
    result_dataset = None

    for id in range(6):
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
        data_source_name="data/data.db",
        query=f"""
                SELECT data.timestamp, data.distance_from_shore, data.distance_from_port, data.speed, data.course, data.lat, data.lon, data.is_fishing, ship_type.id
                FROM data
                JOIN trip ON data.tripId = trip.id
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
