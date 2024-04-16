from typing import Type, Optional

import tensorflow as tf
from tensorflow.python.data.experimental.ops.readers import SqlDatasetV2
from tensorflow.python.data.ops.dataset_ops import DatasetV2


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
                SELECT data.mmsi, data.timestamp, data.distance_from_shore, data.distance_from_port, data.speed, data.course, data.lat, data.lon, data.is_fishing, ship_type.id
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
            tf.double,
            tf.int8,
        ),
    )
    return dataset
