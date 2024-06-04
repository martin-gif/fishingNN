import os

import keras
import tensorflow as tf
from keras import Layer
from keras.src.layers.preprocessing.feature_space import FeatureSpace
from tensorflow.python.data.experimental.ops.readers import SqlDatasetV2
import pandas as pd

PATH_ROOT = "../../../"


def get_dataset_v2(batch_size: int = 50):
    dataset = get_sql_dataset_v2()
    dataset = dataset.map(parse_sql_all_v2)

    feature_space_v2 = get_FeatureSpace_v2(dataset=dataset)
    dataset = dataset.map(lambda x: feature_space_v2(x))
    dataset = dataset.batch(batch_size)

    return dataset


def parse_sql_all_v2(*vals):
    features = {
        # "mmsi": tf.convert_to_tensor(vals[0]),
        "Prediction": tf.convert_to_tensor(vals[1]),
        # "start_lat": tf.convert_to_tensor(vals[2]),
        # "start_lon": tf.convert_to_tensor(vals[3]),
        # "end_lat": tf.convert_to_tensor(vals[4]),
        # "end_lon": tf.convert_to_tensor(vals[5]),
        # "mean_lat": tf.convert_to_tensor(vals[6]),
        # "mean_lon": tf.convert_to_tensor(vals[7]),
        # "loitering_start_timestamp": (tf.convert_to_tensor(vals[8])),
        # "loitering_end_timestamp": tf.convert_to_tensor(vals[9]),
        "loitering_hours": tf.convert_to_tensor(vals[10]),
        # "tot_distance_nm": tf.convert_to_tensor(vals[11]),
        "avg_speed_knots": tf.convert_to_tensor(vals[12]),
        "avg_distance_from_shore_nm": tf.convert_to_tensor(vals[13]),
    }
    return features


def get_FeatureSpace_v2(dataset: tf.data.Dataset):
    path = f"saves/featurespace_v2.keras"
    if os.path.isfile(path):
        reloaded_feature_space = keras.models.load_model(path)
        # print("FeatureSpace inputs:", reloaded_feature_space.get_inputs())
        reloaded_feature_space.get_inputs()  # I don't know why but this is need for the loaded model to work
        return reloaded_feature_space
    else:
        features = FeatureSpace(
            features={
                # "mmsi": FeatureSpace.float_normalized(name="normalized_mmsi"),
                "Prediction": FeatureSpace.string_categorical(num_oov_indices=0),
                # "start_lat": FeatureSpace.float_normalized(),
                # "start_lon": FeatureSpace.float_normalized(),
                # "end_lat": FeatureSpace.float_normalized(),
                # "end_lon": FeatureSpace.float_normalized(),
                # "mean_lat": FeatureSpace.float_normalized(),
                # "mean_lon": FeatureSpace.float_normalized(),
                # "loitering_start_timestamp": FeatureSpace.float_normalized(),
                # "loitering_end_timestamp": FeatureSpace.float_normalized(),
                "loitering_hours": FeatureSpace.float_normalized(),
                # "tot_distance_nm": FeatureSpace.float_normalized(),
                "avg_speed_knots": FeatureSpace.float_normalized(),
                "avg_distance_from_shore_nm": FeatureSpace.float_normalized(),
            },
            output_mode="dict",
        )
        features.adapt(dataset)
        features.save(path)
        # print(features)
        return features


def get_sql_dataset_v2(sql_prompt: str = None) -> SqlDatasetV2:

    if sql_prompt is None:
        prompt = baseSQL()
    else:
        prompt = sql_prompt

    dataset = tf.data.experimental.SqlDataset(
        driver_name="sqlite",
        data_source_name="saves/db_2.db",
        query=prompt,
        output_types=(
            tf.double,
            tf.string,
            tf.double,
            tf.double,
            tf.double,
            tf.double,
            tf.double,
            tf.double,
            tf.string,
            tf.string,
            tf.double,
            tf.double,
            tf.double,
            tf.double,
        ),
    )
    return dataset


def baseSQL() -> str:
    sql = """ 
    WITH mmsi_pred as (
        Select unlabeld.mmsi, labeld.Prediction
            FROM model_labeld as labeld
            LEFT JOIN model_unlabeld as unlabeld
               USING (hours, fishing_hours, average_daily_fishing_hours,
                      fishing_hours_foreign_eez, fishing_hours_high_seas, distance_traveled_km)
            GROUP BY unlabeld.mmsi, labeld.Prediction
        ),
        mmsi_pred_grouped as (
            SELECT a.*
            FROM mmsi_pred a
            LEFT JOIN (
                SELECT *
                FROM mmsi_pred
                GROUP BY mmsi
                HAVING COUNT(mmsi_pred.mmsi) > 1
            ) b
            on a.mmsi = b.mmsi
            AND  a.Prediction = b.Prediction
            WHERE b.mmsi is NULL
        ),
        loitering as (
            SELECT bunker_mmsi as mmsi, start_lat,start_lon,end_lat, end_lon,mean_lat,mean_lon, loitering_start_timestamp, loitering_end_timestamp, loitering_hours, tot_distance_nm, avg_speed_knots, avg_distance_from_shore_nm
            FROM bunker_loitering
            UNION ALL
            SELECT carrier_mmsi as mmsi, start_lat,start_lon,end_lat, end_lon,mean_lat,mean_lon, loitering_start_timestamp, loitering_end_timestamp, loitering_hours, tot_distance_nm, avg_speed_knots, avg_distance_from_shore_nm
            FROM carrier_loitering
        )
    SELECT *
    FROM mmsi_pred_grouped
    JOIN loitering USING (mmsi)
    """
    return sql
