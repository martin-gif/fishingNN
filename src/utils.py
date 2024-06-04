import os

import keras.api.models
import keras.api.utils
import tensorflow as tf
from sqlalchemy import create_engine
from tf_keras.src.utils import FeatureSpace


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


@tf.function
def remove_entry_and_to_list(dictonary, key_list: list, exclude: bool = True):
    """
    Takes a dictionary with lists as values and returns a single list containing all the elements from the lists.

    Args:
    dictonary : A dictionary where the values are lists.
    exclude: if True key_list will be excluded, if False only key_list will be included


    Returns:
    list: A list containing all the elements from the lists in the dictionary.
    """
    tensor_list = []
    for key, value in dictonary.items():
        if exclude:
            if key not in key_list:
                tensor_list.append(value)
        else:
            if key in key_list:
                tensor_list.append(value)

    combined_tensor = tf.concat(tensor_list, axis=-1)
    return combined_tensor


@tf.function
def transform_dataset_dict_to_list_with_X_Y(
    dictonary, lable: str, reduce: bool = False
):
    """

    :param dictionary: Data dictionary
    :param lable: Dictionary key used to split the dataset
    :param reduce: if True only takes the first row of each lable entry
    :return:
    """
    tensor_list = []
    lable_list = []
    for key, value in dictonary.items():
        if key == lable:
            if reduce:
                value = value[:, 0]
            lable_list.append(value)
        else:
            tensor_list.append(value)

    combined_tensor = tf.concat(tensor_list, axis=-1)
    lable_tensor = tf.concat(lable_list, axis=-1)

    return (combined_tensor, lable_tensor)


def get_FeatureSpace(dataset: tf.data.Dataset) -> keras.utils.FeatureSpace:
    path = "Anonymized_AIS_training_data/featurespace.keras"
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
