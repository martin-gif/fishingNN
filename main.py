import keras.utils
import tensorflow as tf
from sqlalchemy import create_engine
from dataloader import fishingDataLoader
from dataset_read import get_tf_sql_dataset_all_typs
from model.nn.shipType import gen_compiled_ship_type_classifier_model


ENGINE = create_engine(url="sqlite:///data/data.db", echo=False)


def parse_sql_data(*vals):

    # features = {
    #     "mmsi": tf.convert_to_tensor(vals[0]),
    #     "timestamp": tf.convert_to_tensor(vals[1]),
    #     "distance_from_shore": tf.convert_to_tensor(vals[2]),
    #     "distance_from_port": tf.convert_to_tensor(vals[3]),
    #     "speed": tf.convert_to_tensor(vals[4]),
    #     "course": tf.convert_to_tensor(vals[5]),
    #     "lat": tf.convert_to_tensor(vals[6]),
    #     "lon": tf.convert_to_tensor(vals[7]),
    #     "is_fishing": tf.convert_to_tensor(vals[8]),
    # }
    features = tf.convert_to_tensor(vals[:9])
    features = tf.transpose(features)
    lable = tf.one_hot(vals[9], depth=6)
    return (features, lable)


def train():
    # get Dataset and preprocess it
    dataset = get_tf_sql_dataset_all_typs(limit_each_class=10000)
    dataset = dataset.shuffle(buffer_size=1000, seed=42)
    dataset = dataset.batch(200)
    dataset = dataset.map(parse_sql_data)
    ds_training, ds_val = keras.utils.split_dataset(dataset, left_size=0.8)

    # generate models
    ship_type_classifier = gen_compiled_ship_type_classifier_model()

    for feature, lable in dataset:
        print(feature.numpy())
        # print("predict:", ship_type_classifier.predict(feature[0]))
        print("lable", lable.numpy())
        break

    # train models
    ship_type_classifier.fit(x=ds_training, epochs=10, validation_data=ds_val)


if __name__ == "__main__":

    loader = fishingDataLoader(input_engine=ENGINE)
    loader.gen_SQL_db()
    train()
