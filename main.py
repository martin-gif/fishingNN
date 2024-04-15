import tensorflow as tf
from sqlalchemy import create_engine
from dataloader import fishingDataLoader
from model.shipType import gen_compiled_ship_type_classifier_model
import keras
from keras.utils import FeatureSpace

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
    batch_size = 200
    # get Dataset
    dataset = tf.data.experimental.SqlDataset(
        driver_name="sqlite",
        data_source_name="data/data.db",
        query="""
            SELECT data.mmsi, data.timestamp, data.distance_from_shore, data.distance_from_port, data.speed, data.course, data.lat, data.lon, data.is_fishing, ship_type.id
            FROM data
            JOIN trip ON data.tripId = trip.id
            JOIN ship_type ON trip.ship_type_id = ship_type.id
            LIMIT 1000;
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
    dataset = dataset.shuffle(buffer_size=10000)
    dataset = dataset.batch(20)

    dataset = dataset.map(parse_sql_data)

    print(next(iter(dataset)))

    # for feature, lable in dataset:
    #     print(feature)
    #     print(lable)
    #     break

    # feature_space = FeatureSpace(
    #     features={
    #         "mmsi": FeatureSpace.float_normalized(),
    #         "timestamp": FeatureSpace.float_normalized(),
    #         "distance_from_shore": FeatureSpace.float_normalized(),
    #         "distance_from_port": FeatureSpace.float_normalized(),
    #         "speed": FeatureSpace.float_normalized(),
    #         "course": FeatureSpace.float_normalized(),
    #         "lat": FeatureSpace.float_normalized(),
    #         "lon": FeatureSpace.float_normalized(),
    #         # "id": FeatureSpace.float_normalized(),
    #     }
    # )
    # feature_space.adapt(dataset)

    # generate models
    ship_type_classifier = gen_compiled_ship_type_classifier_model()

    # train models
    ship_type_classifier.fit(dataset, epochs=10)


if __name__ == "__main__":
    loader = fishingDataLoader(input_engine=ENGINE)
    loader.gen_SQL_db()
    train()
