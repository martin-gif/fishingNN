import os
import warnings

import numpy as np
import pandas as pd
from sqlalchemy import create_engine
from sqlalchemy.orm import Session
from sqlalchemy_utils import database_exists
from tqdm import tqdm

from dbConnector import Base, Shiptype, Ship, Trip


class fishingDataLoader:
    def __init__(self, path="data/data", batch_size=64, *args, **kwargs):
        self.path = path
        self.batch_size = batch_size
        self.file_list = [file for file in os.listdir(self.path) if ".csv" in file]
        self.label_dict = dict(zip(self.file_list, range(len(self.file_list))))
        self.engine = create_engine(url="sqlite:///data.db", echo=False)

    def __files__(self):
        return self.file_list

    def gen_database(self):
        if not database_exists(self.engine.url):
            print("creat DB")
            # TODO remove following line if db works
            # Base.metadata.drop_all(self.engine)  # only wile testing
            Base.metadata.create_all(self.engine)

    def csv_path(self):
        for file in self.file_list:
            path = os.path.join(self.path, file)
            yield path

    def loadAllTrainingData(self):
        files = self.file_list.copy()
        data = pd.DataFrame()
        labels = []

        if files:
            for csv_file in files:
                if csv_file == "unknown.csv":  # skip file with unknown labels
                    continue

                file_path = os.path.join(self.path, csv_file)
                df = pd.read_csv(file_path)

                df = df.drop(columns=["source"])
                data = pd.concat([data, df])
                labels.extend([self.label_dict[str(csv_file)]] * len(df))

        data = data.assign(labels=labels)
        print(data)
        return data.reset_index(drop=True)

    def genSmalerDataset(self, sample: int, folder):
        n, k = divmod(sample, len(self.file_list))
        print(n, k)
        files = self.file_list.copy()
        data = pd.DataFrame()
        for i, csv_file in enumerate(files):
            if csv_file == "unknown.csv":  # skip file with unknown labels
                print("unknown")
                continue

            file_path = os.path.join(self.path, csv_file)
            df = pd.read_csv(file_path)
            df = df.drop(columns=["source"])
            if i >= n:
                df = df.iloc[: n + 1]
            else:
                df = df.iloc[:n]

            data = pd.concat([data, df])
        rows = len(data)
        data.to_csv(os.path.join(folder, f"{rows}.csv"))

    def filter_len(self, data_frame_iter: list, min: int = 0, max: int = 10000):
        result = []
        for df in data_frame_iter:
            rows = len(df)
            if rows <= min or rows >= max:
                # print("dont fit", rows)
                continue
            else:
                # print("fit", rows)
                result.append(df)
        return result

    def remove_rows_between_trips(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        clean rows where ship is not on trip so separation trips later on works better
        :param df: Dataframe with multiple rows where ship is not on a trip
        :return df: cleand Dataframe
        """
        columne_to_split = "distance_from_shore"
        result = df.drop(
            df[
                (
                    (df[columne_to_split].shift() == df[columne_to_split])
                    & (df[columne_to_split] == 0)
                )
            ].index
        )
        return result

    def _upsert_into_db(self, entry: Base):
        with Session(self.engine) as session:
            session.add(entry)
            session.commit()

    def genDatasetFromTrips(
        self, min_length: int = 100, max_length: int = 10000
    ) -> None:
        warnings.simplefilter(
            action="ignore", category=FutureWarning
        )  # suppress Pandas Future warning
        self.gen_database()
        columne_to_split = "distance_from_shore"
        trip_id = 0
        if self.file_list:
            for ship_type_id, path in tqdm(enumerate(self.csv_path())):
                # used for test reasons
                # if "pole_and_line" not in path:
                #     continue

                # generate ship entry and persist save it
                file_basename = os.path.basename(path)
                ship_type = os.path.splitext(file_basename)[0]
                ship = Shiptype(id=ship_type_id, name=ship_type)
                # print(ship_type, ship_type_id)
                self._upsert_into_db(ship)

                # read data and remove rows containing Null value
                data = pd.read_csv(path)
                data = data.dropna()
                group_by_mmsi = data.groupby(by="mmsi")
                # print((df["distance_from_port"] == 0).astype(int).sum(axis=0))
                # print(group_by_mmsi.size())

                list_trips = []
                for shipname, ship in group_by_mmsi:
                    # generate Ship entry and insert it into DB. Needed to make trip unique later
                    shipname_as_int = int(shipname)
                    current_ship = Ship(name=int(shipname_as_int))
                    self._upsert_into_db(current_ship)
                    # remove disturbing rows and split dataset int trips
                    ship = self.remove_rows_between_trips(df=ship)
                    trips = np.split(ship, np.where(ship[columne_to_split] == 0)[0])

                    # check if trip is long enough
                    result_list = self.filter_len(trips, min=min_length, max=max_length)
                    len_result_list = len(result_list)

                    if len_result_list > 0:
                        # Safe trip with id in DB
                        for trip_df in result_list:
                            # print(trip_id)
                            current_trip = Trip(
                                id=trip_id,
                                ship_type_id=ship_type_id,
                                ship_mmsi=shipname_as_int,
                            )
                            self._upsert_into_db(current_trip)
                            trip_id += 1

                            assert isinstance(trip_df, pd.DataFrame)
                            trip_df["tripid"] = trip_id
                            trip_df.to_sql(
                                name="data", con=self.engine, if_exists="append"
                            )


if __name__ == "__main__":
    loader = fishingDataLoader()
    # for path in loader.csv_path():
    #     print(path)
    loader.gen_database()

    d = {"distance_from_shore": [1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 0, 1, 1]}
    df = pd.DataFrame(data=d)
    # print(df)
    cleand_df = loader.remove_rows_between_trips(df=df)
    # print(cleand_df)
    trips = np.split(cleand_df, np.where(cleand_df["distance_from_shore"] == 0)[0])
    for trip in trips:
        print(trip)
