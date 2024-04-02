import glob
import os
import pandas as pd
import tensorflow as tf
import numpy as np
from pandas.core.groupby import DataFrameGroupBy


class fishingDataLoader:
    def __init__(self, path="data/data", batch_size=64, *args, **kwargs):
        self.path = path
        self.batch_size = batch_size
        self.file_list = [file for file in os.listdir(self.path) if ".csv" in file]
        self.label_dict = dict(zip(self.file_list, range(len(self.file_list))))

    def __files__(self):
        return self.file_list

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

    def genDatasetFromTrips(self, sample: int) -> list[list]:
        n, k = divmod(sample, len(self.file_list))
        # print(n, k)
        columne_to_split = "distance_from_shore"
        if self.file_list:
            file_path = os.path.join(self.path, "pole_and_line.csv")
            df = pd.read_csv(file_path)
            group_by_mmsi = df.groupby(by="mmsi")
            print("unique mms's:", len(set(df["mmsi"])))
            # print((df["distance_from_port"] == 0).astype(int).sum(axis=0))
            # print(group_by_mmsi.size())
            list_trips = []
            n = 0
            for shipname, ship in group_by_mmsi:
                # print(ship.size)
                # first split dataframe into trips, where each trip is between to distance_from_port == 0
                # print(int(shipname), end=": ")
                # print((ship["distance_from_port"] == 0).astype(int).sum(), end=" :")
                ship = ship.drop(
                    ship[
                        (
                            (ship[columne_to_split].shift() != ship[columne_to_split])
                            & (ship[columne_to_split] == 0)
                        )
                    ].index
                )
                # print(ship.size, end=": ")
                # print((ship["distance_from_port"] == 0).astype(int).sum())
                # index_to_split = np.where(ship["distance_from_port"] == 0)[0] # gets all indize as a ndarray where condition is true
                # print(index_to_split, index_to_split.size)
                trips = np.split(ship, np.where(ship[columne_to_split] == 0)[0])
                # print(len(trips[0]))
                result_list = self.filter_len(trips, min=60)
                # print(len(result_list))
                if len(result_list) > 0:
                    list_trips.append(result_list)

                if len(list_trips) > 0:
                    print(len(list_trips[n]))
                    n += 1

        return list_trips
