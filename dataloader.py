import glob
import os
import pandas as pd
import tensorflow as tf


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

    def getTrainingData(self):
        batch_size = self.batch_size / (len(self.file_list) - 1)
        files = self.file_list.copy()

        while True:
            data = pd.DataFrame()
            labels = []
            index = 0
            if files:
                for csv_file in files:
                    if csv_file == "unknown.csv":  # skip file with unknown labels
                        continue

                    file_path = os.path.join(self.path, csv_file)
                    df = pd.read_csv(
                        file_path,

                        dtype={
                            'mmsi': 'Float32',
                            'timestamp': 'Float32',
                            'distance_from_shore': 'Float32',
                            'distance_from_port': 'Float32',
                            'speed': 'Float32',
                            'course': 'Float32',
                            'lat': 'Float32',
                            'lon': 'Float32',
                            'is_fishing': 'Float64',

                        },
                    )
                    if int((index + 1) * batch_size) <= len(df):
                        df = df.loc[int(index * batch_size): int((index + 1) * batch_size)]
                    else:
                        df = df.loc[index * batch_size: len(df)]
                        files.remove(csv_file)

                    df = df.drop(columns=["source"])
                    data = pd.concat([data, df])
                    labels.extend([self.label_dict[str(csv_file)]] * len(df))

            yield data, pd.DataFrame(labels)
            index += 1

    def genSmalerDataset(self, sample:int, folder):
        n,k = divmod(sample, len(self.file_list))
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
                df = df.iloc[:n+1]
            else:
                df = df.iloc[:n]

            data = pd.concat([data, df])
        rows = len(data)
        data.to_csv(os.path.join(folder,f'{rows}.csv'))

