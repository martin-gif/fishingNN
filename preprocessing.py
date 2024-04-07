from dataloader import fishingDataLoader
import pandas as pd

data = fishingDataLoader()

data.genDatasetFromTrips()
# print(type(data))
