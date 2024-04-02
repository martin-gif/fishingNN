from dataloader import fishingDataLoader
import pandas as pd

data = fishingDataLoader()

data = data.genDatasetFromTrips(sample=6000)
# print(type(data))
