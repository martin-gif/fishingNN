import sys

import requests
import pandas as pd
import numpy as np
import os
from tqdm import tqdm


def imo_to_csv(file_name: str = "IUUList-20240521.xls", api_key=None):
    if api_key is None:
        api_key = os.environ.get("globalfishingwatchAPI")

    file_name = "IUUList-20240521.xls"
    # prepare Data
    df = pd.read_excel(io=file_name, sheet_name=1, usecols=["CurrentlyListed", "IMO"])
    df = df.dropna()
    df["mmsi"] = "0"
    df = df.astype({"CurrentlyListed": np.bool_, "IMO": np.int32, "mmsi": np.int32})
    df = df[df["CurrentlyListed"] == True]  # nur IUU Schiffe
    df = df.reset_index(drop=True)

    # check every imo to see if GFW has an entry with a MMSI for it
    for entry in tqdm(df.values):
        imo = entry[1]

        request = (
            f"https://gateway.api.globalfishingwatch.org/v2/vessels/advanced-search?datasets=public-global-carrier-vessels:latest,public-global-fishing-vessels:latest,public-global-support-vessels:latest&query=imo%20%3D%20%27"
            + str(imo)
            + "%27&limit=1&offset=0"
        )

        response = requests.get(
            url=request, headers={"Authorization": f"Bearer {api_key}"}
        )
        tmp = (
            response.json()["entries"][0]["mmsi"]
            if len(response.json()["entries"]) > 0
            else 0
        )

        if tmp != 0:
            print(response.json())

        df.loc[df.IMO == imo, "mmsi"] = tmp

    sys.exit()
    # drop rows where no mmsi was found for the imo
    df = df[df["mmsi"] != 0]
    df = df.reset_index(drop=True)

    # safe results to csv
    df.to_csv("imo_to_mmsi.csv")


if __name__ == "__main__":
    imo_to_csv()
