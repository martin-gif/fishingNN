import pandas as pd
import numpy as np
from sqlalchemy import create_engine


def gen_database_v2():
    path_root = "../../../"
    connection = create_engine(url="sqlite:///../../../saves/db_2.db", echo=False)

    df_carrier_loitering = pd.read_csv(
        f"{path_root}data/revealing_the_supply_chain_at_sea_2021/carrier_loitering_v20210408.csv"
    )
    df_bunker_loitering = pd.read_csv(
        f"{path_root}data/revealing_the_supply_chain_at_sea_2021/bunker_loitering_v20210408.csv"
    )
    df_model_unlabeld = pd.read_csv(
        f"{path_root}data/slavery_in_fisheries/s1_training_final.csv"
    )
    df_model_labeld = pd.read_csv(
        f"{path_root}data/slavery_in_fisheries/s4_final_model_predictions.csv"
    )

    df_carrier_loitering.to_sql(name="carrier_loitering", con=connection)
    df_bunker_loitering.to_sql(name="bunker_loitering", con=connection)
    df_model_unlabeld.to_sql(name="model_unlabeld", con=connection)
    df_model_labeld.to_sql(name="model_labeld", con=connection)


if __name__ == "__main__":
    gen_database_v2()
