import pandas as pd


def load_files():
    fishing_vessel = pd.read_csv("data/Fishing_Effort/fishing-vessels-v2.csv")
    loiter_carrier = pd.read_csv(
        "data/revealing_the_supply_chain_at_sea_2021/carrier_loitering_v20210408.csv"
    )
    bunker_carrier = pd.read_csv(
        "data/revealing_the_supply_chain_at_sea_2021/bunker_loitering_v20210408.csv"
    )
    model_unlabeled = pd.read_csv("data/slavery_in_fisheries/s1_training_final.csv")
    model_labeled = pd.read_csv(
        "data/slavery_in_fisheries/s4_final_model_predictions.csv"
    )

    return (
        fishing_vessel,
        loiter_carrier,
        bunker_carrier,
        model_unlabeled,
        model_labeled,
    )


def get_data():
    fishing_vessel, loiter_carrier, bunker_carrier, model_unlabeled, model_labeled = (
        load_files()
    )
    merge_on = [
        "hours",
        "fishing_hours",
        "average_daily_fishing_hours",
        "fishing_hours_foreign_eez",
        "fishing_hours_high_seas",
        "distance_traveled_km",
    ]

    model_df = pd.merge(
        model_unlabeled[merge_on + ["mmsi"]], model_labeled, on=merge_on, how="right"
    )
    grouped_mmsi = (
        model_df.groupby("mmsi")["Prediction"].apply(pd.Series.mode).reset_index()
    )
    grouped_mmsi = grouped_mmsi.drop_duplicates(subset=["mmsi"], keep="last")

    loiter_carrier_data = pd.merge(
        grouped_mmsi[["mmsi", "Prediction"]],
        loiter_carrier,
        left_on="mmsi",
        right_on="carrier_mmsi",
        how="right",
    )
    loiter_bunker_data = pd.merge(
        grouped_mmsi[["mmsi", "Prediction"]],
        bunker_carrier,
        left_on="mmsi",
        right_on="bunker_mmsi",
        how="right",
    )

    loiter_data = pd.concat([loiter_carrier_data, loiter_bunker_data])

    ## Define Class Labeling and encode classes to chosen definitions
    IUU_LABEL = 0
    NON_IUU_LABEL = 1
    UNLABELLED = -1

    loiter_data["loitering_start_timestamp"] = pd.to_datetime(
        loiter_data["loitering_start_timestamp"]
    )
    loiter_data["loitering_start_hour"] = loiter_data[
        "loitering_start_timestamp"
    ].dt.hour
    loiter_data["response"] = loiter_data["Prediction"].apply(
        lambda x: (
            IUU_LABEL
            if x == "Positive"
            else (NON_IUU_LABEL if x == "Negative" else UNLABELLED)
        )
    )

    df_loiter_data_labeled = loiter_data[
        loiter_data["response"] != UNLABELLED
    ].reset_index(drop=True)

    columns = [
        "response",
        "loitering_start_hour",
        "loitering_hours",
        "tot_distance_nm",
        "avg_speed_knots",
        "avg_distance_from_shore_nm",
    ]

    return df_loiter_data_labeled[columns]
