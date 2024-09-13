import nni
import pandas as pd
from copy import deepcopy
import numpy as np
from autogluon.timeseries import TimeSeriesDataFrame, TimeSeriesPredictor
from pathlib import Path
import shutil

TIME_LIMIT = 5*60
PREDICTION_LENGTH = 48
METRIC = "MAPE"


def load_data():
    train_1 = pd.read_csv("../tune/2016_train.csv")
    train_2 = pd.read_csv("../tune/2017_train.csv")
    val_1 = pd.read_csv("../tune/2016_val.csv")
    val_2 = pd.read_csv("../tune/2017_val.csv")

    def prepare_AG_data(dfs: list[pd.DataFrame]):
        # Prepare Data
        melted_dfs = []
        for df in dfs:
            df = deepcopy(df)  # [["Time stamp", "LG 1"]]
            df.loc[:, ["Time stamp"]] = df["Time stamp"].str.replace(r'[^0-9.: ]', '', regex=True).str.strip()
            df.loc[:, ["Time stamp"]] = pd.to_datetime(df["Time stamp"])
            # TODO Normalize Data else MASE means nothing
            # Concat Columns & Assign Ids
            df = df.melt(id_vars=['Time stamp'], var_name='item_id', value_name='target')
            df = df.dropna()
            df.interpolate(method='linear', inplace=True)
            melted_dfs.append(df)
        df = pd.concat(melted_dfs)
        df = TimeSeriesDataFrame.from_data_frame(
            df,
            id_column="item_id",
            timestamp_column="Time stamp"
        )
        return df

    train = prepare_AG_data([train_1, train_2])
    val = prepare_AG_data([val_1, val_2])
    return train, val

def trial(hp):
    train, val = load_data()
    predictor = TimeSeriesPredictor(prediction_length=PREDICTION_LENGTH, freq="15min", eval_metric=METRIC)
    predictor = predictor.fit(
        train,
        hyperparameters={
            # TODO Set model
            "NPTS": hp
        },
        time_limit=TIME_LIMIT,
        skip_model_selection=True,
    )

    results = predictor.evaluate(val)
    nni.report_final_result(results)

    print(results)


if __name__ == "__main__":
    # define default parameters
    hp = {

    }
    # overwrite defaults with trial params
    hp.update(nni.get_next_parameters())
    trial(hp)
