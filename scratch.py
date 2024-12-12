import os
import numpy as np
import seaborn as sn
import kagglehub
import matplotlib.pyplot as plt
import polars as pl
import torch
from torch._prims_common import DeviceLikeType
import torch.nn as nn
from sklearn.metrics import confusion_matrix
from torch.utils.data import DataLoader, Dataset
from torchmetrics import Accuracy, F1Score, MetricCollection, Precision, Recall
from tqdm.notebook import tqdm
from util import custom_train_test_split, GameDataset
from model import TwoTowerModel
from train import train


# %%
cm = confusion_matrix(baseline_labels, baseline_predictions.round())
ax = sn.heatmap(cm, annot=True, fmt="d")
ax.xaxis.tick_top()
ax.invert_yaxis()
ax.invert_xaxis()

# %%
pred_arr = baseline_predictions.numpy()
rev_user_id_map = {value: key for key, value in user_id_map.items()}
rev_game_id_map = {value: key for key, value in game_id_map.items()}

# %%
plt.hist(baseline_predictions.numpy())

# %%
app_id = 1748230
train_data.with_columns(
    [
        pl.col("user_id").replace(rev_user_id_map).alias("user_id"),
        pl.col("app_id").replace(rev_game_id_map).alias("app_id"),
    ]
).filter(pl.col("app_id") == app_id)

# %%
games_metadata_path = os.path.join(path, "games_metadata.json")
games_metadata_df = pl.read_ndjson(games_metadata_path)
games_metadata_df = games_metadata_df.with_columns(pl.col("app_id").cast(pl.Int32))

# %%
games_path = os.path.join(path, "games.csv")
games_df = pl.read_csv(games_path)
games_df = games_df.with_columns(pl.col("app_id").cast(pl.Int32))

with pl.Config(tbl_rows=1000):
    result_df = test_data.with_columns(
        pl.Series("pred", baseline_predictions.numpy()),
    )
    print(result_df.sort("pred", descending=True))
    # test_id = 709735
    # test_id = 739712
    test_id = 329960
    print(
        train_data.with_columns(
            [
                pl.col("user_id").replace(rev_user_id_map).alias("user_id"),
                pl.col("app_id").replace(rev_game_id_map).alias("app_id"),
            ]
        )
        .filter(pl.col("user_id") == rev_user_id_map[test_id])
        .join(games_metadata_df, on="app_id", how="left")
        .join(games_df, on="app_id", how="left")
        .select(["user_id", "app_id", "title", "tags", "is_recommended"])
    )
    result_df = (
        result_df.with_columns(
            [
                pl.col("user_id").replace(rev_user_id_map).alias("user_id"),
                pl.col("app_id").replace(rev_game_id_map).alias("app_id"),
            ]
        )
        .join(games_metadata_df, on="app_id", how="left")
        .join(games_df, on="app_id", how="left")
        .select(["user_id", "app_id", "title", "tags", "is_recommended", "pred"])
    )
    result_df = result_df.filter(pl.col("user_id") == rev_user_id_map[test_id])
    print(result_df.sort("pred", descending=True))
    # result_df.with_columns(pl.when(pl.col("pred") > 0.5).then(1).otherwise(0).alias("pred"))
