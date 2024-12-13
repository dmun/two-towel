"""
%load_ext autoreload
%autoreload 2
"""

import os
import sys
import kagglehub
import polars as pl
import torch
import seaborn as sn
import matplotlib.pyplot as plt
from torch.nn import BCEWithLogitsLoss
from torch.utils.data import DataLoader
from util import custom_train_test_split, GameDataset, compute_metrics, evaluate_model, WeightedGameDataset
from model import TwoTowerModel
from train import train
from sklearn.metrics import confusion_matrix
from torch.profiler import profile, record_function, ProfilerActivity
import polars.selectors as cs


def detect_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")


device = detect_device()

# %%
path = kagglehub.dataset_download("antonkozyriev/game-recommendations-on-steam")
print("Path to dataset files:", path)

# %%
recommendations_path = os.path.join(path, "recommendations.csv")
recommendations_df = pl.read_csv(recommendations_path)
# recommendations_df = recommendations_df.sample(fraction=0.5, seed=42)
reviews_per_user = recommendations_df.group_by("user_id").len()
users_with_more_than_one_review = reviews_per_user.filter(pl.col("len") > 10).get_column("user_id").to_list()
recommendations_df = recommendations_df.filter(pl.col("user_id").is_in(users_with_more_than_one_review))
game_ids = recommendations_df.get_column("app_id").unique()
user_ids = recommendations_df.get_column("user_id").unique()
num_users = user_ids.count()
num_games = game_ids.count()

# %%
num_pos = len(recommendations_df.filter(pl.col("is_recommended") == 1))
num_neg = len(recommendations_df) - num_pos
class_weights = torch.tensor([num_neg / num_pos], device=device)
class_weights

# %%
recommendations_df = recommendations_df.with_columns(
    pl.col("user_id").cast(pl.Int32),
    pl.col("app_id").cast(pl.Int32),
    pl.col("is_recommended").cast(pl.Int8),
    pl.col("hours").cast(pl.Float32),
)
recommendations_df = recommendations_df.drop("date")

user_id_map = {id_: idx for idx, id_ in enumerate(sorted(user_ids.unique()))}
game_id_map = {id_: idx for idx, id_ in enumerate(sorted(game_ids.unique()))}

recommendations_df = recommendations_df.with_columns(
    [
        pl.col("user_id").replace(user_id_map).alias("user_id"),
        pl.col("app_id").replace(game_id_map).alias("app_id"),
    ]
)

# %%
for quantile in [0.25, 0.5, 0.75]:
    recommendations_df = (
        recommendations_df[["app_id", "hours"]]
        .group_by("app_id")
        .quantile(quantile)
        .rename({"hours": f"quantile_{int(quantile*100)}"})
        .join(recommendations_df, on="app_id")
    )

recommendations_df = recommendations_df.drop(cs.matches(".*_right"))

recommendations_df = recommendations_df.with_columns(
    pl.when(pl.col("hours") > pl.col("quantile_75"))
    .then(1.0)
    .when(pl.col("hours") > pl.col("quantile_50"))
    .then(0.5)
    .when(pl.col("hours") > pl.col("quantile_25"))
    .then(0.25)
    .otherwise(0.1)
    .alias("playtime_weight")
)

# %%
train_data, test_data = custom_train_test_split(recommendations_df, "user_id", "is_recommended")
datasets = {
    "train": WeightedGameDataset(train_data),
    "test": WeightedGameDataset(test_data),
}

# %%
baseline_model = TwoTowerModel(num_users, num_games, 32)
train(
    baseline_model,
    datasets,
    loss_fn=BCEWithLogitsLoss(pos_weight=class_weights, reduction="none"),
    lr=0.001,
    num_epochs=10,
    batch_size=10000,
    device=device,
)

# %%
sys.getsizeof(baseline_model.game_embedding.weight.storage())

# %%
test_loader = DataLoader(datasets["test"], batch_size=5000, pin_memory=True, num_workers=7)
baseline_predictions, baseline_labels = evaluate_model(baseline_model, test_loader, device=device)

print("Baseline Model Metrics:")
compute_metrics(baseline_predictions, baseline_labels)

# %%
cm = confusion_matrix(baseline_labels, baseline_predictions.round())
ax = sn.heatmap(cm, annot=True, fmt="d")

ax.set_xlabel("pred")
ax.xaxis.set_label_position("top")
ax.xaxis.tick_top()

ax.set_ylabel("true")
ax.invert_yaxis()
ax.invert_xaxis()

# %%
cm = confusion_matrix([1, 1], [0, 1])
ax = sn.heatmap(cm, annot=True, fmt="d")
ax.xaxis.tick_top()

# %%
torch.save(baseline_model.state_dict(), "weighted_model")
