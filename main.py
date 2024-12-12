import os
import kagglehub
import polars as pl
import torch
from torch.utils.data import DataLoader
from util import custom_train_test_split, GameDataset, compute_metrics, evaluate_model
from model import TwoTowerModel
from train import train


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
# recommendations_df = recommendations_df.sample(fraction=1, seed=42)
reviews_per_user = recommendations_df.group_by("user_id").len()
users_with_more_than_one_review = reviews_per_user.filter(pl.col("len") > 10).get_column("user_id").to_list()
recommendations_df = recommendations_df.filter(pl.col("user_id").is_in(users_with_more_than_one_review))
game_ids = recommendations_df.get_column("app_id").unique()
user_ids = recommendations_df.get_column("user_id").unique()
num_users = user_ids.count()
num_games = game_ids.count()


# %%
recommendations_df = recommendations_df.with_columns(
    pl.col("user_id").cast(pl.Int32),
    pl.col("app_id").cast(pl.Int32),
    pl.col("is_recommended").cast(pl.Int8),
    pl.col("hours").cast(pl.Float32),
    # pl.col('playtime_weight').cast(pl.Float32)
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
train_data, test_data = custom_train_test_split(recommendations_df, "user_id", "is_recommended")
datasets = {
    "train": GameDataset(train_data),
    "test": GameDataset(test_data),
}

# %%
baseline_model = TwoTowerModel(num_users, num_games, 32)
train(
    baseline_model,
    datasets,
    lr=0.001,
    num_epochs=10,
    batch_size=10000,
    device=device,
)


# %%
test_loader = DataLoader(datasets["test"], batch_size=10000, pin_memory=True)
baseline_predictions, baseline_labels = evaluate_model(baseline_model, test_loader, device=device)

print("Baseline Model Metrics:")
compute_metrics(baseline_predictions, baseline_labels)

# %%
torch.save(baseline_model.state_dict(), "baseline_model")
