import numpy as np
import polars as pl
import torch
from torch._prims_common import DeviceLikeType
from torch.utils.data import Dataset
from torchmetrics import Accuracy, F1Score, MetricCollection, Precision, Recall


def custom_train_test_split(
    df: pl.DataFrame,
    user_col: str,
    label_col: str,
    random_state: int = 42,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """
    Split Polars DataFrame by keeping all but one interaction per user in train set.

    Parameters:
    df (pl.DataFrame): Polars DataFrame with user-item interactions
    user_col (str): Name of user ID column
    label_col (str, optional): Name of label column if exists
    random_state (int): Random seed for reproducibility

    Returns:
    tuple: (train_df, test_df) as Polars DataFrames
    """
    np.random.seed(random_state)

    # Add random number column for sampling
    df_with_rand = df.with_columns(pl.lit(np.random.random(len(df))).alias("_random"))

    # Get counts per user
    user_counts = df.group_by(user_col).agg(pl.count().alias("interaction_count"))

    # For users with multiple interactions, select one random interaction for test
    users_multiple = user_counts.filter(pl.col("interaction_count") > 1).select(user_col)

    # Get one random interaction per user for test set
    test_df = (
        df_with_rand.join(users_multiple, on=user_col, how="inner")
        .group_by(user_col)
        .agg(pl.all().sort_by("_random").first())
        .drop("_random")
    )

    # Get all other interactions for train set
    train_df = df_with_rand.join(test_df, on=df.columns, how="anti").drop("_random")

    # Print statistics
    total_users = df.get_column(user_col).n_unique()
    train_users = train_df.get_column(user_col).n_unique()
    test_users = test_df.get_column(user_col).n_unique()

    print("\nSplit Statistics:")
    print(f"Total users: {total_users}")
    print(f"Users in train: {train_users}")
    print(f"Users in test: {test_users}")

    if label_col:
        train_pos_ratio = (train_df.get_column(label_col) == True).mean()
        test_pos_ratio = (test_df.get_column(label_col) == True).mean()
        print(f"\nTrain positive ratio: {train_pos_ratio:.2%}")
        print(f"Test positive ratio: {test_pos_ratio:.2%}")

    avg_train = len(train_df) / train_users
    print(f"\nAverage train interactions per user: {avg_train:.2f}")
    print("Test interactions per user: 1.0")  # By design

    return train_df, test_df


class GameDataset(Dataset):
    def __init__(self, data):
        self.user_ids = torch.tensor(data["user_id"].to_numpy(), dtype=torch.long)
        self.game_ids = torch.tensor(data["app_id"].to_numpy(), dtype=torch.long)
        self.labels = torch.tensor(data["is_recommended"].to_numpy(), dtype=torch.float32)

    def __len__(self):
        return len(self.user_ids)

    def __getitem__(self, idx):
        return self.user_ids[idx], self.game_ids[idx], self.labels[idx]


def evaluate_model(model, loader, device: DeviceLikeType = "cpu"):
    model.eval()
    all_predictions = []
    all_labels = []

    with torch.no_grad():
        for user_ids, game_ids, labels in loader:
            user_ids = user_ids.to(device)
            game_ids = game_ids.to(device)
            labels = labels.to(device)

            predictions = model(user_ids, game_ids)
            all_predictions.append(predictions.cpu())
            all_labels.append(labels.cpu())

    all_predictions = torch.cat(all_predictions)
    all_labels = torch.cat(all_labels)

    return all_predictions, all_labels


metrics = MetricCollection(
    {
        "accuracy": Accuracy(task="binary"),
        "precision": Precision(task="binary"),
        "recall": Recall(task="binary"),
        "f1": F1Score(task="binary"),
    }
)


def compute_metrics(predictions, labels):
    predictions = predictions.view(-1)
    labels = labels.view(-1)

    result = metrics(predictions, labels)
    for name, value in result.items():
        print(f"{name}: {value:.4f}")
