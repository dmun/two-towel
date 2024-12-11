import os

# import os
# os.environ["CUDA_LAUNCH_BLOCKING"] = '1'
import numpy as np
import kagglehub
import matplotlib.pyplot as plt
import polars as pl
import torch
from torch._prims_common import DeviceLikeType
import torch.nn as nn
from sklearn.model_selection import GroupShuffleSplit, train_test_split
from torch.utils.data import DataLoader, Dataset, IterableDataset
from torchmetrics import (
    Accuracy,
    MetricCollection,
    Precision,
    Recall,
    RetrievalNormalizedDCG,
)
from tqdm import tqdm
from plotter import Plotter


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
# users_path = os.path.join(path, "users.csv")
# df = pl.read_csv(users_path)
# users_df = df.sample(fraction=1, seed=42)
# users_df = users_df.filter(pl.col("reviews") > 5)

# %%
recommendations_path = os.path.join(path, "recommendations.csv")
df = pl.read_csv(recommendations_path)
recommendations_df = df.sample(fraction=1, seed=42)
reviews_per_user = recommendations_df.group_by("user_id").len()
users_with_more_than_one_review = (
    reviews_per_user.filter(pl.col("len") > 8).get_column("user_id").to_list()
)
recommendations_df = recommendations_df.filter(
    pl.col("user_id").is_in(users_with_more_than_one_review)
)
game_ids = recommendations_df.get_column("app_id").unique()
user_ids = recommendations_df.get_column("user_id").unique()
recommendations_df

# %%
# recommendations_df["is_recommended"].describe()
# rev_count = recommendations_df.group_by("user_id").count()
# rev_count.sort(by="count")
# plt.hist(rev_count["count"])
# train, test = train_test_split(
#     recommendations_df,
#     test_size=0.2,
#     random_state=42,
#     stratify=recommendations_df["is_recommended"],
# )

# %%
# train["is_recommended"].describe()

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
    users_multiple = user_counts.filter(pl.col("interaction_count") > 1).select(
        user_col
    )

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
    print(f"Test interactions per user: 1.0")  # By design

    return train_df, test_df


# train_users, test_users = train_test_split(
#     user_ids.to_numpy(), test_size=0.2, random_state=42
# )

train_data, test_data = custom_train_test_split(
    recommendations_df, "user_id", "is_recommended"
)


# %%
len(np.intersect1d(train_data, test_data))


# %%
class GameDataset(Dataset):
    def __init__(self, data):
        self.user_ids = torch.tensor(data["user_id"].to_numpy(), dtype=torch.long)
        self.game_ids = torch.tensor(data["app_id"].to_numpy(), dtype=torch.long)
        self.labels = torch.tensor(
            data["is_recommended"].to_numpy(), dtype=torch.float32
        )
        # self.weights = torch.tensor(data['playtime_weight'].to_numpy(), dtype=torch.float32)

    def __len__(self):
        return len(self.user_ids)

    def __getitem__(self, idx):
        # return self.user_ids[idx], self.game_ids[idx], self.labels[idx], self.weights[idx]
        return self.user_ids[idx], self.game_ids[idx], self.labels[idx]


train_dataset = GameDataset(train_data)
test_dataset = GameDataset(test_data)

batch_size = 8096

train_loader = DataLoader(
    train_dataset,
    batch_size=batch_size,
    shuffle=True,
    pin_memory=True,
)
test_loader = DataLoader(
    test_dataset,
    batch_size=batch_size,
    pin_memory=True,
)


# %%
class TwoTowerModel(torch.nn.Module):
    def __init__(self, num_users, num_games, embedding_dim=128):
        super(TwoTowerModel, self).__init__()
        self.user_embedding = torch.nn.Embedding(num_users, embedding_dim)
        self.game_embedding = torch.nn.Embedding(num_games, embedding_dim)

    def forward(self, user_ids, game_ids):
        user_embeds = self.user_embedding(user_ids)
        game_embeds = self.game_embedding(game_ids)

        scores = (user_embeds * game_embeds).sum(dim=1)
        return torch.sigmoid(scores)


# num_users = train_dataset.user_ids.max().item() + 1
# num_games = train_dataset.game_ids.max().item() + 1
num_users = user_ids.count()
num_games = game_ids.count()

baseline_model = TwoTowerModel(num_users, num_games, 20)
print(num_users)
print(num_games)


# %%
class WeightedLoss(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.bce_loss = torch.nn.BCELoss()

    def forward(self, predictions, targets, weights):
        loss = self.bce_loss(predictions, targets)
        weighted_loss = loss * weights
        return weighted_loss.mean()


# %%
def accuracy(out, y):
    return Accuracy(task="binary")(out, y)


# %%
def train_model(
    model: TwoTowerModel,
    loader,
    num_epochs=10,
    lr=0.01,
    optimizer=torch.optim.Adam,
    loss_fn: nn.Module = torch.nn.BCELoss(),
    device: DeviceLikeType = "cpu",
    plot=True,
):
    model = model.to(device)
    optimizer = optimizer(model.parameters(), lr=lr)  # , weight_decay=5e-4)

    if plot:
        plotter = Plotter(
            xlabel="epoch",
            xlim=[1, num_epochs],
            figsize=(10, 5),
            legend=["train loss", "train accuracy", "test loss", "test accuracy"],
        )

    for epoch in range(num_epochs + 1):
        model.train()
        total_loss = 0.0

        for user_ids, game_ids, y in tqdm(loader, position=0, leave=True):
            user_ids = user_ids.to(device)
            game_ids = game_ids.to(device)
            y = y.to(device)

            optimizer.zero_grad()

            out = model(user_ids, game_ids)

            if isinstance(loss_fn, WeightedLoss):
                loss = loss_fn(out, y)
            else:
                loss = loss_fn(out, y)
                # acc = accuracy(out.cpu(), y.cpu())

            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        # if plot and epoch % 10 == 0:
        # if plot:
        with torch.no_grad():
            out, y = evaluate_model(model, test_loader, device=device)
            test_loss = loss_fn(out, y)
            # test_acc = accuracy(out, y)
            # plotter.add(epoch + 1, (loss.item(), acc, test_loss.item(), test_acc))

        print(
            f"Epoch {epoch+1}/{num_epochs}, Loss: {total_loss / len(loader):.2f}, Val Loss: {test_loss:.2f}"
        )


# %%
print("Training Baseline Model...")
train_model(
    baseline_model,
    train_loader,
    lr=0.001,
    num_epochs=10,
    device=device,
)


# %%
def evaluate_model(model, loader, device="cpu"):
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


baseline_predictions, baseline_labels = evaluate_model(
    baseline_model, test_loader, device=device
)

# %%
metrics = MetricCollection(
    {
        "accuracy": Accuracy(task="binary"),
        "precision": Precision(task="binary"),
        "recall": Recall(task="binary"),
        # "ndcg": RetrievalNormalizedDCG(top_k=10)
    }
)


def compute_metrics(predictions, labels):
    predictions = predictions.view(-1)
    labels = labels.view(-1)

    result = metrics(predictions, labels)
    for name, value in result.items():
        print(f"{name}: {value:.4f}")


print("Baseline Model Metrics:")
compute_metrics(baseline_predictions, baseline_labels)

# print("Weighted Loss Model Metrics:")
# compute_metrics(weighted_predictions, weighted_labels)

# %%
pred_arr = baseline_predictions.numpy()
rev_user_id_map = {value: key for key, value in user_id_map.items()}
rev_game_id_map = {value: key for key, value in game_id_map.items()}

# %%
with pl.Config(tbl_rows=1000):
    result_df = test_data.with_columns(
        pl.Series("pred", baseline_predictions.numpy()),
    )
    print(result_df.sort("pred", descending=True))
    test_id = 709735
    print(
        train_data.with_columns(
            [
                pl.col("user_id").replace(rev_user_id_map).alias("user_id"),
                pl.col("app_id").replace(rev_game_id_map).alias("app_id"),
            ]
        ).filter(pl.col("user_id") == rev_user_id_map[test_id])
    )
    result_df = result_df.with_columns(
        [
            pl.col("user_id").replace(rev_user_id_map).alias("user_id"),
            pl.col("app_id").replace(rev_game_id_map).alias("app_id"),
        ]
    )
    result_df = result_df.filter(pl.col("user_id") == rev_user_id_map[test_id])
    print(result_df.sort("pred", descending=True))
    # result_df.with_columns(pl.when(pl.col("pred") > 0.5).then(1).otherwise(0).alias("pred"))
