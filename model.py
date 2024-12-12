import torch



class TwoTowerModel(torch.nn.Module):
    def __init__(self, num_users, num_games, embedding_dim=128):
        super(TwoTowerModel, self).__init__()
        self.user_embedding = torch.nn.Embedding(num_users, embedding_dim)
        self.game_embedding = torch.nn.Embedding(num_games, embedding_dim)

    def forward(self, user_ids, game_ids):
        user_embeds = self.user_embedding(user_ids)
        game_embeds = self.game_embedding(game_ids)
        scores = torch.sum(torch.mul(user_embeds, game_embeds), dim=1)
        return torch.sigmoid(scores)
