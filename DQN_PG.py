import torch
import torch.nn as nn

class Policy(nn.Module):
    def __init__(self, latent_dim, num_actions):
        super(Policy, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(latent_dim * 2, 512), nn.ReLU(),
            nn.Linear(512, 1024), nn.ReLU(),
            nn.Linear(1024, num_actions),
        )

    def forward(self, z, goal, legal_actions_mask):
        x = torch.cat([z, goal], dim=-1)
        print("x", x)
        action_scores = self.model(x)
        # legal_actions_mask=legal_actions_mask.to(torch.float32)
        # action_scores_masked = action_scores - 1e30*legal_actions_mask
        # action_probs = torch.nn.functional.softmax(action_scores_masked, dim=-1)
        primeval_action_probs = torch.nn.functional.softmax(action_scores, dim=-1)
        primeval_action_probs = primeval_action_probs * (1 - legal_actions_mask)
        action_probs = primeval_action_probs / primeval_action_probs.sum(dim=-1, keepdim=True)
        return action_probs
    
class QNetwork(nn.Module):
    def __init__(self, latent_dim, num_actions):
        super(QNetwork, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(latent_dim * 2, 512), nn.ReLU(),
            nn.Linear(512, 1024), nn.ReLU(),
            nn.Linear(1024, num_actions)
        )
    
    def forward(self, z, goal):
        x = torch.cat([z, goal], dim=-1)
        return self.model(x)