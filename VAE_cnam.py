import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
import random

class Encoder(nn.Module):
    def __init__(self, num_channels, latent_dim):
        super(Encoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(num_channels, 32, kernel_size=3, stride=1, padding=1), nn.ReLU(), nn.MaxPool2d(2, 2), # torch.Size([batch_size, 32, 4, 4])
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1), nn.ReLU(), nn.MaxPool2d(2, 2), # torch.Size([batch_size, 64, 2, 2])
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1), nn.ReLU(), nn.MaxPool2d(2, 2), # torch.Size([batch_size, 128, 1, 1])
            nn.Flatten() # torch.Size([batch_size, 128])
        )

        self.linear1 = nn.Linear(128, latent_dim-1)
        self.linear2 = nn.Linear(128, latent_dim-1)
        self.linearLastPlane = nn.Linear(128,1)

    def forward(self, x):
        x = self.encoder(x)
        x_mu = self.linear1(x)
        x_logvar = self.linear2(x)
        scoreLastPlane = torch.sigmoid(self.linearLastPlane(x))
        x_mu = torch.cat([x_mu, scoreLastPlane], dim=1)
        x_logvar = torch.cat([x_logvar, torch.zeros_like(scoreLastPlane)], dim=1)
        return x_mu, x_logvar
    
class Decoder(nn.Module):
    def __init__(self, num_channels, latent_dim):
        super(Decoder, self).__init__()
        self.linear = nn.Linear(latent_dim-1, 128)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1), nn.ReLU(), # torch.Size([batch_size, 64, 2, 2])
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1), nn.ReLU(), # torch.Size([batch_size, 32, 4, 4])
            nn.ConvTranspose2d(32, num_channels, kernel_size=3, stride=2, padding=1, output_padding=1), nn.ReLU(), # torch.Size([batch_size, num_channels, 8, 8])
        )
        
    def forward(self, z):
        gainLastPlane = torch.sigmoid(z[:,-1]).view(-1,1,1,1)
        z_main = z[:,:-1]
        hat_x = F.relu(self.linear(z_main))
        hat_x = hat_x.view(-1, 128, 1, 1)
        hat_x = self.decoder(hat_x)
        hat_x = F.softmax(hat_x.view(hat_x.shape[0], 3, -1), dim=-1).view(hat_x.shape)
        firstTwoPlanes = hat_x[:,:2,:,:]
        LastPlane = hat_x[:,2:,:,:]*gainLastPlane
        hat_x = torch.cat([firstTwoPlanes, LastPlane], dim=1)
        return hat_x
    
class VAE(nn.Module):
    def __init__(self, num_channels, latent_dim, beta):
        super(VAE, self).__init__()
        self.encoder = Encoder(num_channels, latent_dim)
        self.decoder = Decoder(num_channels, latent_dim)
        self.num_channels = num_channels
        self.beta = beta

    def forward(self, x):
        latent_mu, latent_logvar = self.encoder(x)
        z = self.latent_sample(latent_mu, latent_logvar)
        hat_x = self.decoder(z)
        return hat_x, latent_mu, latent_logvar, z

    def latent_sample(self, mu, logvar):
        std = logvar.mul(0.5).exp_()
        eps = torch.empty_like(std).normal_()
        return eps.mul(std).add_(mu)

    def loss(self, hat_x, x, latent_mu, latent_logvar):
        cce_loss = -torch.sum(x * torch.log(hat_x + 1e-9) + (1 - x) * torch.log(1 - hat_x + 1e-9))
        kl_div = -0.5 * torch.sum(1 + latent_logvar - latent_mu.pow(2) - latent_logvar.exp())
        return cce_loss + self.beta * kl_div

def train_and_update_vae(vae, vae_optimizer, positions, batch_size, num_epochs):
    train_dataset = data.TensorDataset(positions) #torch.utils.data.dataset.TensorDataset de tuples de torch.tensor individuels
    for epoch in tqdm(range(1, num_epochs + 1), desc="Entrainement du modèle VAE"):
        train_dataloader = data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True) #torch.utils.data.dataloader.DataLoader de (state_collection_size / batch_size) listes de torch.tensor (1ere dim batch_size)
        avg_loss = 0.
        i=0
        for list in tqdm(train_dataloader):
            i=i+1
            states = list[0]
            reconstructions, latent_mu, latent_logvar, _ = vae(states)
            vae_loss = vae.loss(reconstructions, states, latent_mu, latent_logvar)
            vae_optimizer.zero_grad()
            vae_loss.backward()
            vae_optimizer.step()
            avg_loss += vae_loss.item()
        
        avg_loss /= len(train_dataloader)
        tqdm.write(f"Epoch {epoch}: loss = {avg_loss:.3f}")

def visualize_reconstruction(vae, states):
        reconstructions, _, _, _ = vae(states)
        for i in range(len(states)):
            print(states[i, :], reconstructions[i, :])

def generate_fitted_prior(vae, states):
    latent_mu, _ = vae.encoder(states)
    fitted_prior_mu = latent_mu.mean(dim=0, keepdim=True)
    fitted_prior_std = latent_mu.std(dim=0, keepdim=True)
    return fitted_prior_mu, fitted_prior_std

def generate_covariance_matrix(vae, states):
    latent_mu, _ = vae.encoder(states)
    latent_mean = torch.mean(latent_mu, dim=0, keepdim=True)
    latent_centered = latent_mu - latent_mean
    covariance_matrix = (latent_centered.T @ latent_centered) / (latent_mu.shape[0] - 1)
    return covariance_matrix + 1e-5 * torch.eye(covariance_matrix.shape[0])

def sample_latent_goal(fitted_prior_mu, fitted_prior_std):
    with torch.no_grad():
        epsilon = torch.randn_like(fitted_prior_mu)
        z_sample = fitted_prior_mu + epsilon * fitted_prior_std
    return z_sample

def replace_goals(goals, fitted_prior_mu, fitted_prior_std, prob):
    probs = torch.rand(len(goals))
    
    for k in range(len(goals)):
        if probs[k] < prob:
            goal = sample_latent_goal(fitted_prior_mu, fitted_prior_std)
            goals[k, :] = goal.squeeze()
    return goals

def generate_state_collection_update(replay_buffer, exploration_states, vae_state_collection_size_update, mixture_coeff):
    transitions_buffer_for_update_vae = random.sample(replay_buffer, int(min(vae_state_collection_size_update, len(replay_buffer)) * mixture_coeff))
    states_buffer_for_update_vae = [t[0] for t in transitions_buffer_for_update_vae]
    states_buffer_for_update_vae = torch.cat(states_buffer_for_update_vae, dim=0)
    
    indices_exploration_states_for_update_vae = torch.randperm(exploration_states.size(0))[:int(min(vae_state_collection_size_update, len(replay_buffer)) * (1 - mixture_coeff))]
    exploration_states_for_update_vae = exploration_states[indices_exploration_states_for_update_vae]
    
    states_for_update = torch.cat((states_buffer_for_update_vae, exploration_states_for_update_vae), dim=0)
    return states_for_update

def mahalanobis_distance(z1, z2, covariance_matrix):
    diff = z1 - z2
    inv_cov_matrix = torch.inverse(covariance_matrix)
    dist = torch.sqrt(torch.matmul(torch.matmul(diff, inv_cov_matrix), diff.T))
    return dist