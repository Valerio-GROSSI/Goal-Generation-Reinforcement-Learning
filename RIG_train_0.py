import gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque

from def_module import generate_valid_positions_BlackKingInh1, generate_valid_initial_position_BlackKingInh1, board_to_tensor, generate_legal_actions_mask, decoder_action, select_action_BlackKingInh1
from VAE_cnam import VAE, train_and_update_vae, generate_fitted_prior, visualize_reconstruction, generate_covariance_matrix, sample_latent_goal, replace_goals, mahalanobis_distance, generate_state_collection_update
from DQN_PG import Policy, QNetwork

# Environement
env = gym.make("CustomChess-v0") # obs_env (avant encodage) = chess.Board & act_env = gym_chess.alphazero.MoveEncoding(chess.Move)
num_channels = 3 # 3 canaux pour présence sur l'échiquier roi noir, roi blanc et tour blanche
num_actions = 2048 # on réduit l'espace aux seules actions de notre problème (env.action_space.n = 4672 via encodage des actions propre à ChessAlphaZero-v0)
latent_dim = 65

# Hyperparamètres
vae_state_collection_size = 20000
vae_state_collection_size_update = 5000
mixture_coeff = 0.5
update_vae_every = 10
vae_num_epochs = 200
vae_batch_size = 100
vae_num_epochs_update = 100
vae_batch_size_update = 50
beta = 0.5
vae_learning_rate = 0.001
num_occurences_max = 10

num_episodes = 1000
max_steps = 75 # borne théorique vis à vis de env.legal_actions (en pratique on est borné par 50 avec la règle des 50 coups)
buffer_size = 10000
batch_size = 100
gamma = 0.99
update_target_every = 4 # Fréquence de mise à jour du réseau cible
policy_learning_rate = 0.001
q_network_learning_rate = 0.001
target_q_network_learning_rate = 0.001

vae = VAE(num_channels, latent_dim, beta)
policy = Policy(latent_dim, num_actions)
q_network = QNetwork(latent_dim, num_actions)
target_q_network = QNetwork(latent_dim, num_actions)
target_q_network.load_state_dict(q_network.state_dict())

vae_optimizer = optim.Adam(params=vae.parameters(), lr=vae_learning_rate)
policy_optimizer = optim.Adam(params=policy.parameters(), lr=policy_learning_rate)
q_optimizer = optim.Adam(params=q_network.parameters(), lr=q_network_learning_rate)
target_q_optimizer = optim.Adam(params=target_q_network.parameters(), lr=target_q_network_learning_rate)

# Reinforcement learning with imagined goals
exploration_states = generate_valid_positions_BlackKingInh1(vae_state_collection_size)          # Collect D using various positions (1.)
train_and_update_vae(vae, vae_optimizer, exploration_states, vae_batch_size, vae_num_epochs)      # Train Beta-VAE on D (2.)
fitted_prior_mu, fitted_prior_std = generate_fitted_prior(vae, exploration_states)                # Fit prior to latent encodings {mu(D)} (3.)
# visualize_reconstruction(vae, exploration_states[0:5,:])
covariance_matrix = generate_covariance_matrix(vae, exploration_states)                           # Generate covariance matrix (latent space) for Mahalanobis distance computation

replay_buffer = deque(maxlen=buffer_size)
for episode in range(num_episodes):
    goal = sample_latent_goal(fitted_prior_mu, fitted_prior_std)       # Sample latent goal from prior (5.)
    board = generate_valid_initial_position_BlackKingInh1()          # Sample initial state (6.)
    state_board = env.reset(fen=board.fen())
    state = board_to_tensor(state_board)
    black_action_sens = None
    
    for step in range(max_steps):
        with torch.no_grad():
            z, _  = vae.encoder(state)                                     # Get action (8.)
        legal_actions_mask = generate_legal_actions_mask(env, num_actions)
        torch.set_printoptions(threshold=float('inf'))
        action_probs = policy(z, goal, legal_actions_mask)
        dist = torch.distributions.Categorical(action_probs)
        action = dist.sample()
        next_state, _, done, _ = env.step(decoder_action(action.item())) # Get next_state (9.)
        if done == False:
            #action_random = random.choice(env.legal_actions)            # Coup des noirs
            black_action, black_action_sens = select_action_BlackKingInh1(next_state, env, black_action_sens)
            if black_action != None:
                next_state, _, done, _ = env.step(black_action)
            else:
                done = True
        next_legal_actions_mask = legal_actions_mask                     # Random value
        if done == False:
            next_legal_actions_mask = generate_legal_actions_mask(env, num_actions)
        next_state = board_to_tensor(next_state)
        
        replay_buffer.append((state, action, next_state, goal, legal_actions_mask, next_legal_actions_mask, done)) # Store transition into replay buffer (10.)
        
        if len(replay_buffer)>=batch_size:
            batch = random.sample(replay_buffer, batch_size)                        # Sample transitions from replay buffer (11.)
            states, actions, next_states, goals, legal_actions_mask_stored, next_legal_actions_mask_stored, dones = zip(*batch)
            states = torch.cat(states)
            actions = torch.cat(actions)
            next_states = torch.cat(next_states)
            goals = torch.cat(goals)
            legal_actions_mask_stored = torch.cat(legal_actions_mask_stored)
            next_legal_actions_mask_stored = torch.cat(next_legal_actions_mask_stored)
            dones = torch.tensor(np.array(dones), dtype=torch.float32)

            with torch.no_grad():            
                z, _ = vae.encoder(states)                                            # Encode z = e(s) & z' = e(s') (12.)
                next_z, _ = vae.encoder(next_states)
            goals = replace_goals(goals, fitted_prior_mu, fitted_prior_std, 0.5)      # Probability 0.5 replace goal (13.)
            rewards = mahalanobis_distance(next_z.detach(), goals, covariance_matrix) # Compute new reward (14.)
            q_values = q_network(z, goals).gather(1, actions.unsqueeze(1)).squeeze(1) # Minimize Bellman error (15.)
            with torch.no_grad():
                next_q_values = target_q_network(next_z, goals)
                next_legal_actions_mask_stored = next_legal_actions_mask_stored.bool()
                next_q_values[next_legal_actions_mask_stored] = float('-inf')
                next_q_values = next_q_values.max(1)[0]
            targets = rewards + (gamma * next_q_values * (1 - dones))

            q_loss = (q_values - targets.detach()).pow(2).mean()

            q_optimizer.zero_grad()
            q_loss.backward()
            q_optimizer.step()

            action_probs = policy(z, goals, legal_actions_mask_stored)
            policy_loss = -torch.mean(torch.log(action_probs.gather(1, actions.unsqueeze(1))) * q_values.unsqueeze(1).detach())

            policy_optimizer.zero_grad()
            policy_loss.backward(retain_graph=True)
            policy_optimizer.step()
        
        if episode % update_target_every == 0:
            target_q_network.load_state_dict(q_network.state_dict())
        
        if done:
            break
        state = next_state
    
    new_transitions = []
    total_steps = step + 1
    with torch.no_grad():
        for step in range(total_steps):
            transition = replay_buffer[-(total_steps-step)]
            num_occurences = min(num_occurences_max, total_steps - step)
            for _ in range(num_occurences):
                new_transition = list(transition)
                transition_for_future_state = random.sample(list(replay_buffer)[-(total_steps-step):], 1)   # Sample future state (19.)
                future_state = transition_for_future_state[0][2]
                z_future, _ = vae.encoder(future_state)
                new_transition[3] = z_future
                new_transitions.append(new_transition)
        for k in range(len(new_transitions)):
            replay_buffer.append(tuple(new_transitions[k]))                                                 # Store new transitions into replay buffer (20.)

    if episode % update_vae_every == 0 and len(replay_buffer) > vae_state_collection_size_update:
        states_for_update = generate_state_collection_update(replay_buffer, exploration_states, vae_state_collection_size_update, mixture_coeff)
        train_and_update_vae(vae, vae_optimizer, states_for_update, vae_batch_size_update, vae_num_epochs_update)     # Fine-tune Beta-VAE (23.)

torch.save(vae.state_dict(), "vae_0.pth")
torch.save(policy.state_dict(), "policy_0.pth")
print("Apprentissage du VAE et de la politique terminé et enregistré dans", "vae_0.pth", " et", "policy_0.pth")