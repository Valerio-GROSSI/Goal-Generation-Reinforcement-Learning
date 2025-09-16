import gym # 0.17.3                        # conda create --name chess python=3.9 # conda activate chess # pip install gym
import gym_chess # 0.1.1                   # pip install numpy==1.19.5 # pip install gym_chess # remplacer les np.int par np.int32 dans gym_chess
import chess # 1.11.2                      # pip install chess
import torch                               # pip install torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random                         
from collections import deque
import os
import chess.svg
import cairosvg # 2.7.1                    # pip install cairosvg
import imageio # 2.37.0                    # pip install imageio # pip install imageio[ffmpeg]
import shutil
# Effectuer aussi deux modifications dans gym_chess (voir README)

from def_module import generate_finale_position, generate_valid_initial_position_BlackKingInh1, board_to_tensor, generate_legal_actions_mask, decoder_action, select_action_BlackKingInh1
from VAE_cnam import VAE
from DQN_PG import Policy

def create_directory(episode):
    frames_dir = f"chess_frames_{episode:04d}"
    if not os.path.exists(frames_dir):
        os.makedirs(frames_dir)
    frame_count = 0
    return frame_count, frames_dir
    
def add_frame(env, frames_dir, frame_count):
    board = env.unwrapped._board
    svg_code = chess.svg.board(board)
    frame_filename = os.path.join(frames_dir, f"frame_{frame_count:04d}.png")
    cairosvg.svg2png(bytestring=svg_code.encode('utf-8'), write_to=frame_filename)
    frame_count += 1
    return frame_count

def save_video(frames_dir, frame_count, episode):
    video_filename = f"episode{episode:01d}_probleme_0.mp4"
    fps = 6
    with imageio.get_writer(video_filename, fps=fps) as writer:
        for i in range(frame_count):
            frame_filename = os.path.join(frames_dir, f"frame_{i:04d}.png")
            image = imageio.imread(frame_filename)
            writer.append_data(image)
    print(f"Vidéo sauvegardée sous {video_filename}")

env = gym.make("CustomChess-v0")
num_channels = 3
num_actions = 2048
latent_dim = 65

vae = VAE(num_channels, latent_dim, beta=None)
policy = Policy(latent_dim, num_actions)

vae_path = "vae_0.pth"
policy_path = "policy_0.pth"

if os.path.exists(vae_path) and os.path.exists(policy_path):
    vae.load_state_dict(torch.load(vae_path))
    policy.load_state_dict(torch.load(policy_path))
    print("Paramètres du modèle VAE et du modèle Policy chargés depuis", vae_path, " et", policy_path)
else:
    print("Executer l'entrainement avant le test")

vae.eval()
policy.eval()

# Prenons comme objectif d'emmener le Roi blanc, sa Tour et le roi Noir dans une position précise. Ceci est rendu possible
# car on force le roi noir a rester proche de sa case objectif
fen_position = generate_finale_position()
position = board_to_tensor(chess.Board(fen_position))
goal, _  = vae.encoder(position)

num_episodes=2
for episode in range(num_episodes):
    frame_count, frames_dir = create_directory(episode)

    board = generate_valid_initial_position_BlackKingInh1()
    state_board = env.reset(fen=board.fen())
    state = board_to_tensor(state_board)
    done = False
    black_action_sens = None


    frame_count = add_frame(env, frames_dir, frame_count)

    while not done:
        z, _  = vae.encoder(state)
        legal_actions_mask = generate_legal_actions_mask(env, num_actions)
        action_probs = policy(z, goal, legal_actions_mask) 
        dist = torch.distributions.Categorical(action_probs)
        action = dist.sample()
        next_state, _, done, _ = env.step(decoder_action(action.item()))

        frame_count = add_frame(env, frames_dir, frame_count)

        if done == False:
            #action_random = random.choice(env.legal_actions)            # Coup des noirs
            black_action, black_action_sens = select_action_BlackKingInh1(next_state, env, black_action_sens)
            if black_action != None:
                next_state, _, done, _ = env.step(black_action)

                frame_count = add_frame(env, frames_dir, frame_count)
            else:
                done = True

        next_state = board_to_tensor(next_state)
        state = next_state
    env.close()

    save_video(frames_dir, frame_count, episode)
    shutil.rmtree(frames_dir)

print("Test terminé et vidéos enregistrées")