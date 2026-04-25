# Goal Generation & Reinforcement Learning

This project explores goal-conditioned reinforcement learning using the Reinforcement Learning with Imagined Goals (RIG) approach, proposed by Nair et al. in 2018 (https://arxiv.org/abs/1807.04742), originally used for robotic tasks.

It focuses on learning latent representations of observations and training an agent to achieve diverse goals by sampling targets in a learned latent space.

Please refer to the PDF in the current directory for more detailed information about this project, and for running scrits, apply first the few modifications specified in the file modifs_gym_chess.


## Agent Task

The objective of the chosen study is to generate positions of the chess game and to learn to the agent to achieve the requested configuration. The considered chess positions are those King and white Rook versus Black King, as well as King versus King.

The objective is therefore to learn a latent representation of these positions using a generative model, then to learn to the agent (the Whites) to achieve these specific positions.

A major constraint quickly appeared concerning the use of RIG (or any other method) for the realization by an agent of chess positions. In the paper by Ashvin Nair and Vitchyr Pong, the algorithm is used on robotic tasks, for example moving with a robotic arm all kinds of different objects to certain positions. Contrary to the original application framework of RIG, the agent in the chess environment cannot force a specific position on the chessboard.

From this observation, which applies regardless of the considered chess environment, I decided to drastically restrict the positions of the black king and to guide its actions. I thus considered three problems, leaving each more or less freedom to the Black King in the choice of its moves.

### First task and associated environment

The Black King starts the episode at h1 and only performs actions that bring it closer (except when it finds itself there, being unable to pass over the Rook). The objective for the agent is to move its pieces in such a way as to reach the position White King at h8, White Rook at a8, and Black King at h1.

The idea is that the Black King remains overall in the same area of the chessboard regardless of the actions taken by the agent, and sufficiently far from the agent’s pieces so that the problem can be reduced to a simple positioning of pieces by the agent without considering external influences from the latter.



