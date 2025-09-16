Effectuer deux modifications dans gym_chess en plus de remplacer les np.int par np.int32 dans gym_chess / BoardEncoding :

1. modifier l'init de gym_chess pour l'ajout de l'environnement d'intérêt "CustomChess-v0" :

# -*- coding: utf-8 -*-

"""OpenAI Gym environments for the game of chess.

This package provides two environments for the game of chess: 

 - a basic `Chess-v0` environment, which encodes observations and actions as
   objects of type `chess.Board` and `chess.Move` objects, respectivly 

 - a `ChessAlphaZero-v0' environment, which uses the board and move encoding
   proposed in [Silver et al., 2017]. 

Example:
    >>> import gym
    >>> import gym_chess

    >>> env = gym.make('Chess-v0')
    >>> env = gym.make('ChessAlphaZero-v0')

"""

from gym.envs.registration import register

from gym_chess.envs import Chess
from gym_chess.alphazero import BoardEncoding, MoveEncoding


# def _make_env(encode=False):
#     env = Chess()

#     if encode:
#         env = BoardEncoding(env)
#         env = MoveEncoding(env)

#     return env

def _make_env(encode_board=False, encode_move=False):
    env = Chess()

    if encode_board:
        env = BoardEncoding(env)
    if encode_move:
        env = MoveEncoding(env)

    return env


register(
    id='Chess-v0',
    entry_point='gym_chess:_make_env',
)


register(
    id='ChessAlphaZero-v0',
    entry_point='gym_chess:_make_env',
    kwargs={ 'encode_board': True, 'encode_move': True },
)

register(
    id='CustomChess-v0',
    entry_point='gym_chess:_make_env',
    kwargs={ 'encode_move': True },
)

2. modifier la définition de reset dans gym_chess / Chess pour l'ajout de la fonctionnalité génération d'un board initial sur une position fen spécifiée:

# def reset(self) -> chess.Board:

    #     self._board = chess.Board()
    #     self._ready = True

    #     return self._observation()

    def reset(self, fen=None) -> chess.Board:

        self._board = chess.Board(fen)
        self._ready = True

        return self._observation()