import gym
import gym_chess
import chess
import numpy as np
import random
import torch

# Obtention de l'état au format (3, 8, 8) à partir de l'état encodé sous forme chess.Board
def board_to_matrix(board):
    matrix = np.zeros((3,8,8), dtype=int)
    for square in chess.SQUARES: #range(0, 64)
        piece = board.piece_type_at(square)
        piece_color = board.color_at(square)
        if piece==6 and piece_color==0:
            row, col = divmod(square, 8)
            matrix[0, 7-row, col] = 1 # affichage standard vu de haut / canal présence roi noir
        elif piece==6 and piece_color==1:
            row, col = divmod(square, 8)
            matrix[1, 7-row, col] = 1 # affichage standard vu de haut / canal présence roi blanc
        elif piece==4 and piece_color==1:
            row, col = divmod(square, 8)
            matrix[2, 7-row, col] = 1 # affichage standard vu de haut / canal présence tour blanche
    return matrix

def board_to_tensor(board):
    matrix = board_to_matrix(board)
    state = torch.tensor(matrix, dtype=torch.float32).unsqueeze(0)
    return state

# Obtention des états encodés sous forme fen de toutes les positions possibles de mat pour les blancs dans la configuration roi+tour contre roi
def generate_mate_positions_king_rook_vs_king():
    # positions = set()
    positions =[]
    board = chess.Board('8/8/8/8/8/8/8/8 w HAha - 0 1') # échiquier vide pour remplissage

    for square in chess.SQUARES:
        row, col = divmod(square, 8)
        if row == 0:
            board._set_piece_at(square, 6, 0)
            board._set_piece_at(square+16, 6, 1)
            for _ in range(8):
                if _ <= (square - 2) or _ >= (square + 2):
                    board._set_piece_at(_, 4, 1)
                    # positions.add(board.fen())
                    positions.append(board.fen())
                    board._remove_piece_at(_)
            board._remove_piece_at(square)
            board._remove_piece_at(square+16)

        if row == 7:
            board._set_piece_at(square, 6, 0)
            board._set_piece_at(square-16, 6, 1)
            for _ in range(56,64):
                if _ <= (square - 2) or _ >= (square + 2):
                    board._set_piece_at(_, 4, 1)
                    # positions.add(board.fen())
                    positions.append(board.fen())
                    board._remove_piece_at(_)
            board._remove_piece_at(square)
            board._remove_piece_at(square-16)
        
        if col == 0:
            board._set_piece_at(square, 6, 0)
            board._set_piece_at(square+2, 6, 1)
            for _ in range(8):
                if _ <= (row - 2) or _ >= (row + 2):
                    board._set_piece_at(8*_, 4, 1)
                    # positions.add(board.fen())
                    positions.append(board.fen())
                    board._remove_piece_at(8*_)
            board._remove_piece_at(square)
            board._remove_piece_at(square+2)

        if col == 7:
            board._set_piece_at(square, 6, 0)
            board._set_piece_at(square-2, 6, 1)
            for _ in range(8):
                if _ <= (row - 2) or _ >= (row + 2):
                    board._set_piece_at(8*_+col, 4, 1)
                    # positions.add(board.fen())
                    positions.append(board.fen())
                    board._remove_piece_at(8*_+col)
            board._remove_piece_at(square)
            board._remove_piece_at(square-2)

    return positions

# Obtention de l'état encodé sous forme fen de la position de mat souhaitée lors du test
def generate_mate_position_king_rook_vs_king_Forced():
    board = chess.Board('8/8/8/8/8/8/8/8 w HAha - 0 1') # échiquier vide pour remplissage

    board._set_piece_at(60, 6, 0)
    board._set_piece_at(44, 6, 1)
    board._set_piece_at(56, 4, 1)
    return board.fen()

# Obtention de l'état encodé sous forme fen de la position finale souhaitée lors du test
def generate_finale_position():
    board = chess.Board('8/8/8/8/8/8/8/8 w HAha - 0 1') # échiquier vide pour remplissage

    board._set_piece_at(7, 6, 0)
    board._set_piece_at(63, 6, 1)
    board._set_piece_at(56, 4, 1)
    return board.fen()

# Obtention d'un état (forme chess.Board) de l'échiquier valide comprenant uniquement roi + tour blanche et roi noir, au tour des blancs
def generate_valid_initial_position():
    board = generate_initial_position()
    while board.is_check():
        board = generate_initial_position()
    board.turn = chess.WHITE
    return board

def generate_initial_position():
    board = chess.Board('8/8/8/8/8/8/8/8 b HAha - 0 1') # échiquier vide pour remplissage
     
    square = random.randint(0, 63)
    board._set_piece_at(square, 6, 0)

    square = random.randint(0, 63)
    while board.piece_at(square) is not None: 
        square = random.randint(0, 63)
    board._set_piece_at(square, 6, 1)

    square = random.randint(0, 63)
    while board.piece_at(square) is not None: 
        square = random.randint(0, 63)
    board._set_piece_at(square, 4, 1)

    return board

# Obtention d'un état (forme chess.Board) de l'échiquier valide comprenant uniquement roi + tour blanche et roi noir, au tour des blancs, 
# contrainte Roi Noir situé sur l'une des deux dernières rangées
def generate_valid_initial_position_BlackKingForced():
    board = generate_valid_initial_position()
    black_king_square = board.king(chess.BLACK)
    while black_king_square < 56:
        board = generate_valid_initial_position()
        black_king_square = board.king(chess.BLACK)
    return board

# Obtention d'un état (forme chess.Board) de l'échiquier valide comprenant uniquement roi + tour blanche et roi noir, au tour des blancs, 
# contrainte Roi Noir situé en h1
def generate_valid_initial_position_BlackKingInh1():
    board = generate_valid_initial_position()
    black_king_square = board.king(chess.BLACK)
    while black_king_square != 7:
        board = generate_valid_initial_position()
        black_king_square = board.king(chess.BLACK)
    return board

def generate_valid_matrix_initial_positions(state_collection_size):
    positions = []
    for _ in range(state_collection_size):
        board = generate_valid_initial_position()
        matrix = board_to_matrix(board)
        positions.append(matrix)
    return positions

# en tenseur torch
def generate_valid_initial_positions(state_collection_size):
    positions = generate_valid_matrix_initial_positions(state_collection_size)
    positions = torch.tensor(np.array(positions), dtype=torch.float32)
    return positions

# Obtention d'un état (forme chess.Board) de l'échiquier valide comprenant uniquement roi + tour blanche et roi noir, ou roi blanc et roi noir, au tour des blancs
def generate_valid_position():
    board = generate_position()
    while board.is_check():
        board = generate_position()
    board.turn = chess.WHITE
    return board

def generate_position():
    board = chess.Board('8/8/8/8/8/8/8/8 b HAha - 0 1') # échiquier vide pour remplissage
     
    square = random.randint(0, 63)
    board._set_piece_at(square, 6, 0)

    square = random.randint(0, 63)
    while board.piece_at(square) is not None: 
        square = random.randint(0, 63)
    board._set_piece_at(square, 6, 1)

    if random.random() < 0.6:
        square = random.randint(0, 63)
        while board.piece_at(square) is not None: 
            square = random.randint(0, 63)
        board._set_piece_at(square, 4, 1)

    return board

# Obtention d'un état (forme chess.Board) de l'échiquier valide comprenant uniquement roi + tour blanche et roi noir, ou roi blanc et roi noir, au tour des blancs, 
# contrainte Roi Noir situé sur l'une des deux dernières rangées
def generate_valid_position_BlackKingForced():
    board = generate_valid_position()
    black_king_square = board.king(chess.BLACK)
    while black_king_square < 48:
        board = generate_valid_position()
        black_king_square = board.king(chess.BLACK)
    return board

# Obtention d'un état (forme chess.Board) de l'échiquier valide comprenant uniquement roi + tour blanche et roi noir, ou roi blanc et roi noir, au tour des blancs, 
# contrainte Roi Noir situé en h1
def generate_valid_position_BlackKingInh1():
    board = generate_valid_position()
    black_king_square = board.king(chess.BLACK)
    while black_king_square != 7:
        board = generate_valid_position()
        black_king_square = board.king(chess.BLACK)
    return board

def generate_valid_matrix_positions(state_collection_size):
    positions = []
    for _ in range(state_collection_size):
        board = generate_valid_position()
        matrix = board_to_matrix(board)
        positions.append(matrix)
    return positions

# en tenseur torch
def generate_valid_positions(state_collection_size):
    positions = generate_valid_matrix_positions(state_collection_size)
    positions = torch.tensor(np.array(positions), dtype=torch.float32)
    return positions

def generate_valid_matrix_positions_BlackKingForced(state_collection_size):
    positions = []
    for _ in range(state_collection_size):
        board = generate_valid_position_BlackKingForced()
        matrix = board_to_matrix(board)
        positions.append(matrix)
    return positions

# en tenseur torch
def generate_valid_positions_BlackKingForced(state_collection_size):
    positions = generate_valid_matrix_positions_BlackKingForced(state_collection_size)
    positions = torch.tensor(np.array(positions), dtype=torch.float32)
    return positions

def generate_valid_matrix_positions_BlackKingInh1(state_collection_size):
    positions = []
    for _ in range(state_collection_size):
        board = generate_valid_position_BlackKingInh1()
        matrix = board_to_matrix(board)
        positions.append(matrix)
    return positions

# en tenseur torch
def generate_valid_positions_BlackKingInh1(state_collection_size):
    positions = generate_valid_matrix_positions_BlackKingInh1(state_collection_size)
    positions = torch.tensor(np.array(positions), dtype=torch.float32)
    return positions

# Obtention de l'action encodée suivant uniquement comme actions possibles sur l'échiquier celles d'un roi et d'une tour 
# (on réduit et redéfini l'espace des actions à partir de celui général utilisé par gym_chess pour son env ChessAlphaZero)
# On passe ainsi de 4672 actions à seulement 2048 pour notre problème
def encoder_action(action):
    q_action, r_action = divmod(action, 73)
    q_r_action, r_r_action = divmod(r_action, 7)
    if q_r_action > 7 or (q_r_action % 2 == 1 and r_r_action != 0):
        print("il y a un problème avec les actions légales disponibles")
        return None
    action_encode = q_action * 32 + r_action - 6 * (q_r_action//2)
    return action_encode

def decoder_action(action_encode):
    q_action_encode, r_action_encode = divmod(action_encode, 32)
    q_r_action_encode, r_r_action_encode = divmod(r_action_encode, 8)
    action = q_action_encode * 73 + q_r_action_encode * (8+6) + r_r_action_encode
    return action

# Obtention de l'ensemble des actions légales
def generate_legal_actions(env):
    legal_actions = env.legal_actions
    legal_actions_encode = []
    for idx, action in enumerate(legal_actions):
        action_encode = encoder_action(action)
        legal_actions_encode.append(action_encode)
    return legal_actions_encode

def mask(legal_actions, num_actions):
    if not isinstance(legal_actions, tuple):
        legal_actions = (legal_actions,)
    legal_actions_mask = torch.ones(len(legal_actions), num_actions)
    for idx, legal_actions in enumerate(legal_actions):
        legal_actions_tensor = torch.tensor(np.array(legal_actions), dtype=torch.int64)
        legal_actions_mask[idx, legal_actions_tensor] = 0
    return legal_actions_mask

def generate_legal_actions_mask(env, num_actions):
    legal_actions_encode = generate_legal_actions(env)
    legal_actions_mask = mask(legal_actions_encode, num_actions)
    return legal_actions_mask

def select_action_BlackKing(next_state, env, last_black_action_sens):
    black_king_square = next_state.king(chess.BLACK)
    action_right = np.int64((63 - black_king_square) * 73 - 1 + 43)
    action_left = np.int64((63 - black_king_square) * 73 - 1 + 15)
    action_up = np.int64((63 - black_king_square) * 73 - 1 + 29)
    action_down = np.int64((63 - black_king_square) * 73 - 1 + 1)
    action_down_right = np.int64((63 - black_king_square) * 73 - 1 + 50)
    action_down_left = np.int64((63 - black_king_square) * 73 - 1 + 8)

    if last_black_action_sens != "left":
        if action_up in env.legal_actions:    
            action = action_up
            action_sens = "up"
        elif action_right in env.legal_actions:
            action = action_right
            action_sens = "right"
        elif action_left in env.legal_actions:
            action = action_left
            action_sens = "left"
        elif action_down in env.legal_actions and 56 <= black_king_square <= 63:
            action = action_down
            action_sens = "down"
        elif action_down_right in env.legal_actions and 56 <= black_king_square <= 63:
            action = action_down_right
            action_sens = "down_right"
        elif action_down_left in env.legal_actions and 56 <= black_king_square <= 63:
            action = action_down_left
            action_sens = "down_left"
        else:
            action = None
            action_sens = None
    else:
        if action_up in env.legal_actions:    
            action = action_up
            action_sens = "up"
        elif action_left in env.legal_actions:
            action = action_left
            action_sens = "left"
        elif action_right in env.legal_actions:
            action = action_right
            action_sens = "right"
        elif action_down in env.legal_actions and 56 <= black_king_square <= 63:
            action = action_down
            action_sens = "down"
        elif action_down_left in env.legal_actions and 56 <= black_king_square <= 63:
            action = action_down_left
            action_sens = "down_left"
        elif action_down_right in env.legal_actions and 56 <= black_king_square <= 63:
            action = action_down_right
            action_sens = "down_right"
        else:
            action = None
            action_sens = None

    return action, action_sens

def select_action_BlackKingInh1(next_state, env, last_black_action_sens):
    black_king_square = next_state.king(chess.BLACK)
    action_right = np.int64((63 - black_king_square) * 73 - 1 + 43)
    action_left = np.int64((63 - black_king_square) * 73 - 1 + 15)
    action_up = np.int64((63 - black_king_square) * 73 - 1 + 29)
    action_down = np.int64((63 - black_king_square) * 73 - 1 + 1)
    action_down_right = np.int64((63 - black_king_square) * 73 - 1 + 50)
    action_down_left = np.int64((63 - black_king_square) * 73 - 1 + 8)
    action_up_right = np.int64((63 - black_king_square) * 73 - 1 + 36)
    action_up_left = np.int64((63 - black_king_square) * 73 - 1 + 22)

    if action_down_right in env.legal_actions:
            action = action_down_right
            action_sens = "down_right"
    elif action_right in env.legal_actions:
        action = action_right
        action_sens = "right"
    elif action_down in env.legal_actions:
        action = action_down
        action_sens = "down"
    elif action_left in env.legal_actions:
        action = action_left
        action_sens = "left"
    elif action_up in env.legal_actions:    
            action = action_up
            action_sens = "up"
    elif action_up_left in env.legal_actions:
        action = action_up_left
        action_sens = "up_left"
    else:
        action = None
        action_sens = None

    return action, action_sens

# fen_positions = generate_mate_positions_king_rook_vs_king()
# fen_position = fen_positions[0]
# mate_board = chess.Board(fen_position)
# print(mate_board)

# init_board = generate_valid_initial_position()
# print(init_board)
# matrix = board_to_matrix(init_board)
# print(matrix)
# init_matrix = generate_valid_matrix_initial_positions(state_collection_size=100)
# print(init_matrix[0])
# matrix = generate_valid_matrix_positions(state_collection_size=100)
# print(matrix[0])

# env = gym.make("CustomChess-v0")
# state = env.reset(fen=init_board.fen())
# print(state)
# print(env.legal_actions)
# action = env.legal_actions[0]
# print(action)
# action = encoder_action(action)
# print(action)
# action = decoder_action(action)
# print(action)
# action = encoder_action(action)
# print(action)

# print(env.legal_actions)
# legal_actions = generate_legal_actions(env)
# print(legal_actions)
# num_actions = 2048
# legal_actions_mask = generate_legal_actions_mask(env, num_actions)
# print(legal_actions_mask)

# init_board = generate_valid_initial_position_BlackKingForced()
# env = gym.make("CustomChess-v0")
# state = env.reset(fen=init_board.fen())
# next_state, _, _, _ = env.step(env.legal_actions[0])
# print(next_state)
# print(env.legal_actions)
# action = env.legal_actions[0]
# print(action)
# action = encoder_action(action)
# print(action)
# action = decoder_action(action)
# print(action)
# action = encoder_action(action)
# print(action)
# action_black = select_action_BlackKing(next_state, env)
# print(action_black)

# mat_board = generate_mate_position_king_rook_vs_king_Forced()
# print(mat_board)