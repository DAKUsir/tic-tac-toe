import numpy as np
from itertools import product
WIN_PATTERNS = [
    [0,1,2], [3,4,5], [6,7,8],
    [0,3,6], [1,4,7], [2,5,8],
    [0,4,8], [2,4,6]
]

def check_winner(board):
    for p in WIN_PATTERNS:
        s = board[p[0]] + board[p[1]] + board[p[2]]
        if s == 3: return 1
        if s == -3: return -1
    return 0 if 0 not in board else None

def minimax(board, player):
    res = check_winner(board)
    if res is not None:
        return res * player  # perspective of current player

    best = -2
    for i in range(9):
        if board[i] == 0:
            board[i] = player
            score = -minimax(board, -player)
            board[i] = 0
            if score > best:
                best = score
    return best

def best_move(board, player):
    best_score = -2
    move = None
    for i in range(9):
        if board[i] == 0:
            board[i] = player
            score = -minimax(board, -player)
            board[i] = 0
            if score > best_score:
                best_score = score
                move = i
    return move

def valid_states_with_turn():
    states = []
    for pos in product([1,-1,0], repeat=9):
        x_count = pos.count(1)
        o_count = pos.count(-1)
        if abs(x_count - o_count) > 1:
            continue
        if check_winner(list(pos)) is not None:
            continue
        player = 1 if x_count == o_count else -1
        states.append((list(pos), player))
    return states

states = valid_states_with_turn()


# Dataset generation
X, Y = [], []
for s, player in states:
    mv = best_move(s[:], player)
    if mv is None:
        continue
    X.append(s)
    y = np.zeros(9)
    y[mv] = 1
    Y.append(y)

X = np.array(X, dtype=np.float32)
Y = np.array(Y, dtype=np.float32)

print("Dataset size:", len(X))