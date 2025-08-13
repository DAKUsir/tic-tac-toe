import numpy as np
import random
from Dataset_and_HelperFunction import WIN_PATTERNS, check_winner, minimax, best_move, valid_states_with_turn, states, X, Y
from Numpy_Neural_net import init_net, forward, backward

class TicTacToeNet:
    def __init__(self, dataset_size=None):
        self.X = X[:dataset_size] if dataset_size else X
        self.Y = Y[:dataset_size] if dataset_size else Y
        self.net = init_net()
        self.loss = None

    def train(self, epochs=500, batch_size=128, lr=0.01):
        for e in range(epochs):
            idx = np.random.permutation(len(self.X))
            for i in range(0, len(self.X), batch_size):
                xi = self.X[idx[i:i+batch_size]]
                yi = self.Y[idx[i:i+batch_size]]
                a1, a2, out = forward(self.net, xi)
                backward(self.net, xi, yi, a1, a2, out, lr=lr)
            if (e+1) % 50 == 0:
                _, _, o = forward(self.net, self.X)
                self.loss = -np.mean((self.Y * np.log(o + 1e-9)).sum(axis=1))
        return self.loss

    def predict(self, b):
        _, _, o = forward(self.net, b.reshape(1, -1))
        move = np.argmax(o)
        return move

    def get_loss(self):
        return self.loss

    def get_net(self):
        return self.net