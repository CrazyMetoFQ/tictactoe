import torch
import random
import numpy as np

# Define the Tic Tac Toe game
class TicTacToe:
    def __init__(self, n=3):
        self.n = n
        self.board = [[' ' for _ in range(n)] for _ in range(n)]
        self.player = 'X'

    def print_board(self):
        for row in self.board:
            print('|'.join(row))
        print('')

    def get_state(self):
        """
        Returns the current state of the game as a TicTacToe object.
        """
        state = TicTacToe(n=self.n)
        for i in range(self.n):
            for j in range(self.n):
                if self.board[i][j] == 'X':
                    state.make_move(i, j, 'X')
                elif self.board[i][j] == 'O':
                    state.make_move(i, j, 'O')
        state.player = self.player
        return state


    def make_move(self, row, col, player):
        """
        Makes a move on the board.
        """
        if self.board[row][col] == '-':
            self.board[row][col] = player
            self.player = 'O' if self.player == 'X' else 'X'
        else:
            print('Invalid move')


    def is_valid_move(self, row, col):
        return self.board[row][col] == ' '

    def get_valid_moves(self):
        moves = []
        for i in range(self.n):
            for j in range(self.n):
                if self.is_valid_move(i, j):
                    moves.append((i, j))
        return moves

    def is_game_over(self):
        for i in range(self.n):
            for j in range(self.n):
                if self.board[i][j] == ' ':
                    return False
        return True


# Define the AI player using Q-learning
class QLearningPlayer:
    def __init__(self, n, alpha=0.5, gamma=0.9, epsilon=0.1):
        self.n = n
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.q_values = {}
    
    def get_q_value(self, state, action):
        if (state, action) not in self.q_values:
            self.q_values[(state, action)] = 0.0
        return self.q_values[(state, action)]
    
    def update_q_value(self, state, action, next_state, reward):
        current_q = self.get_q_value(state, action)
        max_next_q = max([self.get_q_value(next_state, (i, j)) for (i, j) in next_state.get_valid_moves()])
        new_q = current_q + self.alpha * (reward + self.gamma * max_next_q - current_q)
        self.q_values[(state, action)] = new_q
    
    def get_move(self, state):
        if np.random.uniform(0, 1) < self.epsilon:
            # Choose a random move
            row, col = np.random.choice(range(self.n)), np.random.choice(range(self.n))
        else:
            # Choose the move with highest Q-value
            q_values = {move: self.get_q_value(state, move) for move in state.get_valid_moves()}
            max_q = max(q_values.values())
            best_moves = [move for move, q in q_values.items() if q == max_q]
            row, col = random.choice(best_moves)
        return row, col




# Define the main function to train and run the game
def main():
    n = 3
    epochs = 100000
    alpha = 0.8
    gamma = 0.95
    epsilon = 0.2
    player = QLearningPlayer(n, alpha=alpha, gamma=gamma, epsilon=epsilon)

    for i in range(epochs):
        game = TicTacToe(n=n)
        while not game.is_game_over():
            if game.player == 'X':
                row, col = player.get_move(game.get_state())
                game.make_move(row, col, 'X')
            else:
                row = int(input('Enter row: '))
                col = int(input('Enter column: '))
                game.make_move(row, col, 'O')
        reward = game.get_reward('X')
        next_state = game.get_state()
        player.update_q_value(game.get_state(), (row, col), next_state, reward)

    # Test the trained player
    game = TicTacToe(n=n)
    while not game.game_over():
        if game.player == 'X':
            row, col = player.get_move(game.get_state())
            game.make_move(row, col, 'X')
        else:
            row = int(input('Enter row: '))
            col = int(input('Enter column: '))
            game.make_move(row, col, 'O')
        print(game)
    print(game.get_winner())

if __name__ == '__main__':
    main()

