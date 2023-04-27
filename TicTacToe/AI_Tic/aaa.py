import random
import numpy as np
import torch

class TicTacToe:
    def __init__(self, n=3):
        self.n = n
        self.board = [[' ' for _ in range(n)] for _ in range(n)]
        self.current_player = 'X'
        self.winner = None
        
    def make_move(self, row, col, player):
        if self.board[row][col] == ' ':
            self.board[row][col] = player
            self.current_player = 'O' if player == 'X' else 'X'
            self.check_winner()
            return True
        return False
    
    def get_valid_moves(self):
        moves = []
        for i in range(self.n):
            for j in range(self.n):
                if self.board[i][j] == ' ':
                    moves.append((i, j))
        return moves
    
    def check_winner(self):
        for i in range(self.n):
            # check rows
            if self.board[i][0] != ' ' and all(self.board[i][j] == self.board[i][0] for j in range(self.n)):
                self.winner = self.board[i][0]
                return self.winner
            
            # check columns
            if self.board[0][i] != ' ' and all(self.board[j][i] == self.board[0][i] for j in range(self.n)):
                self.winner = self.board[0][i]
                return self.winner
            
        # check diagonals
        if self.board[0][0] != ' ' and all(self.board[i][i] == self.board[0][0] for i in range(self.n)):
            self.winner = self.board[0][0]
            return self.winner
        
        if self.board[0][self.n-1] != ' ' and all(self.board[i][self.n-1-i] == self.board[0][self.n-1] for i in range(self.n)):
            self.winner = self.board[0][self.n-1]
            return self.winner
        
        # check if board is full
        if all(self.board[i][j] != ' ' for i in range(self.n) for j in range(self.n)):
            self.winner = ' '
            return self.winner
        
        return None
    
    def get_state(self):
        state = np.zeros((3, self.n, self.n))
        for i in range(self.n):
            for j in range(self.n):
                if self.board[i][j] == 'X':
                    state[0, i, j] = 1
                elif self.board[i][j] == 'O':
                    state[1, i, j] = 1
        if self.current_player == 'O':
            state[2] = np.ones((self.n, self.n))
        return torch.tensor(state, dtype=torch.float32)


class QLearningPlayer:
    def __init__(self, epsilon=0.1, learning_rate=0.1, discount_factor=0.9):
        self.q_values = {}
        self.epsilon = epsilon
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor

    def get_q_value(self, state, action):
        if (state, action) not in self.q_values:
            self.q_values[(state, action)] = 0.0
        return self.q_values[(state, action)]

    def update_q_value(self, state, action, next_state, reward):
        max_q = max([self.get_q_value(next_state, (i, j)) for (i, j) in next_state.get_valid_moves()])
        current_q = self.get_q_value(state, action)
        td_target = reward + self.discount_factor * max_q
        td_error = td_target - current_q
        self.q_values[(state, action)] += self.learning_rate * td_error

    def get_move(self, state):
        if np.random.uniform() < self.epsilon:
            return state.get_random_move()
        else:
            q_values = [self.get_q_value(state, (i, j)) for (i, j) in state.get_valid_moves()]
            max_q = max(q_values)
            if q_values.count(max_q) > 1:
                best_moves = [(i, j) for (i, j) in state.get_valid_moves() if self.get_q_value(state, (i, j)) == max_q]
                return best_moves[np.random.randint(len(best_moves))]
            else:
                index = q_values.index(max_q)
                return state.get_valid_moves()[index]

def main():
    n = 3
    num_episodes = 1000
    player = QLearningPlayer()
    for episode in range(num_episodes):
        game = TicTacToe(n)
        while not game.is_over():
            state = game.get_state()
            row, col = player.get_move(state)
            game.make_move(row, col, 'X')
            next_state = game.get_state()
            reward = game.get_reward('X')
            if not game.is_over():
                row, col = game.get_random_move('O')
                game.make_move(row, col, 'O')
                next_state = game.get_state()
                reward = game.get_reward('X')
            player.update_q_value(state, (row, col), next_state, reward)
        if (episode + 1) % 100 == 0:
            print("Episode {} done".format(episode + 1))

if __name__ == '__main__':
    main()
