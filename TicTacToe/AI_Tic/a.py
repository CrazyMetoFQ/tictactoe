import torch
import random

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
        state = []
        for row in self.board:
            for col in row:
                if col == ' ':
                    state.append(0)
                elif col == 'X':
                    state.append(1)
                else:
                    state.append(2)
        return torch.tensor(state)

    def make_move(self, row, col):
        self.board[row][col] = self.player
        self.player = 'O' if self.player == 'X' else 'X'

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
    def __init__(self, n=3, alpha=0.1, gamma=0.9, eps=0.1):
        self.n = n
        self.alpha = alpha
        self.gamma = gamma
        self.eps = eps
        self.q_table = {}

    def get_q_value(self, state, action):
        if (state, action) not in self.q_table:
            self.q_table[(state, action)] = 0.0
        return self.q_table[(state, action)]

    def update_q_value(self, state, action, next_state, reward):
        old_q = self.get_q_value(state, action)
        next_max_q = max([self.get_q_value(next_state, (i, j)) for (i, j) in next_state.get_valid_moves()])
        new_q = old_q + self.alpha * (reward + self.gamma * next_max_q - old_q)
        self.q_table[(state, action)] = new_q

    def choose_action(self, state, valid_moves):
        if random.random() < self.eps:
            return random.choice(valid_moves)
        else:
            q_values = [self.get_q_value(state, move) for move in valid_moves]
            max_q = max(q_values)
            if q_values.count(max_q) > 1:
                best_moves = [move for move in valid_moves if self.get_q_value(state, move) == max_q]
                return random.choice(best_moves)
            else:
                return valid_moves[q_values.index(max_q)]

# Define the main function to train and run the game
def main(n=3, alpha=0.1, gamma=0.9, eps=0.1, num_episodes=10000):
    # Initialize the game and the
    game = TicTacToe(n=n)
    player = QLearningPlayer(n=n, alpha=alpha, gamma=gamma, eps=eps)

    # Train the AI player
    for i in range(num_episodes):
        state = game.get_state()
        while not game.is_game_over():
            valid_moves = game.get_valid_moves()
            action = player.choose_action(state, valid_moves)
            row, col = action
            game.make_move(row, col)
            next_state = game.get_state()
            if game.is_game_over():
                reward = 1 if game.player == 'O' else -1
            else:
                reward = 0
            player.update_q_value(state, action, next_state, reward)
            state = next_state

        game = TicTacToe(n=n)

    # Play the game against the AI player
    game.print_board()
    while not game.is_game_over():
        if game.player == 'X':
            row = int(input('Enter row: '))
            col = int(input('Enter column: '))
            game.make_move(row, col)
        else:
            state = game.get_state()
            valid_moves = game.get_valid_moves()
            action = player.choose_action(state, valid_moves)
            row, col = action
            game.make_move(row, col)

        game.print_board()

    # Print the winner of the game
    if game.player == 'X':
        print('You win!')
    elif game.player == 'O':
        print('AI wins!')
    else:
        print('Tie game.')

main()