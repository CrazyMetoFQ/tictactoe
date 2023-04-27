import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim


def create_board(n):
    """
    Creates an n x n board filled with zeros
    """
    return np.zeros((n, n), dtype=int)

def place_marker(board, row, col, player):
    """
    Places a player's marker on the board
    """
    board[row][col] = player

def get_next_player(player):
    """
    Returns the next player
    """
    if player == 1:
        return 2
    else:
        return 1

def is_winner(board, player):
    """
    Checks if the player has won
    """
    n = board.shape[0]

    # Check rows
    for i in range(n):
        if np.all(board[i] == player):
            return True

    # Check columns
    for i in range(n):
        if np.all(board[:,i] == player):
            return True

    # Check diagonals
    if np.all(board.diagonal() == player):
        return True

    if np.all(np.fliplr(board).diagonal() == player):
        return True

    return False

def is_tie(board):
    """
    Checks if the game is a tie
    """
    return np.all(board != 0)

def get_move(n):
    """
    Gets the player's move
    """
    while True:
        try:
            row = int(input(f"Enter the row (0 to {n-1}): "))
            col = int(input(f"Enter the column (0 to {n-1}): "))
            break
        except:
            print("Invalid input. Please try again.")

    return row, col

def print_board(board):
    """
    Prints the board
    """
    print(board)

def make_move(board, player, row, col):
    """
    Make a move on the board.

    Args:
        board (list): The current board.
        player (int): The player making the move.
        row (int): The row where the move will be made.
        col (int): The column where the move will be made.

    Returns:
        list: The updated board after making the move.
    """
    board[row][col] = player
    return board



class QNetwork(nn.Module):
    def __init__(self, state_size, action_size, hidden_size):
        super().__init__()
        self.fc1 = nn.Linear(state_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, action_size)

    def forward(self, state):
        state = torch.from_numpy(state).float()  # Convert NumPy array to PyTorch tensor
        x = torch.relu(self.fc1(state))
        q_values = self.fc2(x)
        return q_values


class DQNAgent:
    def __init__(self, state_size, action_size, hidden_size, alpha, gamma, epsilon):
        self.state_size = state_size
        self.action_size = action_size
        self.hidden_size = hidden_size
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.q_network = QNetwork(state_size, action_size, hidden_size)
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=alpha)
        self.loss_fn = nn.MSELoss()

    def get_action(self, state):
        if random.uniform(0, 1) < self.epsilon:
            action = random.randint(0, self.action_size-1)
        else:
            q_values = self.q_network(state)
            action = torch.argmax(q_values).item()
        return action

    def train(self, state, action, reward, next_state, done):
        # Convert state and next_state to PyTorch tensors
        state_tensor = torch.from_numpy(state).float().unsqueeze(0)
        next_state_tensor = torch.from_numpy(next_state).float().unsqueeze(0)

        # Get the Q-value estimates for the current state and next state
        q_values = self.q_network(state_tensor)
        next_q_values = self.q_network(next_state_tensor)

        # Get the Q-value estimate for the action taken in the current state
        action = torch.tensor([action])
        q_value = q_values[0][action]

        # Compute the target Q-value using the Bellman equation
        next_q_value, _ = torch.max(next_q_values, dim=1)
        target_q_value = reward + self.gamma * next_q_value * (1 - done)

        # Compute the loss and update the Q-network
        loss = F.smooth_l1_loss(q_value, target_q_value.detach())
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()


def is_game_over(board):
    # Check if any row is filled with the same player marker
    for row in board:
        if len(set(row)) == 1 and row[0] != 0:
            return True
    # Check if any column is filled with the same player marker
    for col in range(board.shape[1]):
        if len(set(board[:,col])) == 1 and board[0,col] != 0:
            return True
    # Check if any diagonal is filled with the same player marker
    if len(set(board.diagonal())) == 1 and board[0,0] != 0:
        return True
    if len(set(np.fliplr(board).diagonal())) == 1 and board[0,board.shape[1]-1] != 0:
        return True
    # Check if board is full
    if np.all(board != 0):
        return True
    return False


def get_state(board, player):
    state = board.copy()
    state[state == player] = 1
    state[state != player] = 0
    return state.reshape(-1)

def get_reward(board, player):
    if is_winner(board, player):
        return 1.0
    elif is_winner(board, get_next_player(player)):
        return -1.0
    else:
        return 0.0

def play_game(agent, n):
    board = create_board(n)
    current_player = 1
    done = False
    
    while not done:
        state = get_state(board, current_player)
        action = agent.get_action(state)
        row, col = divmod(action, n)  # convert action to (row, col) tuple
        reward = get_reward(board, current_player)
        next_board = make_move(board, current_player, row, col)
        next_state = get_state(next_board, -current_player)
        done = is_game_over(next_board)
        agent.train(state, action, reward, next_state, done)
        board = next_board
        current_player = -current_player

    final_reward = get_reward(board, current_player)
    agent.train(next_state, None, final_reward, None, True)



def train_agent(agent, episodes, n):
    for episode in range(episodes):
        print(f"Episode {episode+1}/{episodes}")
        play_game(agent, n)
        if agent.epsilon > 0.01:
            agent.epsilon *= 0.99






n = 3
state_size = n*n
action_size = n*n
hidden_size = 128
alpha = 0.001
gamma = 0.99
epsilon = 1

agent = DQNAgent(state_size, action_size, hidden_size, alpha, gamma, epsilon)
episodes = 10000

train_agent(agent, episodes, n)

# Play against the AI
board = create_board(n)
current_player = 1
done = False
while not done:
    if current_player == 1:
        row, col = get_move(n)
        place_marker(board, row, col, current_player)
    else:
        state = torch.tensor(get_state(board, current_player)).float().unsqueeze(0)
        q_values = agent.q_network(state)
        action = torch.argmax(q_values).item()
        row, col = np.unravel_index(action, (n, n))
        place_marker(board, row, col, current_player)
    print_board(board)
    if is_winner(board, current_player):
        print(f"Player {current_player} wins!")
        done = True
    elif is_tie(board):
        print("Game is a tie!")
        done = True
    else:
        current_player = get_next_player(current_player)

   

