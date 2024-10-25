## Progammation dynamic
import numpy as np
import random
from TicTacToeAgent.TicTacToe import TicTacToe

class ValueIterationAgent:
    def __init__(self):
        self.game = TicTacToe()
        self.value_function = {}  # V(s)
        self.gamma = 1.0  # Discount factor
        self.epsilon = 0.01  
        self.initialize()  

    def initialize(self):
        """Initialize the value function for the current state."""
        state = tuple(self.game.board)
        self.value_function[state] = 0  # V(current state)==0

    def value_iteration(self):
        """Performs Value Iteration to find the optimal value function."""
        while True:
            delta = 0
            for state in list(self.value_function.keys()):
                self.game.board = list(state)  
                if self.game.is_terminal():
                    self.value_function[state] = self.reward()  
                else:
                    v = self.value_function[state]  
                    action_values = []
                    for action in self.game.available_moves():
                        next_state = self.get_next_state(action)
                        reward = self.reward()  
                        future_value = self.value_function.get(next_state, 0)
                        action_values.append(reward + self.gamma * future_value)

                    self.value_function[state] = max(action_values)  # Update value
                    delta = max(delta, abs(v - self.value_function[state]))

            if delta < self.epsilon:  # Convergence check
                break

    def choose_action(self, state):
        """Choose the best action based on the value function. """
        best_action = None
        best_value = -float('inf')
        for action in self.game.available_moves():
            next_state = self.get_next_state(action)
            value = self.value_function.get(next_state, 0)
            if value > best_value:
                best_value = value
                best_action = action
        return best_action

    def get_next_state(self, action):
        """Get the next state after making a move."""
        game_copy = TicTacToe()  # Create a copy of the game
        game_copy.board = self.game.board.copy()  # Copy the current board state
        game_copy.make_move(action, 'X')  # Assuming 'X' is the current player
        return tuple(game_copy.board)

    def reward(self):
        """Define the reward structure for terminal states."""
        if self.game.current_winner == 'X':
            return 10  # Positive reward for winning
        elif self.game.current_winner == 'O':
            return -10  # Negative reward for losing
        elif not self.game.empty_squares():
            return 0  # Draw
        return 0  # Non-terminal states

    def play_game(self):
        """Play a game using the optimal policy from value iteration."""
        self.game.reset()
        while not self.game.is_terminal():
            state = tuple(self.game.board)
            action = self.choose_action(state)
            if action is not None:  # Ensure a valid action is chosen
                self.game.make_move(action, 'X')
                print("Player X makes a move:")
                self.game.print_board()  # Assuming there's a function to print the board
                
                if self.game.is_terminal():
                    break
                
                available_moves = self.game.available_moves()
                if available_moves:  
                    self.game.make_move(random.choice(available_moves), 'O')
                    print("Player O makes a move:")
                    self.game.print_board()  # Assuming there's a function to print the board

                if self.game.current_winner:
                    break

        if self.game.current_winner:
            print(f"{self.game.current_winner} wins!")
        else:
            print("It's a tie!")

def train_value_iteration_agent():
    agent = ValueIterationAgent()

    # Perform value iteration
    agent.value_iteration()

    print("Training complete!")
    agent.play_game()  # Test the agent after training

if __name__ == "__main__":
    train_value_iteration_agent()
