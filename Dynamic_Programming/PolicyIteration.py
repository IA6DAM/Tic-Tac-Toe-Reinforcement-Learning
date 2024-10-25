## Progammation dynamic
import numpy as np
import random
from TicTacToeAgent.TicTacToe import TicTacToe
class PolicyIterationAgent:
    def __init__(self):
        self.states = []  
        self.policy = {}  
        self.value_function = {}  ## this the V value in our Case
        self.gamma = 0.9  
        self.learning_rate = 0.01 
   
    def generate_all_states(self, game):
        # Generate all possible states (board configurations)
        return [tuple(game.board)]        

    def initialize(self, game):
        # Initialize the policy and value function for all states
        for state in self.generate_all_states(game):
            self.states.append(state)
            self.policy[state] = random.choice(range(9))  # Random initial action
            self.value_function[state] = 0  # Value function starts at 0



    def choose_action(self, state, available_actions):
        # Choose the action based on the current policy
        if state in self.policy:
            return self.policy[state]
        else:
            return random.choice(available_actions)  # Random fallback

    def evaluate_policy(self, game):
        # Policy Evaluation Step: Update the value function
        for state in self.states:
            game.board = list(state)
            if game.is_terminal():
                self.value_function[state] = self.reward(game)
            else:
                next_state = self.get_next_state(game, self.policy[state])
                reward = self.reward(game)
                self.value_function[state] = reward + self.gamma * self.value_function.get(next_state, 0)

    def improve_policy(self, game):
        # Policy Improvement Step: Update the policy based on value function
        for state in self.states:
            game.board = list(state)
            available_actions = game.available_moves()
            best_action = None
            best_value = -float('inf')
            for action in available_actions:
                next_state = self.get_next_state(game, action)
                reward = self.reward(game)  # Reward for taking action a in state s
                value = reward + self.gamma * self.value_function.get(next_state, 0)
                if value > best_value:
                    best_value = value
                    best_action = action
            self.policy[state] = best_action

    def get_next_state(self, game, action):
        # Get the next state after making a move
        game.make_move(action, 'X')  # Assuming X is the current player
        return tuple(game.board)

    def reward(self, game):
        # Define the reward structure
        if game.current_winner == 'X':
            return 1
        elif game.current_winner == 'O':
            return -1
        elif not game.empty_squares():
            return 0  # Draw
        return 0  # For non-terminal states

    def play_game(self, game):
        # Play a game using the current policy
        game.reset()
        while not game.is_terminal():
            state = tuple(game.board)
            action = self.choose_action(state, game.available_moves())
            game.make_move(action, 'X')
            
            # Print the board after the player 'X' makes a move
            print("Player X makes a move:")
            game.print_board()  # Assuming there's a function to print the board
            
            if game.current_winner:
                break
            
            # Opponent plays randomly
            opponent_move = random.choice(game.available_moves())
            game.make_move(opponent_move, 'O')
            
            # Print the board after the opponent 'O' makes a move
            print("Player O makes a move:")
            game.print_board()  # Assuming there's a function to print the board
            
            if game.current_winner:
                break
        
        if game.current_winner:
            print(f"{game.current_winner} wins!")
        else:
            print("It's a tie!")


def train_policy_iteration_agent():
    game = TicTacToe()
    agent = PolicyIterationAgent()
    agent.initialize(game)

    for i in range(3000):  # Train over 3000 iterations
        agent.evaluate_policy(game)
        agent.improve_policy(game)

    print("Training complete!")
    agent.play_game(game)  # Test the agent after training

if __name__ == "__main__":
    train_policy_iteration_agent()
