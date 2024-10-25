import numpy as np
import random
from TicTacToe import TicTacToe

class MCPolicyEvaluator:
    def __init__(self, gamma=1.0):
        self.game = TicTacToe()
        self.Q = {}  # Q(s, a)
        self.Returns = {}  # Returns(s, a)
        self.policy = {}  # π(s)
        self.gamma = gamma  # Discount factor
        self.initialize()

    def initialize(self):
        """Initialize Q-values, policy, and Returns for all state-action pairs."""
        for i in range(3**9):  # Iterate over all possible states (3^9)
            state = self.state_from_index(i)
            self.policy[state] = random.choice(self.game.available_moves()) if self.game.available_moves() else None
            self.Q[state] = {}
            self.Returns[state] = {}
            for move in range(9):  # All possible moves (0-8)
                self.Q[state][move] = 0  # Initialize Q-values to 0
                self.Returns[state][move] = []

    def state_from_index(self, i):
        
        base3 = np.base_repr(i, base=3).zfill(9)
        state = tuple([' ' if x == '0' else ('X' if x == '1' else 'O') for x in base3])
        return state

    def is_possible(self, s):
        """Return a list of possible actions (empty spots) in the current state."""
        return [i for i, spot in enumerate(s) if spot == ' ']

    def set_player(self, s):
        """Determine the current player based on the state (count 'X' and 'O')."""
        x_count = s.count('X')
        o_count = s.count('O')
        return 'X' if x_count <= o_count else 'O'

    def generate_episode(self):
        """Play a full episode using the current policy and return the list of (state, action, reward) tuples."""
        episode = []
        self.game.reset()  
        state = tuple(self.game.board) 
        player = 'X'  # 'X' always starts the game

        while not self.game.is_terminal():
            possible_actions = self.game.available_moves()
            if not possible_actions:
                break  

            action = random.choice(possible_actions)  # Random action following the current policy
            self.game.make_move(action, player)

            next_state = tuple(self.game.board)  # Capture next state as a tuple

            if self.game.current_winner:  # Someone has won
                reward = 1 if self.game.current_winner == 'X' else -1
                episode.append((state, action, reward))
                break
            elif not self.game.available_moves():  # It's a tie
                reward = 0
                episode.append((state, action, reward))
                break
            else:
                reward = 0  # Game continues, no immediate reward

            episode.append((state, action, reward))
            state = next_state  # Move to the next state
            player = 'O' if player == 'X' else 'X'  # Switch player

        return episode

    def print_board(self, state):
        """Print the Tic Tac Toe board."""
        print("\nCurrent Board:")
        for i in range(3):
            print(" | ".join(state[i*3:(i+1)*3]))
            if i < 2:
                print("---------")
        print("\n")

    def update_Q_and_policy(self, episode, reward, first_visit=True):
        """Update Q-values and the policy after an episode."""
        G = reward  # Final reward (either win, loss, or tie)
        cumulative_reward = 0
        visited_state_action_pairs = set()

        for state, action, reward in reversed(episode):
            cumulative_reward = self.gamma * cumulative_reward + reward

            if first_visit:
                if (state, action) not in visited_state_action_pairs:
                    self.Returns[state][action].append(cumulative_reward)
                    self.Q[state][action] = np.mean(self.Returns[state][action])
                    visited_state_action_pairs.add((state, action))
            else:
                self.Returns[state][action].append(cumulative_reward)
                self.Q[state][action] = np.mean(self.Returns[state][action])

            # Update the policy to choose the best action based on the Q-values
            self.policy[state] = max(self.Q[state], key=self.Q[state].get)

        # Print the final board and outcome of the episode
        self.print_board(tuple(self.game.board))
        print(f"Final Reward: {G}, Winner: {self.game.current_winner or 'Draw'}")
        print("-" * 50)

    def train(self, num_episodes=1000, first_visit=True):
        """Train the agent by playing num_episodes games using the current policy."""
        for episode_num in range(num_episodes):
            episode = self.generate_episode()
            if self.game.current_winner:
                reward = 1 if self.game.current_winner == 'X' else -1
            else:
                reward = 0

            self.update_Q_and_policy(episode, reward, first_visit=first_visit)

            print(f"Episode {episode_num + 1}/{num_episodes} completed.\n")


# Instantiate and train the evaluator
evaluator = MCPolicyEvaluator(gamma=0.9)
evaluator.train(num_episodes=1, first_visit=True)
