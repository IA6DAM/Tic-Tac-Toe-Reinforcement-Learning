import numpy as np
import random
import matplotlib.pyplot as plt
from collections import defaultdict
from TicTacToe import TicTacToe

class MCPolicyEvaluator:
    def __init__(self, gamma=1.0):
        self.game = TicTacToe()
        self.Q = defaultdict(lambda: defaultdict(float))  # Q(s, a)
        self.Returns = defaultdict(lambda: defaultdict(list))  # Returns(s, a)
        self.policy = {}  # π(s)
        self.gamma = gamma  # Discount factor
        self.explored_state_action_pairs = set()  # To track explored (s,a) pairs
        self.initialize()

    def initialize(self):
        """Initialize the policy for all state-action pairs."""
        for i in range(3**9):
            state = self.state_from_index(i)
            self.policy[state] = random.choice(self.game.available_moves()) if self.game.available_moves() else None

    def state_from_index(self, i):
        """Convert state index to a tuple representing the board."""
        base3 = np.base_repr(i, base=3).zfill(9)
        state = tuple([' ' if x == '0' else ('X' if x == '1' else 'O') for x in base3])
        return state

    def generate_episode(self):
        """Play a full episode using the current policy and return the list of (state, action, reward) tuples."""
        episode = []
        self.game.reset()
        state = tuple(self.game.board)
        player = 'X'

        while not self.game.is_terminal():
            possible_actions = self.game.available_moves()
            if not possible_actions:
                break

            action = random.choice(possible_actions)
            self.game.make_move(action, player)

            next_state = tuple(self.game.board)
            reward = 1 if self.game.current_winner == 'X' else (-1 if self.game.current_winner == 'O' else 0)

            episode.append((state, action, reward))

            state = next_state
            player = 'O' if player == 'X' else 'X'

            if self.game.is_terminal():
                break

        return episode

    def update_Q_and_policy(self, episode):
        """Update Q-values and policy after an episode."""
        G = 0
        visited_state_action_pairs = set()

        for state, action, reward in reversed(episode):
            G = self.gamma * G + reward
            if (state, action) not in visited_state_action_pairs:
                self.Returns[state][action].append(G)
                self.Q[state][action] = np.mean(self.Returns[state][action])
                visited_state_action_pairs.add((state, action))
                self.explored_state_action_pairs.add((state, action))

            # Update the policy to choose the best action based on the Q-values
            self.policy[state] = max(self.Q[state], key=self.Q[state].get)

    def train(self, num_episodes=1000):
        """Train the agent by playing num_episodes games using the current policy."""
        win_count, draw_count, loss_count = [], [], []
        exploration_progress = []

        for episode_num in range(num_episodes):
            episode = self.generate_episode()

            if self.game.current_winner == 'X':
                win_count.append(1)
                draw_count.append(0)
                loss_count.append(0)
            elif self.game.current_winner == 'O':
                win_count.append(0)
                draw_count.append(0)
                loss_count.append(1)
            else:
                win_count.append(0)
                draw_count.append(1)
                loss_count.append(0)

            # Update Q and policy
            self.update_Q_and_policy(episode)

            # Track exploration progress
            total_state_action_pairs = 9 * (3**9)  # Total possible state-action pairs
            explored_percentage = (len(self.explored_state_action_pairs) / total_state_action_pairs) * 100
            exploration_progress.append(explored_percentage)

        return np.cumsum(win_count), np.cumsum(draw_count), np.cumsum(loss_count), exploration_progress

    def plot_results(self, wins, draws, losses, exploration_progress, num_episodes):
        """Plot the results."""
        episodes = range(num_episodes)

        # Plot percentage of game outcomes
        plt.figure(figsize=(10, 6))
        plt.plot(episodes, wins, label="% Wins", color="blue")
        plt.plot(episodes, draws, label="% Draws", color="red")
        plt.plot(episodes, losses, label="% Losses", color="green")
        plt.xlabel("# Episodes")
        plt.ylabel("%")
        plt.title("Game outcomes with currently learned policy")
        plt.legend()
        plt.show()

        # Plot percentage of (s, a) pairs explored
        plt.figure(figsize=(10, 6))
        plt.plot(episodes, exploration_progress, label="% (s, a) Explored", color="purple")
        plt.xlabel("# Episodes")
        plt.ylabel("% of (s, a) pairs explored")
        plt.title("Exploration of state-action pairs over episodes")
        plt.legend()
        plt.show()

# Instantiate and train the evaluator
evaluator = MCPolicyEvaluator(gamma=0.9)
num_episodes = 100000
wins, draws, losses, exploration_progress = evaluator.train(num_episodes)

# Plot the results
evaluator.plot_results(wins, draws, losses, exploration_progress, num_episodes)
