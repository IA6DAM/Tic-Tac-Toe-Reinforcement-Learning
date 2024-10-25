import random
from TicTacToe import TicTacToe
import numpy as np
import matplotlib.pyplot as plt


class TDPolicyEvaluator:
    def __init__(self, gamma=1.0, alpha=0.1):
        self.game = TicTacToe()
        self.Q = {}  # Q(s, a)
        self.policy = {}  # π(s)
        self.gamma = gamma  # Discount factor
        self.alpha = alpha  
        # List to store TD errors
        self.td_errors = []  
        # List to store cumulative rewards for each episode
        self.cumulative_rewards = []  
        self.episode_rewards = 0  
        self.initialize()

    def initialize(self):
        """Initialize Q-values and policy for all state-action pairs."""
        # Iterate over all possible states (3^9)
        for i in range(3**9):  
            state = self.state_from_index(i)
            self.policy[state] = random.choice(self.game.available_moves()) if self.game.available_moves() else None
            self.Q[state] = {}
             # All possible moves (0-8)
            for move in range(9): 
              # Initialize Q-values to 0
                self.Q[state][move] = 0 

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
         # 'X' always starts the game
        player = 'X' 

        while not self.game.is_terminal():
            possible_actions = self.game.available_moves()
            if not possible_actions:
                break  
            # Random action following the current policy // Try the greedy policy
            action = random.choice(possible_actions)  
            self.game.make_move(action, player)

            next_state = tuple(self.game.board)  

            if self.game.current_winner:  
                reward = 1 if self.game.current_winner == 'X' else -1
                episode.append((state, action, reward, next_state))
                return episode , self.game.current_winner
                # It's a tie
            elif not self.game.available_moves(): 
                reward = 0
                episode.append((state, action, reward, next_state))
                return episode ,"a Tie" 
            else:
                reward = 0 

            episode.append((state, action, reward, next_state))
            state = next_state 
            player = 'O' if player == 'X' else 'X' 

        return episode,None

    def print_board(self, state):
        """Print the Tic Tac Toe board."""
        print("\nCurrent Board:")
        for i in range(3):
            print(" | ".join(state[i*3:(i+1)*3]))
            if i < 2:
                print("---------")
        print("\n")

    def update_q_table(self, state, action, reward, next_state, step):
        """Update Q-values using the TD update rule and track the TD error."""
        if state not in self.Q:
            self.Q[state] = {a: 0 for a in range(9)}
        if next_state not in self.Q:
            self.Q[next_state] = {a: 0 for a in range(9)}
            
        # Get the action with the highest Q-value
        best_next_action = max(self.Q[next_state], key=self.Q[next_state].get)  
         # Calculate TD target
        td_target = reward + self.gamma * self.Q[next_state][best_next_action] 
         # Calculate TD error
        td_error = td_target - self.Q[state][action] 
        # Update Q-value
        self.Q[state][action] += self.alpha * td_error  

        # Store TD error and cumulative reward
        self.td_errors.append(td_error)
        self.episode_rewards += reward

        # Print step information
        print(f'Step: {step}, State: {state}, Action: {action}, Reward: {reward}, Next State: {next_state}')

        # Update the policy to choose the best action based on the Q-values
        self.policy[state] = max(self.Q[state], key=self.Q[state].get)

    def train(self, num_episodes=1000):
        """Train the agen t by playing num_episodes games using the current policy."""
        for episode_num in range(num_episodes):
            self.episode_rewards = 0  # Reset rewards for this episode
            episode,winner = self.generate_episode()

            for step, (state, action, reward, next_state) in enumerate(episode):
             # TD learning update
                self.update_q_table(state, action, reward, next_state, step)  
            self.cumulative_rewards.append(self.episode_rewards)   
             
            if winner == 'Tie':
                print(f"Episode {episode_num + 1}/{num_episodes} completed. Result: Tie.\n")
            else:
                print(f"Episode {episode_num + 1}/{num_episodes} completed. Winner: {winner}\n")
            # Store cumulative rewards after each episode
           
            

        # Plot the results after training
        self.plot_results()

    def plot_results(self):
        """Plot the TD errors and cumulative rewards after training."""
        plt.figure(figsize=(10, 5))

        # Plot TD errors
        plt.subplot(1, 2, 1)
        plt.plot(self.td_errors, label="TD Error")
        plt.xlabel("Step")
        plt.ylabel("TD Error")
        plt.title("TD Error Over Time")
        plt.legend()

        # Plot cumulative rewards
        plt.subplot(1, 2, 2)
        plt.plot(self.cumulative_rewards, label="Cumulative Reward")
        plt.xlabel("Episode")
        plt.ylabel("Cumulative Reward")
        plt.title("Cumulative Reward Over Episodes")
        plt.legend()

        plt.tight_layout()
        plt.show()


# Instantiate and train the evaluator
evaluator = TDPolicyEvaluator(gamma=0.9, alpha=0.1)
evaluator.train(num_episodes=10)
