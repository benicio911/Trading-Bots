import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import load_model
import gym
import os

# Custom Trading Environment
class TradingEnv(gym.Env):
    def __init__(self, initial_balance, data):
        super(TradingEnv, self).__init__()
        # Example Usage
        self.initial_balance = 10000
        self.balance = initial_balance
        self.data = data
        self.current_step = 0
        self.action_space = gym.spaces.Discrete(4)  # Actions: 0 - hold, 1 - buy, 2 - sell, 3 - close
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(data.shape[1],), dtype=np.float32)
        self.position = "none"  # Current position: none, buy, sell
        self.position_price = 0  # Price at which current position was taken
        self.peak_balance = initial_balance  # Highest balance reached
        self.lowest_balance = initial_balance  # Lowest balance reached

    def step(self, action):
        current_price = self._get_current_price()
        previous_price = self._get_previous_price()
        reward = self.calculate_reward(action, current_price, previous_price)
        self.current_step += 1
        done = self.current_step >= len(self.data) - 1
        next_state = self._next_observation()
        return next_state, reward, done, {}

    def reset(self):
        self.balance = self.initial_balance
        self.current_step = 0
        self.position = "none"
        self.position_price = 0
        self.peak_balance = self.initial_balance
        self.lowest_balance = self.initial_balance
        return self._next_observation()

    def _next_observation(self):
        return self.data[self.current_step]

    def _get_current_price(self):
        return self.data[self.current_step][0]  # Assuming the first column is price

    def _get_previous_price(self):
        return self.data[self.current_step - 1][0] if self.current_step > 0 else self.data[0][0]

    def calculate_reward(self, action, current_price, previous_price):
        reward = 0
        transaction_cost = 0.1  # Transaction cost
        alpha = 0.8  # Drawdown penalty scaling factor

        if action == 1:  # Buy
            if self.position == "sell":
                reward = (self.position_price - current_price) - transaction_cost
                self.position = "none"
            elif self.position == "none":
                self.position = "buy"
                self.position_price = current_price

        elif action == 2:  # Sell
            if self.position == "buy":
                reward = (current_price - self.position_price) - transaction_cost
                self.position = "none"
            elif self.position == "none":
                self.position = "sell"
                self.position_price = current_price

        elif action == 3:  # Close
            if self.position == "buy":
                reward = (current_price - self.position_price) - transaction_cost
            elif self.position == "sell":
                reward = (self.position_price - current_price) - transaction_cost
            self.position = "none"

        # Update account balance
        self.balance += reward

        # Calculate drawdown
        current_drawdown = max(0, (self.peak_balance - self.balance) / self.peak_balance)
        drawdown_penalty = -alpha * (current_drawdown ** 2)
        reward += drawdown_penalty

        # Update peak balance and lowest balance if necessary
        if self.balance > self.peak_balance:
            self.peak_balance = self.balance
        if self.balance < self.lowest_balance:
            self.lowest_balance = self.balance

        return reward

# Policy Network for PPO
class PolicyNetwork(tf.keras.Model):
    def __init__(self, num_actions):
        super(PolicyNetwork, self).__init__()
        self.dense1 = Dense(128, activation='relu')
        self.dense2 = Dense(64, activation='relu')
        self.dense3 = Dense(num_actions, activation='softmax')

    def call(self, inputs):
        x = self.dense1(inputs)
        x = self.dense2(x)
        return self.dense3(x)

# Main PPO Training Loop
def train_ppo(env, total_episodes, learning_rate=0.001):
    num_actions = env.action_space.n
    model = PolicyNetwork(num_actions)
    optimizer = Adam(learning_rate)
    model_file_path = "PPO_Model"
        # Check if the model file already exists
    if os.path.exists(model_file_path):
        print(f"Model found at {model_file_path}. Loading model...")
        model = load_model(model_file_path)
    else:
        # Initialize a new model if not found
        model = PolicyNetwork(num_actions)
        optimizer = Adam(learning_rate)

        for episode in range(total_episodes):
            state = env.reset()
            episode_reward = 0
            done = False

            while not done:
                with tf.GradientTape() as tape:
                    action_probs = model(state)
                    action = np.random.choice(num_actions, p=np.squeeze(action_probs))
                    next_state, reward, done, _ = env.step(action)
                    loss = -tf.math.log(action_probs[0, action]) * reward 

                grads = tape.gradient(loss, model.trainable_variables)
                optimizer.apply_gradients(zip(grads, model.trainable_variables))

                state = next_state
                episode_reward += reward
        # Save the trained model
        model.save(model_file_path)
        print(f"Model trained and saved at {model_file_path}")
        print(f"Episode: {episode}, Reward: {episode_reward}")

data = np.random.random((1000, 5))  # Replace with real market data
env = TradingEnv(initial_balance, data)
train_ppo(env, total_episodes=100)