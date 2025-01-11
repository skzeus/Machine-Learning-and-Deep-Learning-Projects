import gymnasium as gym
import time
import numpy as np
import matplotlib.pyplot as plt

# Create environment
env = gym.make("Taxi-v3",render_mode='ansi')
env.reset()
print(env.render())


action_size = env.action_space.n
print("Action size: ", action_size)
state_size = env.observation_space.n
print("State size: ", state_size)

qtable = np.zeros((500,6)) # action size is 6 and state size is 500
episodes = 10000 # num of training episodes
interactions = 999 # max num of interactions per episode
epsilon = 0.99 # e-greedy 0.01 (explore) vs 0.99 (exploit)
alpha = 0.99 # learning rate - 1.
gamma = 0.99 # reward decay rate
debug = 1 # for non-slippery case to observe learning
hist = [] # evaluation history

# Main Q-learning loop
for episode in range(episodes):
    state = env.reset()[0]
    step = 0
    done = False
    total_rewards = 0
    
    for interact in range(interactions):
        # exploitation vs. exploratin by e-greedy sampling of actions
        if np.random.uniform(0, 1) < epsilon:
            action = np.argmax(qtable[state,:])
        else:
            action = np.random.randint(0,6)

        # Observe
        new_state, reward, done, truncated, info = env.step(action)

        # Update Q-table
        qtable[state, action] = qtable[state, action] + alpha * (reward + gamma * np.max(qtable[new_state, :]) - qtable[state, action])
                
        # Our new state is state
        state = new_state
        
        # Check if terminated
        if done == True or truncated == True: 
            break

env.reset()

# fct to evaluate the policy and find the avrg num of rewards and steps taken in 10 episodes
def evaluate_policy(env, Q_table, num_runs=10):
    total_rewards = []
    total_steps = []
    
    for _ in range(num_runs):
        state = env.reset()[0]
        done = False
        rewards = 0
        steps = 0
        
        while not done:
            action = np.argmax(Q_table[state, :])
            next_state, reward, done, truncated, info = env.step(action)
            rewards += reward
            steps += 1
            state = next_state
        
        total_rewards.append(rewards)
        total_steps.append(steps)
    
    avg_reward = np.mean(total_rewards)
    avg_steps = np.mean(total_steps)
    
    return avg_reward, avg_steps


# Step 6: Run evaluation
avg_reward, avg_steps = evaluate_policy(env, qtable)
print(f"Average Total Reward: {avg_reward}")
print(f"Average Number of Steps: {avg_steps}")

