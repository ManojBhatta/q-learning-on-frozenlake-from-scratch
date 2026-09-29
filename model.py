"""
Q-Learning on FrozenLake from Scratch

Assembled from your step-by-step solutions.
"""

import numpy as np

# Step 1 - init_q_table
import numpy as np

def init_q_table(num_states, num_actions):
    """Return a zero-initialized Q-table of shape (num_states, num_actions)."""
    return np.zeros((num_states, num_actions))

# Step 2 - max_q_value
import numpy as np

def max_q_value(q_table, state):
    """Return the maximum Q value across all actions for the given state."""
    return q_table[state].max()

# Step 3 - greedy_action
import numpy as np

def greedy_action(q_table, state):
    """Return the action index with the highest Q value at the given state."""
    q_vals = q_table[state]
    # max_q = np.max(q_vals)
    # max_indices = np.where(q_vals == max_q)[0]
    # return int(np.random.choice(list(max_indices)))
    return int(np.argmax(q_vals))

# Step 4 - sample_random_action
def sample_random_action(action_space):
    return int(action_space.sample())

# Step 5 - should_explore
def should_explore(epsilon, rng):
    """Return True with probability epsilon using the provided numpy Generator."""
    # TODO: draw a uniform sample from rng and compare it to epsilon
    rand = rng.uniform(0, 1.0)
    if rand <= epsilon:
        return True
    else:
        return False

# Step 6 - epsilon_greedy_action
import numpy as np

def epsilon_greedy_action(q_table, state, epsilon, action_space, rng):
    """Return an epsilon-greedy action for the given state."""
    # TODO: with prob epsilon explore via action_space, else pick a max-Q action (random among ties)
    if should_explore(epsilon, rng):
        return sample_random_action(action_space)

    q_vals = q_table[state]
    max_idx = np.flatnonzero(q_vals == q_vals.max())
    return int(rng.choice(max_idx))

# Step 7 - decay_epsilon
def decay_epsilon(epsilon, decay_rate, min_epsilon):
    # TODO: return max(min_epsilon, epsilon * decay_rate)
    return max(min_epsilon, epsilon * decay_rate)

# Step 8 - td_target
def td_target(reward, gamma, q_table, next_state, done):
    # TODO: compute r + gamma * max_a Q(next_state, a), zeroing the bootstrap when done.
    if done:
        return reward
    
    return reward + gamma * max_q_value(q_table, next_state)

# Step 9 - td_error
def td_error(target, q_table, state, action):
    # TODO: return the TD error: target minus current Q(state, action)
    return target - q_table[state, action]

# Step 10 - q_learning_update
def q_learning_update(q_table, state, action, reward, next_state, done, alpha, gamma):
    # TODO: apply Q(s,a) += alpha * (target - Q(s,a)) in place and return the new Q value
    target = td_target(reward, gamma, q_table, next_state, done)
    error = td_error(target, q_table, state, action)
    q_table[state, action] += alpha * error
    return float(q_table[state, action])

# Step 11 - interaction_step
def interaction_step(env, q_table, state, epsilon, alpha, gamma, rng):
    # TODO: select epsilon-greedy action, step env, apply Q-learning update, return (next_state, reward, done)
    action = epsilon_greedy_action(q_table, state, epsilon, env.action_space, rng)
    next_state, reward, terminated, truncated, _ = env.step(action)
    done = terminated or truncated
    q_learning_update(q_table, state, action,reward, next_state, done, alpha, gamma)
    return (int(next_state), float(reward), done)

# Step 12 - run_training_episode
def run_training_episode(env, q_table, epsilon, alpha, gamma, rng, max_steps=200):
    # TODO: reset env, then repeatedly call interaction_step until done or max_steps, returning total reward.
    obs, _ = env.reset()
    step = 0
    total_reward = 0
    while True:
        ns, r, done = interaction_step(env, q_table, obs, epsilon, alpha, gamma, rng)
        step += 1
        total_reward += r
        if step > max_steps or done:
            break
        obs = ns

    return total_reward

# Step 13 - train_q_learning
import numpy as np

def train_q_learning(env, num_episodes, alpha=0.8, gamma=0.95, epsilon_start=1.0, epsilon_min=0.01, epsilon_decay=0.99, seed=0, max_steps=200):
    q_table = np.zeros((env.observation_space.n, env.action_space.n))
    rng = np.random.default_rng(seed)
    env.action_space.seed(seed)
    obs, _ = env.reset(seed=seed)
    epsilon = epsilon_start
    ep_rewards = []
    for ep in range(num_episodes):
        ep_reward = run_training_episode(env, q_table, epsilon, alpha, gamma, rng, max_steps)
        ep_rewards.append(ep_reward)
        epsilon = decay_epsilon(epsilon, epsilon_decay,epsilon_min)

    return (q_table, ep_rewards)

# Step 14 - extract_greedy_policy
def extract_greedy_policy(q_table):
    ns, na = q_table.shape
    policy = np.zeros((ns, ))

    max_q = np.argmax(q_table, axis=1)

    policy = max_q
    return policy

# Step 15 - run_greedy_episode (not yet solved)
# TODO: implement

# Step 16 - evaluate_success_rate (not yet solved)
# TODO: implement

