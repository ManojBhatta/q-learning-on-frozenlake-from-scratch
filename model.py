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
    else:
        return greedy_action(q_table, state)

# Step 7 - decay_epsilon (not yet solved)
# TODO: implement

# Step 8 - td_target (not yet solved)
# TODO: implement

# Step 9 - td_error (not yet solved)
# TODO: implement

# Step 10 - q_learning_update (not yet solved)
# TODO: implement

# Step 11 - interaction_step (not yet solved)
# TODO: implement

# Step 12 - run_training_episode (not yet solved)
# TODO: implement

# Step 13 - train_q_learning (not yet solved)
# TODO: implement

# Step 14 - extract_greedy_policy (not yet solved)
# TODO: implement

# Step 15 - run_greedy_episode (not yet solved)
# TODO: implement

# Step 16 - evaluate_success_rate (not yet solved)
# TODO: implement

