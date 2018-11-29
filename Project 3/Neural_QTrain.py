import gym
import tensorflow as tf
import numpy as np
import random

# General Parameters
# -- DO NOT MODIFY --
ENV_NAME = 'CartPole-v0'
EPISODE = 200000  # Episode limitation
STEP = 200  # Step limitation in an episode
TEST = 10  # The number of tests to run every TEST_FREQUENCY episodes
TEST_FREQUENCY = 100  # Num episodes to run before visualizing test accuracy

# TODO: HyperParameters
GAMMA = 0.9 # discount factor
INITIAL_EPSILON = 0.5 # starting value of epsilon
FINAL_EPSILON = 0.01 # final value of epsilon
EPSILON_DECAY_STEPS = 100 # decay period
HIDDEN_NODES = 100
REPLAY_MEMORY_SIZE = 10000
BATCH_SIZE = 64
replay_buffer = []

# Create environment
# -- DO NOT MODIFY --
env = gym.make(ENV_NAME)
epsilon = INITIAL_EPSILON
STATE_DIM = env.observation_space.shape[0]
ACTION_DIM = env.action_space.n

# Placeholders
# -- DO NOT MODIFY --
state_in = tf.placeholder("float", [None, STATE_DIM])
action_in = tf.placeholder("float", [None, ACTION_DIM])
target_in = tf.placeholder("float", [None])

# TODO: Define Network Graph
#layer one
w1 = tf.Variable(tf.random_normal([STATE_DIM, HIDDEN_NODES], stddev = 0.1))
b1 = tf.Variable(tf.zeros([HIDDEN_NODES]) + 0.1)
l1 = tf.nn.relu(tf.matmul(state_in, w1) + b1)
#layer two
w2 = tf.Variable(tf.random_normal([HIDDEN_NODES, ACTION_DIM], stddev = 0.1))
b2 = tf.Variable(tf.zeros([ACTION_DIM]) + 0.1)

# TODO: Network outputs
q_values = tf.matmul(l1, w2) + b2
q_action = tf.reduce_sum(tf.multiply(q_values, action_in), reduction_indices=1)

# TODO: Loss/Optimizer Definition
loss = tf.reduce_sum(tf.square(target_in - q_action))
optimizer = tf.train.AdamOptimizer().minimize(loss)

# Start session - Tensorflow housekeeping
session = tf.InteractiveSession()
session.run(tf.global_variables_initializer())


# -- DO NOT MODIFY ---
def explore(state, epsilon):
    """
    Exploration function: given a state and an epsilon value,
    and assuming the network has already been defined, decide which action to
    take using e-greedy exploration based on the current q-value estimates.
    """
    Q_estimates = q_values.eval(feed_dict={
        state_in: [state]
    })
    if random.random() <= epsilon:
        action = random.randint(0, ACTION_DIM - 1)
    else:
        action = np.argmax(Q_estimates)
    one_hot_action = np.zeros(ACTION_DIM)
    one_hot_action[action] = 1
    return one_hot_action

def update_buffer(replay_buffer, action, state, next_state, reward, done):
    replay_buffer.append([action, state, next_state, reward, done])
    if len(replay_buffer) > REPLAY_MEMORY_SIZE:
        replay_buffer.pop(0)
    return None

# Main learning loop
for episode in range(EPISODE):

    # initialize task
    state = env.reset()

    # Update epsilon once per episode
    epsilon -= (epsilon - FINAL_EPSILON) / EPSILON_DECAY_STEPS

    # Move through env according to e-greedy policy
    for step in range(STEP):
        action = explore(state, epsilon)
        next_state, reward, done, _ = env.step(np.argmax(action))


        # TODO: Calculate the target q-value.
        # hint1: Bellman
        # hint2: consider if the episode has terminated
        update_buffer(replay_buffer, action, state, next_state, reward, done)
        
        if (len(replay_buffer) > BATCH_SIZE):
            databatch = random.sample(replay_buffer, BATCH_SIZE)
            action_ = [data[0] for data in databatch]
            state_ = [data[1] for data in databatch]
            next_state_ = [data[2] for data in databatch]
            reward_ = [data[3] for data in databatch]
            target = []
            nextstate_q_values = q_values.eval(feed_dict={state_in: next_state_})
            for i in range(0, BATCH_SIZE):
                done_ = databatch[i][4]
                if done_:
                    target.append(reward_[i])
                else:
                    target.append(reward_[i] + GAMMA * np.max(nextstate_q_values[i]))
            session.run([optimizer], feed_dict={
            target_in: target,
            action_in: action_,
            state_in: state_
        })

        # Update
        state = next_state
        if done:
            break

    # Test and view sample runs - can disable render to save time
    # -- DO NOT MODIFY --
    if (episode % TEST_FREQUENCY == 0 and episode != 0):
        total_reward = 0
        for i in range(TEST):
            state = env.reset()
            for j in range(STEP):
                env.render()
                action = np.argmax(q_values.eval(feed_dict={
                    state_in: [state]
                }))
                state, reward, done, _ = env.step(action)
                total_reward += reward
                if done:
                    break
        ave_reward = total_reward / TEST
        print('episode:', episode, 'epsilon:', epsilon, 'Evaluation '
                                                        'Average Reward:', ave_reward)

env.close()
