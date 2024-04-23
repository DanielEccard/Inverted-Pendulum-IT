import collections
import gym
import numpy as np
import statistics
import tensorflow as tf
import tqdm

from matplotlib import pyplot as plt
from typing import Any, List, Sequence, Tuple


# Environment
env = gym.make("CartPole-v1")

# Seed 
seed = 42
env.seed(seed)
tf.random.set_seed(seed)
np.random.seed(seed)

# Limita o epsilon
eps = np.finfo(np.float32).eps.item()

class ActorCritic(tf.keras.Model):

  def __init__(
      self, 
      num_actions: int, # No nosso caso 2
      num_hidden_units: int):
    """Initialize."""
    super().__init__()

    self.common = tf.keras.layers.Dense(num_hidden_units, activation="relu")
    self.actor = tf.keras.layers.Dense(num_actions)
    self.critic = tf.keras.layers.Dense(1)

  def call(self, inputs: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
    x = self.common(inputs)
    return self.actor(x), self.critic(x)

num_actions = env.action_space.n  # 2
num_hidden_units = 128

model = ActorCritic(num_actions, num_hidden_units)


def env_step(action: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:

  state, reward, done, _ = env.step(action) # tira do env as infos
  return (state.astype(np.float32), 
          np.array(reward, np.int32), 
          np.array(done, np.int32))


def tf_env_step(action: tf.Tensor) -> List[tf.Tensor]:
  return tf.numpy_function(env_step, [action], 
                           [tf.float32, tf.int32, tf.int32]) # transforma env step em tensores
def run_episode(
    initial_state: tf.Tensor,  
    model: tf.keras.Model, 
    max_steps: int) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor]:

  action_probs = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True)
  values = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True)
  rewards = tf.TensorArray(dtype=tf.int32, size=0, dynamic_size=True)

  initial_state_shape = initial_state.shape
  state = initial_state

  for t in tf.range(max_steps):
    state = tf.expand_dims(state, 0) # expande para bater com o que o tf quer

    # Tira a ação a partit do modelo
    action_logits_t, value = model(state)

    # proxima ação
    action = tf.random.categorical(action_logits_t, 1)[0, 0]
    action_probs_t = tf.nn.softmax(action_logits_t)

    # guarda valores pro crítico avaliar
    values = values.write(t, tf.squeeze(value))
    action_probs = action_probs.write(t, action_probs_t[0, action])

    # Aplica a ação no env do Gym
    state, reward, done = tf_env_step(action)
    state.set_shape(initial_state_shape)
    rewards = rewards.write(t, reward)

    if tf.cast(done, tf.bool):
      break

  action_probs = action_probs.stack()
  values = values.stack()
  rewards = rewards.stack()

  return action_probs, values, rewards

def get_expected_return(
    rewards: tf.Tensor, 
    gamma: float, 
    standardize: bool = True) -> tf.Tensor:
  """Compute expected returns per timestep."""

  n = tf.shape(rewards)[0]
  returns = tf.TensorArray(dtype=tf.float32, size=n)
  rewards = tf.cast(rewards[::-1], dtype=tf.float32)
  discounted_sum = tf.constant(0.0)
  discounted_sum_shape = discounted_sum.shape
  for i in tf.range(n):
    reward = rewards[i]
    discounted_sum = reward + gamma * discounted_sum
    discounted_sum.set_shape(discounted_sum_shape)
    returns = returns.write(i, discounted_sum)
  returns = returns.stack()[::-1]

  if standardize:
    returns = ((returns - tf.math.reduce_mean(returns)) / 
               (tf.math.reduce_std(returns) + eps))

  return returns
# Huber Loss
huber_loss = tf.keras.losses.Huber(reduction=tf.keras.losses.Reduction.SUM)

def compute_loss(
    action_probs: tf.Tensor,  
    values: tf.Tensor,  
    returns: tf.Tensor) -> tf.Tensor:
  """Computes the combined actor-critic loss."""

  advantage = returns - values

  action_log_probs = tf.math.log(action_probs)
  actor_loss = -tf.math.reduce_sum(action_log_probs * advantage)

  critic_loss = huber_loss(values, returns)

  return actor_loss + critic_loss

optimizer = tf.keras.optimizers.Adam(learning_rate=0.01)

@tf.function
def train_step(
    initial_state: tf.Tensor, 
    model: tf.keras.Model, 
    optimizer: tf.keras.optimizers.Optimizer, 
    gamma: float, 
    max_steps_per_episode: int) -> tf.Tensor:
  """Runs a model training step."""

  with tf.GradientTape() as tape:

    # 1 rodada
    action_probs, values, rewards = run_episode(
        initial_state, model, max_steps_per_episode) 

    # retorno esperado
    returns = get_expected_return(rewards, gamma)

    action_probs, values, returns = [
        tf.expand_dims(x, 1) for x in [action_probs, values, returns]] 

    # loss pro backpropagation
    loss = compute_loss(action_probs, values, returns)

  # Gradientes
  grads = tape.gradient(loss, model.trainable_variables)
  optimizer.apply_gradients(zip(grads, model.trainable_variables))
  episode_reward = tf.math.reduce_sum(rewards)

  return episode_reward

min_episodes_criterion = 100
max_episodes = 10000 # limite superior (esse código é rápido)
max_steps_per_episode = 1000

# "Cartpole-v1 is considered solved if average reward is >= 195 over 100"
# Escolhi 300
reward_threshold = 300
running_reward = 0

# Gamma para mudar o aprendizado
gamma = 0.99

episodes_reward: collections.deque = collections.deque(maxlen=min_episodes_criterion)

with tqdm.trange(max_episodes) as t:
  for i in t:
    initial_state = tf.constant(env.reset(), dtype=tf.float32)
    episode_reward = int(train_step(
        initial_state, model, optimizer, gamma, max_steps_per_episode))

    episodes_reward.append(episode_reward)
    running_reward = statistics.mean(episodes_reward)

    t.set_description(f'Episode {i}')
    t.set_postfix(
        episode_reward=episode_reward, running_reward=running_reward)

    # Show average episode reward every 10 episodes
    if i % 10 == 0:
      pass # print(f'Episode {i}: average reward: {avg_reward}')

    if running_reward > reward_threshold and i >= min_episodes_criterion:  
        break

print(f'\nSolved at episode {i}: average reward: {running_reward:.2f}!')