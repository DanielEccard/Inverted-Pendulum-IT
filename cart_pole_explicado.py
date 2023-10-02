
"""
CartPole

Agente- Observa o ambiente e escolhe uma ação, sendo a recompensa +1
pra cada intervalo de tempo que permaneceu de pé. Se passar de 2.4m ou 24 graus
termina. Os inputs são 4: posição, velocidade linear, velocidade angular e ângulo.
São 2 outputs, andar para a direita ou esquerda. A rede vai prever qual terá melhor
recompensa.

"""

import gymnasium as gym
import math
import random 
import matplotlib
import matplotlib.pyplot as plt
from collections import namedtuple, deque
from itertools import count

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from cartpole_env import CartPoleEnv

env = CartPoleEnv(render_mode="human")

# set up matplotlib
is_ipython = 'inline' in matplotlib.get_backend()
if is_ipython:
    from IPython import display

plt.ion()

# if GPU is to be used
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

"""
 Replay Memory 
 Replay memory é usado para guardar estados, ações e rewards coletados
 pelo agente durante cada iteração em tuples, aumentando a eficiência
 que aprende de experiências passadas, dando mais estabilidade.
    Transition
    state, action -> next_state, reward
"""

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))


class ReplayMemory(object):

    def __init__(self, capacity):
        # Quando chega na capcidade máxima os mais antigos são eliminados
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        # Salva transições e adiciona novas experiências
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        # Amostras random ajudam a descorrelatar no tempo?
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)
 
"""
    DQN
    Queremos achar a política que maximiza o reward, montando assim
    um Q* que indica o reward futuro dado um certo estado e uma ação.
    Aproximamos Q* para Q, que a rede neural irá achar, segundo Bellman
    e depois aplica Huber Loss.
    
"""

class DQN(nn.Module):

    def __init__(self, n_observations, n_actions):
        super(DQN, self).__init__() # inicialização
        self.layer1 = nn.Linear(n_observations, 128) # input+1 hidden layer
        self.layer2 = nn.Linear(128, 128) # +1 hidden layer
        self.layer3 = nn.Linear(128, n_actions) # output, valores de Q para cada ação
    # forward pass com ativação ReLu 
    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        return self.layer3(x)

"""
    Target x online network
    Online- Estima os Qs, com updates frequentes usando gradient descent, escolhendo a ação
    Target- Mesma arquitetura da online, mas os updates são mais lentos. Parametros da online vão pra ela,
    indicando estimativas melhores do Q
"""
BATCH_SIZE = 128 # indica samples no replay
GAMMA = 0.99 # fator de desconto no Q
EPS_START = 0.9 # fator de exploração inicial
EPS_END = 0.05 # fator de exploração mínimo
EPS_DECAY = 1000 # fator de decaimento da exploração
TAU = 0.005 # taxa de update do target network
LR = 1e-4 # learning rate do Adam

# Número de ações
n_actions = env.action_space.n
# Observações
state, info = env.reset()
n_observations = len(state)

policy_net = DQN(n_observations, n_actions).to(device) # policy network
target_net = DQN(n_observations, n_actions).to(device) # target network
target_net.load_state_dict(policy_net.state_dict())

optimizer = optim.AdamW(policy_net.parameters(), lr=LR, amsgrad=True) # usa Adam amsgrad
memory = ReplayMemory(10000) # buffer com 10000 experiências


steps_done = 0


def select_action(state): # estado input, ação output. Toma ação com mais reward com p= epsilon ou ação exploratória p=1-epsilon
    global steps_done
    sample = random.random() # explora ou aproveita
    eps_threshold = EPS_END + (EPS_START - EPS_END) * \
        math.exp(-1. * steps_done / EPS_DECAY) #eps greedy
    steps_done += 1
    if sample > eps_threshold:
        with torch.no_grad():
            # t.max(1) will return the largest column value of each row.
            # second column on max result is index of where max element was
            # se eps < sample, ação com maior reward
            return policy_net(state).max(1)[1].view(1, 1)
    else:
        return torch.tensor([[env.action_space.sample()]], device=device, dtype=torch.long) # se sample< eps ação aleatória


episode_durations = []


def plot_durations(show_result=False):
    plt.figure(1)
    durations_t = torch.tensor(episode_durations, dtype=torch.float)
    if show_result:
        plt.title('Result')
    else:
        plt.clf()
        plt.title('Training...')
    plt.xlabel('Episode')
    plt.ylabel('Duration')
    plt.plot(durations_t.numpy())
    # +100 episódios vê a média
    if len(durations_t) >= 100:
        means = durations_t.unfold(0, 100, 1).mean(1).view(-1)
        means = torch.cat((torch.zeros(99), means))
        plt.plot(means.numpy())

    plt.pause(0.001)  # pause a bit so that plots are updated
    if is_ipython:
        if not show_result:
            display.display(plt.gcf())
            display.clear_output(wait=True)
        else:
            display.display(plt.gcf())


# Training loop
# ^^^^^^^^^^^^^

def optimize_model():
    if len(memory) < BATCH_SIZE: # checa se tem memória antiga
        return
    transitions = memory.sample(BATCH_SIZE)
    # Ele separa os componentes de cada sample(state, actions...) para facilitar cálculos
    # list(zip(*[('a', 1), ('b', 2), ('c', 3), ('d', 4)]))
    # [('a', 'b', 'c', 'd'), (1, 2, 3, 4)]
    batch = Transition(*zip(*transitions))

    # Cria mask de estados não finais
    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                          batch.next_state)), device=device, dtype=torch.bool)
    non_final_next_states = torch.cat([s for s in batch.next_state
                                                if s is not None])
    # Forma tensores com cada batch
    state_batch = torch.cat(batch.state)
    action_batch = torch.cat(batch.action)
    reward_batch = torch.cat(batch.reward)

    # Acha os Q values para as ações no estado atual usando as informações do batch na policy net
    state_action_values = policy_net(state_batch).gather(1, action_batch)

    # Q values na target net
    next_state_values = torch.zeros(BATCH_SIZE, device=device)
    with torch.no_grad():
        next_state_values[non_final_mask] = target_net(non_final_next_states).max(1)[0]
    # Q esperados segundo bellman
    expected_state_action_values = (next_state_values * GAMMA) + reward_batch

    # Huber loss entre Q esperado e Q calculado
    criterion = nn.SmoothL1Loss()
    loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

    # Backpropagation
    optimizer.zero_grad()
    loss.backward()
    # Clip dos gradientes para evitar que fiquem muito grandes?
    torch.nn.utils.clip_grad_value_(policy_net.parameters(), 100)
    optimizer.step() #muda o modelo


if torch.cuda.is_available():
    num_episodes = 600
else:
    num_episodes = 200

for i_episode in range(num_episodes):
    # Reinicia o ambiente
    state, info = env.reset()
    state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
    for t in count():
        action = select_action(state) # seleciona ação
        observation, reward, terminated, truncated, _ = env.step(action.item()) # executa ação
        reward = torch.tensor([reward], device=device)
        done = terminated or truncated

        if terminated:
            next_state = None
        else:
            next_state = torch.tensor(observation, dtype=torch.float32, device=device).unsqueeze(0)

        # Guarda transição
        memory.push(state, action, next_state, reward)

        # Próximo estado
        state = next_state

        # Muda os parâmetros da Q network
        optimize_model()

        # Soft update of the target network's weights
        # θ′ ← τ θ + (1 −τ )θ′
        target_net_state_dict = target_net.state_dict()
        policy_net_state_dict = policy_net.state_dict()
        for key in policy_net_state_dict:
            target_net_state_dict[key] = policy_net_state_dict[key]*TAU + target_net_state_dict[key]*(1-TAU)
        target_net.load_state_dict(target_net_state_dict)

        if done:
            episode_durations.append(t + 1)
            plot_durations()
            break

print('Complete')
plot_durations(show_result=True)
plt.ioff()
plt.show()
