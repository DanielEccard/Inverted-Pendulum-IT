import serial.tools.list_ports
ports = serial.tools.list_ports.comports()
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import Jetson.GPIO as GPIO
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
serialInst = serial.Serial()
portsList = []
posicao_data = []
velocidade_data = []
angulo_data = []
v_angulo_data = []
action_data = []
# Define os números dos pinos GPIO que você deseja usar
pin_11 = 11  # Pino GPIO 11
pin_13 = 13
pin_15 = 15
# Configura o modo de numeração dos pinos
GPIO.setmode(GPIO.BOARD)  # Usando a numeração física dos pinos

# Configura os pinos como saída
GPIO.setup(pin_11, GPIO.OUT)
GPIO.setup(pin_13, GPIO.OUT)
GPIO.setup(pin_15, GPIO.OUT)
class DQN(nn.Module):
    def __init__(self, n_observations, n_actions):
        super(DQN, self).__init__()
        self.layer1 = nn.Linear(n_observations, 128) # input+1 hidden layer
        self.layer2 = nn.Linear(128, 128) # +1 hidden layer
        self.layer3 = nn.Linear(128, n_actions)

    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        return self.layer3(x)
def load_model_and_predict(state, model_path=r'/home/ic/Downloads/policy_net_multi.pth'):
    n_observations = len(state)
    n_actions = 6  # CartPole tem 2 ações: esquerda (0) e direita (1)

    model = DQN(n_observations, n_actions)
    model.load_state_dict(torch.load(model_path))
    model.eval()

    state = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
    with torch.no_grad():
        action = model(state).max(1)[1].item()

    return action  # Retorna 0 para esquerda e 1 para direita

def update_pins(action):
    if action == 0:
        # Ativa o pino 11 e desativa o pino 13
        GPIO.output(pin_11, GPIO.LOW)
        GPIO.output(pin_13, GPIO.LOW)
        GPIO.output(pin_15, GPIO.LOW)
    elif action == 1:
        # Desativa o pino 11 e ativa o pino 13
        GPIO.output(pin_11, GPIO.HIGH)
        GPIO.output(pin_13, GPIO.LOW)
        GPIO.output(pin_15, GPIO.LOW)
    elif action == 2:
        # Desativa o pino 11 e ativa o pino 13
        GPIO.output(pin_11, GPIO.LOW)
        GPIO.output(pin_13, GPIO.HIGH)
        GPIO.output(pin_15, GPIO.LOW)
    elif action == 3:
        # Desativa o pino 11 e ativa o pino 13
        GPIO.output(pin_11, GPIO.HIGH)
        GPIO.output(pin_13, GPIO.HIGH)
        GPIO.output(pin_15, GPIO.LOW)
    elif action == 4:
        # Desativa o pino 11 e ativa o pino 13
        GPIO.output(pin_11, GPIO.LOW)
        GPIO.output(pin_13, GPIO.LOW)
        GPIO.output(pin_15, GPIO.HIGH)
    elif action == 5:
        GPIO.output(pin_11, GPIO.HIGH)
        GPIO.output(pin_13, GPIO.LOW)
        GPIO.output(pin_15, GPIO.HIGH)
    else:
        print("Valor de action inválido. Deve ser 0 ou 1.")


# Identifica as portas seriais disponíveis
ports = serial.tools.list_ports.comports()
portsList = [port.device for port in ports]

# Exibe as portas disponíveis e permite a seleção
print("Portas disponíveis:")
for idx, port in enumerate(portsList):
    print(f"{idx}: {port}")

port_index = int(input("Selecione o índice da porta: "))
portVar = portsList[port_index]
print(f"Porta selecionada: {portVar}")

# Configura e abre a conexão serial
serialInst = serial.Serial()
serialInst.baudrate = 9600
serialInst.port = portVar
serialInst.open()

# Configura o gráfico
fig, (ax1, ax2, ax3, ax4, ax5) = plt.subplots(5, 1, figsize=(10, 8))
fig.suptitle('Dados em Tempo Real')

ax1.set_title('Posição')
ax1.set_xlabel('Tempo')
ax1.set_ylabel('Posição')

ax2.set_title('Velocidade')
ax2.set_xlabel('Tempo')
ax2.set_ylabel('Velocidade')

ax3.set_title('Ação')
ax3.set_xlabel('Tempo')
ax3.set_ylabel('Ação')

ax4.set_title('Ângulo')
ax4.set_xlabel('Tempo')
ax4.set_ylabel('Ângulo')

ax5.set_title('Velocidade Angular')
ax5.set_xlabel('Tempo')
ax5.set_ylabel('V_Ângulo')



# Função de atualização para o gráfico
def update_plot(frame):
    ax1.clear()
    ax2.clear()
    ax3.clear()
    ax4.clear()
    ax5.clear()
    
    ax1.plot(posicao_data, label='Posição')
    ax1.set_title('Posição')
    ax1.set_xlabel('Tempo')
    ax1.set_ylabel('Posição')
    
    ax2.plot(velocidade_data, label='Velocidade', color='orange')
    ax2.set_title('Velocidade')
    ax2.set_xlabel('Tempo')
    ax2.set_ylabel('Velocidade')
    
    ax3.plot(action_data, label='Ação', color='yellow')
    ax3.set_title('Ação')
    ax3.set_xlabel('Tempo')
    ax3.set_ylabel('Ação')

    ax4.plot(angulo_data, label='Ângulo', color='green')
    ax4.set_title('Ângulo')
    ax4.set_xlabel('Tempo')
    ax4.set_ylabel('Ângulo')
    
    ax5.plot(v_angulo_data, label='Velocidade Angular', color='red')
    ax5.set_title('Velocidade Angular')
    ax5.set_xlabel('Tempo')
    ax5.set_ylabel('V_Ângulo')
    
    plt.tight_layout()

ani = FuncAnimation(fig, update_plot, interval=200)  # Atualiza o gráfico a cada 200 msegundo


try:
	while True:
		if serialInst.in_waiting:
			packet1 = serialInst.readline()
			data1 = packet1.decode('utf-8', errors='ignore').rstrip('\n').split()
			if data1[0] == 'Check':
				print(data1)
				posicao = float(data1[2])+2.4
				velocidade = float(data1[4])
				angulo = (float(data1[1]) - 3.70)
				v_angulo = float(data1[3])
				state = [posicao, velocidade, angulo, v_angulo] 
				action = load_model_and_predict(state)
				#Adicionando os dados às listas do plot
				posicao_data.append(posicao)
				velocidade_data.append(velocidade)
				angulo_data.append(angulo)
				v_angulo_data.append(v_angulo)				
				action_data.append(action)	
				update_pins(action)
				print(f"Ação prevista: {action}")  # Imprime 0 (esquerda) ou 1 (direita)
except KeyboardInterrupt:
    # Permite que o loop seja interrompido com Ctrl+C
    print("Loop interrompido pelo usuário")

finally:
    # Limpa a configuração dos pinos GPIO antes de sair
    GPIO.cleanup()
    print("Configuração dos pinos limpa")
    plt.show()
