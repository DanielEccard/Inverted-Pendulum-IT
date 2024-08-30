import serial.tools.list_ports
ports = serial.tools.list_ports.comports()
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import Jetson.GPIO as GPIO
serialInst = serial.Serial()
portsList = []
# Define os números dos pinos GPIO que você deseja usar
pin_11 = 11  # Pino GPIO 11
pin_13 = 13  # Pino GPIO 13
# Configura o modo de numeração dos pinos
GPIO.setmode(GPIO.BOARD)  # Usando a numeração física dos pinos

# Configura os pinos como saída
GPIO.setup(pin_11, GPIO.OUT)
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
def load_model_and_predict(state, model_path=r'/home/ic/Downloads/policy_net.pth'):
    n_observations = len(state)
    n_actions = 2  # CartPole tem 2 ações: esquerda (0) e direita (1)

    model = DQN(n_observations, n_actions)
    model.load_state_dict(torch.load(model_path))
    model.eval()

    state = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
    with torch.no_grad():
        action = model(state).max(1)[1].item()

    return action  # Retorna 0 para esquerda e 1 para direita

def update_pins(action):
    if action == 0:
        GPIO.output(pin_11, GPIO.HIGH)
        print("Action = 1: Pino 11 ativado")
    elif action == 1:
        GPIO.output(pin_11, GPIO.LOW)
        print("Action = 0: Pino 11 desativado")
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

try:
	while True:
		if serialInst.in_waiting:
			packet1 = serialInst.readline()
			data1 = packet1.decode('utf-8', errors='ignore').rstrip('\n').split()
			if data1[0] == 'Check':
				print(data1)
				posicao = float(data1[2])
				velocidade = float(data1[4])
				angulo = (float(data1[1]) - 3.70)
				v_angulo = float(data1[3])
				state = [posicao, velocidade, angulo, v_angulo] 
				action = load_model_and_predict(state)
				update_pins(action)
				print(f"Ação prevista: {action}")  # Imprime 0 (esquerda) ou 1 (direita)
except KeyboardInterrupt:
    # Permite que o loop seja interrompido com Ctrl+C
    print("Loop interrompido pelo usuário")

finally:
    # Limpa a configuração dos pinos GPIO antes de sair
    GPIO.cleanup()
    print("Configuração dos pinos limpa")
