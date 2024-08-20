import serial.tools.list_ports
ports = serial.tools.list_ports.comports()
serialInst = serial.Serial()
portsList = []

for onePort in ports:
    portsList.append(str(onePort))
    print(str(onePort))
val = input("Select Port: COM")
for x in range(0,len(portsList)):
    if portsList[x].startswith("COM" + str(val)):
        portVar = "COM" + str(val)
        print(portVar)
serialInst.baudrate = 9600
serialInst.port = portVar
serialInst.open()
while True:
    if serialInst.in_waiting:
        packet = serialInst.readline()
        data = packet.decode('utf').rstrip('\n').split()
        voltage = data[1]
        angle = data[3]
        angular_velocity = data[7]
        print(voltage)
        print(angle)
        print(angular_velocity)
def load_model_and_predict(state, model_path='policy_net.pth'):
    n_observations = len(state)
    n_actions = 2  # CartPole tem 2 ações: esquerda (0) e direita (1)

    model = DQN(n_observations, n_actions)
    model.load_state_dict(torch.load(model_path))
    model.eval()

    state = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
    with torch.no_grad():
        action = model(state).max(1)[1].item()

    return action  # Retorna 0 para esquerda e 1 para direita

def send_action_via_serial(action, port='COM3', baudrate=9600):
    try:
        with serial.Serial(port, baudrate, timeout=1) as ser:
            ser.write(str(action).encode())
            print(f"Ação {action} enviada via serial.")
    except serial.SerialException as e:
        print(f"Erro de comunicação serial: {e}")

if __name__ == "__main__":
    # Exemplo de estado, substitua pelos valores reais
    state = [0.1, 0.2, 0.3, 0.4] 
    action = load_model_and_predict(state)
    print(f"Ação prevista: {action}")  # Imprime 0 (esquerda) ou 1 (direita)
    
    # Envia a ação prevista via comunicação serial
    send_action_via_serial(action, port='COM3', baudrate=9600)
