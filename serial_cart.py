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
  