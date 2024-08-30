// Define os pinos do encoder
const int encoderPinA = 2; // Pino A do encoder conectado ao pino digital 2
const int encoderPinB = 4; // Pino B do encoder conectado ao pino digital 4
const int dt = 5;
const int fatconvrotacao = 26;
const float fatorconvmetro = 0.0050;
const int fatorconvsegundo = 1000;
int lastAction = -1; // Variável para armazenar o estado anterior de action
// Variáveis para armazenar estado do encoder
float encoderPos = 0.0; // Posição atual do encoder
volatile int encoderPosPrev = 0; // Posição anterior do encoder
volatile long lastUpdate = 0; // Último tempo de atualização
volatile float encoderSpeed = 0; // Velocidade angular do motor
float posOut = 0.0;
const int analogPin = A0;
float previousAngle = 0.0;
unsigned long previousTime = 0;
const int output1 = A1;
const int output2 = A2; 
int action = 0; // saída da IA. Usar serial read

void setup() {
  // Define os pinos do encoder como entrada
  pinMode(encoderPinA, INPUT);
  pinMode(encoderPinB, INPUT);

  pinMode(output1, OUTPUT); // configura pino como saída
  pinMode(output2, OUTPUT); // configura pino como saída
  pinMode(12, INPUT); // Configura o pino digital 2 como entrada
  // Habilita pull-up resistors internos
  digitalWrite(encoderPinA, HIGH);
  digitalWrite(encoderPinB, HIGH);
  attachInterrupt(digitalPinToInterrupt(encoderPinA), updateEncoder, CHANGE);
  // Inicializa a comunicação serial
  Serial.begin(9600);
}

void loop() {
  // Calcula a velocidade do motor a cada segundo
  if (millis() - lastUpdate >= dt) {
    updateEncoder();
    // Calcula a velocidade angular do motor em rotações por minuto (XPM), onde X depende do fator de conv, o 60 pode mudar também
    encoderSpeed = ((float)(encoderPos - encoderPosPrev) / (dt * fatconvrotacao)) * fatorconvmetro * fatorconvsegundo;

    // Exibe a posição e a velocidade do motor no monitor serial
    posOut = mapFloat( encoderPos,  0,  -2084,  2.4,  -2.4);

    // Atualiza a posição anterior do encoder
    encoderPosPrev = encoderPos;
    int sensorValue = analogRead(analogPin);
    float voltage = sensorValue * (5.0 / 1023.0);
    float angle = mapFloat(voltage, 0.0, 3.3, 0.0, 2 * PI);
    unsigned long currentTime = millis();
    float elapsedTime = (currentTime - previousTime) / 1000.0;
    float angularVelocity = (angle - previousAngle) / elapsedTime;

    Serial.print("Check ");
    posOut = mapFloat(encoderPos, 0, -2084, -2.4, 2.4);
    Serial.print(angle);
    Serial.print(" ");            // Separate values with a space
    Serial.print(posOut);
    Serial.print(" ");
    Serial.print(angularVelocity);
    Serial.print(" ");
    Serial.println(encoderSpeed);  // Ends the line with a newline
    previousAngle = angle;
    previousTime = currentTime;
    action = digitalRead(12); 
    if (action != lastAction) {
      if (action == 1) {
        analogWrite(output1, 155);
        analogWrite(output2, 0);  
      } else {
        analogWrite(output2, 155); 
        analogWrite(output1, 0);   
      }
    }
      lastAction = action; // Atualiza lastAction com o valor atual de action
      lastUpdate = millis();
    }
  }
// Função de interrupção para atualizar a posição do encoder
void updateEncoder() {
  // Lê o estado atual dos pinos A e B do encoder
  int a = digitalRead(encoderPinA);
  int b = digitalRead(encoderPinB);

  // Verifica a direção do movimento do encoder
  if (a == b) {
    encoderPos++;
  } else {
    encoderPos--;
  }
}

float mapFloat(float x, float in_min, float in_max, float out_min, float out_max) {
  return (x - in_min) * (out_max - out_min) / (in_max - in_min) + out_min;
}
