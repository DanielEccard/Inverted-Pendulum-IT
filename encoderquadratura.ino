// Define os pinos do encoder
const int encoderPinA = 2; // Pino A do encoder conectado ao pino digital 2
const int encoderPinB = 4; // Pino B do encoder conectado ao pino digital 3
const int dt =5;
const int fatconvrotacao = 26;
const float fatorconvmetro = 0.0050;
const int fatorconvsegundo = 1000;
// Variáveis para armazenar estado do encoder
volatile int encoderPos = 0; // Posição atual do encoder
volatile int encoderPosPrev = 0; // Posição anterior do encoder
volatile long lastUpdate = 0; // Último tempo de atualização
volatile float encoderSpeed = 0; // Velocidade angular do motor

void setup() {
  // Define os pinos do encoder como entrada
  pinMode(encoderPinA, INPUT);
  pinMode(encoderPinB, INPUT);

  // Habilita pull-up resistors internos
  digitalWrite(encoderPinA, HIGH);
  digitalWrite(encoderPinB, HIGH);

  // Configura a função de interrupção para o pino A do encoder
  attachInterrupt(digitalPinToInterrupt(encoderPinA), updateEncoder, CHANGE);

  // Inicializa a comunicação serial
  Serial.begin(9600);
}

void loop() {
  // Calcula a velocidade do motor a cada segundo
  if (millis() - lastUpdate >= dt) {
    // Calcula a velocidade angular do motor em rotações por minuto (XPM), onde X depende do fator de conv, o 60 pode mudar também
    encoderSpeed = ((float)(encoderPos - encoderPosPrev) / (dt * fatconvrotacao)) * fatorconvmetro * fatorconvsegundo ;

  
    // Exibe a posição e a velocidade do motor no monitor serial
    Serial.print("Posição: ");
    Serial.print(encoderPos);
    Serial.print("\tPosição(cm): ");
    Serial.print(encoderPos*0.0192);
    Serial.print("\tVelocidade: ");
    Serial.print(encoderSpeed);
    Serial.println(" m/s");

    // Atualiza a posição anterior do encoder
    encoderPosPrev = encoderPos;

    // Reinicia a contagem de tempo
    lastUpdate = millis();
  }
}

// Função de interrupção para atualizar a posição do encoder
void updateEncoder() {
  // Lê o estado atual dos pinos A e B do encoder
  int a = digitalRead(encoderPinA);
  int b = digitalRead(encoderPinB);

  // Verifica a direção do movimento do encoder
  if (a == HIGH && b == LOW) {
    encoderPos--;  
  } else if (a == LOW && b == HIGH) {
    encoderPos--;
  } else if (a == HIGH && b == HIGH) {
    encoderPos++;
  } else {
    encoderPos++;
  }
}