const int analogPin = A0;
float previousAngle = 0.0;
unsigned long previousTime = 0;

void setup() {
  Serial.begin(9600);
}

void loop() {
  int sensorValue = analogRead(analogPin);

  float voltage = sensorValue * (5.0 / 1023.0);

  float angle = mapFloat(voltage, 0.0, 3.3, 0.0, 2 * PI);

  unsigned long currentTime = millis();
  float elapsedTime = (currentTime - previousTime) / 1000.0; // Convert milliseconds to seconds

  float angularVelocity = (angle - previousAngle) / elapsedTime;

  Serial.print("Voltage: ");
  Serial.print(voltage);
  Serial.print("V\t");
  Serial.print("Angle: ");
  Serial.print(angle);
  Serial.print(" radians\t");
  Serial.print("Angular Velocity: ");
  Serial.print(angularVelocity);
  Serial.println(" radians/s");

  previousAngle = angle;
  previousTime = currentTime;

  delay(100);
}

float mapFloat(float x, float in_min, float in_max, float out_min, float out_max) {
  return (x - in_min) * (out_max - out_min) / (in_max - in_min) + out_min;
}
