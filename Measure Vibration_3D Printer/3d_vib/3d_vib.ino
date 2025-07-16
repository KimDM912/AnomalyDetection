#include <Wire.h>
#include <MPU6050.h>
#include <avr/wdt.h>

MPU6050 mpu;

const unsigned long SAMPLE_INTERVAL_MS = 10;    // 100Hz (10ms)
const float ACCEL_SCALE = 4096.0;               // ±8g 범위
const float FILTER_ALPHA = 0.2;                 // 1차 저역통과필터(16Hz 근처)

float ax_filtered = 0, ay_filtered = 0, az_filtered = 0;
bool header_printed = false;

// 센서 캘리브레이션(초기 오프셋 보정)
float ax_offset = 0, ay_offset = 0, az_offset = 0;

void calibrateMPU() {
  float sum_ax = 0, sum_ay = 0, sum_az = 0;
  const int samples = 500;
  for (int i = 0; i < samples; i++) {
    int16_t ax, ay, az;
    mpu.getAcceleration(&ax, &ay, &az);
    sum_ax += ax / ACCEL_SCALE;
    sum_ay += ay / ACCEL_SCALE;
    sum_az += az / ACCEL_SCALE;
    delay(2);
  }
  ax_offset = sum_ax / samples;
  ay_offset = sum_ay / samples;
  az_offset = (sum_az / samples) - 1.0; // 중력 보정
}

void setup() {
  Serial.begin(250000);
  Wire.begin();
  mpu.initialize();

  mpu.setFullScaleAccelRange(MPU6050_ACCEL_FS_8);
  mpu.setDLPFMode(MPU6050_DLPF_BW_20);

  if (!mpu.testConnection()) {
    Serial.println("ERROR:MPU6050_CONNECTION_FAILED");
    while (1);
  }

  wdt_enable(WDTO_8S);
  delay(500);  // 센서 안정화

  calibrateMPU();

  // 헤더 1회만 출력
  Serial.println("AX,AY,AZ");
  header_printed = true;
}

void loop() {
  wdt_reset();
  static unsigned long lastSampleTime = 0;
  unsigned long currentTime = millis();

  if (currentTime - lastSampleTime >= SAMPLE_INTERVAL_MS) {
    lastSampleTime = currentTime;

    int16_t ax_raw, ay_raw, az_raw;
    mpu.getAcceleration(&ax_raw, &ay_raw, &az_raw);

    // g 단위 변환 및 오프셋 보정
    float ax_g = (ax_raw / ACCEL_SCALE) - ax_offset;
    float ay_g = (ay_raw / ACCEL_SCALE) - ay_offset;
    float az_g = (az_raw / ACCEL_SCALE) - az_offset;

    // 1차 저역통과필터(16Hz)
    ax_filtered = ax_filtered * (1 - FILTER_ALPHA) + ax_g * FILTER_ALPHA;
    ay_filtered = ay_filtered * (1 - FILTER_ALPHA) + ay_g * FILTER_ALPHA;
    az_filtered = az_filtered * (1 - FILTER_ALPHA) + az_g * FILTER_ALPHA;

    // CSV 형식, 소수점 4자리, 공백 없이 출력
    Serial.print(ax_filtered, 4);
    Serial.print(',');
    Serial.print(ay_filtered, 4);
    Serial.print(',');
    Serial.println(az_filtered, 4);
  }
}