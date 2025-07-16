import mysql.connector
import serial
import time

# MySQL 데이터베이스 연결
def connect_to_database():
    return mysql.connector.connect(
        host="localhost",
        user="root",
        password="kimdm912",
        database="3d_printer_vib"
    )

# 시리얼 포트 설정 (보드레이트 250000으로 수정)
arduino_port = "COM4"
baud_rate = 250000  # Arduino 코드와 일치하도록 수정
read_timeout_sec = 5

def init_serial():
    try:
        ser = serial.Serial(arduino_port, baud_rate, timeout=1)
        time.sleep(2)
        print("시리얼 포트 연결 성공")
        return ser
    except Exception as e:
        print(f"시리얼 포트 연결 실패: {e}")
        return None

# 데이터베이스 저장 함수 (소수점 지원)
def save_to_db(accel_x, accel_y, accel_z):
    try:
        db = connect_to_database()
        cursor = db.cursor()
        query = """
            INSERT INTO vib_measure_21
            (accel_x, accel_y, accel_z, timestamp)
            VALUES (%s, %s, %s, NOW())
            """

        cursor.execute(query, (accel_x, accel_y, accel_z))
        db.commit()
        cursor.close()
        db.close()
    except Exception as e:
        print(f"❌ DB 저장 오류: {e}")

# 아두이노 데이터 수신 처리
def read_from_arduino():
    global last_received_time
    arduino = init_serial()
    last_received_time = time.time()

    if arduino is None:
        return

    try:
        while True:
            if arduino.in_waiting > 0:
                line = arduino.readline().decode().strip()
                last_received_time = time.time()

                # 헤더 라인 건너뛰기
                if line == "AX,AY,AZ":
                    print("🔷 헤더 확인")
                    continue
                
                # 데이터 파싱 (예: "0.2071,-0.0588,0.9944")
                if ',' in line:
                    parts = line.split(',')
                    if len(parts) == 3:
                        try:
                            ax = float(parts[0])
                            ay = float(parts[1])
                            az = float(parts[2])
                            print(f"✅ 수신 데이터: X={ax:.4f}, Y={ay:.4f}, Z={az:.4f}")
                            save_to_db(ax, ay, az)
                        except ValueError:
                            print(f"⚠️ 잘못된 데이터 형식: {line}")

            # 5초간 데이터 없을 때 재연결
            if time.time() - last_received_time > read_timeout_sec:
                print("⚠️ 5초간 데이터 없음. 재연결 시도...")
                arduino.close()
                arduino = init_serial()
                last_received_time = time.time()

            time.sleep(0.01)

    except KeyboardInterrupt:
        arduino.close()
        print("🔚 프로그램 종료")

if __name__ == "__main__":
    read_from_arduino()