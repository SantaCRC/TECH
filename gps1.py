import serial, pynmea2
#while True:
with serial.Serial('COM3', 115200, timeout=1) as ser:
    # read up to ten bytes (timeout)
    line = ser.readline().decode()
    linea = pynmea2.parse(line)
    print(linea.latitude)
