#!/usr/bin/python3

import sys
import serial
import time

def getCRC(msg):
    checksum = 0
    for c in msg:
        checksum = checksum ^ ord(c)

    return checksum

def serialSend(ser, msg):
    msg = msg + '='
    ser.write(msg.encode('utf-8') + \
        getCRC(msg).to_bytes(2, 'little') + \
        "\r\n".encode('utf-8'))

def dummyPrint(device):
    print(device)
    ser = serial.Serial(device, 115200, timeout=1)
    ser.close()
    ser.open()
    cnt = 0
    while True:
        flag = 0
        try:
            cnt = cnt + 1
            if (cnt % 10) == 0:
                serialSend(ser, "BPM=100=1")
                serialSend(ser, "VOL=200=2")
                serialSend(ser, "CPU=0=3")
                serialSend(ser, "PRES=3=4")
                serialSend(ser, "FLOW=5=5")
                serialSend(ser, "CPU=40=6")
                serialSend(ser, "TPRESS=4=7")
            elif (cnt % 10) == 1:
                serialSend(ser, "BPM=100=1")
                serialSend(ser, "VOL=200=2")
                serialSend(ser, "CPU=0=3")
                serialSend(ser, "PRES=3.1=4")
                serialSend(ser, "FLOW=5=5")
                serialSend(ser, "CPU=40=6")
                serialSend(ser, "TPRESS=4=7")
            elif (cnt % 10) == 2:
                serialSend(ser, "BPM=100=1")
                serialSend(ser, "VOL=200=2")
                serialSend(ser, "CPU=0=3")
                serialSend(ser, "PRES=2.9=4")
                serialSend(ser, "FLOW=5=5")
                serialSend(ser, "CPU=40=6")
                serialSend(ser, "TPRESS=4=7")
            elif (cnt % 10) == 3:
                serialSend(ser, "BPM=100=1")
                serialSend(ser, "VOL=200=2")
                serialSend(ser, "CPU=0=3")
                serialSend(ser, "PRES=2.7=4")
                serialSend(ser, "FLOW=5=5")
                serialSend(ser, "CPU=40=6")
                serialSend(ser, "TPRESS=4=7")
            elif (cnt % 10) == 4:
                serialSend(ser, "BPM=100=1")
                serialSend(ser, "VOL=200=2")
                serialSend(ser, "CPU=0=3")
                serialSend(ser, "PRES=3.7=4")
                serialSend(ser, "FLOW=5=5")
                serialSend(ser, "CPU=40=6")
                serialSend(ser, "TPRESS=4=7")
            elif (cnt % 10) == 5:
                serialSend(ser, "BPM=100=1")
                serialSend(ser, "VOL=200=2")
                serialSend(ser, "CPU=0=3")
                serialSend(ser, "PRES=3.7=4")
                serialSend(ser, "FLOW=5=5")
                serialSend(ser, "CPU=40=6")
                serialSend(ser, "TPRESS=4=7")
            elif (cnt % 10) == 6:
                serialSend(ser, "BPM=100=1")
                serialSend(ser, "VOL=200=2")
                serialSend(ser, "CPU=0=3")
                serialSend(ser, "PRES=1.7=4")
                serialSend(ser, "FLOW=5=5")
                serialSend(ser, "CPU=40=6")
                serialSend(ser, "TPRESS=4=7")
            elif (cnt % 10) == 7:
                serialSend(ser, "BPM=100=1")
                serialSend(ser, "VOL=200=2")
                serialSend(ser, "CPU=0=3")
                serialSend(ser, "PRES=1.8=4")
                serialSend(ser, "FLOW=5=5")
                serialSend(ser, "CPU=40=6")
                serialSend(ser, "TPRESS=4=7")
            elif (cnt % 10) == 8:
                serialSend(ser, "BPM=100=1")
                serialSend(ser, "VOL=200=2")
                serialSend(ser, "CPU=0=3")
                serialSend(ser, "PRES=3.8=4")
                serialSend(ser, "FLOW=5=5")
                serialSend(ser, "CPU=40=6")
                serialSend(ser, "TPRESS=4=7")
            elif (cnt % 10) == 9:
                serialSend(ser, "BPM=100=1")
                serialSend(ser, "VOL=200=2")
                serialSend(ser, "CPU=0=3")
                serialSend(ser, "PRES=3.8=4")
                serialSend(ser, "FLOW=5=5")
                serialSend(ser, "CPU=40=6")
                serialSend(ser, "TPRESS=4=7")
        except e:
            print(e)
            print("Serial disconnected or not available")
            ser.close()
            break
        time.sleep(0.1)

if __name__ == "__main__":
    if len(sys.argv) >= 2:
        dummyPrint(sys.argv[1])
