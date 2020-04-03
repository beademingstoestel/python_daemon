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
    while True:
        flag = 0
        try:
            serialSend(ser, "BPM=100=1")
            serialSend(ser, "VOL=200=2")
            serialSend(ser, "CPU=0=3")
            serialSend(ser, "PRES=3=4")
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
