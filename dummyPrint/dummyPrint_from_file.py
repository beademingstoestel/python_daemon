#!/usr/bin/python3

import sys
import serial
import time
import bson
import numpy as np


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

def load_data(bson_file_name):
    # load the data from database
    with open(bson_file_name,'rb') as f:
        data = bson.decode_all(f.read())
    # analyse the data
    nbr_data = len(data)
    print("[INFO] The number of data in the database ={}".format(nbr_data))
    values = []
    timestamp = []
    for k in range(nbr_data):
        values.append(float((data[k]['value'])))
        tt = data[k]['loggedAt']
        tmp = (float(tt.time().hour)*3600+float(tt.time().minute)*60+float(tt.time().second))*1e3+float(tt.time().microsecond)/1e3
        timestamp.append(tmp)
    # send back the data
    return values, timestamp, nbr_data

def dummyPrint_from_file(device):
    print(device)
    ser = serial.Serial(device, 115200, timeout=1)
    ser.close()
    ser.open()

    # read recorded data
    # bson file name and load the data
    bson_file_name = 'datasets/data_record_20200401.bson'
    values, timestamp, nbr_data = load_data(bson_file_name)
    # select start/end point to analyze data
    start = 1900
    stop  = nbr_data
    # focus only on the useful part of the data
    xtime = np.array(timestamp[start:stop])
    y = np.array(values[start:stop])
    #plt.plot(xtime,y)
    #plt.grid(True)
    #plt.show()

    values = values[start:stop]
    timestamp = timestamp[start:stop]
    # dt = 0.031 seconds 
    dt = np.mean(np.diff(timestamp)*1e-3)
    nbr_data = len(values)

    cnt = 0
    while True:
        flag = 0
        try:
            serialSend(ser, "BPM=100=1")
            serialSend(ser, "VOL=200=2")
            serialSend(ser, "CPU=0=3")
            # serialSend(ser, "PRES=3=4")
            pres_val="PRES="+str(values[cnt])+"=4"
            serialSend(ser, pres_val)
            serialSend(ser, "FLOW=5=5")
            serialSend(ser, "CPU=40=6")
            serialSend(ser, "TPRESS=4=7")
            cnt = cnt + 1
            if cnt == nbr_data:
                cnt = 0
        except e:
            print(e)
            print("Serial disconnected or not available")
            ser.close()
            break
        time.sleep(dt)

if __name__ == "__main__":
    if len(sys.argv) >= 2:
        dummyPrint_from_file(sys.argv[1])
