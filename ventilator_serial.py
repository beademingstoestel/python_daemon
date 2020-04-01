"""
Ventilator Serial Handler
"""
import serial
import queue
import time
import ventilator_protocol as proto
from datetime import datetime
import traceback


class SerialHandler():

    def __init__(self, db_queue, request_queue, out_queue, alarm_queue, port='/dev/ventilator', baudrate=115200):
        self.port = port
        self.baudrate = baudrate
        self.ser = None

        self.request_queue = request_queue
        self.db_queue = db_queue # Enqueue to
        self.out_queue = out_queue
        self.alarm_queue = alarm_queue
        self.message_id = 0

    def queue_put(self, type, val):
        """
        Send values to all necessary queues

        Args:
            type (str): type to be sent
            val (int): value to be sent
        """
        self.db_queue.put({'type': type, 'val': val})
        self.alarm_queue.put({'type': type, 'val': val})

    def attempt_reconnection(self):
            self.ser = None
            try:
                self.ser = serial.Serial(self.port, self.baudrate)
            except:
                pass

    def run(self, name):
        waiting_for_acks = {}
        self.ser = serial.Serial(self.port, self.baudrate)
        self.ser.reset_input_buffer()
        self.ser.reset_output_buffer()

        print("Starting {}".format(name))
        while True:
            try:
                msg = self.out_queue.get(block=False)
            except queue.Empty:
                msg = None

            if msg != None:
                print("outgoing message: {} with id {}".format(msg, self.message_id))

                msg_bytes = proto.construct_serial_message(msg['type'], msg['val'], self.message_id)

                waiting_for_ack = {'msg': msg, 'sent_at': datetime.utcnow().timestamp()}
                waiting_for_acks[self.message_id] = waiting_for_ack

                # we sent a message with id, so increment it
                self.message_id += 1

                if self.message_id == 256:
                    self.message_id = 0                    

                try:
                    self.ser.write(msg_bytes)
                except:
                    print("Unable to send line ", msg_bytes)
                    self.attempt_reconnection()

            line = ""
            try:
                if (self.ser.inWaiting()>0):
                    line = self.ser.readline()
            except:
                self.attempt_reconnection()

            if line == "":
                continue


            try:
                line = line[:-2] # strip out '\r\n'
                checksum = int(line[-1]) # get raw checksum value
                line = line[:-1]
                calculated_checksum = proto.compute_LRC(line)

                if checksum != calculated_checksum:
                    print(line)
                    print("Checksum does not match, discard")
                    print("key: {},"
                          "val: {},"
                          "checksum: {}, "
                          "calculated_checksum: {}".format(key,
                                                           int(val),
                                                           checksum,
                                                           calculated_checksum))
                    continue

                id = line[-2]
                line = line[:-2]
     
                line = line.decode('ascii')
                tokens = line.split('=')
                key = tokens[0]
                print("Received message: {}".format(line))

                if line.startswith(proto.ack + '='):
                    print("Received ack for id {}".format(id))
                    del waiting_for_acks[id]
                
                if line.startswith(proto.alarm + '='):
                    val = tokens[1]
                    # acknowledge receipt
                    print('Send ACK for id: {}'.format(id))

                    msg_ack_bytes = proto.construct_ack_message(id)

                    try:
                        self.ser.write(msg_ack_bytes)
                    except:
                        print("Unable to send line ", msg_ack_bytes)


                # handle measurements
                for msgtype in proto.measurements:
                    if line.startswith((msgtype + '=')):
                        val = tokens[1]
                        self.queue_put(msgtype, val)

                # handle settings
                for msgtype in proto.settings:
                    if line.startswith((msgtype + '=')):
                        val = tokens[1]
                        if proto.settings_values[msgtype] != val:
                            # send to GUI
                            self.request_queue.put({'type': 'setting',
                                                    'key': msgtype,
                                                    'value': float(val)})
                            # acknowledge receipt
                            print('Send ACK for id: {}'.format(id))

                            msg_ack_bytes = proto.construct_ack_message(id)

                            try:
                                self.ser.write(msg_ack_bytes)
                            except:
                                print("Unable to send line ", msg_ack_bytes)


                # resend messages waiting for ack
                now = datetime.utcnow().timestamp()
                delete = [] 
                for waiting_message in waiting_for_acks.items():
                    if waiting_message[1]['sent_at'] + 1 < now:  #throws error
                        # resend message
                        print("outgoing message: {}", waiting_message[1]['msg'])

                        self.out_queue.put(waiting_message[1]['msg'])
                        delete.append(waiting_message[0]) 
          
                for i in delete:
                    del waiting_for_acks[i] 

            except Exception as e:
                print(e)
                traceback.print_exc()

