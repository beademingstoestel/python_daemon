"""
Breathney emergency ventilator daemon - communication and alarm handler

Copyright (C) 2020  Frank Vanbever et al.
(see AUTHORS for complete list)

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <https://www.gnu.org/licenses/>.

Ventilator Serial Handler
"""
import serial
import queue
import time
import ventilator_protocol as proto
import time
import traceback
from datetime import datetime
import ventilator_log as log

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
        self.counter = 0

    def queue_put(self, type, val, loggedAt):
        """
        Send values to all necessary queues

        Args:
            type (str): type to be sent
            val (int): value to be sent
        """
        self.db_queue.put({'type': type, 'val': val, 'loggedAt': loggedAt})

        if (self.counter % 100) == 0:
            begin = int(round(time.time() * 1000))
            print("serial counter {} time {}".format(self.counter, begin))
            log.INFO(__name__, self.request_queue, "serial counter {} time {}".format(self.counter, begin))
        self.counter = self.counter + 1

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
        log.INFO(__name__, self.request_queue, "Starting {}".format(name))

        while True:
            try:
                msg = self.out_queue.get(block=False)
            except queue.Empty:
                msg = None

            if msg != None:
                print("outgoing message: {} with id {}".format(msg, self.message_id))

                msg_bytes = proto.construct_serial_message(msg['type'], msg['val'], self.message_id)

                waiting_for_ack = {'msg': msg, 'sent_at': time.monotonic()}
                waiting_for_acks[self.message_id] = waiting_for_ack

                # we sent a message with id, so increment it
                self.message_id += 1

                if self.message_id == 256:
                    self.message_id = 0                    

                try:
                    self.ser.write(msg_bytes)
                except:
                    print("Unable to send line ", msg_bytes)
                    log.ERROR(__name__, self.request_queue, "Unable to send line ", msg_bytes)
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

                id = line[-2]
                line = line[:-2]
                line = line.decode('ascii')
                tokens = line.split('=')
                key = tokens[0]
                val = tokens[1]

                if checksum != calculated_checksum:
                    print(line)
                    print("Checksum does not match, discard")
                    print("key: {},"
                          "val: {},"
                          "checksum: {}, "
                          "calculated_checksum: {}".format(key,
                                                           val,
                                                           checksum,
                                                           calculated_checksum))
                    continue

                #print("Received message: {}".format(line))

                if line.startswith(proto.ack + '='):
                    print("Received ack for id {}".format(id))
                    del waiting_for_acks[id]
                
                if line.startswith(proto.alarm + '='):
                    val = tokens[1]
                    self.alarm_queue.put({'type': 'ALARM', 'val': int(float(val)), 'source': 'serial'})
                    # acknowledge receipt
                    print('Send ACK for id: {}'.format(id))

                    msg_ack_bytes = proto.construct_ack_message(id)

                    try:
                        self.ser.write(msg_ack_bytes)
                    except:
                        print("Unable to send line ", msg_ack_bytes)
                        log.ERROR(__name__, self.request_queue, "Unable to send line ", msg_ack_bytes)


                # handle measurements
                for msgtype in proto.measurements:
                    if line.startswith((msgtype + '=')):
                        val = tokens[1]
                        self.queue_put(msgtype, val, datetime.utcnow())

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
                                log.ERROR(__name__, self.request_queue, "Unable to send line ", msg_ack_bytes)


                for msgtype in proto.internal_settings:
                    if line.startswith((msgtype + "=")):
                        print("Received internal settings parameter {}".format(msgtype))

                # resend messages waiting for ack
                now = time.monotonic()
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
                log.ERROR(__name__, self.request_queue, "Exception occurred = {}".format(e))
                traceback.print_exc()



