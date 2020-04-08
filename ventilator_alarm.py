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
"""
import queue
import time
import ventilator_protocol as proto
from enum import Enum
import ventilator_log as log

# BT: Below Threshold
# AT Above Threshold

class AlarmBits(Enum): #todo add more categories?
    NONE                                                = int('00000000000000000000000000000000', 2)  #No alarm
    DATABASE_PROCESSING_PRESSURE_BPM_BT                 = int('00000000000000000000000000000001', 2)  #Breathing per minute too low
    DATABASE_PROCESSING_PRESSURE_IE_RATIO_BT            = int('00000000000000000000000000000010', 2)  #Respiratory rate below threshold
    DATABASE_PROCESSING_PRESSURE_TIME_INHALE_EXHALE_AT  = int('00000000000000000000000000000100', 2)   #inhale time above 10s and exhale time above 10s
    DATABASE_PROCESSING_PRESSURE_DP_AT                  = int('00000000000000000000000000001000', 2)  #Pressure deviate during inhale
    DATABASE_PROCESSING_PRESSURE_DP_PEEP_AT             = int('00000000000000000000000000010000', 2)  #Pressure below peep level detected
    DATABASE_PROCESSING_PRESSURE_AT_BT                  = int('00000000000000000000000000100000', 2)  #Pressure outside the allowed range
    DATABASE_PROCESSING_VOLUME_AT_BT                    = int('00000000000000000000000001000000', 2)  #Volume outside the allowed range
    DATABASE_PROCESSING_VOLUME_NOT_NEAR_ZERO_EBC        = int('00000000000000000000000010000000', 2)  #Volume not near zero at the end of breathing cycle
    DATABASE_PROCESSING_RESERVED1                       = int('00000000000000000000000100000000', 2)  #reserved1
    DATABASE_PROCESSING_RESERVED2                       = int('00000000000000000000001000000000', 2)  #reserved2
    DATABASE_PROCESSING_RESERVED3                       = int('00000000000000000000010000000000', 2)  #reserved3
    DATABASE_PROCESSING_RESERVED4                       = int('00000000000000000000100000000000', 2)  #reserved4
    DATABASE_PROCESSING_RESERVED5                       = int('00000000000000000001000000000000', 2)  #reserved5
    DATABASE_PROCESSING_RESERVED6                       = int('00000000000000000010000000000000', 2)  #reserved6
    DATABASE_PROCESSING_RESERVED7                       = int('00000000000000000100000000000000', 2)  #reserved7
    DATABASE_PROCESSING_RESERVED8                       = int('00000000000000001000000000000000', 2)  #reserved8
    DATABASE_PROCESSING_RESERVED9                       = int('00000000000000010000000000000000', 2)  #reserved9
    DATABASE_PROCESSING_RESERVED10                      = int('00000000000000100000000000000000', 2)  #reserved10
    DATABASE_PROCESSING_EXCEPTION                       = int('00000000000001000000000000000000', 2)  #In database processing an exception occurred
    SERIAL_TIMEOUT                                      = int('10000000000000000000000000000000', 2)  #Serial communication timeout

class AlarmHandler():

    def __init__(self, input_queue, serial_queue, request_queue):
        """
        Alarm Handler constructor

        Args:
            input_queue (queue.Queue): queue on which we receive values
            serial_queue (queue.Queue): queue to notify Controller of alarm
        """
        self.input_queue = input_queue
        self.serial_queue = serial_queue
        self.request_queue = request_queue

        self.alarm_val = 0
        self.alarm_val_serial = 0
        self.alarm_val_processing = 0

        self.time_last_kick_sent = 0
        self.time_last_kick_received = 0

        self.time_watchdog_kick_checked = 0

        self.first_watchdog_kick_received = False
        self.start_time = 0

        self.counter = 0

    def run(self, name):
        print("Starting {}".format(name))
        log.INFO(__name__, self.request_queue, "Starting {}".format(name))
        self.start_time = time.time()
        while True:
            if (self.counter % 10) == 0:
                begin = int(round(time.time() * 1000))
                print("alarm time {}".format(begin))
                log.INFO(__name__, self.request_queue, "alarm counter {} time {}".format(self.counter, begin))
            self.counter = self.counter + 1
            cur_time = time.time()
            # Do we need to kick the watchdog? Only after we've received the first kick
            if self.first_watchdog_kick_received and ((cur_time - self.time_last_kick_sent) > 1 ):
                self.serial_queue.put({'type': proto.alarm, 'val': self.alarm_val_processing}) #TODO alarm_val or alarm_val_processing
                self.time_last_kick_sent = cur_time

            try:
                msg = self.input_queue.get(block=False)
            except queue.Empty:
                msg = None

            if msg != None:
                print("alarm ", msg)
                if msg['type'] == proto.alarm:
                    if msg['source'] == 'serial':
                        self.time_last_kick_received == cur_time
                        if not self.first_watchdog_kick_received:
                            self.first_watchdog_kick_received = True
                        #self.alarm_val_serial = msg['val'] for testing
                    #todo define serial alarm-bitmasks, serial alarm = 0 repeated ?
                    if msg['source'] == 'processing':                        
                        self.alarm_val_processing = msg['val']

                    old_alarm = self.alarm_val
                    self.alarm_val = self.alarm_val_serial | self.alarm_val_processing
                    print("old_alarm ", hex(old_alarm))
                    print("alarm_val ", hex(self.alarm_val))
                    # don't wait on the watchdog kick to put the alarm on the serial queue.
                    if (self.alarm_val > 0) and (old_alarm != self.alarm_val):
                        #self.serial_queue.put({'type': proto.alarm, 'val': self.alarm_val})
                        self.request_queue.put({'type': proto.alarm, 'value': self.alarm_val})

            # Have we received a watchdog kick in time?
            if self.first_watchdog_kick_received and ((cur_time - self.time_watchdog_kick_checked) > 3):
                self.time_watchdog_kick_checked = cur_time
                # Send a watchdog error to the UI every 3 seconds if we lose connection
                if (cur_time - self.time_last_kick_received > 3):
                    self.request_queue.put({'type': proto.alarm, 'value': self.alarm_val | AlarmBits.SERIAL_TIMEOUT.value}) # Error 4: connection timeout

            time.sleep(0.2)

            # Have we received the first watchdog kick in a reasonable timeframe?
            if not self.first_watchdog_kick_received and ((cur_time - self.start_time) > 30):
                #TODO: Raise watchdog timeout alarm.
                pass

