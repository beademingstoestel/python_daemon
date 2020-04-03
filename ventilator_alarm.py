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

        self.time_last_kick_sent = 0
        self.time_last_kick_received = 0

        self.time_watchdog_kick_checked = 0

        self.first_watchdog_kick_received = False
        self.start_time = 0

    def run(self, name):
        print("Starting {}".format(name))
        self.start_time = time.time()
        while True:
            cur_time = time.time()
            # Do we need to kick the watchdog? Only after we've received the first kick
            if self.first_watchdog_kick_received and ((cur_time - self.time_last_kick_sent) > 1 ):
                self.serial_queue.put({'type': proto.alarm, 'val': self.alarm_val})
                self.time_last_kick_sent = cur_time

            try:
                msg = self.input_queue.get(block=False)
            except queue.Empty:
                msg = None

            if msg != None:
                if msg['type'] == "ALARM":
                    self.time_last_kick_received == cur_time
                    if not self.first_watchdog_kick_received:
                        self.first_watchdog_kick_received = True
                    if msg['val'] != 0:
                        self.request_queue.put({'type': 'error', 'value': msg['val']})

            # Have we received a watchdog kick in time?
            if self.first_watchdog_kick_received and ((cur_time - self.time_watchdog_kick_checked) > 3):
                self.time_watchdog_kick_checked = cur_time
                # Send a watchdog error to the UI every 3 seconds if we lose connection
                if (cur_time - self.time_last_kick_received > 3):
                    self.request_queue.put({'type': 'error', 'value': 4}) # Error 4: connection timeout

            time.sleep(0.2)

            # Have we received the first watchdog kick in a reasonable timeframe?
            if not self.first_watchdog_kick_received and ((cur_time - self.start_time) > 30):
                #TODO: Raise watchdog timeout alarm.
                pass

