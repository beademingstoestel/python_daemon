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

Store the received settings in a dictionary

"""
import queue
import time
import ventilator_protocol as proto
import ventilator_log as log


class SettingHandler:
    def __init__(self, serial_queue, request_queue, settings):
        self.serial_queue = serial_queue
        self.settings = settings
        self.request_queue = request_queue
        self.counter = 0

        #add default settings for testing only: don't commit

        self.settings['RR'] = 0.0  # Respiratory rate
        self.settings['VT'] = 0.0  # Tidal Volume
        self.settings['PK'] = 0.0  # Peak Pressure
        self.settings['TS'] = 0.0  # Breath Trigger Threshold
        self.settings['IE'] = 0.0  # Inspiration/Expiration (N for 1/N)
        self.settings['PP'] = 0.0  # PEEP (positive end expiratory pressure)
        self.settings['ADPK'] = 0.0  # Allowed deviation Peak Pressure
        self.settings['ADVT'] = 0.0  # Allowed deviation Tidal Volume
        self.settings['ADPP'] = 0.0  # Allowed deviation PEEP
        self.settings['MODE'] = 0.0  # Machine Mode (Volume Control / Pressure Control)
        self.settings['ACTIVE'] = 0.0  # Machine on / off
        self.settings['PS'] = 0.0  # support pressure
        self.settings['RP'] = 0.0  # ramp time
        self.settings['TP'] = 0.0  # trigger pressure
        self.settings['MT'] = 0.0  # mute
        self.settings['FW'] = 0.0  # firmware version

    def run(self, name):
        print('Starting {}'.format(name))
        log.INFO(__name__, self.request_queue, "Starting {}".format(name))
        while True:
            if (self.counter % 100) == 0:
                begin = int(round(time.time() * 1000))
                print("settings time {}".format(begin))
                log.INFO(__name__, self.request_queue, "settings counter {} time {}".format(self.counter, begin))
            self.counter = self.counter + 1
            try:
                msg = self.serial_queue.get(block=False)
            except queue.Empty:
                msg = None

            if msg != None:
                try:
                    key = msg['type']
                    value = msg['val']
                    if key in proto.settings:
                        self.settings[key] = float(value)
                        print(self.settings)
                except Exception as e:
                    print("Invalid message {}".format(msg))
                    log.ERROR(__name__, self.request_queue, "Invalid message {}".format(msg))
                    print(e)
                    log.ERROR(__name__, self.request_queue, "Exception occurred {}".format(e))

            time.sleep(0.2)
