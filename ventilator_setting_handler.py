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

#store the recevied settings in a dictonary

class SettingHandler:
    def __init__(self, serial_queue, settings):
        self.serial_queue = serial_queue
        self.settings = settings

    def run(self, name):
        print('Starting {}'.format(name))
        while True:
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
                except:
                    print("Invalid message")

            time.sleep(0.2)
