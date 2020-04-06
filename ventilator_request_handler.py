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

class RequestHandler:
    def __init__(self, api_client, request_queue):
        self.request_queue = request_queue
        self.api_client = api_client


    def run(self, name):
        print('Starting {}'.format(name))
        while True:
            try:
                msg = self.request_queue.get(block=False)
            except queue.Empty:
                msg = None

            if msg != None:
                try:
                    if msg['type'] == 'setting':
                        self.api_client.send_setting_float(msg['key'], msg['value'])
                    elif msg['type'] == proto.alarm:
                        self.api_client.send_alarm(msg['value'])
                    elif msg['type'] == 'log':
                        self.api_client.send_log(msg['value'])
                except:
                    print("Invalid message from request")

            time.sleep(0.01)
