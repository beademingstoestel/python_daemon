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
import ventilator_log as log #don't call here the log functions but use rather directly the send_log functions

class RequestHandler:
    def __init__(self, api_client, request_queue):
        self.request_queue = request_queue
        self.api_client = api_client
        self.counter = 0


    def run(self, name):
        print('Starting {}'.format(name))
        self.api_client.send_log(log.LOG_LEVEL_INFO, '{} - Starting {}'.format(__name__, name))
        while True:
            if (self.counter % 100) == 0:
                begin = int(round(time.time() * 1000))
                print("request handler time {}".format(begin))
                log.INFO(__name__, self.request_queue, "request handler counter {} time {}".format(self.counter, begin))
            self.counter = self.counter + 1

            try:
                msg = self.request_queue.get(block=False)
            except queue.Empty:
                msg = None

            if msg != None:
                try:
                    if msg['type'] == proto.setting:
                        self.api_client.send_setting_float(msg['key'], msg['value'])
                    elif msg['type'] == proto.alarm:
                        self.api_client.send_alarm(msg['value'])
                    elif msg['type'] == proto.log:
                        self.api_client.send_log(msg['severity'], msg['value'])
                except Exception as e:
                    print("Invalid message from request = {}".format(msg))
                    self.api_client.send_log(log.LOG_LEVEL_ERROR, "{} - Invalid message from request = {}".format(__name__, msg))
                    print(e)
                    self.api_client.send_log(log.LOG_LEVEL_ERROR, "{} - Exception occurred {}".format(__name__, e))

            time.sleep(0.01)
