#!/usr/bin/python3
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


This daemon has a number of tasks
 1. Get data from serial output and store it in the database
 2. Get alarm setpoints from the UI and sound an alarm when the
    patient needs attention
 3. Ensure the Arduino is still running
"""
import threading
import queue
import time
import sys
import multiprocessing as mp
from ventilator_database import DbClient
from ventilator_serial import SerialHandler
from ventilator_websocket import WebsocketHandler
from ventilator_alarm import AlarmHandler

from ventilator_request import APIRequest
from ventilator_request_handler import RequestHandler
from ventilator_setting_handler import SettingHandler
from database_processing import DatabaseProcessing
from datetime import datetime
from ventilator_alarm import AlarmBits

def run():
    """
    Do setup and start threads
    """
    userport = None
    if len(sys.argv) > 1:
        userport = sys.argv[1]

    api_request = APIRequest("http://localhost:3001")
    api_request.send_setting("startPythonDaemon", datetime.utcnow())

    db_queue = mp.Queue() # Queue for values to write to db
    serial_output_queue = mp.Queue() # Queue for messages to send to controller
    setting_input_queue = mp.Queue()  # Queue for messages to send to controller
    alarm_input_queue = mp.Queue() # Queue for values for Alarm thread
    request_queue = mp.Queue() # Queue with the requests to be sent to the API

    manager = mp.Manager()
    settings = manager.dict()

    if userport:
        ser_handler = SerialHandler(db_queue, request_queue, serial_output_queue, alarm_input_queue, port = userport)
    else:
        ser_handler = SerialHandler(db_queue, request_queue, serial_output_queue, alarm_input_queue)

    websocket_handler = WebsocketHandler(serial_output_queue, setting_input_queue, request_queue)
    alarm_handler = AlarmHandler(alarm_input_queue,serial_output_queue, request_queue)
    request_handler = RequestHandler(api_request, request_queue)
    setting_handler = SettingHandler(setting_input_queue, request_queue, settings) #all settings comes at least from the websocket

    addr = 'mongodb://localhost:27017'
    db_handler = DbClient(db_queue, request_queue, addr)
    database_processing = DatabaseProcessing(settings, alarm_input_queue, request_queue, addr)

    # Thread that handles bidirectional communication
    ser_thread = mp.Process(target=ser_handler.run,
                                  daemon=True,
                                  args=('serial thread',))

    # Thread that handles writing measurement values to the db
    db_thread = mp.Process(target=db_handler.run,
                                 daemon=True,
                                 args=('db thread',))

    # Thread that handles bidirectional websocket communication
    websocket_thread = mp.Process(target=websocket_handler.run,
                                       daemon=True,
                                       args=('websocket thread',))

    # Thread that checks if an alarm should be raised given current measurements
    alarm_thread = mp.Process(target=alarm_handler.run,
                                    daemon=True,
                                    args=('alarm thread',))

    # Thread that sends the received values to the API
    request_thread = mp.Process(target=request_handler.run,
                                    daemon=True,
                                    args=('request thread',))

    # Thread that stores the settings
    setting_thread = mp.Process(target=setting_handler.run,
                                    daemon=True,
                                    args=('setting thread',))

    # Thread that stores the settings
    processing_thread = mp.Process(target=database_processing.run,
                                    daemon=True,
                                    args=('processing thread',))

    ser_thread.start()
    db_thread.start()
    websocket_thread.start()
    alarm_thread.start()
    request_thread.start()
    setting_thread.start()
    processing_thread.start()

    while True:
        # check if all subprocesses are running
        if (ser_thread.is_alive() == False
            or db_thread.is_alive() == False
            or websocket_thread.is_alive() == False
            or alarm_thread.is_alive() == False
            or request_thread.is_alive() == False
            or setting_thread.is_alive() == False
            or processing_thread.is_alive() == False):
                break

        time.sleep(0.1)

    print("One subprocess was terminated, killing other subprocesses")

    ser_thread.kill()
    db_thread.kill()
    websocket_thread.kill()
    alarm_thread.kill()
    request_thread.kill()
    setting_thread.kill()
    processing_thread.kill()

    # Then join before exiting
    ser_thread.join()
    db_thread.join()
    websocket_thread.join()
    alarm_thread.join()
    request_thread.join()
    setting_thread.join()
    processing_thread.join()

if __name__ == "__main__":
    run()
    print("Exiting")
