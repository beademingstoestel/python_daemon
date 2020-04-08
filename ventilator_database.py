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

Ventilator database connection
"""
import queue
from datetime import datetime
from pymongo import MongoClient, errors
import ventilator_log as log
import time

class DbClient():

    def __init__(self, db_queue, request_queue, addr='mongodb://localhost:27017'):
        self.addr = addr
        self.db = None
        self.queue = db_queue
        self.request_queue = request_queue
        self.counter = 0

    def store_pressure(self, msg):
        collection = self.db.pressure_values
        print('store_pressure')
        self.__store_value(collection, msg)
        
    def store_target_pressure(self, msg):
        collection = self.db.targetpressure_values
        print('store_target_pressure')
        self.__store_value(collection, msg)

    def store_volume(self, msg):
        collection = self.db.volume_values
        self.__store_value(collection, msg)

    def store_bpm(self, msg):
        collection = self.db.breathsperminute_values
        self.__store_value(collection, msg)

    def store_trigger(self, msg):
        collection = self.db.trigger_values
        self.__store_value(collection, msg)

    def store_flow(self, msg):
        collection = self.db.flow_values
        self.__store_value(collection, msg)

    def store_cpu(self, msg):
        collection = self.db.cpu_values
        self.__store_value(collection, msg)

    def __store_value(self, collection, msg):
        try:
            collection.insert_one({'value': msg['val'], 'loggedAt': msg['loggedAt']})
        except errors.ConnectionFailure:
            print("Lost connection, client will attempt to reconnect")

    def run(self, name):
        print("Starting {}".format(name))
        log.INFO(__name__, self.request_queue, "Starting {}".format(name))

        # Only start MongoClient after fork()
        # and each child process should have its own instance of the client
        try:
            self.client = MongoClient(self.addr)
        except errors.ConnectionFailure:
            print("Unable to connect, client will attempt to reconnect")
            log.ERROR(__name__, self.request_queue, "Unable to connect, client will attempt to reconnect")

        self.db = self.client.beademing

        while True:
            if (self.counter % 100) == 0:
                begin = int(round(time.time() * 1000))
                print("database time {}".format(begin))
                log.INFO(__name__, self.request_queue, "database counter {} time {}".format(self.counter, begin))
            self.counter = self.counter + 1

            try:
                msg = self.queue.get()
            except queue.Empty:
                msg = None
            if msg != None:		
                try:
                    if msg['type'] == 'BPM':
                        self.store_bpm(msg)
                    elif msg['type'] == 'VOL':
                        self.store_volume(msg)
                    elif msg['type'] == 'TRIG':
                        self.store_trigger(msg)
                    elif msg['type'] == 'PRES':
                        self.store_pressure(msg)
                    elif msg['type'] == 'TPRES':
                        self.store_target_pressure(msg)
                    elif msg['type'] == 'FLOW':
                        self.store_flow(msg)
                    elif msg['type'] == 'CPU':
                        self.store_cpu(msg)
                except Exception as e:
                    print("Invalid message from database = {}".format(msg))
                    log.ERROR(__name__, self.request_queue, "Invalid message from database = {}".format(msg))
                    print(e)
                    log.ERROR(__name__, self.request_queue, "Exception occurred {}".format(e))
