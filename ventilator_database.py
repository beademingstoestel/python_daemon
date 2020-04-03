"""
Ventilator database connection
"""
import queue
from datetime import datetime
from pymongo import MongoClient, errors


class DbClient():

    def __init__(self, db_queue, addr='mongodb://localhost:27017'):
        self.addr = addr
        self.db = None
        self.queue = db_queue

    def store_pressure(self, pressure_val):
        collection = self.db.pressure_values
        self.__store_value(collection, pressure_val)

    def store_volume(self, volume_val):
        collection = self.db.volume_values
        self.__store_value(collection, volume_val)

    def store_bpm(self, breaths_per_minute_val):
        collection = self.db.breathsperminute_values
        self.__store_value(collection, breaths_per_minute_val)

    def store_trigger(self, trigger_val):
        collection = self.db.trigger_values
        self.__store_value(collection, trigger_val)

    def store_flow(self, flow_val):
        collection = self.db.flow_values
        self.__store_value(collection, flow_val)

    def store_cpu(self, cpu_val):
        collection = self.db.cpu_values
        self.__store_value(collection, cpu_val)

    def __store_value(self, collection, val):
        try:
            collection.insert_one({'value': val, 'loggedAt': datetime.utcnow()})
        except errors.ConnectionFailure:
            print("Lost connection, client will attempt to reconnect")

    def last_n_data(self, type_data, N=12000):
        # retrieve the last "N" added measurement N = 12000 data recorded at 200Hz
        if type_data == 'BPM':
            return self.db.breathsperminute_values.find().sort("loggedAt", -1).limit(N)
        elif type_data == 'VOL':
            return self.db.volume_values.find().find().sort("loggedAt", -1).limit(N)
        elif type_data == 'TRIG':
            return self.db.trigger_values.find().sort("loggedAt", -1).limit(N)
        elif type_data == 'PRES':
            return self.db.pressure_values.find().sort("loggedAt", -1).limit(N)
        else:
            print("[ERROR] value type not recognized use: BPM, VOL, TRIG, or PRES")
            return None

    def last_n_values(self, type_data, N=12000):
        # retrieve the last "N" added measurement N = 12000 data recorded at 200Hz
        collection, data_raw, values, timestamp = [], None, [], []
        if type_data == 'BPM':
            collection = self.db.breathsperminute_values
        elif type_data == 'VOL':
            collection = self.db.volume_values
        elif type_data == 'TRIG':
            collection = self.db.trigger_values
        elif type_data == 'PRES':
            collection = self.db.pressure_values
        else:
            print("[ERROR] value type not recognized use: BPM, VOL, TRIG, or PRES")
            return None, None
        # send back data raw format + time stamp
        data_raw = collection.find().sort("loggedAt", -1).limit(N)
        for x in (collection.find({},{ "loggedAt": 0 ,"_id": 0})).sort("loggedAt", -1).limit(100): 
            values.append(x.get('value'))

        return data_raw, values
        
    def run(self, name):
        print("Starting {}".format(name))
        # Only start MongoClient after fork()
        try:
            self.client = MongoClient(self.addr)
        except errors.ConnectionFailure:
            print("Unable to connect, client will attempt to reconnect")
        
        self.db = self.client.beademing
        while True:
            try:
                msg = self.queue.get()
            except queue.Empty:
                continue
            try:
                if msg['type'] == 'BPM':
                    self.store_bpm(msg['val'])
                elif msg['type'] == 'VOL':
                    self.store_volume(msg['val'])
                elif msg['type'] == 'TRIG':
                    self.store_trigger(msg['val'])
                elif msg['type'] == 'PRES':
                    self.store_pressure(msg['val'])
                elif msg['type'] == 'FLOW':
                    self.store_flow(msg['val'])
                elif msg['type'] == 'CPU':
                    self.store_cpu(msg['val'])
            except:
                print("Invalid message from database")

