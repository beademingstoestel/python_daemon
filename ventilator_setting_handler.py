import queue
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
                        self.settings[key] = value
                        #print(self.settings)
                except:
                    print("Invalid message")
