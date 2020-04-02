import queue
import time

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
                    elif msg['type'] == 'error':
                        self.api_client.send_error(msg['value'])
                except:
                    print("Invalid message from request")

            time.sleep(0.01)
