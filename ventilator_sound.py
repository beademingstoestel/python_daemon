import time
from threading import Thread
from playsound import playsound

class SoundPlayer(Thread):


    def __init__(self, sound_path, repeats, sleep_duration):
        """
        Basic Sound Player constructor
        Args:
            sound path (string): string of the path where the sound file is located
            repeat (integer): number of repeats of this file, -1 means infinite repeat
            sleep_duration (float): time between two repeats without playing
        """
        Thread.__init__(self)
        self.sound_path = sound_path
        self.repeats = repeats
        self.sleep_duration = sleep_duration
        self.repeat_cnt = 0

    def terminate(self):
        self.running = False

    def run(self):
        self.running = True
        while self.running and (self.repeats == -1 or self.repeat_cnt <= self.repeats):
            playsound(self.sound_path)
            self.repeat_cnt += 1
            time.sleep(self.sleep_duration)