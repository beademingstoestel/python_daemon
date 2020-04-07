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
import time
from threading import Thread
import sounddevice as sd
import soundfile as sf
import ventilator_log as log

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
            try:
                data, fs = sf.read(self.sound_path, dtype='float32')
                sd.play(data, fs)
                status = sd.wait()
                if status:
                    print('Error during playback: ' + str(status))
            except Exception as e:
                print('Exception occurred during playback {}'.format(e))
            self.repeat_cnt += 1
            time.sleep(self.sleep_duration)
