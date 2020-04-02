# import necessary packages
import numpy as np
from datetime import datetime, date, time, timedelta
import scipy.signal as signal
from scipy.signal import find_peaks
import scipy

"""
process data recorded onthe data base 
and send alarm signal when an anomaly is detected
"""
class PressureMonitor():
    """
    This class get the recorded pressure values and compute the following:
        1) check pressure tracking performance 
            * look for average absolute tracking errors over past cycle that are larger than ?% (of setpoint?))
        2) if pressure > setting*110% (or value from fagg), trigger a warning
        3) proposal: if inhale/exhale state do not change for more than 10s.
    """
    # TODO define
    def __init__(self, pressure_values):
        super().__init__()
        # update very 40ms
        
    def get_nbr_bpm(self, raw_data, median_kernel_size=11):
        # raw_data from the Mongo database
        pvalues, timestamp = [], []
        
        # send back data raw format + time stamp
        for x in (raw_data.find({},{ "_id": 0})): 
            pvalues.append(float(x.get('value')))
            full_time = x.get('loggedAt')
            tmp = (float(full_time.time().hour)*3600+float(full_time.time().minute)*60+float(full_time.time().second))*1e3+float(full_time.time().microsecond)/1e3
            timestamp.append(tmp)
        
        # reverse the order of the element because they are retrieved 
        # in reverse order from the Mongo database
        timestamp.reverse()
        pvalues.reverse()

        # median fileter size = median_kernel_size
        pvalues_filtered = signal.medfilt(pvalues, median_kernel_size)
        # compute the first derivative of the pressure signal
        d_pressure = np.zeros(pvalues_filtered.shape, np.float)
        d_pressure[0:-1] = np.diff(pvalues_filtered)/(np.diff(timestamp)*1e-3)
        d_pressure[-1] = (pvalues_filtered[-1] - pvalues_filtered[-2])/(timestamp[-1] - timestamp[-2])
        
        # https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.find_peaks.html
        # TODO height value to set from the config
        # TODO check if we need to adjust the distance between peaks
        peaks, _ = find_peaks(d_pressure, height=100, distance=50)
        # number of breathing cycle
        number_of_breathing_cycle = len(peaks)
        print("[INFO] The nummber of breathing cycle = {}".format(number_of_breathing_cycle))
        # time of all the breathing cycles loaded from the mongo database
        time_all_breathing_cycle = np.diff(timestamp[peaks])*1e-3
        print("[INFO] Time (in seconds) of all the breathing cycles loaded from the mongo database: {}".foramt(time_all_breathing_cycle))
        # average time of the last # breathing cycle (Ti + Te)
        average_time_breathing_cycle = np.mean(time_all_breathing_cycle)
        print("[INFO] Average time of the last # breathing cycle (Ti + Te) = {} seconds".format(average_time_breathing_cycle))
        
        # compute the BPM from the data analyzed
        total_time_seconds = (timestamp[-1]-timestamp[0])/1e3
        breathing_cycle_per_minute = 60*number_of_breathing_cycle/total_time_seconds
        
        #  send back the following values!
        return number_of_breathing_cycle, average_time_breathing_cycle, breathing_cycle_per_minute


    # TODO
    def run(self):
        return -1














