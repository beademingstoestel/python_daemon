# import necessary packages
import numpy as np
from datetime import datetime, date, time, timedelta
import scipy.signal as signal
from scipy.signal import find_peaks
import scipy


"""
This class get the recorded pressure values and compute the following:
    1) OK -- measure the BPM effective
    2) OK -- check pressure tracking performance 
        * look for average absolute tracking errors over past cycle that are larger than ?% (of setpoint?))
    3) if pressure > setting*110% (or value from fagg), trigger a warning
    4) OK -- proposal: if inhale/exhale state do not change for more than 10s.
        * the function returns all the dt_inhale and dt_exhale
        * check if in the arrray a values is > 10
    5) OK -- ratio between inhale exhale in the database if it is above threshold 
    6) detect when the pressure goes below a "peep" threshold TBD during the exhale period T
"""

class PressureMonitor():
    def __init__(self, raw_data, median_kernel_size=11):
        super().__init__()
        # raw_data from the Mongo database
        self.pvalues, self.timestamp = [], []
        # send back data raw format + time stamp
        for x in (raw_data.find({},{ "_id": 0})): 
            self.pvalues.append(float(x.get('value')))
            full_time = x.get('loggedAt')
            tmp = (float(full_time.time().hour)*3600+float(full_time.time().minute)*60+float(full_time.time().second))*1e3+float(full_time.time().microsecond)/1e3
            self.timestamp.append(tmp)

        # reverse the order of the element because they are retrieved 
        # in reverse order from the Mongo database
        self.timestamp.reverse()
        self.pvalues.reverse()

        # median fileter size = median_kernel_size
        # https://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.signal.medfilt.html
        pvalues_filtered = signal.medfilt(self.pvalues, median_kernel_size)
        # compute the first derivative of the pressure signal
        d_pressure = np.zeros(pvalues_filtered.shape, np.float)
        d_pressure[0:-1] = np.diff(pvalues_filtered)/(np.diff(self.timestamp)*1e-3)
        d_pressure[-1] = (pvalues_filtered[-1] - pvalues_filtered[-2])/(self.timestamp[-1] - self.timestamp[-2])
        
        # https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.find_peaks.html
        # TODO h/d value could be defined as parameters to read from the config file!!
        # TODO check if we need to adjust the distance between peaks
        # get the location of rising edge
        self.ppeaks = self.find_peaks_signal(d_pressure, +1, h=100, d=50)
        # get the location of falling edge
        self.npeaks = self.find_peaks_signal(d_pressure, -1, h=100, d=50)

        # keep only complete breathing cycles
        # should start with peak_positive and end with one as well
        start_pp = self.ppeaks[0]
        end_pp   = self.ppeaks[-1]
        # keep the falling edge in between
        npeaks = npeaks[self.npeaks>start_pp]
        npeaks = npeaks[self.npeaks<end_pp]

    def get_nbr_bpm(self): 
        # number of breathing cycle
        # -1 : to garantee that we have a complete one at the end
        number_of_breathing_cycle = len(self.ppeaks)-1
        print("[INFO] The nummber of breathing cycle = {}".format(number_of_breathing_cycle))
        # time of all the breathing cycles loaded from the mongo database (seconds)
        dtime_all_breathing_cycle = np.diff(self.timestamp[self.ppeaks])*1e-3
        print("[INFO] Time (in seconds) of all the breathing cycles loaded from the mongo database: {}".foramt(dtime_all_breathing_cycle))
        # average time of the last # breathing cycle (Ti + Te)
        average_dtime_breathing_cycle = np.mean(dtime_all_breathing_cycle)
        print("[INFO] Average time of the last # breathing cycle (Ti + Te) = {} seconds".format(average_dtime_breathing_cycle))
        # compute the BPM from the data analyzed
        total_time_seconds = (self.timestamp[self.ppeaks[-1]]-self.timestamp[self.ppeaks[0]])/1e3
        breathing_cycle_per_minute = 60*number_of_breathing_cycle/total_time_seconds
        #  send back the following values!
        return breathing_cycle_per_minute, number_of_breathing_cycle, average_dtime_breathing_cycle

    # TODO check from where we can get the values of the used threshold in this function
    def analyze_inhale_exhale_time(self, threshold_ratio_ie=2.75, threshold_dt_ie=10):
        # combine both list of peaks to measure the Ti and Te
        all_peaks = np.concatenate((self.ppeaks, self.npeaks), axis=0)
        all_peaks = np.sort(all_peaks)
        dtime_inhale_exhale = np.diff(self.timestamp[all_peaks])*1e-3
        # extract the dt for inhale and exhale
        dtime_inhale = dtime_inhale_exhale[0::2]
        dtime_exhale = dtime_inhale_exhale[1::2]
        # compute the ratio exhale/inhale ~ 3 
        ratio_exhale_inhale = dtime_exhale/dtime_inhale
        # nbr of time the ratio is below the predfined threshold
        nbr_ratio_below_threshold = sum(float(num) <= threshold_ratio_ie for num in ratio_exhale_inhale)
        # nbr of time inhale or exhale duration is above the the threshold dt
        nbr_dtinhale_above_threshold = sum(float(num) >= threshold_dt_ie for num in dtime_inhale)
        nbr_dtexhale_above_threshold = sum(float(num) >= threshold_dt_ie for num in dtime_exhale)
        # return 
        return nbr_ratio_below_threshold, nbr_dtinhale_above_threshold, nbr_dtexhale_above_threshold

    def find_peaks_signal(self, signal_x, sign=1, h=100, d=50):
        if abs(sign) == 1:
            peaks, _ = find_peaks(sign*signal_x, height=h, distance=d)
        else:
            peaks = None
            print("[WARNING] sign should be either +1 or -1")
        # send back teh peaks found
        return peaks













