# import necessary packages
import numpy as np
from datetime import datetime, date, time, timedelta
import scipy.signal as signal
from scipy.signal import find_peaks
import scipy
import time
from pymongo import MongoClient, errors

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

Notes
    * param defaults values were chosen when anyalisng the data recorded on the 02/04/2020
"""


class PressureMonitor:
    def __init__(self, raw_data, median_kernel_size=11):
        super().__init__()
        # raw_data from the Mongo database
        self.pvalues, self.timestamp = [], []
        # send back data raw format + time stamp
        for x in (raw_data):
            self.pvalues.append(float(x.get('value')))
            full_time = x.get('loggedAt')
            tmp = (float(full_time.time().hour) * 3600 + float(full_time.time().minute) * 60 + float(
                full_time.time().second)) * 1e3 + float(full_time.time().microsecond) / 1e3
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
        d_pressure[0:-1] = np.diff(pvalues_filtered) / (np.diff(self.timestamp) * 1e-3)
        d_pressure[-1] = (pvalues_filtered[-1] - pvalues_filtered[-2]) / (self.timestamp[-1] - self.timestamp[-2])

        # https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.find_peaks.html
        # TODO h/d value could be defined as parameters to read from the config file!!
        # TODO check if we need to adjust the distance between peaks
        # get the location of rising edge
        self.ppeaks = self.find_peaks_signal(d_pressure, +1, h=100, d=50)
        # get the location of falling edge
        self.npeaks = self.find_peaks_signal(d_pressure, -1, h=100, d=50)

        # keep only complete breathing cycles
        # should start with peak_positive and end with one as well
        try:
            start_pp = self.ppeaks[0]
            end_pp = self.ppeaks[-1]
            # keep the falling edge in between
            self.npeaks = self.npeaks[self.npeaks > start_pp]
            self.npeaks = self.npeaks[self.npeaks < end_pp]
        except:
            raise Exception('no valid data or peaks detected')

    # get the BPM
    def get_nbr_bpm(self):
        # number of breathing cycle
        # -1 : to garantee that we have a complete one at the end
        number_of_breathing_cycle = len(self.ppeaks) - 1
        print("[INFO] The number of breathing cycle = {}".format(number_of_breathing_cycle))
        # time of all the breathing cycles loaded from the mongo database (seconds)
        dtime_all_breathing_cycle =  np.diff(np.array(self.timestamp)[self.ppeaks.astype(int)]) * 1e-3 # np.diff(self.timestamp[self.ppeaks]) * 1e-3
        print("[INFO] Time (in seconds) of all the breathing cycles loaded from the mongo database: {}".format(dtime_all_breathing_cycle))
        # average time of the last # breathing cycle (Ti + Te)
        average_dtime_breathing_cycle = np.mean(dtime_all_breathing_cycle)
        print("[INFO] Average time of the last # breathing cycle (Ti + Te) = {} seconds".format(average_dtime_breathing_cycle))
        # compute the BPM from the data analyzed
        total_time_seconds = (self.timestamp[self.ppeaks[-1]] - self.timestamp[self.ppeaks[0]]) / 1e3
        breathing_cycle_per_minute = 60 * number_of_breathing_cycle / total_time_seconds
        #  send back the following values!
        return breathing_cycle_per_minute, number_of_breathing_cycle, average_dtime_breathing_cycle

    # TODO check from where we can get the values of the threshold for this function
    def analyze_inhale_exhale_time(self, threshold_ratio_ie=2.75, threshold_dt_ie=10):
        # combine both list of peaks to measure the Ti and Te
        all_peaks = np.concatenate((self.ppeaks, self.npeaks), axis=0)
        all_peaks = np.sort(all_peaks)
        dtime_inhale_exhale = np.diff(np.array(self.timestamp)[self.ppeaks.astype(int)]) * 1e-3 # np.diff(self.timestamp[all_peaks]) * 1e-3
        # extract the dt for inhale and exhale
        dtime_inhale = dtime_inhale_exhale[0::2]
        dtime_exhale = dtime_inhale_exhale[1::2]
        # compute the ratio exhale/inhale ~ 3 
        ratio_exhale_inhale = dtime_exhale / dtime_inhale
        # nbr of time the ratio is below the predfined threshold
        nbr_ratio_below_threshold = sum(float(num) <= threshold_ratio_ie for num in ratio_exhale_inhale)
        # nbr of time inhale or exhale duration is above the the threshold dt
        nbr_dtinhale_above_threshold = sum(float(num) >= threshold_dt_ie for num in dtime_inhale)
        nbr_dtexhale_above_threshold = sum(float(num) >= threshold_dt_ie for num in dtime_exhale)
        # return 
        return nbr_ratio_below_threshold, nbr_dtinhale_above_threshold, nbr_dtexhale_above_threshold

    # TODO check from where we can get the values of the desired pressure, threshold and nbr data point for this function
    # nbr_data_point from falling edge and going back
    def check_pressure_tracking_performance(self, pressure_desired=51, threshold_dp=3, nbr_data_point=5):
        """
        data are saved every ~ 0.05s as an inhale state in average ~ 0.75second (from recorded data 02/04/2020)
        we will have during this period 15 data points
        for the  different dp = pressure_desired - pressre_measured 
        we will check the 5 data points before the falling edge
        """
        # measure the absolute differentce to the desired pressure and then take the average of the n measures
        dp_list = []
        for indice_bc in self.npeaks:
            dp = abs(np.array(self.pvalues[indice_bc - nbr_data_point:indice_bc]) - pressure_desired)
            dp_list.append(np.mean(dp))
        # nbr of time inhale or exhale duration is above the the threshold dt
        nbr_dp_above_threshold = sum(float(num) >= threshold_dp for num in dp_list)
        # return
        return nbr_dp_above_threshold, dp_list

    # TODO check from where we can get the values of the peep_value, threshold and nbr data point for this function
    # nbr_data_point from rising edge and going back
    def detect_pressure_below_peep(self, peep_value=10, threshold_dp_peep=5, nbr_data_point=35):
        """
        data are saved every ~0.05s as an exhale state in average ~ 2.50second (from recorded data 02/04/2020)
        we will have during this period 50 data points
        for the peep below  
        we will check the 35 data points before the rising edge
        """
        # measure the absolute differentce to the desired pressure and then take the average of the n measure
        below_peep_list = []
        for indice_bc in self.ppeaks[1:]:
            dp = np.array(self.pvalues[indice_bc - nbr_data_point:indice_bc]) - peep_value
            dp[dp > 0] = 0
            below_peep_list.append(abs(min(dp)))
        # nbr of time inhale or exhale duration is above the the threshold dt
        nbr_dp_peep_above_threshold = sum(float(num) >= threshold_dp_peep for num in below_peep_list)
        # return
        return nbr_dp_peep_above_threshold, below_peep_list

    # TODO check from where we can get the values of the pressure_desired, threshold and nbr data point for this function
    # nbr_data_from rising edge and going forward
    def pressure_peak_overshoot(self, pressure_desired=51, threshold_dp_overshoot=3, nbr_data_point=10):
        """
        data are saved every ~ 0.05s as an inhale state in average ~ 0.75second (from recorded data 02/04/2020)
        we will have during this period 15 data points
        for the  overshoot dp = pressure_desired - pressre_measured 
        we will check the 10 data points after the rising edge
        """
        overshoot_pressure_list = []
        for indice_bc in self.ppeaks:
            dp = np.array(self.pvalues[indice_bc:indice_bc + nbr_data_point]) - pressure_desired
            overshoot_pressure_list.append((max(dp)))
        # nbr of time inhale or exhale duration is above the the threshold dt
        nbr_pressure_overshoot_above_threshold = sum(
            float(num) >= threshold_dp_overshoot for num in overshoot_pressure_list)
        # return
        return nbr_pressure_overshoot_above_threshold, overshoot_pressure_list

    # TODO check from where we can get the values of the pressure_desired, threshold and nbr data point for this function
    # nbr_data_from rising edge and going forward
    def pressure_peak_too_low_high(self, pressure_desired=51, threshold_dp=3):
        # find all the peaks of the loaded breathing cycles
        ind_max_pressure = find_peaks_signal(self.pvalues, 1, h=pressure_desired-threshold_dp, d=50)
        if len(ind_max_pressure)==0:
            # No peak was found in the data loaded -->> rise an alarm !!!
            dpressure_deviate_list = None
            nbr_pressure_deviate = 100
        else:
            dpressure_deviate_list = abs(np.array(self.pvalues)[ind_max_pressure.astype(int)]-pressure_desired)
            # nbr of time inhale or exhale duration is above the the threshold dt
            nbr_pressure_deviate = sum(float(num) >= threshold_dp for num in dpressure_deviate_list)
        # return
        return nbr_pressure_deviate, dpressure_deviate_list

    # find all the peaks that are in the signal
    def find_peaks_signal(self, signal_x, sign=1, h=100, d=50):
        if abs(sign) == 1:
            peaks, _ = find_peaks(sign * signal_x, height=h, distance=d)
        else:
            peaks = None
            print("[WARNING] sign should be either +1 or -1")
        # send back teh peaks found
        return peaks


class VolumeMonitor:
    def __init__(self, raw_data, median_kernel_size=11):
        super().__init__()
        # raw_data from the Mongo database
        self.vvalues, self.timestamp = [], []
        # send back data raw format + time stamp
        for x in (raw_data):
            self.vvalues.append(float(x.get('value')))
            full_time = x.get('loggedAt')
            tmp = (float(full_time.time().hour) * 3600 + float(full_time.time().minute) * 60 + float(
                full_time.time().second)) * 1e3 + float(full_time.time().microsecond) / 1e3
            self.timestamp.append(tmp)

        # reverse the order of the element because they are retrieved 
        # in reverse order from the Mongo database
        self.timestamp.reverse()
        self.vvalues.reverse()

        # median fileter size = median_kernel_size
        # https://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.signal.medfilt.html
        vvalues_filtered = signal.medfilt(self.vvalues, median_kernel_size)
        # compute the first derivative of the pressure signal
        d_volume = np.zeros(vvalues_filtered.shape, np.float)
        d_volume[0:-1] = np.diff(vvalues_filtered) / (np.diff(self.timestamp) * 1e-3)
        d_volume[-1] = (vvalues_filtered[-1] - vvalues_filtered[-2]) / (self.timestamp[-1] - self.timestamp[-2])

        # https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.find_peaks.html
        # TODO h/d value could be defined as parameters to read from the config file!!
        # TODO check if we need to adjust the distance between peaks
        # get the location of rising edge
        self.ppeaks = self.find_peaks_signal(d_volume, +1, h=100, d=50)
        # get the location of falling edge
        self.npeaks = self.find_peaks_signal(d_volume, -1, h=100, d=50)

        # keep only complete breathing cycles
        # should start with peak_positive and end with one as well
        try:
            start_pp = self.ppeaks[0]
            end_pp = self.ppeaks[-1]
            # keep the falling edge in between
            self.npeaks = self.npeaks[self.npeaks > start_pp]
            self.npeaks = self.npeaks[self.npeaks < end_pp]
        except:
            raise Exception('no valid data or peaks detected')

    # TODO check from where we can get the values of the pressure_desired, threshold and nbr data point for this function
    # nbr_data_from rising edge and going forward
    def volume_peak_too_low_high(self, volume_desired=150, threshold_dv=30):
        # find all the peaks of the loaded breathing cycles
        ind_max_volume = find_peaks_signal(self.vvalues, 1, h=volume_desired-threshold_dv, d=50)
        if len(ind_max_volume)==0:
            # No peak was found in the data loaded -->> rise an alarm !!!
            dvolume_deviate_list = None
            nbr_volume_deviate = 100
        else:
            dvolume_deviate_list = abs(np.array(self.vvalues)[ind_max_volume.astype(int)]-volume_desired)
            # nbr of time inhale or exhale duration is above the the threshold dt
            nbr_volume_deviate = sum(float(num) >= threshold_dv for num in dvolume_deviate_list)
        # return
        return nbr_volume_deviate, dvolume_deviate_list

    # find all the peaks that are in the signal
    def find_peaks_signal(self, signal_x, sign=1, h=100, d=50):
        if abs(sign) == 1:
            peaks, _ = find_peaks(sign * signal_x, height=h, distance=d)
        else:
            peaks = None
            print("[WARNING] sign should be either +1 or -1")
        # send back teh peaks found
        return peaks

class DatabaseProcessing:
    def __init__(self, settings, alarm_queue, addr='mongodb://localhost:27017'):
        self.settings = settings
        self.addr = addr
        self.alarm_queue = alarm_queue
        self.alarm_bits = 0
        self.db = None

    def last_n_data(self, type_data, N=1000):
        # retrieve the last "N" added measurement N = 1000 data recorded at 200Hz
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

    def last_n_values(self, type_data, N=1000):
        # retrieve the last "N" added measurement N = 1000 data recorded at 200Hz
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
        for x in (collection.find({}, {"loggedAt": 0, "_id": 0})).sort("loggedAt", -1).limit(100):
            values.append(x.get('value'))

        return data_raw, values

    def run(self, name):
        print("Starting {}".format(name))

        # Only start MongoClient after fork()
        # and each child process should have its own instance of the client
        try:
            self.client = MongoClient(self.addr)
        except errors.ConnectionFailure:
            print("Unable to connect, client will attempt to reconnect")

        self.db = self.client.beademing

        # TODO add a playing sound when a warning is valide !!!!

        while True:
            try:
                data_p = self.last_n_data('PRES')
                pressure_monitor = PressureMonitor(data_p)
                data_v = self.last_n_data('VOL') 
                volume_monitor   = VolumeMonitor(data_v)

                # BT: Below Threshold
                # AT Above Threshold

                # BPM - # Respiratory rate (RR) 
                breathing_cycle_per_minute, number_of_breathing_cycle, average_dtime_breathing_cycle = pressure_monitor.get_nbr_bpm()
                if breathing_cycle_per_minute > self.settings['RR']:
                    print("[INFO] Breathing at %d per minute" % breathing_cycle_per_minute)
                    self.alarm_bits = self.alarm_bits | int('00000001', 2)  # frank will define these bits, example for now 8-bit
                    self.alarm_queue.put({'type': 'error', 'val': self.alarm_bits})
                
                # performance 'IE',   # Inspiration/Expiration (N for 1/N)
                # in the function definition used as default values 2.75 / 10 seconds
                nbr_ratio_BT, nbr_dtinhale_AT, nbr_dtexhale_AT = pressure_monitor.analyze_inhale_exhale_time(threshold_ratio_ie=self.settings['IE'],
                                                                                                            threshold_dt_ie=10)
                if nbr_ratio_BT > 0:
                    print("[INFO] Respiratory rate above threshold : {} ".format(nbr_ratio_BT))
                    self.alarm_bits = self.alarm_bits | int('00000010', 2)  # frank will define these bits, example for now 8-bit
                    self.alarm_queue.put({'type': 'error', 'val': self.alarm_bits})

                if nbr_dtinhale_AT > 0 or nbr_dtexhale_AT > 0:
                    print("[INFO] Breath Trigger Threshold (TS) : {} ".format(nbr_dtinhale_AT))
                    self.alarm_bits = self.alarm_bits | int('00000100', 2)  # frank will define these bits, example for now 8-bit
                    self.alarm_queue.put({'type': 'error', 'val': self.alarm_bits})
                
                # Pressure performance Tracking -- Peak Pressure PK and ADPP Allowed deviation Peak Pressure
                # in the function defaults values are 51 / 3 / (5 for data points)
                nbr_dp_AT, dp_list = pressure_monitor.check_pressure_tracking_performance(pressure_desired=self.settings['PK'],
                                                                        threshold_dp=self.settings['ADPK'],
                                                                        nbr_data_point=5)
                if nbr_dp_AT > 3:
                    print("[INFO] Pressure tracking performance deviate {}".format(nbr_dp_AT))
                    self.alarm_bits = self.alarm_bits | int('00001000', 2)  # frank will define these bits, example for now 8-bit
                    self.alarm_queue.put({'type': 'error', 'val': self.alarm_bits})

                # Pressure below peep value 'PP', # PEEP (positive end expiratory pressure) # 'ADPP', # Allowed deviation PEEP
                # in the function defaults values are 10/5
                nbr_dp_peep_AT, below_peep_list = pressure_monitor.detect_pressure_below_peep(peep_value=self.settings['PP'],
                                                                                            threshold_dp_peep=self.settings['ADPP'],
                                                                                            nbr_data_point=35)
                if nbr_dp_peep_AT > 0:
                    print("[INFO] Pressure below peep level detected {}".format(nbr_dp_peep_AT))
                    self.alarm_bits = self.alarm_bits | int('00010000', 2)  # frank will define these bits, example for now 8-bit
                    self.alarm_queue.put({'type': 'error', 'val': self.alarm_bits})

                
                """
                # TODO confirm that this function is no longer needed
                # Detect when the pressure_peak_overshoot
                # default values are 51 / 3
                nbr_pressure_overshoot_AT, overshoot_pressure_list = pressure_monitor.pressure_peak_overshoot(pressure_desired=self.settings['PK'],
                                                                                                            threshold_dp_overshoot=self.settings['ADPK'],
                                                                                                            nbr_data_point=10)
                if nbr_pressure_overshoot_AT > 0:
                    print("[INFO] Pressure peak overshoot {}".format(nbr_pressure_overshoot_AT))
                    self.alarm_bits = self.alarm_bits | int('00100000', 2)  # frank will define these bits, example for now 8-bit
                    self.alarm_queue.put({'type': 'error', 'val': self.alarm_bits})
                """
                # Detect when the pressure peak of breathing cycle is not in the range defined 
                # default values are PK = 51 / ADPK = 3
                nbr_pressure_AT_BT, deviate_pressure_list = pressure_monitor.pressure_peak_too_low_high(pressure_desired=self.settings['PK'], 
                                                                                                    threshold_dp=self.settings['ADPK'])
                if nbr_pressure_AT_BT > 0:
                    print("[INFO] Pressure outside the allowed range {}".format(nbr_pressure_AT_BT))
                    self.alarm_bits = self.alarm_bits | int('00100000', 2)  # frank will define these bits, example for now 8-bit
                    self.alarm_queue.put({'type': 'error', 'val': self.alarm_bits})


                # function  to find the peak ==>> at the begining peak of the cycle 
                nbr_volume_AT_BT, deviate_volume_list = volume_monitor.volume_peak_too_low_high(volume_desired=self.settings['VT'],
                                                                                                threshold_dv=self.settings['ADVT'])
                if nbr_volume_AT_BT > 0:
                    print("[INFO] Volume outside the allowed range {}".format(nbr_volume_AT_BT))
                    self.alarm_bits = self.alarm_bits | int('01000000', 2)  # frank will define these bits, example for now 8-bit
                    self.alarm_queue.put({'type': 'error', 'val': self.alarm_bits})

                print("processing ", self.settings)


            except Exception as inst:
                print('Exception occurred: ', inst)
                self.alarm_bits = self.alarm_bits | int('01000000', 2)  # frank will define these bits, example for now 8-bit
                self.alarm_queue.put({'type': 'error', 'val': self.alarm_bits})

            time.sleep(0.5)



"""
TODO
* add function to check volume near 0 ==>> at the end of every cycle 
"""