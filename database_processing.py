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
# import necessary packages
import scipy
import time
import numpy as np
import scipy.signal as signal
import ventilator_protocol as proto
import ventilator_log as log
from datetime import datetime, date, timedelta
from pymongo import MongoClient, errors
from scipy.signal import find_peaks
from ventilator_alarm import AlarmBits
from ventilator_sound import SoundPlayer
"""
Notes
    * param defaults values were chosen when anyalisng the data recorded on the 02/04/2020
    * TODO seperate the the classes below!!!
"""
class PressureMonitor:
    """
    Get recorded pressure values and compute the following:
    1) OK -- measure the BPM effective
    2) ?? -- if pressure > setting*110% (or value from fagg), trigger a warning
    3) OK -- proposal: if inhale/exhale state do not change for more than 10s.
    4) OK -- ratio between inhale exhale in the database if it is above threshold 
    5) OK -- detect when the pressure goes below a "peep" threshold TBD during the exhale period T
    6) OK -- detect when the pressure peak is not in the allowed range [Pset-dp : Pset+dp]
    7) ?? -- check pressure tracking performance 
        * look for average absolute tracking errors over past cycle that are larger than ?% (of setpoint?))
        * as setpoint are not availble yet!!! this check is done in a different way:
            == at the end of the inhale period the pressure values are compared to the desired pressure value
    """
    def __init__(self, raw_data, data_TPRES, median_kernel_size=11):
        super().__init__()
        self.pvalues, self.timestamp, self.tpres, self.timetpres = [], [], [], []
        for x in (raw_data):
            self.pvalues.append(float(x.get('value')))
            full_time = x.get('loggedAt')            
            self.timestamp.append(full_time.timestamp() * 1000)
        for x in (data_TPRES):
            self.tpres.append(float(x.get('value')))
            full_time = x.get('loggedAt')            
            self.timetpres.append(full_time.timestamp() * 1000)
        # reverse the order of the element because they are retrieved 
        # in reverse order from the Mongo database
        self.timestamp.reverse()
        self.pvalues.reverse()
        self.tpres.reverse()
        self.timetpres.reverse()
        # find the peaks using the TPRES signal 
        self.ppeaks, self.npeaks = [], []
        y = np.array(self.tpres)
        for k in range(0,len(y)-1):
            if y[k]==0 and y[k+1]>0:
                self.ppeaks.append(k)
            elif y[k]>0 and y[k+1]==0:
                self.npeaks.append(k)
        # keep only complete breathing cycles
        # should start with peak_positive and end with one as well
        self.ppeaks = np.array(self.ppeaks)
        self.npeaks = np.array(self.npeaks)
        try:
            start_pp = self.ppeaks[0]
            end_pp  = self.ppeaks[-1]
            print(start_pp, end_pp)
            # keep the falling edge in between
            self.npeaks = self.npeaks[self.npeaks > start_pp]
            self.npeaks = self.npeaks[self.npeaks < end_pp]
        except:
            raise Exception('PRESSURE no valid data or peaks detected')

    def get_nbr_bpm(self):
        """
        get the BPM / # Respiratory rate (RR) in settings
        Returns:
            TODO add return value explanation
        """
        # number of breathing cycle
        # -1 : to garantee that we have a complete one at the end
        number_of_breathing_cycle = len(self.ppeaks) - 1
        print("[INFO] The number of breathing cycle = {}".format(number_of_breathing_cycle))
        # time of all the breathing cycles loaded from the mongo database (seconds)
        dtime_all_breathing_cycle =  np.diff(np.array(self.timetpres)[self.ppeaks.astype(int)]) * 1e-3 # np.diff(self.timestamp[self.ppeaks]) * 1e-3
        print("[INFO] Time (in seconds) of all the breathing cycles loaded from the mongo database: {}".format(dtime_all_breathing_cycle))
        print(np.array(self.timetpres)[self.ppeaks.astype(int)])            
        # average time of the last # breathing cycle (Ti + Te)
        average_dtime_breathing_cycle = np.mean(dtime_all_breathing_cycle)
        print("[INFO] Average time of the last # breathing cycle (Ti + Te) = {} seconds".format(average_dtime_breathing_cycle))
        # compute the BPM from the data analyzed
        total_time_seconds = (self.timetpres[self.ppeaks[-1]] - self.timetpres[self.ppeaks[0]]) / 1e3
        breathing_cycle_per_minute = 60 * number_of_breathing_cycle / total_time_seconds
        # last breathing cyvle only
        breathing_cycle_per_minute = 60 * 1 / dtime_all_breathing_cycle[-1]
        return breathing_cycle_per_minute, number_of_breathing_cycle, average_dtime_breathing_cycle

    def analyze_inhale_exhale_time(self, threshold_ratio_ie=2.75, threshold_dt_ie=10):
        """
        Args:
            thresthold_ratio_ie (float): 1st param Inspiration/Expiration (N for 1/N) = IE in settings
            theshold_dt_ie (int):  2nd param hardcoded value 10 seconds

        Returns:
            TODO add return value explanation
        """
        # TODO update this function to use the correct timestamp
        # combine both list of peaks to measure the Ti and Te
        all_peaks = np.concatenate((self.ppeaks, self.npeaks), axis=0)
        all_peaks = np.sort(all_peaks)
        dtime_inhale_exhale = np.diff(np.array(self.timetpres)[all_peaks.astype(int)]) * 1e-3 # np.diff(self.timestamp[all_peaks]) * 1e-3
        # extract the dt for inhale and exhale
        dtime_inhale = dtime_inhale_exhale[0::2]
        dtime_exhale = dtime_inhale_exhale[1::2]
        nbr_ele = min(3, len(dtime_inhale), len(dtime_exhale))
        dtime_inhale = dtime_inhale[:nbr_ele]
        dtime_exhale = dtime_exhale[:nbr_ele]
        # compute the ratio exhale/inhale ~ 3 
        ratio_exhale_inhale = dtime_exhale / dtime_inhale
        # nbr of time the ratio is below the predfined threshold
        nbr_ratio_below_threshold = sum(float(num) <= threshold_ratio_ie for num in ratio_exhale_inhale)
        # nbr of time inhale or exhale duration is above the the threshold dt
        nbr_dtinhale_above_threshold = sum(float(num) >= threshold_dt_ie for num in dtime_inhale)
        nbr_dtexhale_above_threshold = sum(float(num) >= threshold_dt_ie for num in dtime_exhale)
        return nbr_ratio_below_threshold, nbr_dtinhale_above_threshold, nbr_dtexhale_above_threshold

    def check_pressure_tracking_performance(self, pressure_desired=51, threshold_dp=3, nbr_data_point=5):
        """
        Args:
            pressure_desired (int): Peak Pressure (PK in settings)
            thresthold_dp   (int): Allowed deviation Peak Pressure (ADPK in settings)
            nbr_data_point (int): from falling edge and going backward
        Returns:
            TODO add return value explanation
        """
        # TODO update this function to use the correct timestamp
        # measure the absolute difference to the desired pressure and then take the average of the n measures
        dp_list = []
        for indice_bc in self.npeaks:
            dp = abs(np.array(self.pvalues[indice_bc - nbr_data_point:indice_bc]) - pressure_desired)
            dp_list.append(np.mean(dp))
        # nbr of time inhale or exhale duration is above the the threshold dt
        nbr_dp_above_threshold = sum(float(num) >= threshold_dp for num in dp_list)
        return nbr_dp_above_threshold, dp_list

    def detect_pressure_below_peep(self, peep_value=10, threshold_dp_peep=5, nbr_data_point=35):
        """
        Args:
            peep_value (int): 'PP' in settings
            threshold_dp_peep (int): Allowed deviation PEEP (ADPP in settings)
            nbr_data_point (int): from rising edge and going back
        Returns:
            TODO add return value explanation
        """
        # find the position of the start /end of the last breathing cycles based on the TREP timestamp
        end_last_bc   = self.ppeaks[-1] 
        end_time_bc   = self.timetpres[end_last_bc]
        exhale_k_bc    = self.npeaks[-1]
        # find the corresponding time stamp in the pressure signal
        start_exhale_time_pressure = np.argwhere(np.array(self.timestamp) > exhale_k_bc)
        start_exhale_time_pressure = start_exhale_time_pressure[0][0]
        end_bc_time_pressure = np.argwhere(np.array(self.timestamp) < end_time_bc)
        end_bc_time_pressure = end_bc_time_pressure[-1][0]
        # get the two point at 0.2s and 0.25s before the inhale start
        tmp = np.array(self.pvalues)
        exhale_val_pressure = tmp[start_exhale_time_pressure : end_bc_time_pressure]
        tmp = np.array(self.timestamp)
        time_exhale_val_pressure = tmp[start_exhale_time_pressure : end_bc_time_pressure]
        p_0200 = exhale_val_pressure[-5]
        t_0200 = time_exhale_val_pressure[-5]
        p_0250 = exhale_val_pressure[-6]
        t_0250 = time_exhale_val_pressure[-6]
        """ TODO check the -5 / -6 enough or do we need to do the following to get correct values
        for ktime in range(0,len(time_exhale_val_pressure)-1):
            if abs(time_exhale_val_pressure[-1]-time_exhale_val_pressure[ktime]-250)<50:
                p_0250 = exhale_val_pressure[ktime]
                t_0250 = time_exhale_val_pressure[ktime]
                break
            else:
                p_0250 = exhale_val_pressure[-6]
                t_0250 = time_exhale_val_pressure[-6]

        for ktime in range(0,len(time_exhale_val_pressure)-1):
            if abs(time_exhale_val_pressure[-1]-time_exhale_val_pressure[ktime]-200)<50:
                p_0200 = exhale_val_pressure[ktime]
                t_0200 = time_exhale_val_pressure[ktime]
                break
            else:
                p_0200 = exhale_val_pressure[-5]
                t_0200 = time_exhale_val_pressure[-5]
        """
        # Eqt check with Bruno/Branimir (FM) and Stijn (VUB)
        if abs(p_0200-peep_value) < threshold_dp_peep:
            nbr_dp_peep_above_threshold = 10
            below_peep_list = p_0200
        elif ( abs((p_0200-p_0250) / (t_0200-t_0250)) > abs((p_0200-exhale_val_pressure[-1])/(p_0200-time_exhale_val_pressure[-1])) ):
            dp_peep_above_threshold = 20
        else:
            dp_peep_above_threshold = 0 
        return dp_peep_above_threshold, p_0200, p_0250 

    def pressure_peak_overshoot(self, pressure_desired=51, threshold_dp_overshoot=3, nbr_data_point=10):
        """
        Args:
            pressure_desired (int): Peak Pressure (PK in settings)
            threshold_dp_overshoot (int): deviation Peak Pressure (ADPK in settings)
            nbr_data_point (int): from rising edge and going forward
        Returns:
            TODO add return value explanation
        """
        # TODO update this function to use the correct timestamp
        overshoot_pressure_list = []
        for indice_bc in self.ppeaks:
            dp = np.array(self.pvalues[indice_bc:indice_bc + nbr_data_point]) - pressure_desired
            overshoot_pressure_list.append((max(dp)))
        # nbr of time inhale or exhale duration is above the the threshold dt
        nbr_pressure_overshoot_above_threshold = sum(float(num) >= threshold_dp_overshoot for num in overshoot_pressure_list)
        return nbr_pressure_overshoot_above_threshold, overshoot_pressure_list

    def pressure_peak_too_low_high(self, pressure_desired=51, threshold_dp=3):
        """
        Args:
            pressure_desired (int): Peak Pressure PK in settings
            threshold_dp (int): Allowed deviation Peak Pressure (ADPK in settings)
        Returns:
            TODO add return value explanation
        """
        # find the position of the start and end of the last breathing cycles
        end_last_bc   = self.ppeaks[-1] 
        start_last_bc = self.ppeaks[-2]         
        start_time_bc = self.timetpres[start_last_bc]
        end_time_bc   = self.timetpres[end_last_bc]
        # find the time stamp in pressure data
        start_bc_time_pressure = np.argwhere(np.array(self.timestamp) > start_time_bc)
        start_bc_time_pressure = start_bc_time_pressure[0][0]
        end_bc_time_pressure = np.argwhere(np.array(self.timestamp) < end_time_bc)
        end_bc_time_pressure = end_bc_time_pressure[-1][0]
        # find the max pressure in the las breathing cycle        
        pvalues_array = np.array(self.pvalues)
        pvalues_last_bc = pvalues_array[start_bc_time_pressure : end_bc_time_pressure]
        val_max_pressure = max(pvalues_last_bc)
        # find the minimun pressure during the inhale period
        exhale_k_bc    = self.npeaks[-1]
        position_max = np.argwhere(pvalues_last_bc == val_max_pressure) + start_bc_time_pressure
        position_max = position_max[0][0]
        val_min_inhale_pressure = min(pvalues_last_bc[position_max : exhale_k_bc])
        # max and min pressure during inhale state should be both in the allowed range
        if abs(val_max_pressure-pressure_desired) < threshold_dp and abs(val_min_inhale_pressure-pressure_desired) < threshold_dp:
            pressure_not_in_allowed_range = 0            
        else:
            pressure_not_in_allowed_range = 10
        return pressure_not_in_allowed_range, val_max_pressure, val_min_inhale_pressure

    def get_IE(self):
        """
        this function used the already loaded pressure reference data, and calculates in the full window of this data 
        how many inhale and exhale cycles there were and especially how long these took, to calculate the average IE ratio
        Args:
            none
        Returns:
            average IE ratio as observed in the data : IE_ratio = exhale_time/ inhale_time
        """
        all_peaks = np.concatenate((self.ppeaks, self.npeaks), axis=0)
        all_peaks = np.sort(all_peaks)
        dtime_inhale_exhale = np.diff(np.array(self.timetpres)[all_peaks.astype(int)])
        # extract the dt for inhale and exhale
        dtime_inhale = dtime_inhale_exhale[0::2]  
        dtime_exhale = dtime_inhale_exhale[1::2]
        # full number of cycles present in data record (in case of incomplete cycles)
        nbr_ele = min(3, len(dtime_inhale), len(dtime_exhale))
        dtime_inhale = dtime_inhale[:nbr_ele]
        dtime_exhale = dtime_exhale[:nbr_ele]
        # compute the ratio exhale/inhale (this are still multiple values) 
        ratio_exhale_inhale = dtime_exhale / dtime_inhale
        # find the average IE to return 
        average_IE = np.mean(ratio_exhale_inhale)
        return average_IE

    def get_PressurePlateau(self): 
        """
        this function used the already loaded pressure measurements, and calculates in the window of this data how much 
        the pressure plateau is during the inhale cycles (the part after the peak pressure occurs). 
        then we calculate the average plateau over all the cycles in this window of data                        
        Args:
            none
        Returns:
            average_PressurePlateau = average pressure plateau as observed in the data 
        """
        pvalues_array = np.array(self.pvalues)
        avg_pressure_per_cycle = []
        for k in range(0, len(self.ppeaks)-1):            
            inhale_pressure = pvalues_array[self.ppeaks[k] : self.npeaks[k]]
            # find pressure peak location                      
            max_ind = np.where(inhale_pressure == max(inhale_pressure)) 
            # take a small offset on max pressure (hardcoded here)
            avg_pressure_per_cycle.append(np.mean(pvalues_array[max_ind[0][0]+6:self.npeaks[k]]))
        # now take the average 
        average_PressurePlateau = np.mean(avg_pressure_per_cycle)
        return average_PressurePlateau

class VolumeMonitor:
    """
    get the recorded volume values and compute the following:
    1) Not tested  -- peak volume per breathing cycle and check if it is in the allowed range [Vset-dv : Vset+dv]
    2) Not tested  -- check if the volume goes near zero at the end of each breathing cycle
    """
    def __init__(self, raw_data, data_TPRES, median_kernel_size=11):
        super().__init__()
        # raw_data from the Mongo database
        self.vvalues, self.timestamp, self.tpres, self.timetpres = [], [], [], []
        # send back data raw format + time stamp
        for x in (raw_data):
            self.vvalues.append(float(x.get('value')))
            full_time = x.get('loggedAt')            
            self.timestamp.append(full_time.timestamp() * 1000)            
        # load the TPRES signal data 
        for x in (data_TPRES):
            self.tpres.append(float(x.get('value')))
            full_time = x.get('loggedAt')            
            self.timetpres.append(full_time.timestamp() * 1000)
        # reverse the order of the element because they are retrieved 
        # in reverse order from the Mongo database
        self.timestamp.reverse()
        self.vvalues.reverse()
        self.tpres.reverse()
        self.timetpres.reverse()
        # find the peaks using the TPRES signal 
        self.ppeaks, self.npeaks = [], []
        y = np.array(self.tpres)
        for k in range(0,len(y)-1):
            if y[k]==0 and y[k+1]>0:
                self.ppeaks.append(k)
            elif y[k]>0 and y[k+1]==0:
                self.npeaks.append(k)
        # keep only complete breathing cycles # should start with peak_positive and end with one as well
        self.ppeaks = np.array(self.ppeaks)
        self.npeaks = np.array(self.npeaks)
        try:
            start_pp = self.ppeaks[0]
            end_pp = self.ppeaks[-1]
            # keep the falling edge in between
            self.npeaks = self.npeaks[self.npeaks > start_pp]
            self.npeaks = self.npeaks[self.npeaks < end_pp]
        except:
            raise Exception('volume no valid data or peaks detected')

    def volume_peak_too_low_high(self, volume_desired=150, threshold_dv=30):
        """
        Args:
            volume_desired : desired volume (VT in settings)
            threshold_dv   : allowed deviation in volume (ADVT in setting)
        Returns:
            volume_deviate = 
            val_max_volume =
        """
        # find the position of the start and end of the last breathing cycles
        end_last_bc   = self.ppeaks[-1] 
        start_last_bc = self.ppeaks[-2]         
        start_time_bc = self.timetpres[start_last_bc]
        end_time_bc   = self.timetpres[end_last_bc]
        # find the time stamp in volume data
        start_bc_time_volume = np.argwhere(np.array(self.timestamp) > start_time_bc)
        start_bc_time_volume = start_bc_time_volume[0][0]
        end_bc_time_volume = np.argwhere(np.array(self.timestamp) < end_time_bc)
        end_bc_time_volume = end_bc_time_volume[-1][0]
        # find the peak volume 
        vvalues_array = np.array(self.vvalues)
        val_max_volume = max(vvalues_array[start_bc_time_volume : end_bc_time_volume])
        if abs(val_max_volume-volume_desired) < threshold_dv:
            volume_deviate   = 0
        else:
            volume_deviate   = 10
        return volume_deviate, val_max_volume

    def detect_volume_not_near_zero_ebc(self, threshold_dv_zero=50):
        """
        function to check volume near 0 ==>> at the end of every cycle
        Args:
            threshold_dv_zero = allowed deviation near the zero level 
        Returns:
            volome_not_near_zero_ebc = volume not near zero at the end of breathing cycle
            val_min_volume = minimum volume found
        """
        # get the position of the last breathing cycle peaks
        end_last_bc   = self.ppeaks[-1]       
        # get the position of the exhale end in the last breathing cycle using TREP time
        end_time_bc   = self.timetpres[end_last_bc]
        end_exhale_time_volume = np.argwhere(np.array(self.timestamp) < end_time_bc)
        end_exhale_time_volume = end_exhale_time_volume[-1][0]
        # get the position of the exhale start in the last breathing cycle using TREP time
        exhale_k_bc    = self.npeaks[-1]
        exhale_time_bc = self.timetpres[exhale_k_bc]
        start_exhale_time_volume = np.argwhere(np.array(self.timestamp) > exhale_time_bc)
        start_exhale_time_volume = start_exhale_time_volume[0][0]
        # find the minimum value during the exhale state
        vvalues_array = np.array(self.vvalues)
        val_min_volume = abs(min(vvalues_array[start_exhale_time_volume : end_exhale_time_volume]))
        if val_min_volume < threshold_dv_zero:
            volome_not_near_zero_ebc = 0
        else:
            volome_not_near_zero_ebc = 10
        return volome_not_near_zero_ebc, val_min_volume

    def get_TidalVolume_lastMinute(self):
        """
        This function returns the total tidal volume of the data as observed over the already built record of data
        Args:
            none (function uses only the data stored in self.vvalues defined elsewhere)
        Returns:
            Volume Last Minute 
        """ 
        time_inhale = (np.array(self.timetpres)[self.ppeaks.astype(int)])
        time_exhale = (np.array(self.timetpres)[self.npeaks.astype(int)])
        tmp_val = np.array(self.vvalues)
        total_TV_ml = 0
        for k in range(0,len(self.ppeaks)-1):
            # find volume time just above the inhale time 
            k1 = np.argwhere(self.timestamp > time_inhale[k])
            k1 = k1[0][0]-min(k1[0][0], 5)
            k2 = np.argwhere(self.timestamp < time_exhale[k])
            k2 = k2[-1][0]+min(10,len(tmp_val)-k2[-1][0])
            if k2<=k1:
                continue
            else:
                total_TV_ml = total_TV_ml + max(tmp_val[k1:k2])
        # convert to total over 60 seconds and return value
        VolumeLastMinute = 60 * total_TV_ml/( (time_inhale[-1]-time_inhale[0]) / 1e3)
        return VolumeLastMinute

class DatabaseProcessing:
    def __init__(self, settings, alarm_queue, request_queue, addr='mongodb://localhost:27017'):
        self.settings = settings
        self.addr = addr
        self.alarm_queue = alarm_queue
        self.request_queue = request_queue
        self.alarm_bits = AlarmBits.NONE.value
        self.previous_alarm_bits = AlarmBits.NONE.value
        self.db = None
        self.previous_mute_setting = 0

    def last_n_data(self, type_data, N=1200):
        """
        retrieve the last "N" added measurement N = 1000

        Returns:
            TODO add return value explanation
        """
        data_TPRES = self.db.targetpressure_values.find().sort("loggedAt", -1).limit(N) 
        if type_data == 'BPM':
            return self.db.breathsperminute_values.find().sort("loggedAt", -1).limit(N), data_TPRES 
        elif type_data == 'VOL':
            return self.db.volume_values.find().sort("loggedAt", -1).limit(N), data_TPRES 
        elif type_data == 'TRIG':
            return self.db.trigger_values.find().sort("loggedAt", -1).limit(N), data_TPRES 
        elif type_data == 'PRES':
            return self.db.pressure_values.find().sort("loggedAt", -1).limit(N), data_TPRES 
        elif type_data == 'TPRES':
            return self.db.targetpressure_values.find().sort("loggedAt", -1).limit(N), None
        else:
            print("[ERROR] value type not recognized use: BPM, VOL, TRIG, or PRES")
            log.ERROR(__name__, self.request_queue, "value type not recognized use: BPM, VOL, TRIG, or PRES")
            return None, None

    def run(self, name):
        print("[INFO] Starting {}".format(name))
        log.INFO(__name__, self.request_queue, "Starting {}".format(name))
        # Only start MongoClient after fork()
        # and each child process should have its own instance of the client
        try:
            self.client = MongoClient(self.addr)
        except errors.ConnectionFailure:
            print("[ERROR] Unable to connect, client will attempt to reconnect")
            log.ERROR(__name__, self.request_queue, "Unable to connect, client will attempt to reconnect")

        self.db = self.client.beademing
        self.sound_player = SoundPlayer('assets/beep.wav', 0, 0)        
                
        while True:
            try:
                self.alarm_bits = AlarmBits.NONE.value
                
                data_p, data_TPRES = self.last_n_data('PRES')
                pressure_monitor = PressureMonitor(data_p, data_TPRES)
                data_v, data_TPRES = self.last_n_data('VOL')
                volume_monitor   = VolumeMonitor(data_v, data_TPRES)

                # BT: Below Threshold
                # AT Above Threshold

                # BPM - # Respiratory rate (RR)
                breathing_cycle_per_minute, number_of_breathing_cycle, average_dtime_breathing_cycle = pressure_monitor.get_nbr_bpm()
                if abs(breathing_cycle_per_minute-self.settings['RR']) > 0.50 :
                    print("[WARNING] Breathing at {} per minute".format(breathing_cycle_per_minute))
                    self.alarm_bits = self.alarm_bits | AlarmBits.DATABASE_PROCESSING_PRESSURE_BPM_BT.value
                
                # performance 'IE',   # Inspiration/Expiration (N for 1/N)
                # in the function definition used as default values 2.75 / 10 seconds
                nbr_ie_ratio_BT, nbr_dtinhale_AT, nbr_dtexhale_AT = pressure_monitor.analyze_inhale_exhale_time(threshold_ratio_ie=self.settings['IE'],
                                                                                                                threshold_dt_ie=10)
                if False: # nbr_ie_ratio_BT > 0:
                    print("[WARNING] # Respiratory rate below threshold : {} ".format(nbr_ie_ratio_BT))
                    self.alarm_bits = self.alarm_bits | AlarmBits.DATABASE_PROCESSING_PRESSURE_IE_RATIO_BT.value

                if False: # nbr_dtinhale_AT > 0 or nbr_dtexhale_AT > 0:
                    print("[WARNING] # inhale time above 10s : {} and # exhale time above 10s :{} ".format(nbr_dtinhale_AT, nbr_dtexhale_AT))
                    self.alarm_bits = self.alarm_bits | AlarmBits.DATABASE_PROCESSING_PRESSURE_TIME_INHALE_EXHALE_AT.value
                
                # Pressure performance Tracking -- Peak Pressure PK and ADPK Allowed deviation Peak Pressure
                # in the function defaults values are 51 / 3 / (5 for data points)
                nbr_dp_AT, dp_list = pressure_monitor.check_pressure_tracking_performance(pressure_desired=self.settings['PK'],
                                                                                            threshold_dp=self.settings['ADPK'],
                                                                                            nbr_data_point=5)
                if False: # nbr_dp_AT > 3:
                    print("[WARNING] # Pressure deviate during inhale {}".format(nbr_dp_AT))
                    self.alarm_bits = self.alarm_bits | AlarmBits.DATABASE_PROCESSING_PRESSURE_DP_AT.value

                # Pressure below peep value 'PP', # PEEP (positive end expiratory pressure) # 'ADPP', # Allowed deviation PEEP
                # in the function defaults values are 10/5
                dp_peep_above_threshold, p_0200, p_0250  = pressure_monitor.detect_pressure_below_peep(peep_value=self.settings['PP'],
                                                                                                        threshold_dp_peep=self.settings['ADPP'],
                                                                                                        nbr_data_point=35)
                if False: # dp_peep_above_threshold > 0:
                    print("[WARNING] # Pressure below peep level detected {}".format(dp_peep_above_threshold))
                    self.alarm_bits = self.alarm_bits | AlarmBits.DATABASE_PROCESSING_PRESSURE_DP_PEEP_AT.value
                """
                # TODO confirm that this function is no longer needed
                # TODO if yes alarm value need to be changed then
                # Detect when the pressure_peak_overshoot
                # default values are 51 / 3
                nbr_pressure_overshoot_AT, overshoot_pressure_list = pressure_monitor.pressure_peak_overshoot(pressure_desired=self.settings['PK'],
                                                                                                            threshold_dp_overshoot=self.settings['ADPK'],
                                                                                                            nbr_data_point=10)
                if nbr_pressure_overshoot_AT > 0:
                    print("[WARNING] Pressure peak overshoot {}".format(nbr_pressure_overshoot_AT))
                    self.alarm_bits = self.alarm_bits | int('00100000', 2)  # frank will define these bits, example for now 8-bit
                """
                # Detect when the pressure peak of breathing cycle is not in the allowed range defined
                # default values are PK = 51 / ADPK = 3
                pressure_AT_BT, val_max_pressure, val_min_inhale_pressure = pressure_monitor.pressure_peak_too_low_high(pressure_desired=self.settings['PK'],
                                                                                                                        threshold_dp=self.settings['ADPK'])
                if pressure_AT_BT > 0:
                    print("[WARNING] # Pressure outside the allowed range {}".format(pressure_AT_BT))
                    self.alarm_bits = self.alarm_bits | AlarmBits.DATABASE_PROCESSING_PRESSURE_AT_BT.value

                # desired volume VT / allowed deviation volume ADVT
                # function  to find the peak volume ==>> at the begining peak of the cycle
                volume_AT_BT, val_max_volume = volume_monitor.volume_peak_too_low_high(volume_desired=self.settings['VT'],
                                                                                        threshold_dv=self.settings['ADVT'])
                if volume_AT_BT > 0:
                    print("[WARNING] # Volume outside the allowed range {}".format(volume_AT_BT))
                    self.alarm_bits = self.alarm_bits | AlarmBits.DATABASE_PROCESSING_VOLUME_AT_BT.value

                # function to check volume near 0 ==>> at the end of every cycle
                # TODO do we have in the settings a value for the threshold_dv_zero? if we need to find value 0 set then the threshold to 0
                volome_not_near_zero_ebc, val_min_volume = volume_monitor.detect_volume_not_near_zero_ebc(threshold_dv_zero=50)
                if volome_not_near_zero_ebc > 0:
                    print("[WARNING] Volume not near zero at the end of breathing cycle {}".format(volome_not_near_zero_ebc))
                    self.alarm_bits = self.alarm_bits | AlarmBits.DATABASE_PROCESSING_VOLUME_NOT_NEAR_ZERO_EBC.value

                if self.alarm_bits > 0:
                    self.alarm_queue.put({'type': proto.alarm, 'val': self.alarm_bits, 'source': 'processing'})
                    #play an alarm
                    print('play beep')
                    if self.previous_alarm_bits == 0:
                        if self.sound_player.is_alive():
                            self.sound_player.terminate()
                            self.sound_player.join()
                        self.sound_player = SoundPlayer('assets/beep.wav', -1, 0.200)
                        self.sound_player.start()
                elif self.alarm_bits == 0 and self.previous_alarm_bits != 0:
                    print('stop beep')
                    if self.sound_player.is_alive():
                        self.sound_player.terminate()
                        self.sound_player.join()
                
                if self.settings['MT'] == 0 and self.previous_mute_setting != 0:
                    print('play beep MT')
                    if self.alarm_bits > 0:
                        if self.sound_player.is_alive():
                            self.sound_player.terminate()
                            self.sound_player.join()
                        self.sound_player = SoundPlayer('assets/beep.wav', -1, 0.200)
                        self.sound_player.start()
                elif self.settings['MT'] == 1 and self.previous_mute_setting == 0:
                    print('stop beep MT')
                    if self.sound_player.is_alive():
                        self.sound_player.terminate()
                        self.sound_player.join()
                    
                self.previous_mute_setting = self.settings['MT']
                self.previous_alarm_bits = self.alarm_bits

                averageIE = pressure_monitor.get_IE()
                averagePressurePlateau = pressure_monitor.get_PressurePlateau()
                volumelastMinute = volume_monitor.get_TidalVolume_lastMinute()

                print("*"*21)
                print("[INFO] Processing Settings", self.settings)
                print("[INFO] BPM = {}".format(breathing_cycle_per_minute))
                print("[INFO] # IE_ratio_BT = {}, # dt_inhale_AT = {}, # dt_exhale_AT = {}  ".format(nbr_ie_ratio_BT, nbr_dtinhale_AT, nbr_dtexhale_AT))
                print("[INFO] # pressure desired not reached {}".format(nbr_dp_AT))
                print("[INFO] # pressure below peep+dp = {} ".format(dp_peep_above_threshold))
                print("[INFO - OK] # peak pressure not in allowed range = {}".format(pressure_AT_BT))
                print("[INFO] Max pressure and min pressure during inhale time = {}".format(val_max_pressure))
                print("[INFO - OK] # peak volume not in allowed range = {}".format(volume_AT_BT))
                print("[INFO] Max volume  during last breathing cycle= {}".format(val_max_volume))
                print("[INFO] # volume not near zero at ebc = {} ".format(volome_not_near_zero_ebc))
                print("[INFO] the average IE = {} ".format(averageIE))
                print("[INFO] the average PressurePlateau = {} ".format(averagePressurePlateau))
                print("[INFO] the average PressurePlateau = {} ".format(volumelastMinute))
                print("*"*21)
                print("alarm_bits ", hex(self.alarm_bits))

                log.INFO(__name__, self.request_queue, "BPM = {}".format(breathing_cycle_per_minute))
                log.INFO(__name__, self.request_queue, "# IE_ratio_BT = {}, # dt_inhale_AT = {}, # dt_exhale_AT = {}  ".format(nbr_ie_ratio_BT, nbr_dtinhale_AT, nbr_dtexhale_AT))
                log.INFO(__name__, self.request_queue, "# pressure desired not reached {}".format(nbr_dp_AT))
                log.INFO(__name__, self.request_queue, "# pressure below peep+dp = {} ".format(dp_peep_above_threshold))
                log.INFO(__name__, self.request_queue, "# peak pressure not in allowed range = {}".format(pressure_AT_BT))
                log.INFO(__name__, self.request_queue, "# peak volume not in allowed range = {}".format(volume_AT_BT))
                log.INFO(__name__, self.request_queue, "# volume not near zero at ebc = {} ".format(volome_not_near_zero_ebc))

            except Exception as e:
                print("[INFO] Processing Settings", self.settings)
                print('[WARNING] Exception occurred: ', e)
                log.ERROR(__name__, self.request_queue, "Exception occurred = {}".format(e))
                self.alarm_bits = self.alarm_bits | AlarmBits.DATABASE_PROCESSING_EXCEPTION.value
                self.alarm_queue.put({'type': proto.alarm, 'val': self.alarm_bits, 'source': 'processing'})

            time.sleep(0.5)
