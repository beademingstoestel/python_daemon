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
    # TODO
    def run(self):
        return -1