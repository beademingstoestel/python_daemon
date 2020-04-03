"""
Settings definition for ventilator
"""

alarm ="ALARM"
ack ="ACK"

measurements = ['BPM',  # Breaths per minute
                'VOL',  # Volume
                'TRIG', # Trigger
                'PRES', # Pressure
                'FLOW', # Liters/min
                'CPU'   # CPU usage
]

settings = ['RR',   # Respiratory rate
            'VT',   # Tidal Volume
            'PK',   # Peak Pressure
            'TS',   # Breath Trigger Threshold
            'IE',   # Inspiration/Expiration (N for 1/N)
            'PP',   # PEEP (positive end expiratory pressure)
            'ADPK', # Allowed deviation Peak Pressure
            'ADVT', # Allowed deviation Tidal Volume
            'ADPP', # Allowed deviation PEEP
            'MODE',  # Machine Mode (Volume Control / Pressure Control)
            'ACTIVE',  # Machine on / off
            'PS', # support pressure
            'RP', # ramp time
            'TP', # trigger pressure
            'MT', # mute
            'FW', # firmware version
]

internal_settings = ['TPRESS'  # Target pressure
]

settings_values = {
    'RR': -1,
    'VT': -1,
    'PK': -1,
    'TS': -1,
    'IE': -1,
    'PP': -1,
    'ADPK': -1,
    'ADVT': -1,
    'ADPP': -1,
    'MODE': -1,
    'ACTIVE': -1,
    'PS': -1,
    'RP': -1,
    'TP': -1,
    'MT': -1,
    'FW': -1,
}

def compute_LRC(bytes):
    checksum = 0
    for byte in bytes:
        checksum ^= byte

    return checksum

def construct_serial_message(key, val, id):
    # every message we send has to have an id
    msg_out = "{}={}=".format(key, val)
    msg_bytes = bytearray(msg_out,'ascii')
    msg_bytes += id.to_bytes(1, byteorder='big')
    msg_bytes += bytearray('=','ascii')
    msg_bytes.append(compute_LRC(msg_bytes))
    msg_bytes += bytearray("\n", 'ascii')
    
    return msg_bytes

def construct_ack_message(id):
    msg_bytes = bytearray("ACK=", 'ascii')
    msg_bytes += id.to_bytes(1, byteorder='big')
    msg_bytes += bytearray('=','ascii')
    msg_bytes.append(compute_LRC(msg_bytes))
    msg_bytes += bytearray("\n", 'ascii')

    return msg_bytes
