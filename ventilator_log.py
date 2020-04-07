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

Ventilator log component
"""

import ventilator_protocol as proto

#TODO future using syslog ? for now use own definitions of log severities # https://en.wikipedia.org/wiki/Syslog

# import syslog

# log severities:
LOG_LEVEL_INFO = 'info'
LOG_LEVEL_ERROR = 'error'
LOG_LEVEL_WARNING = 'warning'
LOG_LEVEL_DEBUG = 'debug'

#TODO adding filename, function (+linenr)

# Wrappers
def INFO(component_name, log_queue, log_message):
    log_queue.put({'type': proto.log, 'severity': LOG_LEVEL_INFO, 'value': "{} - {}".format(component_name, log_message)})


def ERROR(component_name, log_queue, log_message):
    log_queue.put({'type': proto.log, 'severity': LOG_LEVEL_ERROR, 'value': "{} - {}".format(component_name, log_message)})


def WARNING(component_name, log_queue, log_message):
    log_queue.put({'type': proto.log, 'severity': LOG_LEVEL_WARNING, 'value': "{} - {}".format(component_name, log_message)})


def DEBUG(component_name, log_queue, log_message):
    log_queue.put({'type': proto.log, 'severity': LOG_LEVEL_DEBUG, 'value': "{} - {}".format(component_name, log_message)})
