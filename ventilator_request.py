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
import requests
import ventilator_protocol as proto

class APIRequest():

    def __init__(self, base_address):
        self.base_address = base_address

    def __put(self, endpoint, data):
        try:
            r = requests.put(url = self.base_address + endpoint, data = data) 
            data = r.json()

            if not data["result"]:
                print("The request was not successful")
        except requests.RequestException:
            print("Couldn't reach the server")

    def send_setting_float(self, key, val):
        print("Send float setting to server, set to {}".format({key:float(val)}))
        self.__put("/api/settings?returncomplete=false", {key:float(val)})

    def send_setting(self, key, val):
        print("Send setting to server, set to {}".format({key:val}))
        self.__put("/api/settings?returncomplete=false", {key:val})

    def send_alarm(self, val):
        self.__put("/api/settings", {'value':val})
        return
