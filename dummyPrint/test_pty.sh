#!/bin/bash
#
# Test: Create pty devices in /tmp/ttySIM1 /tmp/ttySIM2 and run
# daemon.py and dummyPrint.py with them as argument
#

trap "trap - SIGTERM && kill -- -$$" SIGINT SIGTERM EXIT

socat -d -d PTY,raw,echo=0,link=/tmp/ttySIM1 PTY,raw,echo=0,link=/tmp/ttySIM2 &

python3 ../daemon.py /tmp/ttySIM2 &

python3 ./dummyPrint.py /tmp/ttySIM1
