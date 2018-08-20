"""
timestamp_on_file
~~~~~~~~~~~~~~~~~

Basic script for placing a new-style timestamp on a vent file collected before patient
50 or so. This enables compatibility with current analytics code.
"""
import argparse
import os
import re
import subprocess


parser = argparse.ArgumentParser()
parser.add_argument("file")
args = parser.parse_args()

pattern = re.compile(
    "(?P<year>201[456])-(?P<month>[01]\d)-(?P<day>[0123]\d)"
    "__(?P<hour>[012]\d):(?P<minute>[0-5]\d):(?P<second>[0-5]\d)"
    ".(?P<millis>\d+).csv"
)
match = pattern.search(args.file)
if not match:
    raise Exception("no file-regex match")

dict_ = match.groupdict()
dict_['millis'] = int(dict_['millis']) / 1000
time = "{year}-{month}-{day}-{hour}-{minute}-{second}.{millis}".format(**dict_)
os.system('echo {} > /tmp/time.stamp'.format(time))
os.system('cat /tmp/time.stamp {} > /tmp/vent.file'.format(args.file))
proc = subprocess.Popen(['mv', '/tmp/vent.file', args.file])
proc.communicate()
