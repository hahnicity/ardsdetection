import argparse
import os
import re
import subprocess


def does_file_have_no_timestamp_pat(filename):
    pattern = re.compile(
        "(?P<year>201[456])-(?P<month>[01]\d)-(?P<day>[0123]\d)"
        "__(?P<hour>[012]\d):(?P<minute>[0-5]\d):(?P<second>[0-5]\d)"
        ".(?P<millis>\d+).csv"
    )
    match = pattern.search(filename)
    return match if match else False


def check_if_file_already_has_timestamp(filename):
    with open(filename) as f:
        first_line = f.readline()
        pat = "\d{4}-\d{2}-\d{2}-\d{2}-\d{2}-\d{2}"
        pat2 = "\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}"
        if not re.search(pat, first_line):
            if not re.search(pat2, first_line):
                return False
        return True


def add_timestamp(filename):
    match = does_file_have_no_timestamp_pat(filename)
    if not match:
        raise Exception("no file-to-regex match")
    if check_if_file_already_has_timestamp(filename):
        return
    dict_ = match.groupdict()
    dict_['millis'] = int(dict_['millis']) / 1000
    time = "{year}-{month}-{day}-{hour}-{minute}-{second}.{millis}".format(**dict_)
    os.system('echo {} > /tmp/time.stamp'.format(time))
    os.system('cat /tmp/time.stamp {} > /tmp/vent.file'.format(filename))
    proc = subprocess.Popen(['mv', '/tmp/vent.file', filename])
    proc.communicate()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("file")
    args = parser.parse_args()
    add_timestamp(args.file)


if __name__ == "__main__":
    main()
