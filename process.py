from argparse import ArgumentParser
from glob import glob
import os
from os.path import basename, dirname, join, splitext
import traceback

from algorithms.breath_meta import extract_breath_meta, write_breath_meta
from preprocessing.clear_null_bytes import clear_null_bytes
from preprocessing.clean_datetimed_data import clean_datetimed_data


def get_patient_file(patient_dir):
    """
    Get a patient file that meets a specific criteria of being used

    As a note: the following patients do not have data associated with them

    0050RPI0920150709
    0051RPI0620150711
    0066RPI0620150827
    0069RPI0620150904
    0083RPI1120151120
    0116RPI1521060112
    0117RPI0920160115
    0150RPI0520160216
    0156RPI0520160218

    So we cannot actually use them
    """
    files = glob(os.path.join(patient_dir, "*.csv"))
    for f in files[::-1]:
        if "Brooks" in f or "Jason" in f or "Gold" in f:
            continue
        stat = os.stat(f)
        mb_size = stat.st_size / (1024.0 ** 2)
        # Easy test is it over 3 MB? If so then use it
        if mb_size > 3:
            return f


def main():
    parser = ArgumentParser()
    parser.add_argument("type", choices=["ards", "control"])
    args = parser.parse_args()
    patient_files = []
    file_dir = os.path.dirname(__file__)
    abs_cohort_dir = os.path.join(file_dir, "{}cohort".format(args.type))
    for patient_dir in os.listdir(abs_cohort_dir):
        abs_patient_dir = os.path.join(abs_cohort_dir, patient_dir)
        patient_files.append(get_patient_file(abs_patient_dir))
    patient_files = filter(lambda x: x, patient_files)

    try:
        with open("completed.list.{}".format(args.type), "r") as completed:
            completed_files = [line.strip("\n") for line in completed.readlines()]
    except:
        completed_files = []

    with open("completed.list.{}".format(args.type), "a") as completed:
        with open("errors.list.{}".format(args.type), "w") as errors:
            for p in patient_files:
                if p not in completed_files:
                    base_no_ext = splitext(basename(p))[0]
                    patient_dir = basename(dirname(p))
                    outfile = join(
                        dirname(p),
                        "{}_{}_breath_meta.csv".format(patient_dir, base_no_ext)
                    )
                    try:
                        stringio = clear_null_bytes(p)
                        stringio = clean_datetimed_data(None, stringio)
                        array = extract_breath_meta(stringio)
                        write_breath_meta(array, outfile)
                        completed.write(p + "\n")
                    except Exception as err:
                        for a in err.args:
                            errors.write(a + "\n")
                            traceback.print_exc(file=errors)



if __name__ == "__main__":
    main()
