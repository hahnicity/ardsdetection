from argparse import ArgumentParser
from glob import glob
from os import listdir
from os.path import abspath, dirname, join
from shutil import move


def main():
    parser = ArgumentParser()
    parser.add_argument("type", choices=["ardscohort", "controlcohort"])
    args = parser.parse_args()
    dirs = listdir(args.type)
    mapping = {"ardscohort": "0159RPI1820160219", "controlcohort": "0204RPI1020160331"}
    for dir in dirs:
        patient_dir = join(abspath(dirname(__file__)), args.type, dir)
        files = glob("{}/{}*".format(patient_dir, mapping[args.type]))
        # should only be one.
        if len(files) > 1:
            raise Exception("stuff")
        for path in files:  # ensures that empty dirs dont crash us
            # DEBUG
            #print(path, join(patient_dir, "{}_breath_meta.csv".format(dir)))
            move(path, join(patient_dir, "{}_breath_meta.csv".format(dir)))


if __name__ == "__main__":
    main()
