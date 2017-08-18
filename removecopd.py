import csv
import shutil


def main():
    with open("cohort-20160420.csv") as ards:
        reader = csv.reader(ards)
        ards = [line[0] for line in reader]

    with open("copd.csv") as f:
        reader = csv.reader(f)
        for line in reader:
            if line[0] in ards:
                try:
                    shutil.rmtree("ardscohort/{}".format(line[0]))
                except:
                    pass
            else:
                try:
                    shutil.rmtree("controlcohort/{}".format(line[0]))
                except:
                    pass


main()
