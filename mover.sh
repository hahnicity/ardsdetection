#!/usr/local/bin/bash

ardsmover() {
    IFS="\n" read -r -a array <<< $(cat cohort-20160420.csv)
    echo $array
    box_sync=/Users/greg/Box\ Sync/BedsideToCloud\ Pilot\ Project/PVI/RPi\ Data
    for el in ${array[@]}
    do
        rsync -r "${box_sync}/${el}" "."
    done
}

controlmover() {
    # patients 112 and 82 have two separate folders. What to do with them??
    IFS="\n" read -r -a array <<< $(cat controlcohort.csv)
    box_sync=/Users/greg/Box\ Sync/BedsideToCloud\ Pilot\ Project/PVI/RPi\ Data
    for el in ${array[@]}
    do
        padded_digits=$(expr 4 - ${#el})
        padding=$(head -c ${padded_digits} < /dev/zero| tr '\0' '0')
        dir=$(find "$box_sync" -depth 1 -type d -name "${padding}${el}RPI*")
        echo $dir
        if [[ -z $dir ]]; then
            continue
        fi
        rsync -r "${dir}" "."
    done
}

main() {
    tomove=$1
    if [[ $tomove -eq "control" ]]; then
        controlmover
    elif [[ $tomove -eq "ards" ]]; then
        ardsmover
    fi
}

main $1
