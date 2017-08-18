#!/bin/bash

main() {
    local ip=$1
    if [[ -z $ip ]]; then
        echo "enter an ip/hostname to ssh to!"
        exit 1
    fi

    ssh -i /Users/greg/.ssh/gr-pair-gondolin.pem ec2-user@$ip "mkdir -p  ecs251-final-project"
    scp -i /Users/greg/.ssh/gr-pair-gondolin.pem -r ../ecs251-final-project/*.py ec2-user@$ip:~/ecs251-final-project
    scp -i /Users/greg/.ssh/gr-pair-gondolin.pem -r ../ecs251-final-project/*.sh ec2-user@$ip:~/ecs251-final-project
    scp -i /Users/greg/.ssh/gr-pair-gondolin.pem -r ../ecs251-final-project/*.txt ec2-user@$ip:~/ecs251-final-project
    scp -i /Users/greg/.ssh/gr-pair-gondolin.pem -r ../ecs251-final-project/ardscohort ec2-user@$ip:~/ecs251-final-project
    scp -i /Users/greg/.ssh/gr-pair-gondolin.pem -r ../ecs251-final-project/controlcohort ec2-user@$ip:~/ecs251-final-project
}

main $1
