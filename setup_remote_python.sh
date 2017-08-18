#!/bin/bash

main() {
    local ips=$(aws ec2 describe-instances | grep PublicIpAddress | awk '{print $2}' | tr -d \" | tr -d ,)
    for i in $ips; do
        setup_instance $i &
    done
}

setup_instance() {
    local ip=$1
    if [[ -z $ip ]]; then
        echo "input an ip!"
        exit 1
    fi
    local identity_path=/Users/greg/.ssh/gr-pair-gondolin.pem
    local user="ec2-user"
    local dir="ecs251-final-project"
    local options="-o StrictHostKeyChecking=no -o CheckHostIP=no"
    ssh $options -i $identity_path $user@$ip "mkdir -p $dir"
    scp $options -i $identity_path bootstrap.sh requirements.txt $user@$ip:~/$dir/
    ssh $options -i $identity_path $user@$ip "cd ~/$dir; ./bootstrap.sh"
}

main
