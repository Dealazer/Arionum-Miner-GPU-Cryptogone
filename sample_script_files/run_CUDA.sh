#!/bin/bash

# please change pool address, wallet address and worker ID to yours
# adjust -b & -t value as described in the README and FAQ
# -u means use all devices, you can also use -d to specify list of devices (ex: -d 0,2,5)
devices="-u"
worker=""
pool="http://aro.cool"
wallet="4bb66RkoTCz63XPBvMyHfWRE1vWif21j77m1kNVNRd7o4YtJdEQWY7BsRVxAYoTdMexSVFGFaekrc3UATTSERwmQ"
threads="1"
batches="64"
stats_node_url=""
stats_node_secret=""
test_mode="false" # set this to true to bench the miner performance

extraPrm=""
if [ "test_mode" = "true" ]; then
    extraPrm="--test-mode"
fi

# set this to false if you do not want miner to auto relaunch after crash
relaunch_miner_on_crash="true"

while :
do
	./arionum-cuda-miner "$devices" "$extraPrm" -b "$batches" -t "$threads" -p "$pool" -a "$wallet" -i "$worker" -n "$stats_node_url" -s "$stats_node_secret"

	if [ "$relaunch_miner_on_crash" = "true" ]; then
		echo "miner crashed, relaunching in 5 seconds ..."
		sleep 5
	else
		break
	fi
done
