#!/bin/bash
set -e

# please change pool & wallet address to yours
# adjust -b & -t value as described in the README and FAQ
pool="http://aro.cool"
wallet="4bb66RkoTCz63XPBvMyHfWRE1vWif21j77m1kNVNRd7o4YtJdEQWY7BsRVxAYoTdMexSVFGFaekrc3UATTSERwmQ"
threads="1"
batches="48"

# -u means use all device, you can also use -d to specify list of devices (ex: -d 0,2,5)
./arionum-cuda-miner -u -b "$batches" -t "$threads" -p "$pool" -a "$wallet"

