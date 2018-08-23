@echo OFF

REM please change pool & wallet address to yours
REM adjust -b & -t value as described in the README and FAQ
set pool="http://aro.cool"
set wallet="4bb66RkoTCz63XPBvMyHfWRE1vWif21j77m1kNVNRd7o4YtJdEQWY7BsRVxAYoTdMexSVFGFaekrc3UATTSERwmQ"
set threads="2"
set batches="24"

REM -u means use all device, you can also use -d to specify list of devices (ex: -d 0,2,5)
arionum-opencl-miner.exe -u -b %batches% -t %threads% -p %pool% -a %wallet%

pause