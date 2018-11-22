@echo OFF

REM please change pool address, wallet address and worker ID to yours
REM adjust -b & -t value as described in the README and FAQ
REM -u means use all devices, you can also use -d to specify list of devices (ex: -d 0,2,5)
set devices="-u"
set worker=""
set pool="http://aro.cool"
set wallet="4bb66RkoTCz63XPBvMyHfWRE1vWif21j77m1kNVNRd7o4YtJdEQWY7BsRVxAYoTdMexSVFGFaekrc3UATTSERwmQ"
set threads="1"
set batches="64"
set stats_node_url=""
set stats_node_secret=""

REM set this to true to bench the miner performance
set test_mode="false"
set extraPrm=""
if %test_mode% == "true" (
	set extraPrm="--test-mode"
)

arionum-opencl-miner.exe %devices% %extraPrm% -b %batches% -t %threads% -p %pool% -a %wallet% -i %worker% -n %stats_node_url% -s %stats_node_secret%

pause
