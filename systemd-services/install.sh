#!/bin/bash 

if [[ $EUID > 0 ]]
then 
  echo "Please run with super-user privileges"
  exit 1
else
	cp ./delaycalibration.service /etc/systemd/system/
	cp ./delaymodel.service /etc/systemd/system/
	systemctl disable delaycalibration.service
	systemctl disable delaymodel.service
	systemctl daemon-reload
	systemctl enable delaycalibration.service
	systemctl enable delaymodel.service
fi