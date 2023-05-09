#!/bin/bash 

if [[ $EUID > 0 ]]
then 
  echo "Please run with super-user privileges"
  exit 1
else
	cp ./calibration_gain_collator.service /etc/systemd/system/
	cp ./delaymodel.service /etc/systemd/system/
	cp ./delaylogger.service /etc/systemd/system/
	systemctl disable calibration_gain_collator.service
	systemctl disable delaymodel.service
	systemctl disable delaylogger.service
	systemctl daemon-reload
	systemctl enable calibration_gain_collator.service
	systemctl enable delaymodel.service
	systemctl enable delaylogger.service
fi