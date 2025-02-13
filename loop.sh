#!/bin/sh

ROUND_TRACKING_FILE=.round-tracking-file

if [ ! -e $ROUND_TRACKING_FILE ]
then
    ROUND=1
else
    ROUND=$(< $ROUND_TRACKING_FILE)
fi

while true
do
    ./process_round.py --round $ROUND
    ./train.py --round $ROUND --round-tracking-file $(ROUND_TRACKING_FILE)
    ROUND=$(< $ROUND_TRACKING_FILE)
    if ./report-script.py --metric accuracy --validation --patience 3 --round $ROUND
    then
	# Successful. Worth continuing
	continue
    else
	break
    fi
done
