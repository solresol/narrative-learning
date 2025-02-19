#!/bin/sh

if [ "$ROUND_TRACKING_FILE" = "" ]
then
    ROUND_TRACKING_FILE=.round-tracking-file
fi

if [ ! -e $ROUND_TRACKING_FILE ]
then
    ROUND=1
else
    ROUND=$(< $ROUND_TRACKING_FILE)
fi

while true
do
    ./process_round.py --round $ROUND
    ./train.py --round-id $ROUND --round-tracking-file $ROUND_TRACKING_FILE
    ROUND=$(< $ROUND_TRACKING_FILE)
    if ./report-script.py --metric accuracy --validation --patience 3 
    then
	# Successful. Worth continuing
	continue
    else
	break
    fi
done
