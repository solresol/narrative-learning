#!/bin/bash

if [ "$ROUND_TRACKING_FILE" = "" ]
then
    ROUND_TRACKING_FILE=.round-tracking-file
fi

echo "Using $ROUND_TRACKING_FILE to track the current round"

if [ ! -e $ROUND_TRACKING_FILE ]
then
    ROUND=1
    echo "Using a default of 1"
else
    ROUND=$(< $ROUND_TRACKING_FILE)
fi

echo "Starting at round $ROUND"

while true
do
    uv run process_round.py --round $ROUND --loop
    uv run train.py --round-id $ROUND --round-tracking-file $ROUND_TRACKING_FILE
    ROUND=$(< $ROUND_TRACKING_FILE)
    if uv run report-script.py --metric accuracy --validation --patience 3 
    then
	# Successful. Worth continuing
	continue
    else
	break
    fi
done
