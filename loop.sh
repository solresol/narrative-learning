#!/bin/bash

if [ "$ROUND_TRACKING_FILE" = "" ]
then
    echo "ROUND_TRACKING_FILE is not defined"
    exit 1
fi

echo "Using $ROUND_TRACKING_FILE to track the current round"

if [ "$NARRATIVE_LEARNING_DATABASE" = "" ]
then
    echo "NARRATIVE_LEARNING_DATABASE is not defined"
    exit 1
fi

echo "Storing rounds and prompts in $NARRATIVE_LEARNING_DATABASE"

if [ ! -e $ROUND_TRACKING_FILE ]
then
    ROUND=1
    echo "Using a default of 1"
else
    ROUND=$(< $ROUND_TRACKING_FILE)
fi

echo "Starting at round $ROUND"

if [ "$NARRATIVE_LEARNING_PATIENCE" = "" ]
then
    NARRATIVE_LEARNING_PATIENCE=3
fi

echo "I will give up once I have had $NARRATIVE_LEARNING_PATIENCE rounds without an improvement"


while uv run report-script.py --metric accuracy --validation --patience $NARRATIVE_LEARNING_PATIENCE
do
    uv run process_round.py --round $ROUND --loop --progress-bar
    # This runs train one more time than is actually necessary
    uv run train.py --round-id $ROUND --round-tracking-file $ROUND_TRACKING_FILE
    ROUND=$(< $ROUND_TRACKING_FILE)
done

BEST_ROUND=$(uv run report-script.py --best)

if [ "$BEST_ROUND" != "" ]
then
    OUTFILE="${NARRATIVE_LEARNING_DATABASE%.sqlite}.best-round.txt"
    echo $BEST_ROUND > $OUTFILE
fi
