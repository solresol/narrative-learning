#!/bin/bash

ROUND=1
echo "Starting at round $ROUND"

echo "Starting at round $ROUND"

if [ "$NARRATIVE_LEARNING_PATIENCE" = "" ]
then
    NARRATIVE_LEARNING_PATIENCE=3
fi

echo "I will give up once I have had $NARRATIVE_LEARNING_PATIENCE rounds without an improvement"


while uv run report-script.py --metric accuracy --validation --patience $NARRATIVE_LEARNING_PATIENCE
do
    uv run process_round.py --round $ROUND --loop --progress-bar || exit 1
    # This runs train one more time than is actually necessary
    uv run train.py --round-id $ROUND --verbose || exit 1
    ROUND=$((ROUND + 1))
done

BEST_ROUND=$(uv run report-script.py --best)

if [ "$BEST_ROUND" != "" ]
then
    echo "Best round: $BEST_ROUND"
fi
