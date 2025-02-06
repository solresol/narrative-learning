# narrative-learning

What if a text-based explanation was the machine learning model?


## Explanation

Normally we train up a model, and then we create an explanation so that human beings can understand it. Ideally, understanding should be sufficient for a human being to reproduce the activity of the computer -- it should be (English) language human-readable text. But now that computers can act on the instructions from human-readable text, the human-readable text could be the model itself. If so, we don't need the model: we could iterate (the LLM improving the human-readable text; evaluate the results). Small studies would be "how dumb can the evaluating model be?"; "how dumb can the training/prompt-improver model be?"; "how sensitive is it to the amount of history or the number of examples given?"; "how does it compare to other classification techniques?"

### Side question

How would we do narrative learning regressors?


## Data prep

`python initialise_database.py`

Maybe do this, but this might be a bad idea after a few inferences. It's not needed
for any other step.

`sqlite3 titanic_medical.sqlite ".dump" > titanic_medical.sql`

Check to see if it works... this should produce a long list:

`./process_round.py --list --round 1`


## Operation

This will get things going.

`./process_round.py --list --round 1` 

I haven't implemented the actual training yet.
