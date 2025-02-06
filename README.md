# narrative-learning

What if a text-based explanation was the machine learning model?

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
