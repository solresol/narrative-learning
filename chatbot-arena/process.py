#!/usr/bin/env uv

# Eventually this should download and update the date on the elo.date file.

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--elo", default="elo.csv")
args = parser.parse_args()
import pandas
df = pandas.read_csv(args.elo)
for m in df.Model:
    print(m)
#print(df)
#print(df.columns)
#print(df[['Model', 'Arena Score', '95% CI']])
