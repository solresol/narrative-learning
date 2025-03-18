#!/bin/bash

rm -f outputs/wisconsin_results.csv outputs/titanic_results.csv outputs/southgermancredit_results.csv \
    && make outputs/wisconsin_results.csv outputs/titanic_results.csv outputs/southgermancredit_results.csv \
    && git add outputs/wisconsin_results.csv outputs/titanic_results.csv outputs/southgermancredit_results.csv \
    && git commit -m"Latest experimental results"
