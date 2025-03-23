#!/bin/bash

rm -f outputs/wisconsin_results.csv outputs/titanic_results.csv outputs/southgermancredit_results.csv \
    && make outputs/wisconsin_results.csv outputs/titanic_results.csv outputs/southgermancredit_results.csv \
    && git add outputs/wisconsin_results.csv outputs/titanic_results.csv outputs/southgermancredit_results.csv \
    && make outputs/impact-of-samples.tex outputs/model_details.tex outputs/titanic_by_model_size.png outputs/wisconsin_by_model_size.png outputs/southgermancredit_by_model_size.png \
    && git add -f outputs/impact-of-samples.tex outputs/sample-count-impact-chart.png outputs/model_details.tex outputs/titanic_by_model_size.png outputs/wisconsin_by_model_size.png outputs/southgermancredit_by_model_size.png \
    && git commit -m"Latest experimental results"
