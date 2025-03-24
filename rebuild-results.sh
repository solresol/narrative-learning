#!/bin/bash

rm -f outputs/wisconsin_results.csv outputs/titanic_results.csv outputs/southgermancredit_results.csv \
    && make outputs/wisconsin_results.csv outputs/titanic_results.csv outputs/southgermancredit_results.csv \
    && git add outputs/wisconsin_results.csv outputs/titanic_results.csv outputs/southgermancredit_results.csv \
    && make outputs/impact-of-samples.tex outputs/model_details.tex outputs/titanic_by_model_size.png outputs/wisconsin_by_model_size.png outputs/southgermancredit_by_model_size.png \
    && make outputs/herdan-model-size-trend.png \
    && make outputs/wisconsin_error_rate_by_herdan.png outputs/southgermancredit_error_rate_by_herdan.png outputs/titanic_error_rate_by_herdan.png \
    && make outputs/wisconsin_error_rate_by_prompt_wordcount.png outputs/wisconsin_error_rate_by_reasoning_wordcount.png outputs/wisconsin_error_rate_by_cumulative_wordcount.png \
    && make outputs/titanic_error_rate_by_prompt_wordcount.png outputs/titanic_error_rate_by_reasoning_wordcount.png outputs/titanic_error_rate_by_cumulative_wordcount.png \
    && make outputs/southgermancredit_error_rate_by_prompt_wordcount.png outputs/southgermancredit_error_rate_by_reasoning_wordcount.png outputs/southgermancredit_error_rate_by_cumulative_wordcount.png \
    && git add -f outputs/impact-of-samples.tex \
    && git add -f outputs/sample-count-impact-chart.png \
    && git add -f outputs/model_details.tex \
    && git add -f outputs/titanic_by_model_size.png outputs/wisconsin_by_model_size.png outputs/southgermancredit_by_model_size.png \
    && git add -f outputs/wisconsin_model_projection.tex outputs/titanic_model_projection.tex  outputs/southgermancredit_model_projection.tex \
    && git add -f outputs/herdan-model-size-trend.png outputs/herdan-model-size-definitions.tex \
    && git add -f outputs/wisconsin_error_rate_by_herdan.png outputs/wisconsin_error_rate_by_herdan_pvalue.tex outputs/wisconsin_error_rate_by_herdan_slope.tex outputs/southgermancredit_error_rate_by_herdan.png outputs/southgermancredit_error_rate_by_herdan_pvalue.tex outputs/southgermancredit_error_rate_by_herdan_slope.tex outputs/titanic_error_rate_by_herdan.png outputs/titanic_error_rate_by_herdan_pvalue.tex outputs/titanic_error_rate_by_herdan_slope.tex \
    && git add -f outputs/wisconsin_error_rate_by_prompt_wordcount.png outputs/wisconsin_error_rate_by_prompt_wordcount_pvalue.tex outputs/wisconsin_error_rate_by_prompt_wordcount_slope.tex \
    && git add -f outputs/wisconsin_error_rate_by_reasoning_wordcount.png outputs/wisconsin_error_rate_by_reasoning_wordcount_pvalue.tex outputs/wisconsin_error_rate_by_reasoning_wordcount_slope.tex \
    && git add -f outputs/wisconsin_error_rate_by_cumulative_wordcount.png outputs/wisconsin_error_rate_by_cumulative_wordcount_pvalue.tex outputs/wisconsin_error_rate_by_cumulative_wordcount_slope.tex \
    && git add -f outputs/titanic_error_rate_by_prompt_wordcount.png outputs/titanic_error_rate_by_prompt_wordcount_pvalue.tex outputs/titanic_error_rate_by_prompt_wordcount_slope.tex \
    && git add -f outputs/titanic_error_rate_by_reasoning_wordcount.png outputs/titanic_error_rate_by_reasoning_wordcount_pvalue.tex outputs/titanic_error_rate_by_reasoning_wordcount_slope.tex \
    && git add -f outputs/titanic_error_rate_by_cumulative_wordcount.png outputs/titanic_error_rate_by_cumulative_wordcount_pvalue.tex outputs/titanic_error_rate_by_cumulative_wordcount_slope.tex \
    && git add -f outputs/southgermancredit_error_rate_by_prompt_wordcount.png outputs/southgermancredit_error_rate_by_prompt_wordcount_pvalue.tex outputs/southgermancredit_error_rate_by_prompt_wordcount_slope.tex \
    && git add -f outputs/southgermancredit_error_rate_by_reasoning_wordcount.png outputs/southgermancredit_error_rate_by_reasoning_wordcount_pvalue.tex outputs/southgermancredit_error_rate_by_reasoning_wordcount_slope.tex \
    && git add -f outputs/southgermancredit_error_rate_by_cumulative_wordcount.png outputs/southgermancredit_error_rate_by_cumulative_wordcount_pvalue.tex outputs/southgermancredit_error_rate_by_cumulative_wordcount_slope.tex \
    && git commit -m"Latest experimental results"
