.PRECIOUS: %.sqlite
.SECONDARY: %.sqlite

# Real datasets, mutated
WISCONSIN_DATASET := wisconsin_exoplanets
TITANIC_DATASET := titanic_medical
SGC_DATASET := sgc_coral

# Fake datasets
ESPIONAGE_DATASET := espionage
TIMETRAVEL_INSURANCE_DATASET := timetravel_insurance
POTIONS_DATASET := potions

# Models not to add (they generally don't work)... phi falcon falcon10 gemma llamaphi deepseek qwq llama
# More problematic models: anthropic anthropic10
# It turns out that anthropic and anthropic10 were using haiku for training, and sonnet for inference! No wonder it was so expensive and lackluster
MODELS := openai openai10 openai10o1 openai45 openai4510 openailong openaio1 anthropic37 anthropic3710 gemini geminipro gemini10 geminipro10 anthropic anthropic10 gemini25 openaio3 openaio310 openai41 openai4110 openai35 opus40 opus4010 sonnet40 sonnet4010
TEMPLATES_DIR := dbtemplates
RESULTS_DIR := results

.PHONY: all wisconsin titanic sgc espionage timetravel_insurance potions wisconsin-distributions titanic-distributions southgermancredit-distributions espionage-distributions timetravel_insurance-distributions potions-distributions ensembles ensembles5

all: wisconsin titanic southgermancredit espionage timetravel_insurance potions ensembles outputs/impact-of-samples.tex outputs/model_details.tex outputs/titanic_by_model_size.png outputs/wisconsin_by_model_size.png outputs/southgermancredit_by_model_size.png outputs/espionage_by_model_size.png outputs/timetravel_insurance_by_model_size.png outputs/potions_by_model_size.png outputs/titanic_by_elo.png outputs/wisconsin_by_elo.png outputs/southgermancredit_by_elo.png outputs/espionage_by_elo.png outputs/timetravel_insurance_by_elo.png outputs/potions_by_elo.png
	echo Done

ensembles: outputs/titanic_ensemble_summary.txt outputs/wisconsin_ensemble_summary.txt outputs/southgermancredit_ensemble_summary.txt outputs/espionage_ensemble_summary.txt outputs/timetravel_insurance_ensemble_summary.txt outputs/potions_ensemble_summary.txt
        @echo "All ensemble results are stored in the database"
    @echo "Ensembles now use model release dates from the database to track temporal progress"

ensembles5: outputs/titanic_ensemble5_summary.txt outputs/wisconsin_ensemble5_summary.txt outputs/southgermancredit_ensemble5_summary.txt outputs/espionage_ensemble5_summary.txt outputs/timetravel_insurance_ensemble5_summary.txt outputs/potions_ensemble5_summary.txt
        @echo "All k=5 ensemble results are stored in the database"

outputs/model_details.tex: model_details.json make_model_size_table.py
	uv run make_model_size_table.py --output outputs/model_details.tex

outputs/herdan-model-size-trend.png outputs/herdan-model-size-definitions.tex:  outputs/titanic_results.csv  outputs/wisconsin_results.csv outputs/southgermancredit_results.csv outputs/espionage_results.csv outputs/timetravel_insurance_results.csv outputs/potions_results.csv results_lexicostatistics_model_trend.py 
	uv run results_lexicostatistics_model_trend.py --output outputs/herdan-model-size-trend.png --latex outputs/herdan-model-size-definitions.tex outputs/wisconsin_results.csv outputs/titanic_results.csv outputs/southgermancredit_results.csv outputs/espionage_results.csv outputs/timetravel_insurance_results.csv outputs/potions_results.csv

outputs/promptwc-model-size-trend.png outputs/promptwc-model-size-definitions.tex:  outputs/titanic_results.csv  outputs/wisconsin_results.csv outputs/southgermancredit_results.csv outputs/espionage_results.csv outputs/timetravel_insurance_results.csv outputs/potions_results.csv results_wordcount_model_trend.py 
	uv run results_wordcount_model_trend.py --wordcount-type prompt --output outputs/promptwc-model-size-trend.png --latex outputs/promptwc-model-size-definitions.tex outputs/wisconsin_results.csv outputs/titanic_results.csv outputs/southgermancredit_results.csv outputs/espionage_results.csv outputs/timetravel_insurance_results.csv outputs/potions_results.csv

outputs/reasoningwc-model-size-trend.png outputs/reasoningwc-model-size-definitions.tex:  outputs/titanic_results.csv  outputs/wisconsin_results.csv outputs/southgermancredit_results.csv outputs/espionage_results.csv outputs/timetravel_insurance_results.csv outputs/potions_results.csv results_wordcount_model_trend.py 
	uv run results_wordcount_model_trend.py --wordcount-type reasoning --output outputs/reasoningwc-model-size-trend.png --latex outputs/reasoningwc-model-size-definitions.tex outputs/wisconsin_results.csv outputs/titanic_results.csv outputs/southgermancredit_results.csv outputs/espionage_results.csv outputs/timetravel_insurance_results.csv outputs/potions_results.csv

outputs/cumulativewc-model-size-trend.png outputs/cumulativewc-model-size-definitions.tex:  outputs/titanic_results.csv  outputs/wisconsin_results.csv outputs/southgermancredit_results.csv outputs/espionage_results.csv outputs/timetravel_insurance_results.csv outputs/potions_results.csv results_wordcount_model_trend.py 
	uv run results_wordcount_model_trend.py --wordcount-type cumulative --output outputs/cumulativewc-model-size-trend.png --latex outputs/cumulativewc-model-size-definitions.tex outputs/wisconsin_results.csv outputs/titanic_results.csv outputs/southgermancredit_results.csv outputs/espionage_results.csv outputs/timetravel_insurance_results.csv outputs/potions_results.csv


outputs/impact-of-samples.tex outputs/sample-count-impact-chart.png: outputs/titanic_results.csv outputs/wisconsin_results.csv outputs/southgermancredit_results.csv outputs/espionage_results.csv outputs/timetravel_insurance_results.csv outputs/potions_results.csv resultssampleimpact.py
	uv run resultssampleimpact.py --pivot outputs/samples-pivot-table.csv --image outputs/sample-count-impact-chart.png --stats-results outputs/impact-of-samples.txt --brief-stats outputs/impact-of-samples.tex outputs/southgermancredit_results.csv outputs/titanic_results.csv outputs/wisconsin_results.csv outputs/espionage_results.csv outputs/timetravel_insurance_results.csv outputs/potions_results.csv

######################################################################

# Wisconsin

wisconsin: outputs/wisconsin_results.csv
	echo All Wisconsin results are ready

# Should depend on ... wisconsin-databases wisconsin-estimates wisconsin-results wisconsin-prompts wisconsin-best wisconsin-baseline
outputs/wisconsin_results.csv: $(RESULTS_DIR)/$(WISCONSIN_DATASET).baseline.json
	uv run create_task_csv_file.py --task wisconsin --env-dir envs/wisconsin --output outputs/wisconsin_results.csv --model-details model_details.json --baseline $(RESULTS_DIR)/$(WISCONSIN_DATASET).baseline.json --progress

wisconsin-baseline: $(RESULTS_DIR)/$(WISCONSIN_DATASET).baseline.json
	echo Baseline created

$(RESULTS_DIR)/$(WISCONSIN_DATASET).baseline.json: configs/$(WISCONSIN_DATASET).config.json $(RESULTS_DIR)/$(WISCONSIN_DATASET)-baseline.sqlite baseline.py
	uv run baseline.py --config configs/$(WISCONSIN_DATASET).config.json --database $(RESULTS_DIR)/$(WISCONSIN_DATASET)-baseline.sqlite --output $(RESULTS_DIR)/$(WISCONSIN_DATASET).baseline.json

wisconsin-databases: $(foreach model,$(MODELS),$(RESULTS_DIR)/$(WISCONSIN_DATASET)-$(model).sqlite)
	echo All Wisconsin databases are initialized

wisconsin-estimates: $(foreach model,$(MODELS),$(RESULTS_DIR)/$(WISCONSIN_DATASET)-$(model).estimate.txt)
	echo All Wisconsin estimates are ready

wisconsin-results: $(foreach model,$(MODELS),$(RESULTS_DIR)/$(WISCONSIN_DATASET)-$(model).results.csv)
	echo All Wisconsin results are ready

wisconsin-prompts: $(foreach model,$(MODELS),$(RESULTS_DIR)/$(WISCONSIN_DATASET)-$(model).decoded-best-prompt.txt)
	echo All Wisconsin prompts are ready

wisconsin-best: $(foreach model,$(MODELS),$(RESULTS_DIR)/$(WISCONSIN_DATASET)-$(model).best-round.txt)
	echo All Wisconsin best-round files are ready

wisconsin-distributions: $(foreach model,$(MODELS),outputs/$(WISCONSIN_DATASET)-$(model).distribution.png)
	echo All Wisconsin distribution files are ready

outputs/$(WISCONSIN_DATASET)-%.distribution.png outputs/$(WISCONSIN_DATASET)-%.distribution.txt: envs/wisconsin/%.env
	uv run resultdistribution.py --env-file $< --distribution-image outputs/$(WISCONSIN_DATASET)-$*.distribution.png --fitted-distribution outputs/$(WISCONSIN_DATASET)-$*.distribution.txt

$(RESULTS_DIR)/$(WISCONSIN_DATASET)-%.best-round.txt: $(RESULTS_DIR)/$(WISCONSIN_DATASET)-%.sqlite
	. ./envs/wisconsin/$*.env && ./loop.sh

$(RESULTS_DIR)/$(WISCONSIN_DATASET)-%.sqlite: $(TEMPLATES_DIR)/$(WISCONSIN_DATASET).sql
	sqlite3 $@ < $<

$(RESULTS_DIR)/$(WISCONSIN_DATASET)-%.estimate.txt: $(RESULTS_DIR)/$(WISCONSIN_DATASET)-%.best-round.txt
	. ./envs/wisconsin/$*.env && uv run report-script.py --estimate accuracy > $@

$(RESULTS_DIR)/$(WISCONSIN_DATASET)-%.results.csv: $(RESULTS_DIR)/$(WISCONSIN_DATASET)-%.best-round.txt
	. ./envs/wisconsin/$*.env && uv run report-script.py --train --validation --test --show
	. ./envs/wisconsin/$*.env && uv run report-script.py --train --validation --test --csv $@

$(RESULTS_DIR)/$(WISCONSIN_DATASET)-%.decoded-best-prompt.txt: $(RESULTS_DIR)/$(WISCONSIN_DATASET)-%.best-round.txt
	. ./envs/wisconsin/$*.env && uv run decode.py --round-id $(shell cat $<) --encoding-instructions conversions/breast_cancer --verbose --output $@


outputs/wisconsin_by_model_size.png outputs/wisconsin_model_pvalue.tex: outputs/wisconsin_results.csv results_chart_by_size.py
	uv run results_chart_by_size.py --dataset Wisconsin --input outputs/wisconsin_results.csv --image outputs/wisconsin_by_model_size.png --pvalue outputs/wisconsin_model_pvalue.tex --projection outputs/wisconsin_model_projection.tex

outputs/wisconsin_by_elo.png outputs/wisconsin_elo_pvalue.tex: outputs/wisconsin_results.csv results_chart_by_elo.py chatbot-arena/elo.csv chatbot-arena/translation.json
	uv run results_chart_by_elo.py --dataset Wisconsin --input outputs/wisconsin_results.csv --elo chatbot-arena/elo.csv --elo-translation chatbot-arena/translation.json --output outputs/wisconsin_by_elo.png --pvalue outputs/wisconsin_elo_pvalue.tex

outputs/wisconsin_error_rate_by_herdan.png outputs/wisconsin_error_rate_by_herdan_pvalue.tex  outputs/wisconsin_error_rate_by_herdan_slope.tex: outputs/wisconsin_results.csv results_error_rate_by_herdan.py
	uv run results_error_rate_by_herdan.py --show  --image-output outputs/wisconsin_error_rate_by_herdan.png --pvalue-output outputs/wisconsin_error_rate_by_herdan_pvalue.tex  --slope-output outputs/wisconsin_error_rate_by_herdan_slope.tex outputs/wisconsin_results.csv

outputs/wisconsin_error_rate_by_prompt_wordcount.png outputs/wisconsin_error_rate_by_prompt_wordcount_pvalue.tex outputs/wisconsin_error_rate_by_prompt_wordcount_slope.tex: outputs/wisconsin_results.csv results_error_rate_by_wordcount.py
	uv run results_error_rate_by_wordcount.py --wordcount-type prompt --show --image-output outputs/wisconsin_error_rate_by_prompt_wordcount.png --pvalue-output outputs/wisconsin_error_rate_by_prompt_wordcount_pvalue.tex --slope-output outputs/wisconsin_error_rate_by_prompt_wordcount_slope.tex outputs/wisconsin_results.csv

outputs/wisconsin_error_rate_by_reasoning_wordcount.png outputs/wisconsin_error_rate_by_reasoning_wordcount_pvalue.tex outputs/wisconsin_error_rate_by_reasoning_wordcount_slope.tex: outputs/wisconsin_results.csv results_error_rate_by_wordcount.py
	uv run results_error_rate_by_wordcount.py --wordcount-type reasoning --show --image-output outputs/wisconsin_error_rate_by_reasoning_wordcount.png --pvalue-output outputs/wisconsin_error_rate_by_reasoning_wordcount_pvalue.tex --slope-output outputs/wisconsin_error_rate_by_reasoning_wordcount_slope.tex outputs/wisconsin_results.csv

outputs/wisconsin_error_rate_by_cumulative_wordcount.png outputs/wisconsin_error_rate_by_cumulative_wordcount_pvalue.tex outputs/wisconsin_error_rate_by_cumulative_wordcount_slope.tex: outputs/wisconsin_results.csv results_error_rate_by_wordcount.py
	uv run results_error_rate_by_wordcount.py --wordcount-type cumulative --show --image-output outputs/wisconsin_error_rate_by_cumulative_wordcount.png --pvalue-output outputs/wisconsin_error_rate_by_cumulative_wordcount_pvalue.tex --slope-output outputs/wisconsin_error_rate_by_cumulative_wordcount_slope.tex outputs/wisconsin_results.csv

outputs/wisconsin_ensemble_summary.txt: configs/$(WISCONSIN_DATASET).config.json results_ensembling.py postgres-schemas/model_release_dates.sql postgres-schemas/ensemble_results.sql
        uv run results_ensembling.py --env-dir envs/wisconsin --progress-bar --summary outputs/wisconsin_ensemble_summary.txt

outputs/wisconsin_ensemble5_summary.txt: configs/$(WISCONSIN_DATASET).config.json results_ensembling.py postgres-schemas/model_release_dates.sql postgres-schemas/ensemble_results.sql
        uv run results_ensembling.py --env-dir envs/wisconsin --progress-bar --summary outputs/wisconsin_ensemble5_summary.txt --k 5


# How we created the dataset
obfuscations/breast_cancer: conversions/breast_cancer obfuscation_plan_generator.py datasets/breast_cancer.csv
	uv run obfuscation_plan_generator.py --csv-file datasets/breast_cancer.csv  --obfuscation-plan obfuscations/breast_cancer --guidelines conversions/breast_cancer

# Should add obfuscations/breast_cancer as a dependency, but that means the template seems to get rebuilt
# too often. I guess it's because obfuscations/breast_cancer gets opened and updated by initialise_database.py
$(TEMPLATES_DIR)/$(WISCONSIN_DATASET).sql configs/$(WISCONSIN_DATASET).config.json: datasets/breast_cancer.csv initialise_database.py
	uv run initialise_database.py --database $(TEMPLATES_DIR)/$(WISCONSIN_DATASET).sqlite --source datasets/breast_cancer.csv --config-file configs/$(WISCONSIN_DATASET).config.json --obfuscation obfuscations/breast_cancer --verbose
	sqlite3 $(TEMPLATES_DIR)/$(WISCONSIN_DATASET).sqlite .dump > $(TEMPLATES_DIR)/$(WISCONSIN_DATASET).sql
	rm -f $(TEMPLATES_DIR)/$(WISCONSIN_DATASET).sqlite


######################################################################

# How we created the dataset
obfuscations/titanic: conversions/titanic obfuscation_plan_generator.py datasets/titanic.csv
	uv run obfuscation_plan_generator.py --csv-file datasets/titanic.csv  --obfuscation-plan obfuscations/titanic --guidelines conversions/titanic

$(TEMPLATES_DIR)/$(TITANIC_DATASET).sql configs/titanic_medical.config.json: datasets/titanic.csv initialise_database.py
	uv run initialise_database.py --database $(TEMPLATES_DIR)/titanic_medical.sqlite --source datasets/titanic.csv --config-file configs/titanic_medical.config.json --obfuscation obfuscations/titanic --verbose
	sqlite3 $(TEMPLATES_DIR)/$(TITANIC_DATASET).sqlite .dump > $(TEMPLATES_DIR)/$(TITANIC_DATASET).sql
	rm -f $(TEMPLATES_DIR)/$(TITANIC_DATASET).sqlite


$(RESULTS_DIR)/$(TITANIC_DATASET)-%.best-round.txt: $(RESULTS_DIR)/$(TITANIC_DATASET)-%.sqlite
	. ./envs/titanic/$*.env && ./loop.sh

$(RESULTS_DIR)/$(TITANIC_DATASET)-%.sqlite: $(TEMPLATES_DIR)/$(TITANIC_DATASET).sql
	sqlite3 $@ < $<

$(RESULTS_DIR)/$(TITANIC_DATASET)-%.estimate.txt: $(RESULTS_DIR)/$(TITANIC_DATASET)-%.best-round.txt
	. ./envs/titanic/$*.env && uv run report-script.py --estimate accuracy > $@

$(RESULTS_DIR)/$(TITANIC_DATASET)-%.results.csv: $(RESULTS_DIR)/$(TITANIC_DATASET)-%.best-round.txt
	. ./envs/titanic/$*.env && uv run report-script.py --train --validation --test --show
	. ./envs/titanic/$*.env && uv run report-script.py --train --validation --test --csv $@

$(RESULTS_DIR)/$(TITANIC_DATASET)-%.decoded-best-prompt.txt: $(RESULTS_DIR)/$(TITANIC_DATASET)-%.best-round.txt
	. ./envs/titanic/$*.env && uv run decode.py --round-id $(shell cat $<) --encoding-instructions conversions/breast_cancer --verbose --output $@

titanic: outputs/titanic_results.csv
	echo All Titanic results are ready

# Again, should depend on titanic-estimates titanic-results titanic-prompts titanic-best titanic-baseline
# I'm not sure if this globbing will work
outputs/titanic_results.csv: $(RESULTS_DIR)/$(TITANIC_DATASET).baseline.json
	uv run create_task_csv_file.py --task titanic --env-dir envs/titanic --output outputs/titanic_results.csv --model-details model_details.json --baseline $(RESULTS_DIR)/$(TITANIC_DATASET).baseline.json  --progress

titanic-baseline: $(RESULTS_DIR)/$(TITANIC_DATASET).baseline.json
	echo Baseline created

$(RESULTS_DIR)/$(TITANIC_DATASET).baseline.json: configs/$(TITANIC_DATASET).config.json $(RESULTS_DIR)/$(TITANIC_DATASET)-baseline.sqlite baseline.py
	uv run baseline.py --config configs/$(TITANIC_DATASET).config.json --database $(RESULTS_DIR)/$(TITANIC_DATASET)-baseline.sqlite --output $(RESULTS_DIR)/$(TITANIC_DATASET).baseline.json

# Add these targets to depend on all models
titanic-databases: $(foreach model,$(MODELS),$(RESULTS_DIR)/$(TITANIC_DATASET)-$(model).sqlite)
	@echo All Titanic databases are initialised


titanic-estimates: $(foreach model,$(MODELS),$(RESULTS_DIR)/$(TITANIC_DATASET)-$(model).estimate.txt)
	echo All Titanic estimates are ready

titanic-results: $(foreach model,$(MODELS),$(RESULTS_DIR)/$(TITANIC_DATASET)-$(model).results.csv)
	echo All Titanic results are ready

titanic-prompts: $(foreach model,$(MODELS),$(RESULTS_DIR)/$(TITANIC_DATASET)-$(model).decoded-best-prompt.txt)
	echo All Titanic prompts are ready

titanic-best: $(foreach model,$(MODELS),$(RESULTS_DIR)/$(TITANIC_DATASET)-$(model).best-round.txt)
	echo All Titanic best-round files are ready

titanic-distributions: $(foreach model,$(MODELS),outputs/$(TITANIC_DATASET)-$(model).distribution.png)
	echo All Titanic distribution files are ready

outputs/$(TITANIC_DATASET)-%.distribution.png outputs/$(TITANIC_DATASET)-%.distribution.txt: envs/titanic/%.env
	uv run resultdistribution.py --env-file $< --distribution-image outputs/$(TITANIC_DATASET)-$*.distribution.png --fitted-distribution outputs/$(TITANIC_DATASET)-$*.distribution.txt

outputs/titanic_by_model_size.png outputs/titanic_model_pvalue.tex: outputs/titanic_results.csv results_chart_by_size.py
	uv run results_chart_by_size.py --dataset Titanic --input outputs/titanic_results.csv --image outputs/titanic_by_model_size.png --pvalue outputs/titanic_model_pvalue.tex --projection outputs/titanic_model_projection.tex

outputs/titanic_by_elo.png outputs/titanic_elo_pvalue.tex: outputs/titanic_results.csv results_chart_by_elo.py chatbot-arena/elo.csv chatbot-arena/translation.json
	uv run results_chart_by_elo.py --dataset Titanic --input outputs/titanic_results.csv --elo chatbot-arena/elo.csv --elo-translation chatbot-arena/translation.json --output outputs/titanic_by_elo.png --pvalue outputs/titanic_elo_pvalue.tex

outputs/titanic_error_rate_by_herdan.png outputs/titanic_error_rate_by_herdan_pvalue.tex  outputs/titanic_error_rate_by_herdan_slope.tex: outputs/titanic_results.csv results_error_rate_by_herdan.py
	uv run results_error_rate_by_herdan.py --show  --image-output outputs/titanic_error_rate_by_herdan.png --pvalue-output outputs/titanic_error_rate_by_herdan_pvalue.tex  --slope-output outputs/titanic_error_rate_by_herdan_slope.tex outputs/titanic_results.csv

outputs/titanic_error_rate_by_prompt_wordcount.png outputs/titanic_error_rate_by_prompt_wordcount_pvalue.tex outputs/titanic_error_rate_by_prompt_wordcount_slope.tex: outputs/titanic_results.csv results_error_rate_by_wordcount.py
	uv run results_error_rate_by_wordcount.py --wordcount-type prompt --show --image-output outputs/titanic_error_rate_by_prompt_wordcount.png --pvalue-output outputs/titanic_error_rate_by_prompt_wordcount_pvalue.tex --slope-output outputs/titanic_error_rate_by_prompt_wordcount_slope.tex outputs/titanic_results.csv

outputs/titanic_error_rate_by_reasoning_wordcount.png outputs/titanic_error_rate_by_reasoning_wordcount_pvalue.tex outputs/titanic_error_rate_by_reasoning_wordcount_slope.tex: outputs/titanic_results.csv results_error_rate_by_wordcount.py
	uv run results_error_rate_by_wordcount.py --wordcount-type reasoning --show --image-output outputs/titanic_error_rate_by_reasoning_wordcount.png --pvalue-output outputs/titanic_error_rate_by_reasoning_wordcount_pvalue.tex --slope-output outputs/titanic_error_rate_by_reasoning_wordcount_slope.tex outputs/titanic_results.csv

outputs/titanic_error_rate_by_cumulative_wordcount.png outputs/titanic_error_rate_by_cumulative_wordcount_pvalue.tex outputs/titanic_error_rate_by_cumulative_wordcount_slope.tex: outputs/titanic_results.csv results_error_rate_by_wordcount.py
	uv run results_error_rate_by_wordcount.py --wordcount-type cumulative --show --image-output outputs/titanic_error_rate_by_cumulative_wordcount.png --pvalue-output outputs/titanic_error_rate_by_cumulative_wordcount_pvalue.tex --slope-output outputs/titanic_error_rate_by_cumulative_wordcount_slope.tex outputs/titanic_results.csv

outputs/titanic_ensemble_summary.txt: configs/$(TITANIC_DATASET).config.json results_ensembling.py postgres-schemas/model_release_dates.sql postgres-schemas/ensemble_results.sql
        uv run results_ensembling.py --env-dir envs/titanic --progress-bar --summary outputs/titanic_ensemble_summary.txt

outputs/titanic_ensemble5_summary.txt: configs/$(TITANIC_DATASET).config.json results_ensembling.py postgres-schemas/model_release_dates.sql postgres-schemas/ensemble_results.sql
        uv run results_ensembling.py --env-dir envs/titanic --progress-bar --summary outputs/titanic_ensemble5_summary.txt --k 5



######################################################################


# How we created the dataset
obfuscations/southgermancredit: conversions/southgermancredit obfuscation_plan_generator.py datasets/southgermancredit.csv
	uv run obfuscation_plan_generator.py --csv-file datasets/southgermancredit.csv  --obfuscation-plan obfuscations/southgermancredit --guidelines conversions/southgermancredit

$(TEMPLATES_DIR)/$(SGC_DATASET).sql configs/$(SGC_DATASET).config.json: datasets/southgermancredit.csv initialise_database.py
	uv run initialise_database.py --database $(TEMPLATES_DIR)/$(SGC_DATASET).sqlite --source datasets/southgermancredit.csv --config-file configs/$(SGC_DATASET).config.json --obfuscation obfuscations/southgermancredit --verbose
	sqlite3 $(TEMPLATES_DIR)/$(SGC_DATASET).sqlite .dump > $(TEMPLATES_DIR)/$(SGC_DATASET).sql
	rm -f $(TEMPLATES_DIR)/$(SGC_DATASET).sqlite


$(RESULTS_DIR)/$(SGC_DATASET)-%.best-round.txt: $(RESULTS_DIR)/$(SGC_DATASET)-%.sqlite
	. ./envs/southgermancredit/$*.env && ./loop.sh

$(RESULTS_DIR)/$(SGC_DATASET)-%.sqlite: $(TEMPLATES_DIR)/$(SGC_DATASET).sql
	sqlite3 $@ < $<

$(RESULTS_DIR)/$(SGC_DATASET)-%.estimate.txt: $(RESULTS_DIR)/$(SGC_DATASET)-%.best-round.txt
	. ./envs/southgermancredit/$*.env && uv run report-script.py --estimate accuracy > $@

$(RESULTS_DIR)/$(SGC_DATASET)-%.results.csv: $(RESULTS_DIR)/$(SGC_DATASET)-%.best-round.txt
	. ./envs/southgermancredit/$*.env && uv run report-script.py --train --validation --test --show
	. ./envs/southgermancredit/$*.env && uv run report-script.py --train --validation --test --csv $@

$(RESULTS_DIR)/$(SGC_DATASET)-%.decoded-best-prompt.txt: $(RESULTS_DIR)/$(SGC_DATASET)-%.best-round.txt
	. ./envs/southgermancredit/$*.env && uv run decode.py --round-id $(shell cat $<) --encoding-instructions conversions/southgermancredit --verbose --output $@

southgermancredit: outputs/southgermancredit_results.csv
	echo All SouthGermanCredit results are ready

# Again, should depend on southgermancredit-estimates southgermancredit-results southgermancredit-prompts southgermancredit-best southgermancredit-baseline
outputs/southgermancredit_results.csv: $(RESULTS_DIR)/$(SGC_DATASET).baseline.json
	uv run create_task_csv_file.py --task southgermancredit --env-dir envs/southgermancredit --output outputs/southgermancredit_results.csv --model-details model_details.json --baseline $(RESULTS_DIR)/$(SGC_DATASET).baseline.json  --progress

southgermancredit-baseline: $(RESULTS_DIR)/$(SGC_DATASET).baseline.json
	echo Baseline created

$(RESULTS_DIR)/$(SGC_DATASET).baseline.json: configs/$(SGC_DATASET).config.json $(RESULTS_DIR)/$(SGC_DATASET)-baseline.sqlite baseline.py
	uv run baseline.py --config configs/$(SGC_DATASET).config.json --database $(RESULTS_DIR)/$(SGC_DATASET)-baseline.sqlite --output $(RESULTS_DIR)/$(SGC_DATASET).baseline.json

# Add these targets to depend on all models
southgermancredit-databases: $(foreach model,$(MODELS),$(RESULTS_DIR)/$(SGC_DATASET)-$(model).sqlite)
	@echo All SouthGermanCredit databases are initialised


southgermancredit-estimates: $(foreach model,$(MODELS),$(RESULTS_DIR)/$(SGC_DATASET)-$(model).estimate.txt)
	echo All SouthGermanCredit estimates are ready

southgermancredit-results: $(foreach model,$(MODELS),$(RESULTS_DIR)/$(SGC_DATASET)-$(model).results.csv)
	echo All SouthGermanCredit results are ready

southgermancredit-prompts: $(foreach model,$(MODELS),$(RESULTS_DIR)/$(SGC_DATASET)-$(model).decoded-best-prompt.txt)
	echo All SouthGermanCredit prompts are ready

southgermancredit-best: $(foreach model,$(MODELS),$(RESULTS_DIR)/$(SGC_DATASET)-$(model).best-round.txt)
	echo All SouthGermanCredit best-round files are ready

southgermancredit-distributions: $(foreach model,$(MODELS),outputs/$(SGC_DATASET)-$(model).distribution.png)
	echo All SouthGermanCredit distribution files are ready

outputs/$(SGC_DATASET)-%.distribution.png outputs/$(SGC_DATASET)-%.distribution.txt: envs/southgermancredit/%.env
	uv run resultdistribution.py --env-file $< --distribution-image outputs/$(SGC_DATASET)-$*.distribution.png --fitted-distribution outputs/$(SGC_DATASET)-$*.distribution.txt

outputs/southgermancredit_by_model_size.png outputs/southgermancredit_model_pvalue.tex: outputs/southgermancredit_results.csv results_chart_by_size.py
	uv run results_chart_by_size.py --dataset "South German Credit" --input outputs/southgermancredit_results.csv --image outputs/southgermancredit_by_model_size.png --pvalue outputs/southgermancredit_model_pvalue.tex --projection outputs/southgermancredit_model_projection.tex

outputs/southgermancredit_by_elo.png outputs/southgermancredit_elo_pvalue.tex: outputs/southgermancredit_results.csv results_chart_by_elo.py chatbot-arena/elo.csv chatbot-arena/translation.json
	uv run results_chart_by_elo.py --dataset "South German Credit" --input outputs/southgermancredit_results.csv --elo chatbot-arena/elo.csv --elo-translation chatbot-arena/translation.json --output outputs/southgermancredit_by_elo.png --pvalue outputs/southgermancredit_elo_pvalue.tex


outputs/southgermancredit_error_rate_by_herdan.png outputs/southgermancredit_error_rate_by_herdan_pvalue.tex  outputs/southgermancredit_error_rate_by_herdan_slope.tex: outputs/southgermancredit_results.csv results_error_rate_by_herdan.py
	uv run results_error_rate_by_herdan.py --show  --image-output outputs/southgermancredit_error_rate_by_herdan.png --pvalue-output outputs/southgermancredit_error_rate_by_herdan_pvalue.tex  --slope-output outputs/southgermancredit_error_rate_by_herdan_slope.tex outputs/southgermancredit_results.csv

outputs/southgermancredit_error_rate_by_prompt_wordcount.png outputs/southgermancredit_error_rate_by_prompt_wordcount_pvalue.tex outputs/southgermancredit_error_rate_by_prompt_wordcount_slope.tex: outputs/southgermancredit_results.csv results_error_rate_by_wordcount.py
	uv run results_error_rate_by_wordcount.py --wordcount-type prompt --show --image-output outputs/southgermancredit_error_rate_by_prompt_wordcount.png --pvalue-output outputs/southgermancredit_error_rate_by_prompt_wordcount_pvalue.tex --slope-output outputs/southgermancredit_error_rate_by_prompt_wordcount_slope.tex outputs/southgermancredit_results.csv

outputs/southgermancredit_error_rate_by_reasoning_wordcount.png outputs/southgermancredit_error_rate_by_reasoning_wordcount_pvalue.tex outputs/southgermancredit_error_rate_by_reasoning_wordcount_slope.tex: outputs/southgermancredit_results.csv results_error_rate_by_wordcount.py
	uv run results_error_rate_by_wordcount.py --wordcount-type reasoning --show --image-output outputs/southgermancredit_error_rate_by_reasoning_wordcount.png --pvalue-output outputs/southgermancredit_error_rate_by_reasoning_wordcount_pvalue.tex --slope-output outputs/southgermancredit_error_rate_by_reasoning_wordcount_slope.tex outputs/southgermancredit_results.csv

outputs/southgermancredit_error_rate_by_cumulative_wordcount.png outputs/southgermancredit_error_rate_by_cumulative_wordcount_pvalue.tex outputs/southgermancredit_error_rate_by_cumulative_wordcount_slope.tex: outputs/southgermancredit_results.csv results_error_rate_by_wordcount.py
	uv run results_error_rate_by_wordcount.py --wordcount-type cumulative --show --image-output outputs/southgermancredit_error_rate_by_cumulative_wordcount.png --pvalue-output outputs/southgermancredit_error_rate_by_cumulative_wordcount_pvalue.tex --slope-output outputs/southgermancredit_error_rate_by_cumulative_wordcount_slope.tex outputs/southgermancredit_results.csv

outputs/southgermancredit_ensemble_summary.txt: configs/$(SGC_DATASET).config.json results_ensembling.py postgres-schemas/model_release_dates.sql postgres-schemas/ensemble_results.sql
        uv run results_ensembling.py --env-dir envs/southgermancredit --progress-bar --summary outputs/southgermancredit_ensemble_summary.txt

outputs/southgermancredit_ensemble5_summary.txt: configs/$(SGC_DATASET).config.json results_ensembling.py postgres-schemas/model_release_dates.sql postgres-schemas/ensemble_results.sql
        uv run results_ensembling.py --env-dir envs/southgermancredit --progress-bar --summary outputs/southgermancredit_ensemble5_summary.txt --k 5


######################################################################

# Espionage dataset -- 0% noise
$(TEMPLATES_DIR)/$(ESPIONAGE_DATASET).sql configs/$(ESPIONAGE_DATASET).config.json:
	uv run ./random_classification_data_generator.py --number-of-data-points 200 --feature1-name "SecretHandshakeQuality" --feature1-mean 70 --feature1-stddev 10 --feature2-name "AccentThickness" --feature2-mean 30 --feature2-stddev 8 --target-column-name "AgentStatus" --primary-key-name "AgentID" --table-name "espionage_agents"  --splits-table-name "espionage_splits" --random-seed 42  --class-deciding-expression "SecretHandshakeQuality - AccentThickness > 35" --false-class-name "Loyal"  --true-class-name "DoubleAgent" --noise 0.0 --holdout 0.2 --validation 0.5 --output-db-file $(TEMPLATES_DIR)/$(ESPIONAGE_DATASET).sqlite --config-file  configs/$(ESPIONAGE_DATASET).config.json --use-uuid
	sqlite3 $(TEMPLATES_DIR)/$(ESPIONAGE_DATASET).sqlite .dump > $(TEMPLATES_DIR)/$(ESPIONAGE_DATASET).sql
	rm -f $(TEMPLATES_DIR)/$(ESPIONAGE_DATASET).sqlite

$(RESULTS_DIR)/$(ESPIONAGE_DATASET)-%.best-round.txt: $(RESULTS_DIR)/$(ESPIONAGE_DATASET)-%.sqlite
	. ./envs/espionage/$*.env && ./loop.sh

$(RESULTS_DIR)/$(ESPIONAGE_DATASET)-%.sqlite: $(TEMPLATES_DIR)/$(ESPIONAGE_DATASET).sql
	sqlite3 $@ < $<

$(RESULTS_DIR)/$(ESPIONAGE_DATASET)-%.estimate.txt: $(RESULTS_DIR)/$(ESPIONAGE_DATASET)-%.best-round.txt
	. ./envs/espionage/$*.env && uv run report-script.py --estimate accuracy > $@

$(RESULTS_DIR)/$(ESPIONAGE_DATASET)-%.results.csv: $(RESULTS_DIR)/$(ESPIONAGE_DATASET)-%.best-round.txt
	. ./envs/espionage/$*.env && uv run report-script.py --train --validation --test --show
	. ./envs/espionage/$*.env && uv run report-script.py --train --validation --test --csv $@

espionage: outputs/espionage_results.csv
	echo All Espionage results are ready

outputs/espionage_results.csv: $(RESULTS_DIR)/$(ESPIONAGE_DATASET).baseline.json
	uv run create_task_csv_file.py --task espionage --env-dir envs/espionage --output outputs/espionage_results.csv --model-details model_details.json --baseline $(RESULTS_DIR)/$(ESPIONAGE_DATASET).baseline.json --progress

espionage-baseline: $(RESULTS_DIR)/$(ESPIONAGE_DATASET).baseline.json
	echo Baseline created

$(RESULTS_DIR)/$(ESPIONAGE_DATASET).baseline.json: configs/$(ESPIONAGE_DATASET).config.json $(RESULTS_DIR)/$(ESPIONAGE_DATASET)-baseline.sqlite baseline.py
	uv run baseline.py --config configs/$(ESPIONAGE_DATASET).config.json --database $(RESULTS_DIR)/$(ESPIONAGE_DATASET)-baseline.sqlite --output $(RESULTS_DIR)/$(ESPIONAGE_DATASET).baseline.json

# Add these targets to depend on all models
espionage-databases: $(foreach model,$(MODELS),$(RESULTS_DIR)/$(ESPIONAGE_DATASET)-$(model).sqlite)
	@echo All Espionage databases are initialised

espionage-estimates: $(foreach model,$(MODELS),$(RESULTS_DIR)/$(ESPIONAGE_DATASET)-$(model).estimate.txt)
	echo All Espionage estimates are ready

espionage-results: $(foreach model,$(MODELS),$(RESULTS_DIR)/$(ESPIONAGE_DATASET)-$(model).results.csv)
	echo All Espionage results are ready

espionage-best: $(foreach model,$(MODELS),$(RESULTS_DIR)/$(ESPIONAGE_DATASET)-$(model).best-round.txt)
	echo All Espionage best-round files are ready

espionage-distributions: $(foreach model,$(MODELS),outputs/$(ESPIONAGE_DATASET)-$(model).distribution.png)
	echo All Espionage distribution files are ready

outputs/$(ESPIONAGE_DATASET)-%.distribution.png outputs/$(ESPIONAGE_DATASET)-%.distribution.txt: envs/espionage/%.env
	uv run resultdistribution.py --env-file $< --distribution-image outputs/$(ESPIONAGE_DATASET)-$*.distribution.png --fitted-distribution outputs/$(ESPIONAGE_DATASET)-$*.distribution.txt

outputs/espionage_by_model_size.png outputs/espionage_model_pvalue.tex: outputs/espionage_results.csv results_chart_by_size.py
	uv run results_chart_by_size.py --dataset Espionage --input outputs/espionage_results.csv --image outputs/espionage_by_model_size.png --pvalue outputs/espionage_model_pvalue.tex --projection outputs/espionage_model_projection.tex

outputs/espionage_by_elo.png outputs/espionage_elo_pvalue.tex: outputs/espionage_results.csv results_chart_by_elo.py chatbot-arena/elo.csv chatbot-arena/translation.json
	uv run results_chart_by_elo.py --dataset Espionage --input outputs/espionage_results.csv --elo chatbot-arena/elo.csv --elo-translation chatbot-arena/translation.json --output outputs/espionage_by_elo.png --pvalue outputs/espionage_elo_pvalue.tex

outputs/espionage_error_rate_by_herdan.png outputs/espionage_error_rate_by_herdan_pvalue.tex outputs/espionage_error_rate_by_herdan_slope.tex: outputs/espionage_results.csv results_error_rate_by_herdan.py
	uv run results_error_rate_by_herdan.py --show --image-output outputs/espionage_error_rate_by_herdan.png --pvalue-output outputs/espionage_error_rate_by_herdan_pvalue.tex --slope-output outputs/espionage_error_rate_by_herdan_slope.tex outputs/espionage_results.csv

outputs/espionage_error_rate_by_prompt_wordcount.png outputs/espionage_error_rate_by_prompt_wordcount_pvalue.tex outputs/espionage_error_rate_by_prompt_wordcount_slope.tex: outputs/espionage_results.csv results_error_rate_by_wordcount.py
	uv run results_error_rate_by_wordcount.py --wordcount-type prompt --show --image-output outputs/espionage_error_rate_by_prompt_wordcount.png --pvalue-output outputs/espionage_error_rate_by_prompt_wordcount_pvalue.tex --slope-output outputs/espionage_error_rate_by_prompt_wordcount_slope.tex outputs/espionage_results.csv

outputs/espionage_error_rate_by_reasoning_wordcount.png outputs/espionage_error_rate_by_reasoning_wordcount_pvalue.tex outputs/espionage_error_rate_by_reasoning_wordcount_slope.tex: outputs/espionage_results.csv results_error_rate_by_wordcount.py
	uv run results_error_rate_by_wordcount.py --wordcount-type reasoning --show --image-output outputs/espionage_error_rate_by_reasoning_wordcount.png --pvalue-output outputs/espionage_error_rate_by_reasoning_wordcount_pvalue.tex --slope-output outputs/espionage_error_rate_by_reasoning_wordcount_slope.tex outputs/espionage_results.csv

outputs/espionage_error_rate_by_cumulative_wordcount.png outputs/espionage_error_rate_by_cumulative_wordcount_pvalue.tex outputs/espionage_error_rate_by_cumulative_wordcount_slope.tex: outputs/espionage_results.csv results_error_rate_by_wordcount.py
	uv run results_error_rate_by_wordcount.py --wordcount-type cumulative --show --image-output outputs/espionage_error_rate_by_cumulative_wordcount.png --pvalue-output outputs/espionage_error_rate_by_cumulative_wordcount_pvalue.tex --slope-output outputs/espionage_error_rate_by_cumulative_wordcount_slope.tex outputs/espionage_results.csv

outputs/espionage_ensemble_summary.txt: configs/$(ESPIONAGE_DATASET).config.json results_ensembling.py postgres-schemas/model_release_dates.sql postgres-schemas/ensemble_results.sql
        uv run results_ensembling.py --env-dir envs/espionage --progress-bar --summary outputs/espionage_ensemble_summary.txt --no-decodex

outputs/espionage_ensemble5_summary.txt: configs/$(ESPIONAGE_DATASET).config.json results_ensembling.py postgres-schemas/model_release_dates.sql postgres-schemas/ensemble_results.sql
        uv run results_ensembling.py --env-dir envs/espionage --progress-bar --summary outputs/espionage_ensemble5_summary.txt --k 5 --no-decodex

######################################################################

# Time travel insurance- 10% noise
$(TEMPLATES_DIR)/$(TIMETRAVEL_INSURANCE_DATASET).sql configs/$(TIMETRAVEL_INSURANCE_DATASET).config.json:
	uv run ./random_classification_data_generator.py --number-of-data-points 200 --feature1-name "TimelineDeviation" --feature1-mean 12 --feature1-stddev 3  --feature2-name "ParadoxCount"  --feature2-mean 5  --feature2-stddev 2  --target-column-name "PolicyClaim"  --primary-key-name "IncidentID"  --table-name "time_travel_incidents"  --splits-table-name "time_travel_splits"  --random-seed 42  --class-deciding-expression "TimelineDeviation + 2 * ParadoxCount > 20"  --false-class-name "Denied"  --true-class-name "Approved" --noise 0.10  --holdout 0.2  --validation 0.5   --output-db-file $(TEMPLATES_DIR)/$(TIMETRAVEL_INSURANCE_DATASET).sqlite  --config-file configs/$(TIMETRAVEL_INSURANCE_DATASET).config.json --use-uuid
	sqlite3 $(TEMPLATES_DIR)/$(TIMETRAVEL_INSURANCE_DATASET).sqlite .dump > $(TEMPLATES_DIR)/$(TIMETRAVEL_INSURANCE_DATASET).sql
	rm -f $(TEMPLATES_DIR)/$(TIMETRAVEL_INSURANCE_DATASET).sqlite

$(RESULTS_DIR)/$(TIMETRAVEL_INSURANCE_DATASET)-%.best-round.txt: $(RESULTS_DIR)/$(TIMETRAVEL_INSURANCE_DATASET)-%.sqlite
	. ./envs/timetravel_insurance/$*.env && ./loop.sh

$(RESULTS_DIR)/$(TIMETRAVEL_INSURANCE_DATASET)-%.sqlite: $(TEMPLATES_DIR)/$(TIMETRAVEL_INSURANCE_DATASET).sql
	sqlite3 $@ < $<

$(RESULTS_DIR)/$(TIMETRAVEL_INSURANCE_DATASET)-%.estimate.txt: $(RESULTS_DIR)/$(TIMETRAVEL_INSURANCE_DATASET)-%.best-round.txt
	. ./envs/timetravel_insurance/$*.env && uv run report-script.py --estimate accuracy > $@

$(RESULTS_DIR)/$(TIMETRAVEL_INSURANCE_DATASET)-%.results.csv: $(RESULTS_DIR)/$(TIMETRAVEL_INSURANCE_DATASET)-%.best-round.txt
	. ./envs/timetravel_insurance/$*.env && uv run report-script.py --train --validation --test --show
	. ./envs/timetravel_insurance/$*.env && uv run report-script.py --train --validation --test --csv $@

timetravel_insurance: outputs/timetravel_insurance_results.csv
	echo All Time Travel Insurance results are ready

outputs/timetravel_insurance_results.csv: $(RESULTS_DIR)/$(TIMETRAVEL_INSURANCE_DATASET).baseline.json
	uv run create_task_csv_file.py --task timetravel_insurance --env-dir envs/timetravel_insurance --output outputs/timetravel_insurance_results.csv --model-details model_details.json --baseline $(RESULTS_DIR)/$(TIMETRAVEL_INSURANCE_DATASET).baseline.json --progress

timetravel_insurance-baseline: $(RESULTS_DIR)/$(TIMETRAVEL_INSURANCE_DATASET).baseline.json
	echo Baseline created

$(RESULTS_DIR)/$(TIMETRAVEL_INSURANCE_DATASET).baseline.json: configs/$(TIMETRAVEL_INSURANCE_DATASET).config.json $(RESULTS_DIR)/$(TIMETRAVEL_INSURANCE_DATASET)-baseline.sqlite baseline.py
	uv run baseline.py --config configs/$(TIMETRAVEL_INSURANCE_DATASET).config.json --database $(RESULTS_DIR)/$(TIMETRAVEL_INSURANCE_DATASET)-baseline.sqlite --output $(RESULTS_DIR)/$(TIMETRAVEL_INSURANCE_DATASET).baseline.json

# Add these targets to depend on all models
timetravel_insurance-databases: $(foreach model,$(MODELS),$(RESULTS_DIR)/$(TIMETRAVEL_INSURANCE_DATASET)-$(model).sqlite)
	@echo All Time Travel Insurance databases are initialised

timetravel_insurance-estimates: $(foreach model,$(MODELS),$(RESULTS_DIR)/$(TIMETRAVEL_INSURANCE_DATASET)-$(model).estimate.txt)
	echo All Time Travel Insurance estimates are ready

timetravel_insurance-results: $(foreach model,$(MODELS),$(RESULTS_DIR)/$(TIMETRAVEL_INSURANCE_DATASET)-$(model).results.csv)
	echo All Time Travel Insurance results are ready

timetravel_insurance-best: $(foreach model,$(MODELS),$(RESULTS_DIR)/$(TIMETRAVEL_INSURANCE_DATASET)-$(model).best-round.txt)
	echo All Time Travel Insurance best-round files are ready

timetravel_insurance-distributions: $(foreach model,$(MODELS),outputs/$(TIMETRAVEL_INSURANCE_DATASET)-$(model).distribution.png)
	echo All Time Travel Insurance distribution files are ready

outputs/$(TIMETRAVEL_INSURANCE_DATASET)-%.distribution.png outputs/$(TIMETRAVEL_INSURANCE_DATASET)-%.distribution.txt: envs/timetravel_insurance/%.env
	uv run resultdistribution.py --env-file $< --distribution-image outputs/$(TIMETRAVEL_INSURANCE_DATASET)-$*.distribution.png --fitted-distribution outputs/$(TIMETRAVEL_INSURANCE_DATASET)-$*.distribution.txt

outputs/timetravel_insurance_by_model_size.png outputs/timetravel_insurance_model_pvalue.tex: outputs/timetravel_insurance_results.csv results_chart_by_size.py
	uv run results_chart_by_size.py --dataset "Time Travel Insurance" --input outputs/timetravel_insurance_results.csv --image outputs/timetravel_insurance_by_model_size.png --pvalue outputs/timetravel_insurance_model_pvalue.tex --projection outputs/timetravel_insurance_model_projection.tex

outputs/timetravel_insurance_by_elo.png outputs/timetravel_insurance_elo_pvalue.tex: outputs/timetravel_insurance_results.csv results_chart_by_elo.py chatbot-arena/elo.csv chatbot-arena/translation.json
	uv run results_chart_by_elo.py --dataset "Time Travel Insurance" --input outputs/timetravel_insurance_results.csv --elo chatbot-arena/elo.csv --elo-translation chatbot-arena/translation.json --output outputs/timetravel_insurance_by_elo.png --pvalue outputs/timetravel_insurance_elo_pvalue.tex

outputs/timetravel_insurance_error_rate_by_herdan.png outputs/timetravel_insurance_error_rate_by_herdan_pvalue.tex outputs/timetravel_insurance_error_rate_by_herdan_slope.tex: outputs/timetravel_insurance_results.csv results_error_rate_by_herdan.py
	uv run results_error_rate_by_herdan.py --show --image-output outputs/timetravel_insurance_error_rate_by_herdan.png --pvalue-output outputs/timetravel_insurance_error_rate_by_herdan_pvalue.tex --slope-output outputs/timetravel_insurance_error_rate_by_herdan_slope.tex outputs/timetravel_insurance_results.csv

outputs/timetravel_insurance_error_rate_by_prompt_wordcount.png outputs/timetravel_insurance_error_rate_by_prompt_wordcount_pvalue.tex outputs/timetravel_insurance_error_rate_by_prompt_wordcount_slope.tex: outputs/timetravel_insurance_results.csv results_error_rate_by_wordcount.py
	uv run results_error_rate_by_wordcount.py --wordcount-type prompt --show --image-output outputs/timetravel_insurance_error_rate_by_prompt_wordcount.png --pvalue-output outputs/timetravel_insurance_error_rate_by_prompt_wordcount_pvalue.tex --slope-output outputs/timetravel_insurance_error_rate_by_prompt_wordcount_slope.tex outputs/timetravel_insurance_results.csv

outputs/timetravel_insurance_error_rate_by_reasoning_wordcount.png outputs/timetravel_insurance_error_rate_by_reasoning_wordcount_pvalue.tex outputs/timetravel_insurance_error_rate_by_reasoning_wordcount_slope.tex: outputs/timetravel_insurance_results.csv results_error_rate_by_wordcount.py
	uv run results_error_rate_by_wordcount.py --wordcount-type reasoning --show --image-output outputs/timetravel_insurance_error_rate_by_reasoning_wordcount.png --pvalue-output outputs/timetravel_insurance_error_rate_by_reasoning_wordcount_pvalue.tex --slope-output outputs/timetravel_insurance_error_rate_by_reasoning_wordcount_slope.tex outputs/timetravel_insurance_results.csv

outputs/timetravel_insurance_error_rate_by_cumulative_wordcount.png outputs/timetravel_insurance_error_rate_by_cumulative_wordcount_pvalue.tex outputs/timetravel_insurance_error_rate_by_cumulative_wordcount_slope.tex: outputs/timetravel_insurance_results.csv results_error_rate_by_wordcount.py
	uv run results_error_rate_by_wordcount.py --wordcount-type cumulative --show --image-output outputs/timetravel_insurance_error_rate_by_cumulative_wordcount.png --pvalue-output outputs/timetravel_insurance_error_rate_by_cumulative_wordcount_pvalue.tex --slope-output outputs/timetravel_insurance_error_rate_by_cumulative_wordcount_slope.tex outputs/timetravel_insurance_results.csv

outputs/timetravel_insurance_ensemble_summary.txt: configs/$(TIMETRAVEL_INSURANCE_DATASET).config.json results_ensembling.py postgres-schemas/model_release_dates.sql postgres-schemas/ensemble_results.sql
        uv run results_ensembling.py --env-dir envs/timetravel_insurance --progress-bar --summary outputs/timetravel_insurance_ensemble_summary.txt --no-decodex

outputs/timetravel_insurance_ensemble5_summary.txt: configs/$(TIMETRAVEL_INSURANCE_DATASET).config.json results_ensembling.py postgres-schemas/model_release_dates.sql postgres-schemas/ensemble_results.sql
        uv run results_ensembling.py --env-dir envs/timetravel_insurance --progress-bar --summary outputs/timetravel_insurance_ensemble5_summary.txt --k 5 --no-decodex

######################################################################

# Potions dataset -- 20% noise
$(TEMPLATES_DIR)/$(POTIONS_DATASET).sql configs/$(POTIONS_DATASET).config.json:
	uv run ./random_classification_data_generator.py --number-of-data-points 200 --feature1-name "FizzIntensity"  --feature1-mean 40  --feature1-stddev 12  --feature2-name "ColourShift"  --feature2-mean 15  --feature2-stddev 5  --target-column-name "PotionEffectiveness"  --primary-key-name "PotionBatchID"  --table-name "magic_potions"  --splits-table-name "potion_splits"  --random-seed 42   --class-deciding-expression "3 * FizzIntensity + ColourShift > 140"  --false-class-name "Ineffective"  --true-class-name "Effective"  --noise 0.20  --holdout 0.2  --validation 0.5  --output-db-file $(TEMPLATES_DIR)/$(POTIONS_DATASET).sqlite  --config-file configs/$(POTIONS_DATASET).config.json --use-uuid
	sqlite3 $(TEMPLATES_DIR)/$(POTIONS_DATASET).sqlite .dump > $(TEMPLATES_DIR)/$(POTIONS_DATASET).sql
	rm -f $(TEMPLATES_DIR)/$(POTIONS_DATASET).sqlite

$(RESULTS_DIR)/$(POTIONS_DATASET)-%.best-round.txt: $(RESULTS_DIR)/$(POTIONS_DATASET)-%.sqlite
	. ./envs/potions/$*.env && ./loop.sh

$(RESULTS_DIR)/$(POTIONS_DATASET)-%.sqlite: $(TEMPLATES_DIR)/$(POTIONS_DATASET).sql
	sqlite3 $@ < $<

$(RESULTS_DIR)/$(POTIONS_DATASET)-%.estimate.txt: $(RESULTS_DIR)/$(POTIONS_DATASET)-%.best-round.txt
	. ./envs/potions/$*.env && uv run report-script.py --estimate accuracy > $@

$(RESULTS_DIR)/$(POTIONS_DATASET)-%.results.csv: $(RESULTS_DIR)/$(POTIONS_DATASET)-%.best-round.txt
	. ./envs/potions/$*.env && uv run report-script.py --train --validation --test --show
	. ./envs/potions/$*.env && uv run report-script.py --train --validation --test --csv $@

potions: outputs/potions_results.csv
	echo All Potions results are ready

outputs/potions_results.csv: $(RESULTS_DIR)/$(POTIONS_DATASET).baseline.json
	uv run create_task_csv_file.py --task potions --env-dir envs/potions --output outputs/potions_results.csv --model-details model_details.json --baseline $(RESULTS_DIR)/$(POTIONS_DATASET).baseline.json --progress

potions-baseline: $(RESULTS_DIR)/$(POTIONS_DATASET).baseline.json
	echo Baseline created

$(RESULTS_DIR)/$(POTIONS_DATASET).baseline.json: configs/$(POTIONS_DATASET).config.json $(RESULTS_DIR)/$(POTIONS_DATASET)-baseline.sqlite baseline.py
	uv run baseline.py --config configs/$(POTIONS_DATASET).config.json --database $(RESULTS_DIR)/$(POTIONS_DATASET)-baseline.sqlite --output $(RESULTS_DIR)/$(POTIONS_DATASET).baseline.json

# Add these targets to depend on all models
potions-databases: $(foreach model,$(MODELS),$(RESULTS_DIR)/$(POTIONS_DATASET)-$(model).sqlite)
	@echo All Potions databases are initialised

potions-estimates: $(foreach model,$(MODELS),$(RESULTS_DIR)/$(POTIONS_DATASET)-$(model).estimate.txt)
	echo All Potions estimates are ready

potions-results: $(foreach model,$(MODELS),$(RESULTS_DIR)/$(POTIONS_DATASET)-$(model).results.csv)
	echo All Potions results are ready

potions-best: $(foreach model,$(MODELS),$(RESULTS_DIR)/$(POTIONS_DATASET)-$(model).best-round.txt)
	echo All Potions best-round files are ready

potions-distributions: $(foreach model,$(MODELS),outputs/$(POTIONS_DATASET)-$(model).distribution.png)
	echo All Potions distribution files are ready

outputs/$(POTIONS_DATASET)-%.distribution.png outputs/$(POTIONS_DATASET)-%.distribution.txt: envs/potions/%.env
	uv run resultdistribution.py --env-file $< --distribution-image outputs/$(POTIONS_DATASET)-$*.distribution.png --fitted-distribution outputs/$(POTIONS_DATASET)-$*.distribution.txt

outputs/potions_by_model_size.png outputs/potions_model_pvalue.tex: outputs/potions_results.csv results_chart_by_size.py
	uv run results_chart_by_size.py --dataset "Potions" --input outputs/potions_results.csv --image outputs/potions_by_model_size.png --pvalue outputs/potions_model_pvalue.tex --projection outputs/potions_model_projection.tex

outputs/potions_by_elo.png outputs/potions_elo_pvalue.tex: outputs/potions_results.csv results_chart_by_elo.py chatbot-arena/elo.csv chatbot-arena/translation.json
	uv run results_chart_by_elo.py --dataset "Potions" --input outputs/potions_results.csv --elo chatbot-arena/elo.csv --elo-translation chatbot-arena/translation.json --output outputs/potions_by_elo.png --pvalue outputs/potions_elo_pvalue.tex

outputs/potions_error_rate_by_herdan.png outputs/potions_error_rate_by_herdan_pvalue.tex outputs/potions_error_rate_by_herdan_slope.tex: outputs/potions_results.csv results_error_rate_by_herdan.py
	uv run results_error_rate_by_herdan.py --show --image-output outputs/potions_error_rate_by_herdan.png --pvalue-output outputs/potions_error_rate_by_herdan_pvalue.tex --slope-output outputs/potions_error_rate_by_herdan_slope.tex outputs/potions_results.csv

outputs/potions_error_rate_by_prompt_wordcount.png outputs/potions_error_rate_by_prompt_wordcount_pvalue.tex outputs/potions_error_rate_by_prompt_wordcount_slope.tex: outputs/potions_results.csv results_error_rate_by_wordcount.py
	uv run results_error_rate_by_wordcount.py --wordcount-type prompt --show --image-output outputs/potions_error_rate_by_prompt_wordcount.png --pvalue-output outputs/potions_error_rate_by_prompt_wordcount_pvalue.tex --slope-output outputs/potions_error_rate_by_prompt_wordcount_slope.tex outputs/potions_results.csv

outputs/potions_error_rate_by_reasoning_wordcount.png outputs/potions_error_rate_by_reasoning_wordcount_pvalue.tex outputs/potions_error_rate_by_reasoning_wordcount_slope.tex: outputs/potions_results.csv results_error_rate_by_wordcount.py
	uv run results_error_rate_by_wordcount.py --wordcount-type reasoning --show --image-output outputs/potions_error_rate_by_reasoning_wordcount.png --pvalue-output outputs/potions_error_rate_by_reasoning_wordcount_pvalue.tex --slope-output outputs/potions_error_rate_by_reasoning_wordcount_slope.tex outputs/potions_results.csv

outputs/potions_error_rate_by_cumulative_wordcount.png outputs/potions_error_rate_by_cumulative_wordcount_pvalue.tex outputs/potions_error_rate_by_cumulative_wordcount_slope.tex: outputs/potions_results.csv results_error_rate_by_wordcount.py
	uv run results_error_rate_by_wordcount.py --wordcount-type cumulative --show --image-output outputs/potions_error_rate_by_cumulative_wordcount.png --pvalue-output outputs/potions_error_rate_by_cumulative_wordcount_pvalue.tex --slope-output outputs/potions_error_rate_by_cumulative_wordcount_slope.tex outputs/potions_results.csv

outputs/potions_ensemble_summary.txt: configs/$(POTIONS_DATASET).config.json results_ensembling.py postgres-schemas/model_release_dates.sql postgres-schemas/ensemble_results.sql
        uv run results_ensembling.py --env-dir envs/potions --progress-bar --summary outputs/potions_ensemble_summary.txt --no-decodex

outputs/potions_ensemble5_summary.txt: configs/$(POTIONS_DATASET).config.json results_ensembling.py postgres-schemas/model_release_dates.sql postgres-schemas/ensemble_results.sql
        uv run results_ensembling.py --env-dir envs/potions --progress-bar --summary outputs/potions_ensemble5_summary.txt --k 5 --no-decodex
