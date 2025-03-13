.PRECIOUS: %.sqlite
.SECONDARY: %.sqlite

WISCONSIN_DATASET := wisconsin_exoplanets
TITANIC_DATASET := titanic_medical
#To add to models... MODELS := gemini (eventually)
# Models not to add (they generally don't work)... phi falcon falcon10 gemma llamaphi
MODELS := anthropic10 openai openai10 openai10o1 anthropic llama openai45 openailong deepseek qwq
TEMPLATES_DIR := dbtemplates
RESULTS_DIR := results

.PHONY: all wisconsin titanic

all: wisconsin titanic
	echo Done


######################################################################

# Wisconsin

# Define variables for reuse

# Main targets
.PHONY: all wisconsin

all: wisconsin


wisconsin: wisconsin_results.txt
	echo All Wisconsin results are ready

# I'm not sure that this next rule is right. In particular, whether globbing works in a Makefile.
# I could change the program to take a directory and a file prefix
wisconsin_results.txt: wisconsin-estimates wisconsin-results wisconsin-prompts wisconsin-best wisconsin-baseline
	uv run create_task_csv_file.py --task wisconsin --env-dir envs/wisconsin --output wisconsin_results.csv --results results/wisconsin_exoplanets*.txt

wisconsin-baseline: $(RESULTS_DIR)/$(WISCONSIN_DATASET).baseline.json
	echo Baseline created

$(RESULTS_DIR)/$(WISCONSIN_DATASET).baseline.json: configs/$(WISCONSIN_DATASET).config.json $(RESULTS_DIR)/$(WISCONSIN_DATASET)-baseline.sqlite
	uv run baseline.py --config configs/$(WISCONSIN_DATASET).config.json --database $(RESULTS_DIR)/$(WISCONSIN_DATASET)-baseline.sqlite --output $(RESULTS_DIR)/$(WISCONSIN_DATASET).baseline.json

# Add these targets to depend on all models
wisconsin-estimates: $(foreach model,$(MODELS),$(RESULTS_DIR)/$(WISCONSIN_DATASET)-$(model).estimate.txt)
	echo All Wisconsin estimates are ready

wisconsin-results: $(foreach model,$(MODELS),$(RESULTS_DIR)/$(WISCONSIN_DATASET)-$(model).results.csv)
	echo All Wisconsin results are ready

wisconsin-prompts: $(foreach model,$(MODELS),$(RESULTS_DIR)/$(WISCONSIN_DATASET)-$(model).decoded-best-prompt.txt)
	echo All Wisconsin prompts are ready

wisconsin-best: $(foreach model,$(MODELS),$(RESULTS_DIR)/$(WISCONSIN_DATASET)-$(model).best-round.txt)
	echo All Wisconsin best-round files are ready

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


# How we created the dataset
obfuscations/breast_cancer: conversions/breast_cancer obfuscation_plan_generator.py datasets/breast_cancer.csv
	uv run obfuscation_plan_generator.py --csv-file datasets/breast_cancer.csv  --obfuscation-plan obfuscations/breast_cancer --guidelines conversions/breast_cancer

# Should add obfuscations/breast_cancer as a dependency, but that means the template seems to get rebuilt
# too often. I guess it's because obfuscations/breast_cancer gets opened and updated by initialise_database.py
$(TEMPLATES_DIR)/wisconsin_exoplanets.sql configs/wisconsin_exoplanets.config.json: datasets/breast_cancer.csv initialise_database.py
	uv run initialise_database.py --database $(TEMPLATES_DIR)/wisconsin_exoplanets.sqlite --source datasets/breast_cancer.csv --config-file configs/wisconsin_exoplanets.config.json --obfuscation obfuscations/breast_cancer --verbose
	sqlite3 $(TEMPLATES_DIR)/$(WISCONSIN_DATASET).sqlite .dump > $(TEMPLATES_DIR)/wisconsin_exoplanets.sql
	rm -f $(TEMPLATES_DIR)/$(WISCONSIN_DATASET).sqlite


######################################################################

# How we created the dataset
obfuscations/titanic: conversions/titanic obfuscation_plan_generator.py datasets/titanic.csv
	uv run obfuscation_plan_generator.py --csv-file datasets/titanic.csv  --obfuscation-plan obfuscations/titanic --guidelines conversions/titanic

$(TEMPLATES_DIR)/$(TITANIC_DATASET).sql configs/titanic_medical.config.json: datasets/titanic.csv initialise_database.py
	uv run initialise_database.py --database $(TEMPLATES_DIR)/titanic_medical.sqlite --source datasets/titanic.csv --config-file configs/titanic_medical.config.json --obfuscation obfuscations/titanic --verbose
	sqlite3 $(TEMPLATES_DIR)/$(TITANIC_DATASET).sqlite .dump > $(TEMPLATES_DIR)/$(TITANIC_DATASET).sql
	rm -f $(TEMPLATES_DIR)/$(TEMPLATE_DATASET).sqlite


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

titanic: titanic_results.txt
	echo All Titanic results are ready

# Again, I'm not sure if this globbing will work
titanic_results.txt: titanic-estimates titanic-results titanic-prompts titanic-best titanic-baseline
	uv run create_task_csv_file.py --task titanic --env-dir envs/titanic --output titanic_results.csv --results results/titanic_medical*.txt

titanic-baseline: $(RESULTS_DIR)/$(TITANIC_DATASET).baseline.json
	echo Baseline created

$(RESULTS_DIR)/$(TITANIC_DATASET).baseline.json: configs/$(TITANIC_DATASET).config.json $(RESULTS_DIR)/$(TITANIC_DATASET)-baseline.sqlite
	uv run baseline.py --config configs/$(TITANIC_DATASET).config.json --database $(RESULTS_DIR)/$(TITANIC_DATASET)-baseline.sqlite --output $(RESULTS_DIR)/$(TITANIC_DATASET).baseline.json

# Add these targets to depend on all models
titanic-estimates: $(foreach model,$(MODELS),$(RESULTS_DIR)/$(TITANIC_DATASET)-$(model).estimate.txt)
	echo All Titanic estimates are ready

titanic-results: $(foreach model,$(MODELS),$(RESULTS_DIR)/$(TITANIC_DATASET)-$(model).results.csv)
	echo All Titanic results are ready

titanic-prompts: $(foreach model,$(MODELS),$(RESULTS_DIR)/$(TITANIC_DATASET)-$(model).decoded-best-prompt.txt)
	echo All Titanic prompts are ready

titanic-best: $(foreach model,$(MODELS),$(RESULTS_DIR)/$(TITANIC_DATASET)-$(model).best-round.txt)
	echo All Titanic best-round files are ready


	echo Titanic is ready

######################################################################

