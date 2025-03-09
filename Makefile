WISCONSIN_DATASET := wisconsin_exoplanets
TITANIC_DATASET := titanic_medical
#To add to models... MODELS := openai phi
MODELS := anthropic10 falcon10 gemma openai10 openai-o1-10anthropic falcon llamaphi openai openai45 openailong
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

wisconsin: wisconsin-estimates wisconsin-results wisconsin-prompts wisconsin-best
	echo Wisconsin is ready

# Add these targets to depend on all models
wisconsin-estimates: $(foreach model,$(MODELS),$(RESULTS_DIR)/$(WISCONSIN_DATASET)-$(model).estimate.txt)
	echo All Wisconsin estimates are ready

wisconsin-results: $(foreach model,$(MODELS),$(RESULTS_DIR)/$(WISCONSIN_DATASET)-$(model).results.csv)
	echo All Wisconsin results are ready

wisconsin-prompts: $(foreach model,$(MODELS),$(RESULTS_DIR)/$(WISCONSIN_DATASET)-$(model).decoded-best-prompt.txt)
	echo All Wisconsin prompts are ready

wisconsin-best: $(foreach model,$(MODELS),$(RESULTS_DIR)/$(WISCONSIN_DATASET)-$(model).best-round.txt)
	echo All Wisconsin best-round files are ready

# Pattern rule for .best-round.txt files. This is super-generic. It would be better to specialise this
# for Wisconsin, and then have another one for the Titanic data
$(RESULTS_DIR)/%.best-round.txt: $(RESULTS_DIR)/%.sqlite
	. ./envs/$(firstword $(subst _, ,$(basename $*)))/$(lastword $(subst -, ,$(basename $*))).env && ./loop.sh


$(RESULTS_DIR)/$(WISCONSIN_DATASET)-%.sqlite: $(TEMPLATES_DIR)/$(WISCONSIN_DATASET).sqlite
	cp $< $@

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
$(TEMPLATES_DIR)/wisconsin_exoplanets.sqlite configs/wisconsin_exoplanets.config.json: datasets/breast_cancer.csv initialise_database.py
	uv run initialise_database.py --database $(TEMPLATES_DIR)/wisconsin_exoplanets.sqlite --source datasets/breast_cancer.csv --config-file configs/wisconsin_exoplanets.config.json --obfuscation obfuscations/breast_cancer --verbose


######################################################################


titanic: results/titanic_medical-gemma.results.csv results/titanic_medical-gemma.decoded-best-prompt.txt results/titanic_medical-gemma.estimate.txt \
     results/titanic_medical-anthropic.results.csv results/titanic_medical-anthropic.decoded-best-prompt.txt results/titanic_medical-anthropic.estimate.txt \
     results/titanic_medical-anthropic-10example.results.csv results/titanic_medical-anthropic-10example.decoded-best-prompt.txt results/titanic_medical-anthropic-10example.estimate.txt \
     results/titanic_medical-falcon.results.csv results/titanic_medical-falcon.decoded-best-prompt.txt results/titanic_medical-falcon.estimate.txt \
     results/titanic_medical-falcon-10examples.results.csv results/titanic_medical-falcon-10examples.decoded-best-prompt.txt results/titanic_medical-falcon-10examples.estimate.txt \
     results/titanic_medical-openai.results.csv results/titanic_medical-openai.decoded-best-prompt.txt results/titanic_medical-openai.estimate.txt \
     results/titanic_medical-openai-10examples.results.csv results/titanic_medical-openai-10examples.decoded-best-prompt.txt results/titanic_medical-openai-10examples.estimate.txt \
     results/titanic_medical-openai-o1-10examples.results.csv results/titanic_medical-openai-o1-10examples.decoded-best-prompt.txt results/titanic_medical-openai-o1-10examples.estimate.txt \
     results/titanic_medical-llama-phi.results.csv results/titanic_medical-llama-phi.decoded-best-prompt.txt results/titanic_medical-llama-phi.estimate.txt
	echo Titanic is ready

# Gemma2, 3 examples
results/titanic_medical-gemma.estimate.txt: results/titanic_medical-gemma.best-round.txt
	. ./envs/titanic/gemma.env && uv run report-script.py --estimate accuracy > $@
results/titanic_medical-gemma.results.csv: results/titanic_medical-gemma.best-round.txt
	. ./envs/titanic/gemma.env && uv run report-script.py --train --validation --test --show
	. ./envs/titanic/gemma.env && uv run report-script.py --train --validation --test --csv $@
results/titanic_medical-gemma.decoded-best-prompt.txt: results/titanic_medical-gemma.best-round.txt
	. ./envs/titanic/gemma.env && uv run decode.py --round-id $(shell cat $<) --encoder-program initialise_titanic.py --verbose --output $@
results/titanic_medical-gemma.best-round.txt: results/titanic_medical-gemma.sqlite
	. ./envs/titanic/gemma.env && ./loop.sh
results/titanic_medical-gemma.sqlite:
	. ./envs/titanic/gemma.env && uv run initialise_titanic.py

# Anthropic 3 examples
results/titanic_medical-anthropic.estimate.txt: results/titanic_medical-anthropic.best-round.txt
	. ./envs/titanic/anthropic.env && uv run report-script.py --estimate accuracy > $@
results/titanic_medical-anthropic.results.csv: results/titanic_medical-anthropic.best-round.txt
	. ./envs/titanic/anthropic.env && uv run report-script.py --train --validation --test --show
	. ./envs/titanic/anthropic.env && uv run report-script.py --train --validation --test --csv $@
results/titanic_medical-anthropic.decoded-best-prompt.txt: results/titanic_medical-anthropic.best-round.txt
	. ./envs/titanic/anthropic.env && uv run decode.py --round-id $(shell cat $<) --encoder-program initialise_titanic.py --verbose --output $@
results/titanic_medical-anthropic.best-round.txt: results/titanic_medical-anthropic.sqlite
	. ./envs/titanic/anthropic.env && ./loop.sh
results/titanic_medical-anthropic.sqlite:
	. ./envs/titanic/anthropic.env && uv run initialise_titanic.py

# Anthropic 10 examples
results/titanic_medical-anthropic-10example.estimate.txt: results/titanic_medical-anthropic-10example.best-round.txt
	. ./envs/titanic/anthropic10.env && uv run report-script.py --estimate accuracy > $@
results/titanic_medical-anthropic-10example.results.csv: results/titanic_medical-anthropic-10example.best-round.txt
	. ./envs/titanic/anthropic10.env && uv run report-script.py --train --validation --test --show
	. ./envs/titanic/anthropic10.env && uv run report-script.py --train --validation --test --csv $@
results/titanic_medical-anthropic-10example.decoded-best-prompt.txt: results/titanic_medical-anthropic-10example.best-round.txt
	. ./envs/titanic/anthropic10.env && uv run decode.py --round-id $(shell cat $<) --encoder-program initialise_titanic.py --verbose --output $@
results/titanic_medical-anthropic-10example.best-round.txt: results/titanic_medical-anthropic-10example.sqlite
	. ./envs/titanic/anthropic10.env && ./loop.sh
results/titanic_medical-anthropic-10example.sqlite:
	. ./envs/titanic/anthropic10.env && uv run initialise_titanic.py

# Falcon 3 examples
results/titanic_medical-falcon.estimate.txt: results/titanic_medical-falcon.best-round.txt
	. ./envs/titanic/falcon.env && uv run report-script.py --estimate accuracy > $@
results/titanic_medical-falcon.results.csv: results/titanic_medical-falcon.best-round.txt
	. ./envs/titanic/falcon.env && uv run report-script.py --train --validation --test --show
	. ./envs/titanic/falcon.env && uv run report-script.py --train --validation --test --csv $@
results/titanic_medical-falcon.decoded-best-prompt.txt: results/titanic_medical-falcon.best-round.txt
	. ./envs/titanic/falcon.env && uv run decode.py --round-id $(shell cat $<) --encoder-program initialise_titanic.py --verbose --output $@
results/titanic_medical-falcon.best-round.txt: results/titanic_medical-falcon.sqlite
	. ./envs/titanic/falcon.env && ./loop.sh
results/titanic_medical-falcon.sqlite:
	. ./envs/titanic/falcon.env && uv run initialise_titanic.py

# Falcon 10 examples
results/titanic_medical-falcon-10examples.estimate.txt: results/titanic_medical-falcon-10examples.best-round.txt
	. ./envs/titanic/falcon10.env && uv run report-script.py --estimate accuracy > $@
results/titanic_medical-falcon-10examples.results.csv: results/titanic_medical-falcon-10examples.best-round.txt
	. ./envs/titanic/falcon10.env && uv run report-script.py --train --validation --test --show
	. ./envs/titanic/falcon10.env && uv run report-script.py --train --validation --test --csv $@
results/titanic_medical-falcon-10examples.decoded-best-prompt.txt: results/titanic_medical-falcon-10examples.best-round.txt
	. ./envs/titanic/falcon10.env && uv run decode.py --round-id $(shell cat $<) --encoder-program initialise_titanic.py --verbose --output $@
results/titanic_medical-falcon-10examples.best-round.txt: results/titanic_medical-falcon-10examples.sqlite
	. ./envs/titanic/falcon10.env && ./loop.sh
results/titanic_medical-falcon-10examples.sqlite:
	. ./envs/titanic/falcon10.env && uv run initialise_titanic.py

# OpenAI 3 examples
results/titanic_medical-openai.estimate.txt: results/titanic_medical-openai.best-round.txt
	. ./envs/titanic/openai.env && uv run report-script.py --estimate accuracy > $@
results/titanic_medical-openai.results.csv: results/titanic_medical-openai.best-round.txt
	. ./envs/titanic/openai.env && uv run report-script.py --train --validation --test --show
	. ./envs/titanic/openai.env && uv run report-script.py --train --validation --test --csv $@
results/titanic_medical-openai.decoded-best-prompt.txt: results/titanic_medical-openai.best-round.txt
	. ./envs/titanic/openai.env && uv run decode.py --round-id $(shell cat $<) --encoder-program initialise_titanic.py --verbose --output $@
results/titanic_medical-openai.best-round.txt: results/titanic_medical-openai.sqlite
	. ./envs/titanic/openai.env && ./loop.sh
results/titanic_medical-openai.sqlite:
	. ./envs/titanic/openai.env && uv run initialise_titanic.py

# OpenAI 10 examples
results/titanic_medical-openai-10examples.estimate.txt: results/titanic_medical-openai-10examples.best-round.txt
	. ./envs/titanic/openai10.env && uv run report-script.py --estimate accuracy > $@
results/titanic_medical-openai-10examples.results.csv: results/titanic_medical-openai-10examples.best-round.txt
	. ./envs/titanic/openai10.env && uv run report-script.py --train --validation --test --show
	. ./envs/titanic/openai10.env && uv run report-script.py --train --validation --test --csv $@
results/titanic_medical-openai-10examples.decoded-best-prompt.txt: results/titanic_medical-openai-10examples.best-round.txt
	. ./envs/titanic/openai10.env && uv run decode.py --round-id $(shell cat $<) --encoder-program initialise_titanic.py --verbose --output $@
results/titanic_medical-openai-10examples.best-round.txt: results/titanic_medical-openai-10examples.sqlite
	. ./envs/titanic/openai10.env && ./loop.sh
results/titanic_medical-openai-10examples.sqlite:
	. ./envs/titanic/openai10.env && uv run initialise_titanic.py

# OpenAI O1 10 examples
results/titanic_medical-openai-o1-10examples.estimate.txt: results/titanic_medical-openai-o1-10examples.best-round.txt
	. ./envs/titanic/openai-o1-10.env && uv run report-script.py --estimate accuracy > $@
results/titanic_medical-openai-o1-10examples.results.csv: results/titanic_medical-openai-o1-10examples.best-round.txt
	. ./envs/titanic/openai-o1-10.env && uv run report-script.py --train --validation --test --show
	. ./envs/titanic/openai-o1-10.env && uv run report-script.py --train --validation --test --csv $@
results/titanic_medical-openai-o1-10examples.decoded-best-prompt.txt: results/titanic_medical-openai-o1-10examples.best-round.txt
	. ./envs/titanic/openai-o1-10.env && uv run decode.py --round-id $(shell cat $<) --encoder-program initialise_titanic.py --verbose --output $@
results/titanic_medical-openai-o1-10examples.best-round.txt: results/titanic_medical-openai-o1-10examples.sqlite
	. ./envs/titanic/openai-o1-10.env && ./loop.sh
results/titanic_medical-openai-o1-10examples.sqlite:
	. ./envs/titanic/openai-o1-10.env && uv run initialise_titanic.py

# LLaMA-Phi
results/titanic_medical-llama-phi.estimate.txt: results/titanic_medical-llama-phi.best-round.txt
	. ./envs/titanic/llama-phi.env && uv run report-script.py --estimate accuracy > $@
results/titanic_medical-llama-phi.results.csv: results/titanic_medical-llama-phi.best-round.txt
	. ./envs/titanic/llama-phi.env && uv run report-script.py --train --validation --test --show
	. ./envs/titanic/llama-phi.env && uv run report-script.py --train --validation --test --csv $@
results/titanic_medical-llama-phi.decoded-best-prompt.txt: results/titanic_medical-llama-phi.best-round.txt
	. ./envs/titanic/llama-phi.env && uv run decode.py --round-id $(shell cat $<) --encoder-program initialise_titanic.py --verbose --output $@
results/titanic_medical-llama-phi.best-round.txt: results/titanic_medical-llama-phi.sqlite
	. ./envs/titanic/llama-phi.env && ./loop.sh
results/titanic_medical-llama-phi.sqlite:
	. ./envs/titanic/llama-phi.env && uv run initialise_titanic.py

