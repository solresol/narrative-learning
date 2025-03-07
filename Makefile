all:

obfuscations/breast_cancer: conversions/breast_cancer obfuscation_plan_generator.py datasets/breast_cancer.csv
	uv run obfuscation_plan_generator.py --csv-file datasets/breast_cancer.csv  --obfuscation-plan obfuscations/breast_cancer --guidelines conversions/breast_cancer

# should also have obfuscations/breast_cancer
dbtemplates/wisconsin_exoplanets.sqlite configs/wisconsin_exoplanets.config.json: datasets/breast_cancer.csv initialise_database.py
	uv run initialise_database.py --database dbtemplates/wisconsin_exoplanets.sqlite --source datasets/breast_cancer.csv --config-file configs/wisconsin_exoplanets.config.json --obfuscation obfuscations/breast_cancer --verbose


old-all: results/titanic_medical-gemma.results.csv results/titanic_medical-gemma.decoded-best-prompt.txt results/titanic_medical-gemma.estimate.txt \
     results/titanic_medical-anthropic.results.csv results/titanic_medical-anthropic.decoded-best-prompt.txt results/titanic_medical-anthropic.estimate.txt \
     results/titanic_medical-anthropic-10example.results.csv results/titanic_medical-anthropic-10example.decoded-best-prompt.txt results/titanic_medical-anthropic-10example.estimate.txt \
     results/titanic_medical-falcon.results.csv results/titanic_medical-falcon.decoded-best-prompt.txt results/titanic_medical-falcon.estimate.txt \
     results/titanic_medical-falcon-10examples.results.csv results/titanic_medical-falcon-10examples.decoded-best-prompt.txt results/titanic_medical-falcon-10examples.estimate.txt \
     results/titanic_medical-openai.results.csv results/titanic_medical-openai.decoded-best-prompt.txt results/titanic_medical-openai.estimate.txt \
     results/titanic_medical-openai-10examples.results.csv results/titanic_medical-openai-10examples.decoded-best-prompt.txt results/titanic_medical-openai-10examples.estimate.txt \
     results/titanic_medical-openai-o1-10examples.results.csv results/titanic_medical-openai-o1-10examples.decoded-best-prompt.txt results/titanic_medical-openai-o1-10examples.estimate.txt \
     results/titanic_medical-llama-phi.results.csv results/titanic_medical-llama-phi.decoded-best-prompt.txt results/titanic_medical-llama-phi.estimate.txt
	echo Done

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

