RACO ?= raco

.PHONY: setup setup-python shim reference weights model-reference compile test fmt clean

setup-python:
	sh tools/setup-python.sh

shim:
	racket tools/gen-wrapper.rkt

reference:
	.venv/bin/python tools/gen-reference.py

weights:
	mkdir -p weights/SmolLM2-135M
	curl -L -o weights/SmolLM2-135M/model.safetensors \
	  https://huggingface.co/HuggingFaceTB/SmolLM2-135M/resolve/main/model.safetensors

model-reference:
	.venv/bin/python tools/gen-model-reference.py

setup:
	$(RACO) pkg install --skip-installed --link --name zhulong ./src
	$(RACO) pkg install --skip-installed --link ./models

compile:
	$(RACO) setup zhulong models

test:
	$(RACO) test -x ./src ./models ./tests

fmt:
	find src models tests -name '*.rkt' ! -name 'info.rkt' ! -name '.#*' | xargs $(RACO) fmt -i

clean:
	find . -name compiled -type d -prune -exec rm -rf {} +
