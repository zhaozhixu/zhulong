RACO ?= raco

.PHONY: setup setup-python compile test fmt clean

setup-python:
	sh tools/setup-python.sh

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
