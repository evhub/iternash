.PHONY: clean-docs
clean-docs: clean docs

.PHONY: all
all: clean docs run

.PHONY: test
test: clean install run

.PHONY: setup
setup:
	pip install -U setuptools pip coconut-develop[watch]

.PHONY: build
build:
	coconut setup.coco --no-tco --strict
	coconut iternash-source iternash --no-tco --strict --jobs sys

.PHONY: dev
dev: build
	pip install -Ue .[examples,dev]

.PHONY: install
install: build
	pip install -Ue .[examples]

.PHONY: docs
docs: dev
	pdoc ./iternash -o ./docs --force

.PHONY: run
run:
	python ./iternash/examples/myopia_unit_testing.py
	python ./iternash/examples/self_prisoner_dilemma.py

.PHONY: clean
clean:
	rm -rf ./dist ./build
	-find . -name '*.pyc' -delete
	-find . -name '__pycache__' -delete
	-find . -name '*.bbopt.pickle' -delete
	-find . -name '*.bbopt.json' -delete

.PHONY: wipe
wipe: clean
	rm -rf ./iternash ./setup.py ./docs

.PHONY: upload
upload: wipe docs
	python3 setup.py sdist bdist_wheel
	pip3 install -U --ignore-installed twine
	twine upload dist/*

.PHONY: watch
watch: install
	coconut iternash-source iternash --watch --no-tco --strict
