.PHONY: clean-docs
clean-docs: clean docs

.PHONY: all
all: clean docs run

.PHONY: all-pd
all-pd: clean docs run-pd

.PHONY: all-driver
all-driver: clean docs run-driver

.PHONY: test
test: clean install run

.PHONY: setup
setup:
	pip install -U setuptools pip coconut-develop[watch]

.PHONY: build
build:
	coconut setup.coco --no-tco --strict
	coconut itergame-source itergame --no-tco --strict --jobs sys

.PHONY: dev
dev: build
	pip install -Ue .[examples,dev]

.PHONY: install
install: build
	pip install -Ue .[examples]

.PHONY: docs
docs: dev
	pdoc ./itergame -o ./docs --force

.PHONY: run
run: run-pd run-driver

.PHONY: run-pd
run-pd:
	python ./itergame/examples/self_prisoner_dilemma.py

.PHONY: run-driver
run-driver:
	python ./itergame/examples/absent_minded_driver.py

.PHONY: clean
clean:
	rm -rf ./dist ./build
	-find . -name '*.pyc' -delete
	-find . -name '__pycache__' -delete
	-find . -name '*.bbopt.pickle' -delete
	-find . -name '*.bbopt.json' -delete

.PHONY: wipe
wipe: clean
	rm -rf ./itergame ./setup.py ./docs

.PHONY: upload
upload: wipe docs
	python3 setup.py sdist bdist_wheel
	pip3 install -U --ignore-installed twine
	twine upload dist/*

.PHONY: upload-old
upload-old: export INSTALL_OLD_ITERNASH=TRUE
upload-old: upload

.PHONY: watch
watch: install
	coconut itergame-source itergame --watch --no-tco --strict
