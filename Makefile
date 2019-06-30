.PHONY: docs
docs: dev
	pdoc ./iternash -o ./docs --force

.PHONY: test
test: clean install
	python ./iternash/examples/absent_minded_driver.py

.PHONY: dev
dev: build
	pip install -Ue .[examples,dev]

.PHONY: install
install: build
	pip install -Ue .[examples]

.PHONY: build
build:
	coconut setup.coco --no-tco --strict
	coconut iternash-source iternash --no-tco --strict --jobs sys

.PHONY: setup
setup:
	pip install -U setuptools pip coconut-develop[watch]

.PHONY: upload
upload: wipe docs
	python3 setup.py sdist bdist_wheel
	pip3 install -U --ignore-installed twine
	twine upload dist/*

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

.PHONY: watch
watch: install
	coconut iternash-source iternash --watch --no-tco --strict
