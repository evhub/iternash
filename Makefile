.PHONY: clean-docs
clean-docs: clean docs

.PHONY: all
all: clean-docs run

.PHONY: test-pd
test-pd: clean build run-pd

.PHONY: test-driver
test-driver: clean build run-driver

.PHONY: test-logistic
test-logistic: clean build run-logistic

.PHONY: run
run: run-pd run-driver run-logistic

.PHONY: run-pd
run-pd:
	python ./itergame/examples/self_prisoner_dilemma.py

.PHONY: run-driver
run-driver:
	python ./itergame/examples/absent_minded_driver.py

.PHONY: run-logistic
run-logistic:
	python ./itergame/examples/logistic_success_curve.py

.PHONY: test
test: clean install run

.PHONY: setup
setup:
	pip install -U setuptools pip coconut-develop[watch]

.PHONY: build
build:
	coconut setup.coco --no-tco --strict
	coconut itergame-source itergame --no-tco --strict --jobs sys

.PHONY: force-build
force-build:
	coconut setup.coco --no-tco --strict --force
	coconut itergame-source itergame --no-tco --strict --jobs sys --force

.PHONY: dev
dev: build
	pip install -Ue .[examples,dev]

.PHONY: install
install: build
	pip install -Ue .[examples]

.PHONY: docs
docs: dev
	pdoc ./itergame -o ./docs --force

.PHONY: clean
clean:
	rm -rf ./dist ./build
	-find . -name '*.pyc' -delete
	-C:/GnuWin32/bin/find.exe . -name '*.pyc' -delete
	-find . -name '__pycache__' -delete
	-C:/GnuWin32/bin/find.exe . -name '__pycache__' -delete
	-find . -name '*.bbopt.pickle' -delete
	-C:/GnuWin32/bin/find.exe . -name '*.bbopt.pickle' -delete
	-find . -name '*.bbopt.json' -delete
	-C:/GnuWin32/bin/find.exe . -name '*.bbopt.json' -delete

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
