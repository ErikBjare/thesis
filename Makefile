all: dist

dist: docs tex notebooks

.PHONY: docs
docs:
	cd docs/sphinx && poetry run make html

tex: dist/thesis.pdf dist/goaldocument.pdf

.PHONY: notebooks
notebooks:
	cd notebooks && poetry run make

dist/%.pdf: docs/tex/build/%.pdf
	mkdir -p dist
	cp $< dist

%.png: %.dot
	dot -Tpng $< -o $@

docs/tex/build/%.pdf: docs/tex/%.tex docs/tex/*.bib docs/tex/gqm.png docs/tex/content/*.tex
	latexmk $< -pdf -shell-escape -cd -output-directory=build -interaction=nonstopmode -file-line-error

precommit:
	make typecheck
	make test

test:
	poetry run pytest

format:
	poetry run black src/ tests/ scripts/
	bash -c 'nbqa black notebooks/*.ipynb --nbqa-mutate'

typecheck:
	poetry run mypy
	bash -c 'poetry run nbqa mypy notebooks/*.ipynb'

clean:
	rm -rfv docs/tex/build
	#rm docs/gcm.png
	cd docs/tex && rm -fv *.aux *.bbl *.bcf *.cfg  *.blg *.dvi *.log *.pdf *.run.xml *.toc *.in *.markdown.* *.out *.tdo

git-config:
	git config --local filter.notebook.clean "poetry run jupyter nbconvert --ClearOutputPreprocessor.enabled=True --stdin --to=notebook --stdout"
	git config --local filter.notebook.smudge "poetry run jupyter nbconvert --ClearOutputPreprocessor.enabled=True --stdin --to=notebook --stdout"
	git config --local filter.notebook.required true

jupyter:
	# From: https://stackoverflow.com/a/47296960/965332
	poetry run pip3 install ipykernel
	poetry run bash -c 'python -m ipykernel install --user --name=`basename $$VIRTUAL_ENV`'
