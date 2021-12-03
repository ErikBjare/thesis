all: dist

dist: docs tex notebooks

.PHONY: docs
docs:
	cd docs/sphinx && poetry run make html

tex: dist/thesis.pdf dist/goaldocument.pdf dist/popsci.pdf dist/presentation.pdf

lint-tex:
	# Check 'as seen in' should probably be just 'seen in'
	! git grep 'as seen' docs/tex/content/
	# Contractions
	! git grep "'ve" docs/tex/content/
	! git grep "'t" docs/tex/content/
	! git grep "there's" docs/tex/content/
	! git grep "it's" docs/tex/content/
	! git grep "organic" docs/tex/  # should be 'naturalistic'
	# Consecutive cites (should use a single \cite with all refs)
	! git grep -E '[\]cite\{[a-zA-Z\_\0-9]+}[\]cite'

lint-textidote:
	# Textidote loads config from docs/tex/.textidote
	cd docs/tex; textidote

.PHONY: notebooks
notebooks:
	cd notebooks && poetry run make

dist/%.pdf: docs/tex/build/%.pdf
	mkdir -p dist
	cp $< dist

%.png: %.dot
	dot -Tpng -Gdpi=300 $< -o $@

imgs: docs/tex/img/method.png docs/tex/img/method-analysis.png docs/tex/img/gqm.png

docs/tex/build/%.pdf: docs/tex/%.tex docs/tex/*.bib imgs docs/tex/content/*.tex docs/tex/figures/*.tex
	latexmk $< -pdf -shell-escape -cd -output-directory=build -interaction=nonstopmode -file-line-error -r docs/tex/.latexmkrc

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
	bash -c 'poetry run nbqa mypy notebooks/Main.ipynb notebooks/Signal.ipynb notebooks/Activity.ipynb notebooks/PPG.ipynb --ignore-missing-imports'

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
