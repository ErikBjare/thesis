all: dist

dist: dist/thesis.pdf dist/goaldocument.pdf

dist/%.pdf: docs/tex/build/%.pdf
	mkdir -p dist
	cp $< dist

%.png: %.dot
	dot -Tpng $< -o $@

docs/tex/build/%.pdf: docs/tex/%.tex docs/tex/*.bib docs/tex/gqm.png
	latexmk $< -pdf -shell-escape -cd -output-directory=build -interaction=nonstopmode -file-line-error

precommit:
	make typecheck
	make test

test:
	poetry run pytest

format:
	black src/ tests/ scripts/

typecheck:
	poetry run mypy

clean:
	rm -r docs/build
	rm -r docs/tex/build
	#rm docs/gcm.png
	#rm -f *.aux *.bbl *.bcf *.cfg  *.blg *.dvi *.log *.pdf *.run.xml *.toc *.in *.markdown.* *.out *.tdo
