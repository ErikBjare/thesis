all: dist

dist: docs/build/goaldocument.pdf docs/build/thesis.pdf
	mkdir -p dist
	cp $< dist

%.png: %.dot
	dot -Tpng $< -o $@

docs/build/%.pdf: docs/%.tex docs/*.bib docs/gcm.png
	latexmk $< -pdf -shell-escape -cd -output-directory=build -interaction=nonstopmode -file-line-error

test:
	poetry run pytest src/eegwatch/*.py

clean:
	rm -r docs/build
	#rm docs/gcm.png
	#rm -f *.aux *.bbl *.bcf *.cfg  *.blg *.dvi *.log *.pdf *.run.xml *.toc *.in *.markdown.* *.out *.tdo
