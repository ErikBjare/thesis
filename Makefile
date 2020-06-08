all: dist

dist: docs/build/goaldocument.pdf docs/build/thesis.pdf
	mkdir -p dist
	cp $< dist

docs/build/%.pdf: docs/%.tex docs/*.bib
	latexmk $< -pdf -cd -output-directory=build

clean:
	rm -r docs/build
	#rm -f *.aux *.bbl *.bcf *.cfg  *.blg *.dvi *.log *.pdf *.run.xml *.toc *.in *.markdown.* *.out *.tdo
