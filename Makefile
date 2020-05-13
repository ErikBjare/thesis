
all: goaldocument.pdf thesis.pdf

dist: *.pdf
	mkdir -p dist
	cp *.pdf dist

goaldocument.pdf: goaldocument.tex
	# Not sure why, but we need to run pdflatex twice for it to work with biber
	pdflatex --shell-escape goaldocument.tex
	biber goaldocument
	pdflatex --shell-escape goaldocument.tex

goaldocument.tex: *.bib

thesis.pdf: thesis.tex
	pdflatex --shell-escape thesis.tex
	biber thesis
	pdflatex --shell-escape thesis.tex

clean:
	rm -f *.aux *.bbl *.bcf *.cfg  *.blg *.dvi *.log *.pdf *.run.xml *.toc *.in *.markdown.*
