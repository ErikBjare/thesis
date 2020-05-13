
all: goaldocument.pdf thesis.pdf

goaldocument.pdf: goaldocument.tex
	pdflatex --shell-escape goaldocument.tex
	biber goaldocument
	pdflatex --shell-escape goaldocument.tex

thesis.pdf: thesis.tex
	pdflatex --shell-escape thesis.tex
	biber thesis
	pdflatex --shell-escape thesis.tex

