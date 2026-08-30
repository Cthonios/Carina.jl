# Carina theory manual.
#
#   make          build carina-theory.pdf
#   make clean    remove auxiliary files, keep the PDF
#   make purge    remove auxiliary files and the PDF
#
# Deliberately does not use latexmk: it is not present in a base TeX Live
# installation, and the passes this document needs are few and known.  Two
# lualatex passes bracket bibtex, then a third resolves the citations the
# bibliography just defined.

MAIN := carina-theory
TEX  := lualatex -interaction=nonstopmode -halt-on-error
BIB  := bibtex

.PHONY: all clean purge

all: $(MAIN).pdf

$(MAIN).pdf: $(MAIN).tex preamble.tex refs.bib $(wildcard chapters/*.tex)
	$(TEX) $(MAIN).tex
	-$(BIB) $(MAIN)
	$(TEX) $(MAIN).tex
	$(TEX) $(MAIN).tex

clean:
	rm -f $(MAIN).aux $(MAIN).bbl $(MAIN).blg $(MAIN).log $(MAIN).out \
	      $(MAIN).toc $(MAIN).lof $(MAIN).lot $(MAIN).loa $(MAIN).fls \
	      chapters/*.aux

purge: clean
	rm -f $(MAIN).pdf
