# SEC5L paper.
#
#   make          build main.pdf
#   make watch    rebuild continuously on save
#   make clean    remove auxiliary files, keep the PDF
#   make purge    remove auxiliary files and the PDF
#
# pdflatex rather than the lualatex the theory manual uses: this document
# carries `times` and `epsfig`, which are happier there, and it was written
# against Overleaf's default pdflatex toolchain.

MAIN    := main
LATEXMK := latexmk -pdf -interaction=nonstopmode -halt-on-error

.PHONY: all watch clean purge

all: $(MAIN).pdf

$(MAIN).pdf: $(MAIN).tex references.bib
	$(LATEXMK) $(MAIN).tex

watch:
	$(LATEXMK) -pvc $(MAIN).tex

clean:
	$(LATEXMK) -c

purge:
	$(LATEXMK) -C
