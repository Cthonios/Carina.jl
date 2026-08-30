# HW3L research note.
#
#   make          build note.pdf
#   make watch    rebuild continuously on save
#   make clean    remove auxiliary files, keep the PDF
#   make purge    remove auxiliary files and the PDF

MAIN    := note
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
