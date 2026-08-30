# Carina theory manual.
#
#   make          build carina-theory.pdf
#   make watch    rebuild continuously on save
#   make clean    remove auxiliary files, keep the PDF
#   make purge    remove auxiliary files and the PDF
#
# latexmk drives the rerun-until-stable loop, including BibTeX and the extra
# pass a shifted table of contents forces.  Getting that loop right by hand is
# possible but not worth doing: a fixed three-pass recipe silently leaves stale
# page numbers whenever an edit moves a chapter across a page boundary.

MAIN    := carina-theory
LATEXMK := latexmk -lualatex -interaction=nonstopmode -halt-on-error

.PHONY: all watch clean purge

all: $(MAIN).pdf

$(MAIN).pdf: $(MAIN).tex preamble.tex refs.bib $(wildcard chapters/*.tex)
	$(LATEXMK) $(MAIN).tex

watch:
	$(LATEXMK) -pvc $(MAIN).tex

clean:
	$(LATEXMK) -c
	rm -f chapters/*.aux

purge:
	$(LATEXMK) -C
	rm -f chapters/*.aux
