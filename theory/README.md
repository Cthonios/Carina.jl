# Carina theory manual

The mathematics behind the code: formulation, constitutive models, and every
solution algorithm Carina implements.

This is the third of three documents, split by the question each answers:

| Document | Answers |
|---|---|
| `docs/` (Documenter) | What do I write in the input deck? |
| `benchmark_report.md` | How fast is it, on what hardware? |
| `theory/` (this) | Why do the algorithms work? |

## Building

```
make          # -> carina-theory.pdf
make watch    # rebuild continuously on save
make clean    # remove auxiliary files, keep the PDF
make purge    # remove auxiliary files and the PDF
```

The PDF is a build product and is not tracked.

### Requirements

`lualatex`, `bibtex` and `latexmk`, plus the LaTeX packages `boldtensors`,
`algorithm`, `algorithmicx` (for `algpseudocode`), `booktabs` and `microtype`.

On Fedora these are `texlive-scheme-medium` or, if you would rather add only
what is missing from a smaller installation:

```
sudo dnf install texlive-boldtensors texlive-algorithms \
                 texlive-algorithmicx texlive-booktabs \
                 texlive-microtype latexmk
```

On Debian and Ubuntu, `texlive-latex-extra texlive-science latexmk` covers all
of them.

## Conventions

`preamble.tex` defines every notation macro in one place, so a symbol cannot
drift between chapters. Tensors and matrices go through `\tensor{}`, vectors
through `\vect{}`, fourth-order tensors through `\fourth{}`; all three wrap the
`boldtensors` active-character interface (`~X`, `"X`) so those characters stay
out of the chapter sources.

Each chapter opens with the source files implementing it. Where the code
departs from a textbook statement of an algorithm, the departure is the point
of the section rather than a footnote to it — those departures are usually the
result of a measurement, and the measurement is cited from
`benchmark/evidence/`.
