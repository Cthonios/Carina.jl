# Research

Formulations that are ahead of the code.

Carina keeps its written material in three places, split by how each relates to
the implementation:

| Directory | Relationship to the code |
|---|---|
| `docs/` | What to write in an input deck for what the code does today |
| `theory/` | Why the algorithms the code **implements** work |
| `research/` | Formulations the code does **not** implement yet |

The distinction between `theory/` and `research/` is the one that matters.
The theory manual's premise is that every algorithm in it can be checked
against the source, and a chapter describing something unimplemented would
quietly break that contract with the reader. Work lives here until there is
something to check it against.

## Graduating to `theory/`

A formulation moves out of `research/` when the code implements it. At that
point the material is rewritten for the theory manual rather than copied: the
two documents have different jobs. A paper argues that a formulation is
correct and worth having; a theory chapter explains what the code does and why,
and names the file doing it. The paper stays here as the record of the
argument.

## Contents

- **`sec5l/`** — SEC5L, a five-field mixed tetrahedral element on the symmetric
  elasticity complex with logarithmic strain. Mota and Foulk. In progress; not
  implemented.
- **`hw3l/`** — HW3L, a three-field Hu–Washizu tetrahedron in logarithmic
  strain. Argues that volumetric locking and spurious soft modes are two
  directions of one inf-sup condition, and that a stable pair removes both
  without stabilization. Research note; not implemented.

The two are independent attacks on the same problem and share no machinery.
Where one fails the other may not, which is the point of keeping both.
