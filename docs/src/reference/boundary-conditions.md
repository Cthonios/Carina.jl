# Boundary conditions

The optional top-level `boundary conditions` section holds two lists,
`dirichlet` and `neumann`. Body forces live in their own top-level section,
documented at the bottom of this page.

```yaml
boundary conditions:
  dirichlet:
    - side set: ssz-
      component: z
      function: "0.0"
  neumann:
    - side set: ssy+
      component: y
      function: "1.0e6 * t"
```

!!! note "`dirichlet` and `neumann` are matched case-insensitively"
    `dirichlet`, `Dirichlet`, and `DIRICHLET` are all accepted, as are the
    corresponding spellings of `neumann`. The lowercase form is used
    throughout this documentation and in every bundled example.

    A key that is not one of the two — a genuine typo such as `dirchlet:` —
    produces a warning with a suggestion:

    ```
    [WARNING] Unknown key "dirchlet" in boundary conditions. Did you mean "dirichlet"?
    ```

    Carina releases before this behaviour matched the key exactly, so
    `Dirichlet:` was silently skipped and the run proceeded with no boundary
    conditions. If you are reading an older input file that behaved oddly,
    that is the likely cause.

Both lists are optional; omitting the whole section is legal and yields no
boundary conditions.

## Dirichlet (essential)

Prescribes displacement components on a side set or a node set.

| Key | Required | Description |
|---|---|---|
| `side set` | one of the two | Exodus side set name. |
| `node set` | one of the two | Exodus node set name. |
| `component` | yes | `x`, `y`, or `z`. |
| `function` | yes | Expression in `x`, `y`, `z`, `t`. See [Function expressions](functions.md). |

Exactly one of `side set` / `node set` must be present. If both appear,
`side set` wins. If neither does:

```
Dirichlet BC entry must specify side_set or node_set.
```

Entry keys **are** validated, so a typo like `sideset:` produces an unknown-key
warning with a suggestion.

Dirichlet conditions are enforced by **DOF elimination**, not by penalty. The
resulting system is a genuinely reduced `n_free × n_free` problem with no
artificial conditioning — Carina asserts this at startup, because penalty
enforcement would pollute the spectrum with eigenvalues around `10⁶·tr(K)/n`
and wreck CG convergence.

## Neumann (natural)

Applies a traction over a side set, or a point load at the nodes of a node set.
The two are distinguished by which key you use.

| Key | Required | Description |
|---|---|---|
| `side set` | one of the two | Surface traction, integrated over the face. Units: force per area (Pa). |
| `node set` | one of the two | Point load, applied directly at each node in the set. Units: force (N). |
| `component` | yes | `x`, `y`, or `z`. |
| `function` | yes | Expression in `x`, `y`, `z`, `t`. |

```yaml
boundary conditions:
  neumann:
    - side set: ssy+          # traction, N/m²
      component: y
      function: "1.0e6 * t"
    - node set: tip_nodes     # point load, N, at every node in the set
      component: z
      function: "-100.0 * t"
```

A point load applies the function value at **each node** of the set
individually — it is not a total force distributed over the set. A 100 N
function on a 20-node set applies 2000 N in total.

### Sign convention

Positive function values produce force or traction in the **positive**
component direction. This is the natural convention, and it is what you should
write. Internally Carina negates the value, because FEC adds the Neumann
contribution to the residual `R = F_int − F_ext`; that detail does not surface
in the input file.

## Body forces

The optional top-level `body forces` section applies a volumetric force
density to an element block.

```yaml
body forces:
  - block: all
    component: z
    function: "-9.81 * 2700.0"
```

| Key | Required | Default | Description |
|---|---|---|---|
| `block` | no | `all` | Element block name, or `all`. |
| `component` | yes | — | `x`, `y`, or `z`. |
| `function` | yes | — | Expression in `x`, `y`, `z`, `t`. Units: force per volume (N/m³). |

The section accepts either a single mapping or a list of them; a bare mapping
is wrapped into a one-element list. Entry keys are validated.

Unlike Neumann conditions, body forces are **not** sign-flipped — the value you
write is used directly. For gravity in −z, write `-9.81 * ρ` as above, using
your material's density.
