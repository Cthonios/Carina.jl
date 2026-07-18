# Function expressions

Every `function` key — in Dirichlet and Neumann conditions, body forces, and
initial conditions — and the `displacement` key of a traveling-wave condition
take a string expression that Carina compiles into a scalar function.

```yaml
function: "1.0e-3 * sin(pi * x) * t"
```

## Variable namespace

Exactly four variables are in scope, in this order:

| Variable | Meaning |
|---|---|
| `x`, `y`, `z` | Coordinates of the node or integration point |
| `t` | Current simulation time |

Time is deliberately the **last** variable, which is what makes the symbolic
time and space derivatives used elsewhere (notably the traveling-wave initial
condition) well defined.

Expressions support the usual arithmetic — `+`, `-`, `*`, `/`, `^` — along with
`sin`, `cos`, `tan`, `exp`, `log`, `sqrt`, `abs`, `atan`, and the constant
`pi`.

A constant is a perfectly good expression:

```yaml
function: "0.0"
```

Quote your expressions. YAML will otherwise interpret some of them — a bare
`1.0e-3` is fine, but anything containing `:` or starting with punctuation is
not.

## Named bindings

An expression may be prefixed with `name=value;` bindings, separated by
semicolons. The **last** semicolon-separated segment is the expression; every
earlier segment must be a binding.

```yaml
function: "a=1.0e-3; tc=2.5e-4; a*exp(-(t-tc)^2/(2*tc^2))"
```

Bindings are textually inlined before parsing, so the example above becomes

```
(1.0e-3)*exp(-(t-(2.5e-4))^2/(2*(2.5e-4)^2))
```

Two properties follow from this being a substitution rather than an assignment:

- **Later bindings may use earlier ones.** `tc=2.5e-4; tau=tc/2; ...` is legal;
  `tau` expands to `((2.5e-4)/2)`.
- **Substitution respects word boundaries.** A binding named `tc` will not
  match the leading `t` of `tc*t`, and a binding named `a` will not corrupt
  `atan`.

A segment before the final one that lacks an `=` is an error:

```
Expected `name=value` in expression binding fragment: "..."
```

Bindings are a readability device with no runtime cost — the substituted
expression is compiled once and evaluated as a plain scalar function, with no
per-call parsing.

## Examples

A ramp:

```yaml
function: "1.0e-3 * t"
```

A spatially varying hold:

```yaml
function: "0.005 * (1.0 - z)"
```

A Gaussian pulse in time, using bindings for legibility:

```yaml
function: "a=1.0e-3; tc=2.5e-4; w=5.0e-5; a*exp(-(t-tc)^2/(2*w^2))"
```

A twist about the z axis, applied as an initial velocity in `x`:

```yaml
function: "omega=50.0; -omega*y"
```
