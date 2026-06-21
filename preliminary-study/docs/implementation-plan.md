# Plan for first version

## Contents

1. **Scope**
2. **Network connectivity**
   - 2.1 Entities
   - 2.2 Storage for code: array layout
   - 2.3 Relation to assembly: the two loops
   - 2.4 Recommendation
   - 2.5 Handling pre-defined node and edge parameters
3. **Jacobian evaluation**
   - 3.1 The same source serves value and derivative
   - 3.2 Complex-step-safe primitives
   - 3.3 The seed is sparse — reuse the CSC endpoint table
   - 3.4 Worked example
4. **Variable storage**
   - 4.1 Three categories by provenance
   - 4.2 The refresh DAG, and "recomputed, not read"
   - 4.3 Pre-indexing: a parse-time field registry
   - 4.4 How `@njit` addresses a column — two tiers
   - 4.5 One index namespace, two physical homes
5. **Thermodynamic state management and support for reacting flows**
   - 5.1 The boundary: a single `thermo_update`
   - 5.2 Two call sites: pre- and post-solve
   - 5.3 Selecting a model: an integer id
   - 5.4 Model configuration: `thermo_params`
   - 5.5 Composition through the scalar registry
   - 5.6 Equilibrium closure (the MVP)
   - 5.7 Detailed chemistry: separate the kernel from the integrator
6. **Solver design**
   - 6.1 Solver-specific terms: `solver_params`
   - 6.2 Element-reported quantities: `solver_aux`
   - 6.3 Carrying state between iterations: a `SolverState`
7. **Routine skeleton: signatures first**
   - 7.0 Pre-defined input (what the user supplies)
   - 7.1 Parse time — build the immutable solve bundles
   - 7.2 Solve time — physics layer (`@njit`)
   - 7.3 Solve time — linear layer (scipy sparse)
   - 7.4 Solve time — control layer (Python)
   - 7.5 Placeholder helpers
8. **The perturbation (acoustic) network**
   - 8.1 What is reused, what is new
   - 8.2 The acoustic operator `A(ω)`
   - 8.3 The duct node and acoustic element faces
   - 8.4 Acoustic parameters and the perturbation state
   - 8.5 The three analyses as drivers
   - 8.6 Routine skeleton: signatures first
9. **Object-oriented layer: the user-facing shell**
   - 9.1 The surface: objects, and the two consumers
   - 9.2 The element catalog and acoustic faces
   - 9.3 User-supplied initial conditions

## 1. Scope

Earlier experiments and prototypes led to one decisive outcome: handling every flow
regime within a single solver algorithm, and demanding that it be perfectly robust,
is unproductive. The first version therefore draws an explicit line around the scope
of the project.

The object of interest is the mean-flow solution, which becomes an ingredient of the
downstream perturbation-analysis study. The practical cases in view are subsonic;
seamless supersonic capability would matter only for intakes, nozzles, and perhaps
scramjet combustors, which lie outside the present scope. The first version does not
pursue supersonic solution capability or the algorithms it would require. It retains
the existing core solver, which ran robustly across the test cases and removed the
excessive sensitivity to initial conditions.

The scope line itself is informed by the `dev` branch, where an internal
shock-position degree of freedom was studied to let a supersonic outlet emerge. Those
studies showed the discontinuity is inherent and must be seeded rather than emerging
on its own — exactly the capability being deferred. The line is therefore drawn at
smooth, subsonic mean-flow solutions: the solver stays robust on the cases of
interest, and supersonic/shock handling is left to a later version.

This first version also covers the **perturbation (acoustic) network** built on the
converged mean flow — the consistency goal that motivated the whole architecture
(§8). It is scoped to match: **subsonic, and either flowing or quiescent**. The two
singular operating points $\bar M = 0$ and $\bar M = 1$ are handled as worked out in
`theory.md` §12.6 (the choked boundary by its analytic
one-way limit; the quiescent $\bar M = 0$ case automatically); supersonic acoustic
propagation is deferred with the supersonic mean flow. Crucially the acoustic layer
is **not numerically heavy**: it reuses the mean-flow kernels for its one expensive
object (the converged Jacobian) and is otherwise small sparse linear algebra over a
frequency sweep — so, unlike the mean-flow core, it introduces **no new `@njit`
code** and lives entirely in the Python / SciPy layer.

## 2. Network connectivity

In principle, the relevant connectivity matrix would depened on the role of edges and nodes of the network regarding the assembly of linear equation systems. For the present case of interest, equations are written for each node, and state vectors (e.g. solution variable vectors) are stored at edges.

### 2.1 Entities

We model the network as a **directed graph** plus a notion of *ports*.

All indices — nodes, edges, and ports — are **0-based**.

- **Nodes** $\mathcal{N} = \{n_0, n_1, \dots, n_{N-1}\}$. Conservation equations are assembled at nodes, they can be thought as control volumes.

- **Edges** $\mathcal{E} = \{e_0, e_1, \dots, e_{E-1}\}$. Edges carry the state, the solution vector lives on edges and equations are written in terms of edge states.

- **Ports**. A port is a *local attachment slot on a node*.
 Ports are how a node refers to its own edges without knowing global edge indices, so the local port index is identical to the local edge index within the node.

We represent this information as $\mathbf{P} \in \mathbb{Z}^{N \times E}$ in the range $0 \le P_{ne} \le d_n - 1$ denoting the local index (port) at which edge $e$ is attached to node $n$.
It relates the local edge indices within a node to the global edge indices.

Each edge is **directed** and has two endpoints — a **tail** (source) and a **head** (target).
For an  edge $e$ incident to node $n$, we denote the direction of $e$ **relative** to $n$ as $\sigma_{ne}$ defined as follows:
$$
\sigma_{ne} =
\begin{cases}
+1 & \text{tail at } n \;\;(e \text{ points away from } n,\ \text{outgoing}),\\
-1 & \text{head at } n \;\;(e \text{ points toward } n,\ \text{incoming}).
\end{cases}
$$
The global network topology is represented using incidence matrix $\mathbf{A}$:
$$
A_{ne} =
\begin{cases}
\sigma_{ne} & e \; \text{is an edge of } n,\\
\;\;\,0 & \text{otherwise}.
\end{cases}
$$
$\mathbf{A}$ and $\mathbf{P}$ completely define the network structure.
It can be further compacted to a single matrix $\mathbf{G}$:
$$
G_{ne} =
\begin{cases}
\sigma_{ne} \left( p_{ne} + 1 \right) & e \; \text{is an edge of } n,\\
\;\;\,0 & \text{otherwise}.
\end{cases}
$$
The code will use $\mathbf{A}$ and $\mathbf{P}$.

### 2.2 Storage for code: array layout (NumPy / Numba compatible)

$\mathbf{A}$ and $\mathbf{P}$ are conceptual dense objects; in code we never materialize them dense. Both the node view and the edge view reduce to **flat fixed-dtype integer arrays** — no Python lists, tuples, dicts, or per-edge objects — so the assembly loops are `@njit`/`nopython`-compatible and vectorizable. There are exactly two such views, and they are the **CSR and CSC of the one shared sparsity pattern**.

**Node-row view = CSR of the pattern.** The compressed-sparse-row triple over the $N\times E$ pattern of $\mathbf{A}$, carrying orientation and port as parallel `data` channels:

| array | dtype, length | meaning |
|---|---|---|
| `row_ptr` | `int[N+1]` | node $n$ owns the slot range `k ∈ [row_ptr[n], row_ptr[n+1])`; `row_ptr[n+1]−row_ptr[n] = d_n` |
| `col_edge` | `int[nnz]` | `col_edge[k]` = global edge index at slot $k$ |
| `orient` | `int8[nnz]` | `orient[k] = o_{ne} ∈ {+1,−1}` — the data of $\mathbf{A}$ |
| `port` | `int[nnz]` | `port[k]` = 0-based local port — the data of $\mathbf{P}$ |

with `nnz = ` $\sum_n d_n = 2E - b$ ($b$ = number of boundary half-edges). If slots are emitted in port order then `k − row_ptr[n]` *equals* the local port $p$, so `port` is recoverable from the offset; we keep it explicit for generality. This is precisely `CSR(A)` plus one extra `data` array, and it answers **node → its incident edges** by slicing one contiguous run.

**Edge-column view = CSC of the same pattern.** CSC is the transpose layout: `CSC(A) = CSR(A`$^\top$`)`, grouping the *same* `nnz` incidences by edge (column) instead of by node (row). In general it would need its own `col_ptr` (`int[E+1]`) and `row_node` (`int[nnz]`). But in a graph **every interior edge column has exactly two nonzeros** (one tail `+1`, one head `−1`), so `col_ptr` is the fixed arithmetic sequence `[0, 2, 4, …, 2E]` and carries no information. We therefore collapse the CSC of this pattern into **four fixed-width arrays of length $E$** (a struct-of-arrays — still flat, no ragged rows):

| array | dtype, length | meaning |
|---|---|---|
| `tail_node`, `head_node` | `int[E]` | the two endpoint node indices of edge $e$ |
| `tail_port`, `head_port` | `int[E]` | their 0-based local ports |

with the convention **tail = the `+1` endpoint (source), head = the `−1` endpoint (target)**. Row $e$ of these four arrays *is* column $e$ of $\mathbf{A}$/$\mathbf{P}$ read out: the nonzeros are `(tail_node[e], +1, tail_port[e])` and `(head_node[e], −1, head_port[e])`. This replaces the old loosely-defined "edge record" tuple: it is the CSC column of edge $e$, stored as a row across four arrays.

**The relation, precisely.** The node view and the edge view are the **CSR and CSC of one shared pattern** — the same `nnz` incidences, grouped two ways. CSR is ragged (variable degree $d_n$, hence the `row_ptr` indirection); CSC is regular here (every column has length 2, so its `col_ptr` degenerates to the fixed-width table). Either is obtained from the other in a single $O(\text{nnz})$ counting pass. Materializing both costs `nnz + 4E ≈ 6E` integers total and removes every search from assembly.

**Worked example (the edge-column / CSC view).** For the network in `docs/examples/ConnectivityDemonstrator.yaml` (a UI export, included as an example input file; all indices 0-based), the four arrays are:

| $e$ | `tail_node[e]` | `tail_port[e]` | `head_node[e]` | `head_port[e]` |
|---|---|---|---|---|
| 0 | 0 | 0 | 1 | 0 |
| 1 | 1 | 1 | 2 | 0 |
| 2 | 1 | 2 | 3 | 1 |
| 3 | 2 | 2 | 3 | 0 |
| 4 | 2 | 1 | 4 | 0 |
| 5 | 3 | 2 | 4 | 1 |
| 6 | 4 | 2 | 5 | 0 |

Row $e=3$: edge 3's state couples node-rows `tail_node[3]=2` and `head_node[3]=3` at ports 2 and 0 — an $O(1)$ lookup, no scan. The full set of representations (the CSR node-row view, this CSC edge-column view, the trivial `col_ptr`, and the round-trip between them) is the subject of §2.2 above.

### 2.3 Relation to assembly: the two loops

The shared pattern *is* the block-sparsity of the Jacobian: row $n$ has nonzero blocks exactly at the columns of edges incident to $n$. Assembly walks the pattern in **both directions**, and each direction is served cheaply by exactly one of the two views.

**Loop 1 — node residuals (node → edges, uses CSR).** One block row per node; iterate its contiguous CSR slots, pull each incident edge's state, apply the flux with the stored sign:

```text
for n in range(N):                       # one (block) row of the system
    R = 0
    for k in range(row_ptr[n], row_ptr[n+1]):     # O(d_n), contiguous
        e = col_edge[k]                  # incident edge (global index)
        s = orient[k]                    # +1 outgoing, -1 incoming
        R += s * flux(state[e], local_port = port[k])
    residual[n] = R                      # mass + pressure-type rows live here
```

**Loop 2 — edge equations & scatter (edge → its two nodes, uses CSC).** One block per edge; the four arrays hand back the two endpoint rows directly:

```text
for e in range(E):                       # one (block) column of the system
    t, pt = tail_node[e], tail_port[e]   # the +1 endpoint  (O(1))
    h, ph = head_node[e], head_port[e]   # the -1 endpoint  (O(1))

    # (a) the per-edge total-enthalpy transport equation (THEORY §6.2)
    #     reads the two endpoint donors H[t], H[h] in O(1):
    residual_edge[e] = h_t[e] - ( theta(mdot[e]) * H[t]
                                + (1 - theta(mdot[e])) * H[h] )

    # (b) scatter this edge's state/Jacobian block into the two rows it couples:
    J[t, e] += d_flux_d_state(state[e], local_port = pt)
    J[h, e] -= d_flux_d_state(state[e], local_port = ph)
```

**Exact use case of the edge view.** It is the *only* structure that, given an edge $e$, returns its two endpoint node-rows and local ports in $O(1)$. Two parts of assembly need exactly that and nothing else: (a) evaluating the edge-owned total-enthalpy transport equation, which needs the donor enthalpies $H_{\text{tail}(e)}$ and $H_{\text{head}(e)}$ of its two endpoints (THEORY §6.1–6.2); and (b) scattering the edge's state / Jacobian block into the two node-rows it couples. With only the node-row CSR, each edge would have to scan $O(N)$ rows to rediscover its own endpoints — which the CSC view eliminates.

### 2.4 Recommendation

Store the unpacked pair $\mathbf{A}$ and $\mathbf{P}$ (one shared CSR pattern, two parallel data arrays) for node-row assembly, plus a **per-edge endpoint table** for the scatter direction. Demote the packed $\mathbf{G}$ to an optional on-disk/interchange format only. The reasoning:

1. **The memory argument for $\mathbf{G}$ is gone.** Because $\mathbf{A}$ and $\mathbf{P}$ have identical sparsity, they share a single `indptr`/`indices` and differ only by one extra `data` array of length `nnz` (the ports). That is the entire cost of "two matrices" — not a second sparse structure.
2. **No decode in the hot loop.** $\mathbf{G}$ forces `sign` / `abs` / `-1` on every entry touched during assembly; $\mathbf{A}$/$\mathbf{P}$ hand back orientation and the bare 0-based port directly at slot $k$. Assembly is the inner loop, so it should pay nothing.
3. **No offset, no signed-zero hazard.** With 0-based ports, $\mathbf{G}$ *must* carry the $\pm(p+1)$ offset to keep port $0$ distinct from a structural zero. $\mathbf{A}$/$\mathbf{P}$ avoids it entirely, since $\mathbf{A}$'s pattern already carries connectivity.
4. **$\mathbf{A}$ is reusable as-is.** It is a standard signed incidence matrix, so it drops straight into graph operations ($\mathbf{A}\mathbf{A}^\top$ for node adjacency / graph-Laplacian, Jacobian block-sparsity, connectivity checks). $\mathbf{G}$ cannot be used this way without first being unpacked.

**Why also keep the edge-column (CSC) view.** $\mathbf{A}$/$\mathbf{P}$ in node-row CSR only serve the *node $\to$ edges* direction cheaply (slice row $n$). Assembly also needs *edge $\to$ its two nodes and ports* — the scatter direction and the donor lookup for the edge transport equation — which is the **transpose / CSC** of the very same pattern, collapsed (because every column has exactly two nonzeros) into the four fixed-width arrays `tail_node, tail_port, head_node, head_port`. It answers that in $O(1)$. CSR and CSC are the same `nnz` incidences grouped by row vs. by column, so neither direction ever requires a search; see the two assembly loops above.

### 2.5 Handling pre-defined node and edge based parameters

Both nodes and edges may require certain model parameters to be present
(e.g. an inlet's prescribed mass flow rate, an area-change element's two areas,
a boolean "prevent-reverse-flow" switch). The constraint is that the assembly
loops are `@njit` kernels: whatever they read inside the per-element loop has to
be plain contiguous arrays indexed by integers — not Python objects, and not
anything that hashes strings at run time. The goal is therefore to carry an
arbitrary, per-element-type set of named parameters while presenting only flat
arrays to the kernels.

We reconcile the two by splitting the lifecycle: names and schemas are resolved
once in Python (**parse time**), and only integer-indexed arrays cross into the
`@njit` solve (**solve time**). Mixed dtypes go to separate arrays
(`float`/`int`/`bool`); heterogeneous per-entity schemas are handled by packing
CSR-style, so an entity needing three floats owns three slots and one needing
none owns zero.

#### Storage schema

Per entity kind (`node`, `edge`) and per dtype, a flat value buffer paired with
a CSR pointer array:

```text
node_fparam : float64[nnz_f]   node_fparam_ptr : int64[N_nodes + 1]
node_iparam : int64[nnz_i]     node_iparam_ptr : int64[N_nodes + 1]
node_bparam : bool[nnz_b]      node_bparam_ptr : int64[N_nodes + 1]
edge_fparam : float64[...]     edge_fparam_ptr : int64[N_edges + 1]
edge_iparam : int64[...]       edge_iparam_ptr : int64[N_edges + 1]
edge_bparam : bool[...]        edge_bparam_ptr : int64[N_edges + 1]
```

Entity `n`'s block of a given dtype is `param[ptr[n] : ptr[n+1]]`. `ptr` is
monotonic non-decreasing with `ptr[0] = 0` and `ptr[-1] = nnz`; an entity that
owns no parameter of that dtype has `ptr[n] == ptr[n+1]`. This is the same CSR
offset device used for the connectivity rows.

A Python-side **name→slot map** fixes each parameter's offset within its block:

```text
(entity, param_name) -> i     # i = offset inside that entity's dtype block
```

(key by element *type* if schemas are declared per type rather than per
instance). The map lives only at parse time; kernels carry the resulting `i`
values as compile-time constants.

#### Conversion (parse time, Python, runs once)

1. Parse each node/edge into an ordinary dict `{param_name: value}`.
2. **Promote uniform fields.** A parameter present on *every* node (or *every*
   edge) is not a per-entity parameter — it is uniform field data. Move it to
   the dense store (below) and drop it from the dicts.
3. Walk entities in index order; for each dtype append the remaining values to
   the matching flat buffer, advancing `ptr`, and record each
   `(entity, name) -> i` in the name→slot map.

#### Access (solve time, `@njit`)

The per-element loop receives the flat buffers and their `_ptr` arrays — nothing
else. Entity `n`'s `i`-th parameter of a dtype is a single indexed read:

```text
node_fparam[node_fparam_ptr[n] + i]      # edges mirror: edge_fparam[edge_fparam_ptr[e] + i]
```

`int`/`bool` follow the same form against their own buffers. The `i` values are
the compile-time constants from the name→slot map, so each element routine
hard-codes the offsets of the parameters it declared and binds them to named
locals. No dict, string, or Python object crosses the JIT boundary.

#### Worked example

Network `n0 -(e0)-> n1 -(e1)-> n2` (3 nodes, 2 edges):

| entity | float params  | int params | bool params |
|--------|---------------|------------|-------------|
| `n0`   | `npf0`, `npf1`| —          | —           |
| `n1`   | `npf2`        | `npi0`     | —           |
| `n2`   | `npf3`        | —          | —           |
| `e0`   | —             | `epi0`     | —           |
| `e1`   | `epf0`        | `epi0`     | —           |

Parsed dicts:

```python
nodes = [{"npf0": 2.0, "npf1": 3.5}, {"npf2": 7.0, "npi0": 4}, {"npf3": 1.5}]
edges = [{"epi0": 1}, {"epi0": 0, "epf0": 9.0}]
```

`epi0` is on *every* edge, so step 2 promotes it to the dense store and drops it:
`edge_data[row("epi0"), :] = [1, 0]`. That leaves `epf0` (on `e1` only) as the
sole packed edge parameter; no node parameter is universal, so node dicts pass
through. Flattening the remainder (step 3):

```text
node_fparam     = [2.0, 3.5, 7.0, 1.5]   node_fparam_ptr = [0, 2, 3, 4]
node_iparam     = [4]                     node_iparam_ptr = [0, 0, 1, 1]
node_bparam     = []                      node_bparam_ptr = [0, 0, 0, 0]
edge_fparam     = [9.0]                   edge_fparam_ptr = [0, 0, 1]
edge_iparam     = []                      edge_iparam_ptr = [0, 0, 0]

name->slot:  (n0,npf0)->0  (n0,npf1)->1  (n1,npf2)->0  (n1,npi0)->0  (n2,npf3)->0  (e1,epf0)->0
```

Element routines read by the access rule, binding offsets to named locals:

```python
@njit
def residual_massflow_inlet(n, node_fparam, node_fparam_ptr, ...):
    base = node_fparam_ptr[n]
    npf0 = node_fparam[base + 0]   # i = 0
    npf1 = node_fparam[base + 1]   # i = 1
    # ... build this element's residual rows from npf0, npf1, ...

@njit
def residual_area_change(e, edge_fparam, edge_fparam_ptr, ...):
    epf0 = edge_fparam[edge_fparam_ptr[e] + 0]   # i = 0
    # ...
```

Tracing the example: `n1` → `node_fparam[node_fparam_ptr[1] + 0] = node_fparam[2]
= 7.0` (`npf2`) and `node_iparam[node_iparam_ptr[1] + 0] = 4` (`npi0`); `e1` →
`edge_fparam[edge_fparam_ptr[1] + 0] = 9.0` (`epf0`).

#### Dense store for uniform fields

The packed-CSR form above is for *type-specific* parameters that vary element to
element. Fields that *every* entity carries — solution-adjacent and dependent
per-edge quantities (area, pressures, temperatures, mass flux, …) and any
parameter promoted in step 2 — share a uniform schema and are simplest and
fastest as plain dense arrays: `edge_data[field, edge]` and
`node_data[field, node]`, column = entity, row = named field via a fixed name
list. Both forms reuse the connectivity layer's CSR offset convention, so there
is one storage idiom across the whole solver.

## 3. Jacobian evaluation

We evaluate the Jacobian by **complex-step differentiation** (CSD), which worked
robustly in the prototype without any external AD library. The derivative of a
real residual $R$ with respect to a real unknown $x$ is

$$\frac{\partial R}{\partial x} = \frac{\operatorname{Im}\,R(x + i h)}{h} + \mathcal{O}(h^2),$$

evaluated by perturbing $x$ along the imaginary axis. Unlike a finite
difference it has **no subtractive cancellation**, so $h$ can be taken
absurdly small (e.g. `1e-30`) and the result is accurate to full machine
precision with no step-size tuning. The cost is that every residual must be
evaluated in complex arithmetic.

Reconciling this with JIT code rests on three facts.

### 3.1 The same source serves value and derivative

Numba's `nopython` mode supports `complex128` natively. If a residual routine is
written **dtype-generic** — it never hard-codes `float64`, and it derives its
accumulators from the dtype of the state it is handed — then numba compiles a
second, `complex128` specialization of the *same source* on first call with a
complex argument. The real call returns the residual; the complex call, seeded
on one unknown, returns a Jacobian column. No hand-written derivatives, no
duplicated kernels.

```python
@njit
def node_residual(n, state, row_ptr, col_edge, orient, out):
    R = state[col_edge[row_ptr[n]]] * 0.0      # adopt state's dtype (float64 OR complex128)
    for k in range(row_ptr[n], row_ptr[n + 1]):
        x = state[col_edge[k]]
        R += orient[k] * x * cstep_abs(x)      # example flux = x·|x|  (orient[k] = ±1)
    out[0] = R
```

### 3.2 Complex-step-safe primitives

CSD requires the residual to be *holomorphic* in the perturbed unknown along the
real axis. Three everyday operations break that and must be replaced by `@njit`
helpers used uniformly in every element routine:

- **`abs`** — the built-in returns the modulus $\sqrt{x\bar x}$, a real number
  that annihilates the imaginary seed. Replace with a sign taken from the real
  part only (`cstep_abs` below).
- **comparisons / branch conditions** — `if mdot > 0`, `min`/`max`, clamps,
  upwinding switches must test `.real`, otherwise the infinitesimal seed can flip
  a branch and corrupt the derivative.
- **`.conjugate()`, `np.real`, casting to `float`** in the differentiated path —
  forbidden; each discards the seed.

```python
H = 1e-30

@njit
def cstep_abs(z):            # holomorphic |·| : sign from the real part
    return z if z.real >= 0.0 else -z

@njit
def cstep_gt(a, b):         # branch on the real part so the seed can't flip it
    return a.real > b.real
```

> **Hazard.** The edge transport equation in *Loop 2* uses
> `theta(mdot[e])`, an upwinding switch on the sign of `mdot`. That is exactly a
> branch condition: it must route through `cstep_gt` (compare `mdot.real`), or
> the donor-enthalpy derivatives will be wrong at flow reversal. Any future
> limiter or `min`/`max` clamp inherits the same requirement.

### 3.3 The seed is sparse — reuse the CSC endpoint table

A naïve CSD perturbs one unknown and re-evaluates *all* residuals — $E\cdot$
`N_SOLVE` full sweeps. But the Jacobian's block-sparsity **is** the shared
connectivity pattern (§2.3): perturbing an unknown on edge
$e$ changes only the residuals of the nodes incident to $e$ — its two endpoints,
which the CSC table `tail_node` / `head_node` returns in $O(1)$. So the driver
loops columns (edges), seeds each of the edge's `N_SOLVE` band-1 unknowns, and
recomputes **only those two node-rows**, writing the two nonzero blocks of that
column:

```python
state_c = state.astype(np.complex128)            # complex working copy
for e in range(E):                               # one block column per edge
    t, h = tail_node[e], head_node[e]            # only rows that depend on edge e
    for j in range(N_SOLVE):                     # one band-1 unknown at a time
        state_c[e, j] = state[e, j] + 1j * H     # seed
        for n in (t, h):
            node_residual(n, state_c, ...)        # complex eval, affected rows only
            J[n, e][:, j] = out.imag / H          # exact derivative, this column
        state_c[e, j] = state[e, j]              # un-seed
```

Edge-owned equations (the per-edge transport residual) differentiate by the same
recipe, perturbing their own state and the two donor endpoints.

### 3.4 Worked example

Take the network `n0 -(e0)-> n1 -(e1)-> n2` from before, one unknown per edge
(`N_SOLVE = 1`), state `x = [2.0, 3.0]`, and the flux `x·|x|` used above.
Node `n1` is interior: `e0` is incoming (`orient = -1`), `e1` outgoing
(`orient = +1`), so

$$R(n_1) = -\,x_0\,|x_0| \;+\; x_1\,|x_1|.$$

Seed column `e0` with `x0 → 2.0 + iH` and recompute the rows incident to `e0`,
namely `tail_node[0]=n0` and `head_node[0]=n1`:

$$R(n_1) = -(2{+}iH)\,\mathrm{cstep\_abs}(2{+}iH) + (3)(3)
        = -(2{+}iH)^2 + 9 = 5 + H^2 - 4iH,$$

so `J[n1,e0] = Im(R)/H = -4`, matching the analytic `∂R/∂x0 = -2|x0| = -4`.
Seeding column `e1` likewise gives `J[n1,e1] = +6 = 2|x1|`. These land at exactly
`n1`'s two CSR slots (columns `e0`, `e1`) — the driver never touched any other
row, and the tiny `H` cost nothing in accuracy.

## 4. Variable storage

The full state at an edge is recoverable from the few unknowns that build up our
solution vector; everything else — Mach number, speed of sound, density — is a
*dependent* quantity. These belong to different categories, and the code design
should say so explicitly, because the category of a field is exactly what tells us
**who writes it, in what order, and whether it may be read on the differentiated
path**. We categorize by *provenance*, not by where the bytes live (all of it can
share the one dense store from §2.5).

### 4.1 Three categories by provenance

| band | category | examples | written by |
|---|---|---|---|
| 1 | **primary unknowns** — the source of truth | `mdot, p, h_t`, `Z_el…` | the state update (the Newton vector *is* this band) |
| 2 | **thermo-derived** — opaque to the solver | `T, rho, c, W`, species, rate sources | only `thermo_update`, across the AD-3 boundary |
| 3 | **flow-derived** — solver-owned kinematics | `u = mdot/(ρA)`, `M = u/c`, `q = ½ρu²` | a small `flow_derive`, the solver-side mirror of `thermo_update` |

Nothing derived ever writes band 1; nothing in band 3 crosses the thermo boundary.
`flow_derive` carries the same dtype-generic, complex-clean discipline as
`thermo_update` (§3), but lives on our side of the boundary.

### 4.2 The refresh DAG, and "recomputed, not read"

The categories form a strict downstream chain:

```
unknowns (mdot, p, h_t, Z_el)
   → closure adapter:  h = h_t − ½u²  with  ρ = ρ(Z_el, h, p)   # small fixed point, calls thermo
   → thermo_update     → band 2  (T, ρ, c, W, …)
   → flow_derive       → band 3  (u, M, q, …)
```

Each pass reads only upstream bands and writes its own. The `h_t→h` step is the one
non-trivial link — a mini fixed point, because `h` needs `u` needs `ρ` needs `h` —
and it stays solver-owned, *outside* `thermo_update`, as the closure adapter (R-B1.1).

A single pure routine `derive_edge(x_edge, thermo_params) → (band 2, band 3)` runs
this pipeline, and is used **two ways**:

- **On the residual / PRE (complex) path**, bands 2–3 are recomputed **fresh, inline,
  in order** from the seeded unknown — **never read back** from the dense store.
  Reading the stale real column would drop the complex seed (the same hazard flagged
  for thermo and for `theta(mdot)` upwinding). *Derived fields are recomputed, not
  read, on the differentiated path.*
- **On the real path**, `derive_edge` runs once per accepted step to **refill the
  cache**: the dense columns for bands 2–3 exist for output, diagnostics, lagged
  sources, and warm starts, and are **read-only to the residual**.

So bands 2–3 in `edge_data` are a real-valued cache; the authoritative value on the
differentiated path is always the freshly recomputed one.

### 4.3 Pre-indexing: a parse-time field registry

Every field needs a fixed integer index for `@njit` access, and every index needs a
human-friendly name. The only thing that varies between runs is the **width of the
vector-valued fields** — `Z_el` is `n_elem` wide, species `n_species` wide; every
scalar is known ahead of time. We resolve those widths *once*, at parse time (the
same lifecycle as `thermo_configure()`), and freeze a single authoritative name↔index
map. A `FieldRegistry` is appended to in band order, then finalized:

```python
reg = FieldRegistry()
reg.add_unknowns("mdot", "p", "h_t")          # band 1 — fixed core
reg.add_unknowns(*thermo.element_names())     # band 1 tail — Z_C, Z_H, Z_O, … (n_elem wide)
reg.add_thermo("T", "rho", "c", "W")          # band 2 — fixed core
reg.add_thermo(*thermo.species_names())       # band 2 tail — Y_O2, Y_CO2, … (n_species, may be empty)
reg.add_flow("u", "M", "q")                   # band 3 — solver-owned kinematics
layout = reg.finalize()
```

`finalize()` assigns **contiguous indices in band order** and returns:

- `EDGE_N_VARS`, plus the three band-boundary indices, so a whole band slices as a
  range (`edge_data[reg.THERMO, e]`);
- `names: list[str]` (index→name) and `index[name]` (name→index) — **Python-only**,
  for YAML, output headers, restart files, diagnostics. No string ever reaches `@njit`.
- `field_layout: int64[::1]` — the handful of *flexible* offsets (`n_elem`, `Z_el`
  start, `n_species`, species start, band edges), threaded read-only exactly like
  `thermo_params`.
- `scale: float64[::1]` — a per-variable reference magnitude (`mdot_ref`, `p_ref`,
  …), one entry per index. The solver needs it because the residual rows carry
  **different physical units** — mass in kg/s, momentum in N, energy in W, `Z_el`
  dimensionless — so a raw `‖R‖` is dominated by the largest-unit rows and silently
  ignores the rest. Dividing each residual (and `Δx`, and the tolerances) by `scale`
  makes the convergence test dimensionless and uniform. The registry emits it because
  it is already the single authority on what each index *is*; the solver should not
  keep a parallel table.

The flexible columns are **named by the model**, not by us. `thermo_configure()`
already resolves `n_elem`/`n_species`; it also returns a small **field manifest** —
its ordered list of contributed columns (`["Z_C","Z_H",…]`, `["Y_O2",…]`) — and the
registry concatenates `core unknowns + thermo manifest + flow-derived`. Whoever knows
the mechanism is the one that names the slots.

### 4.4 How `@njit` addresses a column — two tiers

- **Truly-fixed scalars** (`mdot=0, p=1, h_t=2, T, rho, c, W, u, M, q`) never depend
  on the model, so they are plain **module-level `int64` constants** — numba
  constant-folds them, free.
- **Model-flexible offsets** are read once per kernel from `field_layout`
  (`z0 = field_layout[Z_EL_START]; ne = field_layout[N_ELEM]`, then slice
  `x[z0:z0+ne, e]`). Loop-invariant, negligible, and it mirrors how `thermo_params.ti`
  already carries offsets. A swapped mechanism changes only `n_elem`/`n_species` and
  the manifest — **no `@njit` signature changes**, because flexibility lives in array
  *length* plus `field_layout`, never in *type* (the `thermo_params` rule again).

### 4.5 One index namespace, two physical homes

Indices are numbered **once, 0…`EDGE_N_VARS`−1, across all three bands**, so "every
index has exactly one human name" stays literally true. The physical arrays then split
by provenance, which falls straight out of the band boundaries:

- band 1 → the **solution vector** `x[var, edge]` (`N_SOLVE = 3 + n_elem` wide), owned
  and updated by the linear solve;
- bands 2–3 → the **derived cache** `edge_data[field, edge]`, refilled on the real path,
  recomputed-not-read on the complex path.

The registry is the single place that knows the band edges, hence the single place that
knows which array an index lands in. (Node-side state carries the same taxonomy; most
dependent quantities are on edges, so the bands matter most there.)

## 5. Thermodynamic state management and support for reacting flows

The state carried at an edge is a few conserved unknowns — `(mdot, p, h_t)` today,
`(mdot, p, h_t, Z_el…)` once composition is transported. Everything the residuals
actually read — `T`, `rho`, the sound speed `c`, molar mass `W`, later species and
reaction-rate sources — is a *function of a thermodynamic point* `(composition,
h, p)`. Thermo state management puts exactly one such function family behind the
solver so element code never names a concrete gas. This is the boundary
`reactive-flow-requirements.md` draws (AD-3): the **thermo model owns thermodynamic
functions of `(Z_el, h, p)`**; the **solver owns the map from edge unknowns to that
point** (the closure adapter, R-B1.1) and the network balances. We fix the
interface the `@njit` solver sees and leave the model behind it — perfect gas,
equilibrium kernel, table, finite-rate — replaceable. *The model internals are out
of scope here; this section specifies only what crosses the boundary and when.*

A naming note up front. Composition is transported as a **vector of element mass
fractions** `Z_el` (C, H, O, N, …; ~4–5 scalars), which is what makes arbitrary
fuel/oxidiser combinations expressible. The single **mixture fraction** is only the
two-stream special case, and it is **expanded to `Z_el` before any thermo call**
(R-B4.7) — so the interface never sees a mixture fraction, only `Z_el` (or, later,
species).

### 5.1 The boundary: a single `thermo_update`

The solver talks to the model through **one routine**. It is handed a thermodynamic
point and a mode, and it **populates the derived-state fields** for that edge — it
does not return a concrete gas object:

```python
thermo_update(model_id, tf, ti,    # which model + its (opaque) mechanism block
              Z_el, h, p,          # the thermodynamic point at this edge
              mode,                # what to populate
              out)                 # derived fields written back
```

| data | direction | note |
|---|---|---|
| `model_id`, `tf`, `ti` | in | model selector + its flat mechanism block (below) |
| `Z_el, h, p` | in | the point — element mass fractions, static enthalpy, pressure |
| `mode` | in | `STATE` (`T, rho, c, W…`), `+SPECIES`, `+RATES` |
| `out` | out | `T, rho, a_frozen, a_eq, W`; species `Y`; rate sources `wdot` |

Two things stay deliberately *outside* this call, to keep the AD-3 boundary clean:
the **`h_t → h` kinetic-energy fixed point** (`h = h_t − u²/2`, `u = mdot/(ρA)`) is
the closure adapter's job (R-B2.2) and may invoke `thermo_update` several times;
and **mixture-fraction expansion** happens upstream. So `thermo_update` only ever
sees a static `(Z_el, h, p)`.

`thermo_update` is **dtype-generic**, exactly like the residual kernel (see
§3): handed a real state slice it writes real fields; handed
a complex (seeded) slice it writes complex fields. The *same source* therefore
serves a value evaluation and a Jacobian column — there is no separate "complex
thermo."

### 5.2 Two call sites: pre- and post-solve

`thermo_update` fires at two points in a Newton iteration, and that split is the
whole reason the interface is shaped this way:

| | **PRE** (building R / J, before the linear solve) | **POST** (after Δx, or at convergence) |
|---|---|---|
| arithmetic | complex-capable, on the CSD path | real |
| writes | only fields the residual rows read (`T, rho, a_eq, W`) | non-residual fields: full species, diagnostics, **lagged** rate sources |
| mandatory? | yes — makes `R(x)` evaluable | optional for equilibrium; load-bearing for lagged kinetics |

The **PRE** call is what makes the residual evaluable, so it lives *inside*
residual assembly and **rides the sparse CSD seed**: when edge `e`'s unknown is
seeded, `thermo_update(e)` is recomputed on the complex working copy *before* its
two incident node rows are recomputed (endpoints from the CSC table). It must never
be a one-shot real pre-pass that residuals later read from `edge_data` — a stale
real value would silently drop the derivative. The **POST** call runs on the
converged real state to populate what the solve did not need: species for output,
and (for lagged kinetics) the frozen rate sources the next iterate will read.

For an **equilibrium** model the PRE call alone suffices — equilibrium is a closure,
so species are slaved to `(Z_el, h, p)` and the residual has all it needs; POST is
then pure diagnostics.

### 5.3 Selecting a model: an integer id

`nopython` has no virtual dispatch, so we select the backend with an `int64`
`model_id` and a thin switch — every backend shares the `thermo_update` signature:

```python
PERFECT_GAS, EQ_KERNEL, EQ_TABLE = 0, 1, 2     # EQ_* = equilibrium backends

@njit
def thermo_update(model_id, tf, ti, Z_el, h, p, mode, out):
    if   model_id == PERFECT_GAS: _pg_update(tf, ti, Z_el, h, p, mode, out)
    elif model_id == EQ_KERNEL:   _eqk_update(tf, ti, Z_el, h, p, mode, out)
    else:                         _eqt_update(tf, ti, Z_el, h, p, mode, out)
```

This is robust, not a compromise: the branch is on an **integer**, never a
differentiation variable, so it is complex-step-clean; numba **constant-folds** a
pinned `model_id`, inlining straight to the one backend with the dead arms removed
(zero dispatch cost); and adding a backend is one `elif`, invisible to every
element routine. Selection is **global** by default (`PerfectGas` reproduces
today's behaviour bit-for-bit, R-B6.1), because reaction is an *element* concern,
not a gas concern — equilibrium-everywhere (R-B5.1) and a reactor element (R-B5.2)
are configurations over the *same* gas. If mixed gases are ever needed, promote
`model_id` to a per-edge `edge_data` row and index `model_id[e]`; nothing else
changes.

The mechanism a backend needs (species data, element matrix, …) crosses the
boundary as a per-model **flat data block** `tf : float64[…]`, `ti : int64[…]` —
the same dense-store idiom used for uniform edge fields, parsed once and read-only
at solve time. Its precise byte layout stays the backend's private business; the
bundle that carries it, and its per-model contents, are described next.

### 5.4 Model configuration: `thermo_params`

The model is built **once, pre-solve** — the same parse-time/solve-time split used
for connectivity and parameters — and the result is threaded through every Newton
iteration unchanged. A Python `thermo_configure()` reads the user's thermo inputs
(mechanism file, backend choice, options) and emits an immutable bundle:

```text
thermo_params = (model_id, tf, ti)        # immutable, read-only at solve time
    model_id : int64
    tf       : float64[::1]   # every real model constant, one flat C-contiguous blob
    ti       : int64[::1]     # every integer model constant + sub-block offsets
```

Models with an inner solve additionally get a separate, solver-owned **warm-start
buffer** `warm : float64[:, ::1]` (per edge), refreshed on the real path only and
read frozen by CSD columns (§5.2). It is kept *apart* from
`thermo_params` precisely because it is mutable, whereas `thermo_params` never
changes during a solve.

The contents and lengths of `tf`/`ti` are entirely model-dependent — but their
**type is not**, and numba keys compilation on type, not length. Every backend
presents the identical `(int64, float64[::1], int64[::1])`, so a *single*
`thermo_update` compiles for all of them; `PerfectGas` simply ships a length-2 `tf`
and the equilibrium kernel a length-10⁴ one. (The arrays are declared C-contiguous
because *layout* — unlike length — is part of the numba type and would otherwise
force a second specialization.) The per-model variety therefore lives only in array
length and in the compile-time offset constants each backend uses to read its own
sub-blocks — never in a per-model object *type*, which numba could not pass through
one signature.

What the bundle holds in each case (the data *categories*, not the private byte
layout):

```text
PerfectGas    model_id = PERFECT_GAS
              tf = [ cp, R ]                          # one constant gas
              ti = [ ]                                # no composition (n_elem = 0)
              warm : none                             # no inner solve

Equilibrium   model_id = EQ_KERNEL
              tf = [ species NASA polynomials | molar masses | T breakpoints | ref ]
              ti = [ n_species, n_elem | element×species matrix | sub-block offsets ]
              warm : element potentials, per edge     # equilibrium-solve warm start

Finite-rate   model_id = FINITE_RATE
              tf = [ …all of Equilibrium… | Arrhenius (A, b, Ea) | third-body eff. ]
              ti = [ …all of Equilibrium… | reaction stoichiometry | n_reactions ]
              warm : reactor inner-solve state, per edge
```

Equilibrium is a superset of perfect gas, and finite-rate a superset of equilibrium
— the kinetic data is *appended* over the same species thermo underneath. That
nesting is the storage-level reflection of "the `τ→∞` limit recovers equilibrium":
the finite-rate backend reuses the equilibrium sub-blocks verbatim.

### 5.5 Composition through the scalar registry

`Z_el` enters the state through the **scalar registry** (R-B4.1): `h_t` is the
first registered scalar, each element of `Z_el` adds one more, so the band-1
solve width is `N_SOLVE = 3 + n_elem` and the system stays square at
`(3 + n_elem)·E`. The model *declares*
`n_elem`; the registry turns that into the per-edge slot count the storage scheme
already accommodates, and the donor rule advects the extra slots with no new
element code (R-B4.3; realizability R-B4.5). The two assembly loops are unchanged —
they read the extra slots like any other edge state. Choking uses the
**equilibrium** sound speed `a_eq` from `thermo_update` (R-B3.1).

### 5.6 Equilibrium closure (the MVP)

With an equilibrium model, edge states are `f(Z_el, h, p)` and equilibrium holds at
every edge with **no special element and no source term**; combustion appears where
streams of differing `Z_el` mix at junctions (R-B5.1). The one care point is that
the equilibrium solve has an *inner* iteration, and a real-only iterative solve on
the differentiated path is the near-dealbreaker from §3. Two
sanctioned escapes (R-A6.1): make the inner solve **branch-free** so the complex
step propagates through it natively, or converge it on the real part and return an
**analytic sensitivity block** spliced by the implicit-function theorem (the
`state.solve_density` pattern). Either way the `thermo_update` caller stays
oblivious: value on a real call, correct derivative on a seeded call.

### 5.7 Detailed chemistry: separate the kernel from the integrator

Finite-rate chemistry is a later *addition*, not a rewrite, and the design turns on
one distinction:

- the **chemistry kernel** — the pointwise rate evaluator `wdot(Y, T, p)` — is
  small, branch-free, and **complex-clean**; it is what `thermo_update(mode=+RATES)`
  exposes;
- the **chemistry integrator** — the stiff solver that advances a reactor to its
  answer (BDF/Rosenbrock marching, or a steady-PSR root-find) — is full of adaptive
  stepping and branches, is **real-only**, and must be kept **off the flow path**.

Keeping the two separate is what stops chemistry stiffness from contaminating the
flow solve. A **reactor element** nests an *inner* chemistry solve: a real
integrator drives the reactor's species to their converged values for the given
inflow, and the flow Newton sees only the **converged input→output map plus its
exact reduced sensitivity** — obtained by the implicit-function theorem on the
converged reactor residual `G = 0`, using the complex-clean *kernel* (not the
integrator) to form the blocks:

```text
Y_out = integrate_reactor(Y_in, h, p, tau, ...)      # real, stiff, branchy — contained
# expose to flow Newton:  dY_out/d_inputs = -(dG/dY_out)^{-1} (dG/d_inputs)
#   G = mdot (Y_out - Y_in) - V W wdot(Y_out, ...) ,   dG/(.) via CSD on the kernel
```

This gives one clean spectrum, selected per kernel:

- **Fully coupled** (kernel complex-transparent, source in PRE) — CSD lifts the
  stiff source Jacobian for free; best *local* rate, but the stiff block enters the
  flow Jacobian's conditioning, so it leans on continuation.
- **Nested + IFT** (integrator separated, sensitivity spliced) — the **recommended**
  path when the integrator cannot be made complex-clean: stiffness is contained
  inside the element, yet the flow Newton still receives the *exact reduced*
  Jacobian, so it stays a true Newton on the flow variables — separation without
  the lagging penalty.
- **Lagged** (source frozen in POST) — the escape hatch: simplest, any real
  integrator, but only linear/Picard convergence and fragile near
  ignition/extinction.

A note on convergence, to avoid overstating it: the exact (coupled or IFT-reduced)
Jacobian gives the best *local* rate, **not** an unconditional guarantee — stiff
chemistry is ill-conditioned and near-singular at turning points, so robustness is
a **separate, orthogonal lever**: globalization via the planned Damköhler / source
homotopy (R-B5.5), continued from the equilibrium solution. Jacobian quality earns
the *rate*; continuation earns the *reach*. The `τ→∞` limit of any of these recovers
the equilibrium model exactly (R-B5.4; detailed balance R-A5.2).

## 6. Solver design

The solver supports multiple solution algorithms, though only one is implemented
today. It is separate from the network: it consumes the network — connectivity,
element parameters, and a means to update the Jacobian — rather than being part of
it. As with thermodynamic state management, this calls for an explicit solver
interface. The added difficulty is that elements may need solver-specific
quantities, a case the design accounts for from the outset (the `solver_aux` hook
below).

The solution steps are separated into focused routines rather than one monolithic
call. The core is `iterate`, which advances the solution by a single iteration; the
Newton–Raphson controls and the other stabilization knobs sit on top of it.

The boundary mirrors the thermo one, pointed the other way. Thermo is a function
family the network *calls*; the solver is a consumer that *calls the network*. It
sees only a state vector `x`, a residual, and a sparse Jacobian — never an edge,
a gas, or `Z_el`. The contract is one routine:

```python
assemble(x, solver_params) -> (R, J)        # R: residual, J: scipy sparse Jacobian
```

This is the *conceptual* contract — what the solver sees is just `x` in, `(R, J)`
out. The concrete kernel additionally threads the immutable parse-time bundles
(`conn`, `layout`, `thermo_params`) as read-only context, so the real signature is
`assemble(state, conn, layout, thermo_params, solver_params)`; none of those extra
arguments are solver state, so conceptually it remains `assemble(x; params)`.

`J` is assembled explicitly as a sparse matrix (the CSD sparse seed of
§3 fills it column by column) and handed to a **scipy sparse
solver** for the linear step. That linear solve, and the Newton/stabilization
control loop around it, both live *above* the `@njit` line — they run once per
iteration, not once per edge, so they are not hot, and they are free to be branchy
and real-only (the same kernel-vs-integrator split as detailed chemistry). The only
solver-aware code that crosses into `@njit` is a single optional residual term,
described next.

### 6.1 Solver-specific terms: `solver_params`

Most globalization is *transparent* to the network — a line search or trust-region
step is pure algebra on `R` and `J` and needs no cooperation from element code. But
some stabilizers are *intrusive*: they inject a term into the residual itself.

- **Pseudo-transient continuation** adds `(V/Δt)(U − U_old)` to each element
  residual — it needs `x_old` and a (possibly per-edge) timestep.
- **Physical continuation** — the Damköhler / source ramp we keep for stiff
  chemistry — needs a scalar `λ` the residual reads.

These cross the boundary the same way `thermo_params` does: a **read-only bundle
threaded into `assemble`**, built once and passed unchanged through the iteration.

```text
solver_params = (method_id, dt_inv, x_old, cont_lambda, …)   # immutable at solve time
    method_id   : int64          # which strategy (mostly a Python-side concern)
    dt_inv      : float64[…]      # 1/Δt, per-edge or scalar; 0 ⇒ no pseudo-transient
    x_old       : float64[…]      # previous accepted state (pseudo-transient only)
    cont_lambda : float64         # continuation parameter; 1 ⇒ full physics
```

For **pure Newton** the bundle is null/identity (`dt_inv = 0`, `cont_lambda = 1`)
and the extra term is a guarded no-op — so the `@njit` residual signature is
**invariant across solvers**, exactly the length-and-type-stable trick used for
`thermo_params`. The augmentation is read as *data* and applied behind a flag; it is
never a branch on a differentiation variable, so it stays complex-step-clean. The
deliberate cost is that the residual kernel must *know about* the augmentation slot
even when it is unused — one extra additive term, which we accept.

Note the asymmetry with thermo. There the whole physics kernel differs per backend,
so `model_id` must switch *inside* `@njit`. Here every method evaluates the *same*
`R(x)`/`J(x)`; what differs is the outer loop (Python) plus this one additive term
(data). So `method_id` barely crosses the JIT line — "which solver" is a strategy
choice in the control layer, not a kernel switch.

### 6.2 Element-reported quantities: `solver_aux`

The dependency also runs the other way: a solver may need a quantity only the
element can compute. Pseudo-transient needs a local timestep, which means a local
wave speed / CFL number; continuation may want a local stiffness indicator to drive
its schedule. This is an **output from element to solver**, opposite in direction to
`solver_params`.

We expose it as an **optional element hook**, kept *off* the residual path:

```python
solver_aux(x_edge, …) -> aux        # e.g. local |λ|_max for a CFL-based Δt
```

It is optional — absent for solvers that do not ask, so plain Newton pays nothing —
and it is a diagnostic-style read, not part of `R`, so it never needs to be
complex-clean. The control layer calls it between iterations (e.g. to refresh
`dt_inv` for the next `solver_params`), keeping the intrusive feedback loop —
element reports CFL → solver sets `Δt` → residual reads `dt_inv` — entirely outside
the differentiated path.

### 6.3 Carrying state between iterations: a `SolverState`

The control loop has to remember things across iterations — the previous accepted
state for pseudo-transient, the continuation `λ`, the residual-norm history, iterate
counters. We bundle these in a `SolverState`, but with **one firm rule, the same one
that governs `thermo_params`: it never reaches `@njit`.** Numba cannot take a Python
object as a kernel argument, so at every boundary `SolverState` *decomposes into bare
arrays and scalars*; the kernels see `x`, `x_old`, `dt_inv`, `cont_lambda`, never the
object. It is a convenience container for the Python control layer, not a thing the
differentiated path knows about.

That rule also keeps **ownership** clean, which is the real reason to be careful here:

| owned by | members | lifecycle |
|---|---|---|
| **network** | `x`, `edge_data`/`node_data`, `warm` | njit-visible; `x` updated by the linear solve, derived caches refilled on the real path |
| **solver** | `x_old`, `cont_lambda`, residual-norm history, iterate counters, the active `method_id` | control-layer only; never enters a kernel |

The network half is exactly the state the residual already reads and writes; the
solver half is pure bookkeeping the network must stay ignorant of. Keeping
`SolverState` a thin bag that *unpacks* at the boundary — rather than an object passed
downward — is what stops the solver's bookkeeping from leaking into the network and
preserves the "solver consumes the network" separation this section is built on.

## 7. Routine skeleton: signatures first

This section collects everything above into one walk-through of the call graph, as
**empty routines — signatures only**. No bodies yet; the point is to pin down, for
each routine, *what it takes, what it returns, and what it mutates in place*, and to
make the parse-time / solve-time and Python / `@njit` lines explicit. We start from
the pre-defined input the user supplies and end at a converged state.

Conventions in the stubs below:

- `# in` read-only argument · `# out` value returned · `# mod` mutated in place.
- **`@njit`** marks the dtype-generic, complex-clean kernels (same source serves the
  real value and the complex Jacobian column); everything else is plain Python and may
  be branchy and real-only.
- Bundles are the immutable tuples already defined: `conn`, `layout`, `thermo_params`,
  `solver_params`; `state` is the `SolverState` bag, unpacked to bare arrays at every
  `@njit` boundary.

### 7.0 Pre-defined input (what the user supplies)

Data, not routines — the raw problem statement, parsed from YAML:

```text
raw_topology      # incidence: edges, their endpoint nodes and ports (e.g. docs/examples/ConnectivityDemonstrator.yaml)
node_params_raw   # per-node predefined fields — boundary conditions live here:
                  #   a mass-flow-inlet or pressure-outlet IS a node type, with its
                  #   imposed quantity (fixed mdot, p, T, …) carried as a node parameter
edge_params_raw   # per-edge predefined fields (area A, loss coeffs, length, …)
thermo_inputs     # backend choice + mechanism file/options
solver_inputs     # method choice + knobs (tolerances, relaxation, continuation schedule)
```

### 7.1 Parse time — build the immutable solve bundles (Python, runs once)

```python
def build_connectivity(raw_topology):
    """Parse the incidence list into the CSR (node→edges) and CSC (edge→endpoints)
    views of the one shared sparsity pattern. Runs one O(nnz) counting pass."""
    # in  : raw_topology
    # out : conn = (row_ptr, col_edge, orient, port,            # CSR (node→edges)
    #               tail_node, head_node, tail_port, head_port, # CSC (edge→endpoints)
    #               N, E)
    degree = count_incidences_per_node(raw_topology)          # d_n
    row_ptr = prefix_sum(degree)                              # int[N+1]
    for each (edge, endpoint) incidence in port order:        # fills CSR data channels
        col_edge[k], orient[k], port[k] = edge, sign, local_port
    for e in edges:                                           # fills the CSC table
        tail_node[e], tail_port[e] = +1 endpoint of e
        head_node[e], head_port[e] = -1 endpoint of e
    return conn

def build_registry(core_unknowns, thermo_manifest, flow_fields):
    """Number every field once, in band order, and freeze the authoritative
    name↔index map, flexible offsets, and per-variable convergence scales."""
    # in  : the three band manifests (thermo_manifest comes from thermo_configure)
    # out : layout = (EDGE_N_VARS, N_SOLVE, BAND1_END, BAND2_END,
    #                 field_layout, scale, names, index)
    reg = FieldRegistry()
    reg.add_unknowns(*core_unknowns)        # band 1  (mdot, p, h_t, Z_el…)
    reg.add_thermo(*thermo_manifest)        # band 2  (T, rho, c, W, species…)
    reg.add_flow(*flow_fields)              # band 3  (u, M, q)
    return reg.finalize()                   # assigns contiguous indices, computes the tuple

def thermo_configure(thermo_inputs):
    """Build the chosen thermo model once: pick its integer backend id, flatten its
    mechanism into the opaque (tf, ti) block, and report the columns it contributes."""
    # out : thermo_params = (model_id, tf, ti)   # immutable, read-only at solve time
    #     + thermo_manifest (element_names, species_names) feeding build_registry
    model_id      = select_backend(thermo_inputs)             # PERFECT_GAS / EQ_KERNEL / …
    tf, ti        = flatten_mechanism(thermo_inputs)          # backend-private layout
    thermo_manifest = (element_names, species_names)          # names the flexible columns
    return (model_id, tf, ti), thermo_manifest

def solver_configure(solver_inputs, layout):
    """Translate the user's solver choice into the read-only solver_params bundle and
    the control knobs. Pure Newton ⇒ null/identity bundle (no intrusive term)."""
    # out : solver_params, controls
    method_id   = select_method(solver_inputs)
    dt_inv      = zeros(...)        # 0 ⇒ no pseudo-transient (set per schedule otherwise)
    cont_lambda = 1.0               # 1 ⇒ full physics
    x_old       = None              # filled on the first accepted step
    controls    = (tol, max_iter, relaxation, continuation_schedule)
    return (method_id, dt_inv, x_old, cont_lambda), controls

def allocate_state(conn, layout):
    """Allocate the bare solve-time arrays and wrap them in the SolverState bag.
    node_data is then populated from the predefined node parameters (incl. BCs)."""
    # out : state : SolverState
    x         = zeros(layout.N_SOLVE,    conn.E)     # band 1, owned by the linear solve
    edge_data = zeros(layout.EDGE_N_VARS, conn.E)    # full width: bands 2-3 are the live
                                                     # derived cache; band-1 rows are left
                                                     # unused so one index namespace addresses
                                                     # both x and edge_data without an offset
    node_data = zeros(n_node_fields,      conn.N)    # node state + predefined fields (BCs)
    warm      = zeros(n_warm,             conn.E)    # thermo warm-start (real path only)
    return SolverState(x, edge_data, node_data, warm,
                       x_old=None, cont_lambda=1.0, history=[], counters=0)

def initial_guess(state, conn, layout, thermo_params):
    """Seed state.x from the imposed node-boundary values and a first real-path
    derive_edge so edge_data is consistent before the first residual."""
    # mod : state.x, state.edge_data
    # in  : boundary values already live in state.node_data (inlet/outlet node params)
    propagate_boundary_values(state.x, state.node_data, conn)   # mdot/p/h_t starting field
    for e in range(conn.E):                                     # real pass to fill the cache
        b2, b3 = derive_edge(state.x[:, e], thermo_params)
        state.edge_data[:, e] = concat(b2, b3)
```

### 7.2 Solve time — physics layer (`@njit`, dtype-generic, complex-clean)

```python
@njit
def thermo_update(model_id, tf, ti, Z_el, h, p, mode, out):
    """Populate the derived-state fields of one thermodynamic point (Z_el, h, p).
    Dtype-generic: real slice → real fields, complex slice → Jacobian column.
    Model internals are out of scope; this is only the integer-id dispatch."""
    # mod : out  (T, rho, c, W; species; rate sources — per mode)
    if   model_id == PERFECT_GAS: _pg_update(tf, ti, Z_el, h, p, mode, out)
    elif model_id == EQ_KERNEL:   _eqk_update(tf, ti, Z_el, h, p, mode, out)
    else:                         _eqt_update(tf, ti, Z_el, h, p, mode, out)

@njit
def closure_adapter(x_edge, thermo_params):
    """Solve the kinetic-energy fixed point h = h_t − ½u² (u from mdot/ρA, ρ from
    thermo) that maps the unknown h_t to the static h the thermo point needs."""
    # iterate thermo_update(STATE) ↔ h until h settles; converge on .real (complex-clean)
    return h

@njit
def flow_derive(x_edge, band2):
    """Kinematic band-3 fields (u, M, q) from the edge unknowns and the thermo point."""
    return band3

@njit
def derive_edge(x_edge, thermo_params):
    """Per-edge refresh DAG as one pure routine: closure → thermo → flow. Recomputed
    inline on the residual path (never read back); run real once per step to fill cache."""
    h     = closure_adapter(x_edge, thermo_params)
    thermo_update(..., STATE, band2)
    band3 = flow_derive(x_edge, band2)
    return band2, band3

@njit
def residual(x, conn, layout, thermo_params, solver_params, R, edge_data, node_data):
    """Assemble the network residual: per-edge enthalpy-transport equations (CSC loop)
    and per-node balances (CSR loop). PRE thermo recomputed inline so it rides the seed."""
    # mod : R, edge_data
    for e in range(conn.E):                          # Loop-2 (CSC): edge transport equation
        derive_edge(...)                             # PRE — recomputed-not-read
        R[edge_row(e)] = edge_transport(e, ...)      # h_t vs upwinded donor enthalpy (.real switch)
    for n in range(conn.N):                          # Loop-1 (CSR): node balance
        R[node_row(n)] = node_balance(n, ...) - imposed(node_data, n)   # BCs as imposed term
    apply_solver_term(R, x, solver_params)           # guarded pseudo-transient / continuation

@njit
def jacobian(x, conn, layout, thermo_params, solver_params, J, edge_data, node_data):
    """Complex-step Jacobian along the connectivity pattern: seed each edge unknown
    and recompute only its two endpoint node-rows, reading Im(R)/h into J."""
    # mod : J
    for e in range(conn.E):
        for v in range(layout.N_SOLVE):              # each band-1 unknown of edge e
            seed_and_recompute_endpoint_rows(e, v, ...)   # local seed → the two CSC rows → J block

def assemble(state, conn, layout, thermo_params, solver_params):
    """Thin Python wrapper: unpack the SolverState arrays, run the two @njit kernels,
    and hand back R with J as a scipy sparse matrix for the linear solve."""
    # out : (R, J)
    R = empty(n_rows)
    residual(state.x, conn, layout, thermo_params, solver_params,
             R, state.edge_data, state.node_data)
    J = empty_sparse(...)
    jacobian(state.x, conn, layout, thermo_params, solver_params,
             J, state.edge_data, state.node_data)
    return R, to_scipy_sparse(J)
```

### 7.3 Solve time — linear layer (scipy sparse, not `@njit`)

```python
def linear_solve(J, R):
    """Solve the sparse Newton system J·dx = −R with a scipy sparse solver
    (direct LU for v1; an iterative/matrix-free option can drop in here later)."""
    # out : dx
    dx = scipy_sparse_solve(J, -R)
    return dx
```

### 7.4 Solve time — control layer (Python: branchy, real-only)

```python
def solver_aux(x_edge, band2):
    """Optional element→solver hook: report a quantity only the element can compute
    (local |λ|_max for a CFL-based Δt, a stiffness indicator). Off the residual path,
    so it is real-only and never needs to be complex-clean."""
    # out : aux
    return local_wave_speed(x_edge, band2)        # e.g. |u| + c, for the pseudo-transient Δt

def converged(R, dx, scale, tol):
    """Convergence test on the scale-nondimensionalized residual and step, so rows of
    different physical units (mass, momentum, energy, Z_el) are weighted comparably."""
    # out : bool
    return norm(R / scale) < tol and norm(dx / scale) < tol

def step(state, dx, alpha):
    """Apply one accepted Newton update with relaxation / line-search factor alpha."""
    # mod : state.x
    state.x += alpha * dx.reshape(state.x.shape)

def thermo_post(state, conn, layout, thermo_params):
    """POST real pass after an accepted step: populate the non-residual fields —
    full species for output, frozen rate sources for lagged kinetics, diagnostics —
    and refresh the warm-start buffer."""
    # mod : state.edge_data, state.warm
    for e in range(conn.E):
        thermo_update(*thermo_params, ..., STATE | SPECIES | RATES, state.edge_data[:, e])

def iterate(state, conn, layout, thermo_params, solver_params):
    """One Newton iteration: assemble R/J, solve for the step, globalize, apply it,
    then refresh derived state. Records the residual norm into state.history."""
    # mod : state ;  out : info
    R, J  = assemble(state, conn, layout, thermo_params, solver_params)
    dx    = linear_solve(J, R)
    alpha = globalize(state, R, dx)               # line search / trust region (transparent)
    step(state, dx, alpha)
    thermo_post(state, conn, layout, thermo_params)
    state.history.append(norm(R / layout.scale)); state.counters += 1
    return info(R, dx, alpha)

def solve(state, conn, layout, thermo_params, solver_params, controls):
    """Outer driver: iterate until converged or max_iter, advancing the continuation /
    pseudo-transient schedule between steps (refreshing dt_inv via solver_aux). Expects
    state already seeded by the caller (seed_initial), so user ICs are never clobbered."""
    # mod : state ;  out : (state, status)
    for it in range(controls.max_iter):
        info = iterate(state, conn, layout, thermo_params, solver_params)
        if converged(info.R, info.dx, layout.scale, controls.tol): break
        solver_params = advance_schedule(solver_params, state, controls)  # dt_inv, cont_lambda, x_old
    return state, status(state)
```

The dependency order is strictly downstream: §7.0 input → §7.1 bundles → `solve` drives
`iterate`, which calls `assemble` (§7.2) → `linear_solve` (§7.3) → `step`/`thermo_post`
(§7.4). The bodies above are **skeletons, not implementations** — they fix the control flow
and the data each routine touches, deliberately stopping short of the arithmetic (the
element flux laws, the line-search policy, the continuation schedule, the thermo
backends) and of any performance detail (flat-array indexing, in-place seeding,
buffer reuse). The named helpers they call are placeholders, to be worked out and
renamed as we implement each in turn; what each is meant to do:

### 7.5 Placeholder helpers

*Parse time (§7.1):*
- `count_incidences_per_node` — node degree $d_n$ for every node.
- `prefix_sum` — cumulative offsets of the degrees → the CSR `row_ptr`.
- `select_backend` / `select_method` — map a user input string to the integer `model_id` / `method_id`.
- `flatten_mechanism` — pack the chosen mechanism into the opaque `(tf, ti)` block.
- `propagate_boundary_values` — seed `x` from the imposed inlet/outlet node values.
- `concat` — join the two derived bands into one `edge_data` column.

*Physics (§7.2):*
- `_pg_update` / `_eqk_update` / `_eqt_update` — the per-backend thermo kernels (internals out of scope).
- `edge_row` / `node_row` — index of an edge's / node's residual row in `R`.
- `edge_transport` — the per-edge total-enthalpy transport residual (THEORY §6.2): `h_t` vs the upwinded donor enthalpy.
- `node_balance` — the per-node conservation sum over incident edges (the CSR loop body).
- `imposed` — the value a boundary-condition node fixes (zero for an interior node).
- `apply_solver_term` — add the guarded pseudo-transient / continuation term to `R`.
- `seed_and_recompute_endpoint_rows` — CSD-seed unknown `v` of edge `e`, recompute only its two CSC endpoint rows, write the `J` block.
- `empty` / `empty_sparse` / `to_scipy_sparse` — allocate `R`, the fixed-pattern `J`, and convert `J` to scipy sparse.

*Linear & control (§7.3–§7.4):*
- `scipy_sparse_solve` — the sparse linear solve of `J·dx = −R`.
- `local_wave_speed` — element wave speed ($|u|+c$) feeding the CFL-based `Δt`.
- `globalize` — choose the step factor `alpha` (line search / trust region).
- `advance_schedule` — update `dt_inv` / `cont_lambda` / `x_old` per the continuation schedule.
- `norm` — vector norm of its (scale-nondimensionalized) argument.
- `info` / `status` — small result records: per-iteration `(R, dx, alpha)` and the final convergence verdict.

## 8. The perturbation (acoustic) network

The mean-flow solution is an *ingredient*, not the end goal: on top of a converged
mean state the project builds a **linear acoustic / perturbation network** — the
consistency target named in `../README.md`. Its mathematics
(linearization about the mean, characteristic variables, the length-bearing duct,
the storage and source terms, the three target computations) is derived in full in
`theory.md` §12; that chapter is the
**perturbation-theory** reference. This section is the **engineering** counterpart —
where the capability lands in the architecture of §§2–7, what it reuses, and what
little it adds. One fact organizes everything:

> The acoustic network is a **second analysis over the same compiled network and
> the same converged state** — not a second solver. It reuses the connectivity
> (§2), the complex-step machinery (§3), the field registry (§4) and the frozen mean
> thermo state (§5). What it adds — operator assembly and a small family of drivers
> — is light: a sweep of sparse linear solves and one eigenproblem, all **above the
> `@njit` line**. The acoustic layer introduces **no new JIT kernel**.

Once multiple physics *models* are out of scope (§9), this is the only abstraction
boundary left, and it is a real one: **compile once, analyze two ways** — solve for
the mean flow, then probe it acoustically.

Scope mirrors §1: subsonic, flowing **or** quiescent. The two singular operating
points are handled as established in `theory.md` §12.6 — the
choked boundary by its analytic one-way limit, the quiescent $\bar M = 0$ case
automatically (the divergent upwind-linearization term multiplies a *vanishing* mean
total-enthalpy difference, so it cancels for any regularization width). A
near-stagnant edge that bridges a real mean enthalpy gap is *guarded*, not solved
(ibid. §4.5b). Supersonic acoustic propagation is deferred with the supersonic mean
flow, and the assembler refuses a duct edge whose mean state is supersonic.

### 8.1 What is reused, what is new

| layer | mean flow | the acoustic network |
|---|---|---|
| **connectivity** (§2) | CSR/CSC of the incidence pattern | **reused verbatim** — $A(\omega)$ has the *same* block-sparsity as $J$; duct length, element volume and flame data ride the §2.5 per-element parameter store |
| **CSD Jacobian** (§3) | assembles $J(x)$ each Newton step | **reused** — $J_{\mathrm{alg}}$ *is* the converged Jacobian (un-regularized variant, `theory.md` §12.6); the storage $M$ comes from the same complex-step trick on a transient-flux operator |
| **variable storage** (§4) | bands 1–3, real | **extended** — the perturbation unknown $\hat x \in \mathbb{C}$ mirrors the band-1 layout; the registry's names / indices / scales carry over to acoustic I/O |
| **thermo** (§5) | `thermo_update` on the differentiated path | **read-only** — $A(\omega)$ reads the *frozen* mean $\bar c, \bar\rho, \dots$ from `edge_data`; no thermo call on the acoustic path |
| **solver** (§6) | Newton control loop on `assemble → (R,J)` | **new sibling drivers** — frequency sweep, nonlinear eigensolver, inverse solve; real-only Python + SciPy, exactly the §6 control-layer altitude |
| **OO shell** (§9) | `Network → Solver → Solution` | **new consumer** — `Acoustics(problem, solution)`, beside `Solver` |

The net new code is therefore: one operator-assembly routine, a handful of analytic
*stamps* (duct, flame, boundary), and three thin drivers. No connectivity code, no
new kernels, no thermo work.

### 8.2 The acoustic operator $A(\omega)$

At each frequency the perturbation problem is the complex linear system

$$
A(\omega)\,\hat x = \hat b,
\qquad
A(\omega) = \underbrace{J_{\mathrm{alg}}}_{\text{algebraic}}
          + \underbrace{\mathrm i\omega M}_{\text{storage}}
          + \underbrace{P(\omega)}_{\text{propagation}}
          + \underbrace{S(\omega)}_{\text{sources}}
$$

(forcing $\hat b$ for scattering, $\hat b = 0$ for stability). The four blocks come
from four different mechanisms, and only two depend on $\omega$:

- **$J_{\mathrm{alg}}$ — reuse.** The converged steady Jacobian, formed by the §3
  machinery. For acoustics it is (optionally) re-formed with the regularizations off
  and a few polish steps — `theory.md` §12.6. **Real, fixed**
  across the whole sweep.
- **$M$ — one complex-step.** The volume/storage block the steady residual dropped
  ($\partial_t\!\int_V U\,\mathrm dV$, perturbation theory §4.1). It is obtained by
  complex-stepping a dtype-generic **transient-flux operator** $U(x)$ at $\bar x$ —
  the very trick of §3, but evaluated **once** over $E$ edges, so it needs *no* JIT:
  a plain Python function suffices. **Real, fixed** across the sweep.
- **$P(\omega)$ — analytic stamp.** For each duct node, its two continuity rows and
  the entropy-advection row are replaced by the phase relations $e^{-\mathrm
  i\omega\tau_\pm}, e^{-\mathrm i\omega\tau_0}$, built in characteristic variables
  through the **existing** `characteristics.py` blocks $T_e, L_e$. $\omega$-dependent.
- **$S(\omega)$ — analytic stamp.** The flame transfer block (an $n$–$\tau$ closure
  or a measured FTF) coupling a heat-release row to a reference-edge velocity
  functional. $\omega$-dependent; the term that makes $A$ non-self-adjoint.
- **Boundaries.** Terminal-edge reflection coefficients $\hat g = \mathcal
  R(\omega)\,\hat f$, diagonal in characteristics — a third small stamp.

**Two kinds of "complex," kept at different stages.** The complex-step seed
($h \to 0$, an *extraction* device) yields the **real** matrices $J_{\mathrm{alg}}$
and $M$; the frequency $\omega$ is **genuinely** complex physics. The pipeline is:
complex-step at the *real* mean state $\bar x \Rightarrow$ real $J_{\mathrm{alg}}, M$;
*then* assemble the complex $A(\omega)$. One must never complex-step a complex base
state — the two roles of $\mathrm i$ do not mix.

**Per-$\omega$ efficiency.** $J_{\mathrm{alg}}$ and $M$ are assembled once and
cached; a sweep restamps only $P(\omega)$, $S(\omega)$ and the $\mathrm i\omega$
scaling — the same "fixed pattern, cheap per-step update" discipline the Newton loop
uses for $J$. $A(\omega)$ reuses the §2 sparsity throughout.

### 8.3 The duct node and acoustic element faces

Wave propagation needs the one element the mean flow does not: a **length-bearing,
lossless, constant-area duct**. It is modeled as a **2-port node**, not an edge —
edges carry a single state triple, but a duct relates *two distinct* end-station
states, which are exactly its two incident edge-states (the same way an area-change
node relates the two edges it joins). This keeps the §2 convention ("equations at
nodes, state at edges") and the §2 sparsity intact.

The duct has **two faces**:

- a **mean face** — equal-area continuity, the $L$-independent limit of
  `IsentropicAreaChange`; it rides the steady solve and leaves $\bar x$ untouched
  (its `length` is metadata read *only* by the acoustic assembly);
- an **acoustic face** — the $P(\omega)$ phase stamp of §8.2.

This generalizes: **every element type gains an optional acoustic face.** The §9
catalog entry (`ElementSpec`) declares, beside its steady `residual_id`, an optional
**acoustic stamp id** (`acoustic_id`). The *default* face for any element is "my
acoustic block is the CSD linearization of my steady rows" — i.e. the
$J_{\mathrm{alg}}$ path — so **every existing element is acoustically live for
free**. Only three element types override the default: the duct (adds $P$), a
finite-volume element (adds $M$), and a flame (adds $S$). "Each element has a mean
face and an acoustic face" is the founding consistency goal of the project, made
literal in the catalog.

### 8.4 Acoustic parameters and the perturbation state

**Parameters.** The acoustic metadata — duct `length`, element control `volume`
$V_v$, `effective_length` $L_{\mathrm{eff}}$ (the end-correction inertance of the
perturbation theory §4.3), flame `n` and `tau`, terminal `reflection` coefficients —
are **ordinary §2.5 per-element parameters**. They are parsed, promoted-if-uniform,
and packed exactly like the mean-flow parameters, with one defining property: they
**never enter $R$**, so they are invisible to the mean solve and read only when an
acoustic stamp asks for them. No new storage mechanism.

**State.** The perturbation unknown $\hat x$ is **complex** and **mirrors the band-1
layout** — one $(\hat{\dot m}, \hat p, \hat h_t)$ triple per edge, addressed by the
same registry indices as the mean unknowns. Mode shapes and scattering entries are
therefore named by the same strings as everything else (§9). The mean state $\bar x$
and its derived bands are read **frozen** throughout an analysis — a clean constant
read, with none of the "recomputed-not-read" staleness hazard of §4 (there is no
differentiation w.r.t. $\bar x$ on the acoustic path, and $\bar x$ does not change
during a sweep).

### 8.5 The three analyses as drivers

The three target computations of the perturbation theory (§5 there) are three
**drivers** — the acoustic analogues of the §6 Newton control loop, and like it
**real-only Python + SciPy, off the JIT line**:

- **Scattering / transfer matrix.** Pick two stations; apply two linearly
  independent forcings $\hat b^{(1)}, \hat b^{(2)}$; solve $A(\omega)\hat x = \hat b$
  per frequency over a sweep; read the wave amplitudes $w = L_e\hat x$ at the chosen
  ports to fill the $2\times2$ $\mathbf S(\omega)$ (or $\mathbf T$). Cost: two complex
  sparse solves per $\omega$.
- **Stability.** Set $\hat b = 0$; a mode exists iff $\det A(\omega) = 0$, a
  **nonlinear eigenvalue problem** in complex $\omega$ (through $\mathrm i\omega$, the
  duct phases, and the FTF). Solve by Newton on the determinant from acoustic-mode
  seeds, or by **Beyn's contour integral** to capture every mode in a region without
  seeds. This driver owns the one real numerical hazard of the acoustic layer:
  $e^{-\mathrm i\omega\tau}$ over/underflows for complex $\omega$, so it must scale
  the phases (and prefer the contour method for robustness).
- **Black-box identification.** Leave one element's block symbolic; from a measured
  global response $\mathbf S_{\mathrm{meas}}(\omega)$ the assembled relation becomes a
  **linear matrix equation** for that element's $B(\omega)$ over the measured band —
  the formal inverse of scattering.

All three thread the shared bundles and call `assemble_acoustic`; none touches a
kernel. They are `Solver`-siblings in every structural sense (§9).

### 8.6 Routine skeleton: signatures first

As in §7, signatures only — `# in` read-only, `# out` returned, `# mod` mutated.
Everything here is plain Python (NumPy / SciPy sparse); **nothing is `@njit`**. The
shared bundles `conn`, `layout`, `thermo_params` are the §7 ones; `acoustic_params`
is the new read-only bundle of the §8.4 metadata; `x_bar` is a converged
`Solution`'s frozen state.

```python
def build_acoustic_blocks(x_bar, conn, layout, thermo_params):
    """Frequency-independent, built once: the algebraic block and the storage block.
    J_alg is the converged §3 Jacobian, optionally un-regularized + polished
    (theory.md §12.6); M is the complex-step of the transient-
    flux operator at x_bar (once over E edges, so no JIT)."""
    # out : (J_alg, M)        # real scipy.sparse, cached for the whole sweep
    J_alg = steady_jacobian(x_bar, conn, layout, thermo_params, regularized=False)
    M     = complex_step(transient_flux, x_bar, conn, layout, thermo_params)
    return J_alg, M

def assemble_acoustic(omega, blocks, x_bar, conn, layout, acoustic_params):
    """Stamp A(ω) for one frequency: cached J_alg + iω·M, then the ω-dependent
    analytic stamps. Returns a complex scipy.sparse matrix on the §2 pattern."""
    # out : A(omega)
    A = blocks.J_alg.astype(complex) + 1j*omega * blocks.M
    stamp_propagation(A, omega, x_bar, conn, layout, acoustic_params)  # P(ω): duct phases via T_e/L_e
    stamp_sources(A, omega, x_bar, acoustic_params)                    # S(ω): flame n–τ / FTF
    stamp_boundaries(A, omega, x_bar, acoustic_params)                 # reflection coefficients
    return A

def scattering(prob, solution, ports, omegas, acoustic_params):
    """Two independent forcings per ω → the S/T-matrix spectrum at the chosen ports."""
    # out : S(omega) over omegas
    blocks = build_acoustic_blocks(solution.x, prob.conn, prob.layout, prob.thermo_params)
    for omega in omegas:
        A      = assemble_acoustic(omega, blocks, solution.x, prob.conn, prob.layout, acoustic_params)
        x1, x2 = sparse_solve(A, b_up), sparse_solve(A, b_down)
        S[omega] = wave_amplitudes(x1, x2, ports, prob.layout)
    return S

def modes(prob, solution, region, acoustic_params):
    """Nonlinear eigenproblem det A(ω)=0 over a complex-ω region (Newton from seeds, or
    Beyn contour). Returns (frequency, growth rate, mode shape) per mode."""
    # out : [(omega, growth, shape), ...]
    blocks = build_acoustic_blocks(solution.x, prob.conn, prob.layout, prob.thermo_params)
    A_of   = lambda omega: assemble_acoustic(omega, blocks, solution.x,
                                             prob.conn, prob.layout, acoustic_params)
    return nonlinear_eig(A_of, region)        # det A = 0 ; scales e^{-iωτ} internally

def identify(prob, solution, unknown_block, measured, omegas, acoustic_params):
    """Inverse problem: solve the linear matrix equation for the symbolic element block
    B(ω) from a measured global response over the band."""
    # out : B(omega) over omegas
```

The dependency order follows the natural build order (`theory.md` §12, with the
closed-form validation checks of §12.7): duct + `assemble_acoustic` → `scattering`
(validate one duct between reflecting ends) → `modes` (validate the $\omega \approx
n\pi\bar c/L$ duct modes at zero growth) → storage $M$ (validate a Helmholtz
resonator) → flame $S$ → `identify`. Each step has a closed-form acoustic check, in
the spirit of `tests/test_characteristics.py`.

#### Placeholder helpers

- `steady_jacobian` — the §3 CSD assembly, exposed with a `regularized` flag (`theory.md` §12.6).
- `transient_flux` — the dtype-generic $\partial_t U$ operator whose complex-step is $M$.
- `stamp_propagation` / `stamp_sources` / `stamp_boundaries` — the three analytic stamps writing $P$, $S$ and the reflection rows.
- `sparse_solve` — a complex sparse linear solve (SciPy).
- `wave_amplitudes` — read $(f,g,h)$ at named ports via $L_e$ → the scattering columns.
- `nonlinear_eig` — the Newton-on-$\det$ / Beyn driver for $\det A(\omega) = 0$, with phase scaling.

## 9. Object-oriented layer: the user-facing shell

Everything above is a **functional core** — flat, immutable bundles and `@njit` kernels.
This section sketches the **imperative shell** that wraps it: the objects the user
actually holds. The shell's one job is to **translate human-friendly input into the
pre-defined input of §7.0 and drive the §7.1–§7.4 routines** — it *coordinates and names,
and implements no numerics*. Any arithmetic that shows up in a method here is a smell;
it belongs in the core.

Three decisions fix the architecture:

- **One concrete `Network`, not a model framework.** An earlier ambition — make
  `Network` a generic entry point hosting *many* physics models via an injected
  `NetworkModel` — is **dropped**: this version targets compressible flow and only
  that. `Network` is therefore one concrete class holding the generic toolset (graph
  building, string addressing, the registry, lazy result views, the compile valve)
  *and* this physics's element catalog (§9.2) directly. The lesson from the earlier framework still
  binds — `Network` is **never subclassed** (subclassing there grew reflection hooks
  and pushed physics into constructors); the toolset just lives in one place, with
  the per-element variability isolated in the catalog rather than an injected object.
- **Compile once, analyze two ways.** The real axis of variation is not "many models"
  but **two analyses over one compiled network**: `compile()` produces a single
  immutable `CompiledProblem`, consumed both by the mean-flow `Solver` (→ `Solution`)
  and by the `Acoustics` driver (§8) on that solution. Neither analysis owns the
  network; both read the same frozen bundles.
- **Strings are the user's handle.** Nodes, edges and fields are addressed by name;
  the registry resolves names to indices once, at compile.

The same `# in` / `# out` / `# mod` conventions apply. None of these classes cross the
`@njit` line — they assemble the bundles that do.

### 9.1 The surface: objects, and the two consumers

```python
class Network:
    """The one object the user builds and holds: this physics's graph + the generic
    toolset + the element catalog (§9.2). Stores raw input; compiles lazily on first
    solve. Concrete and never subclassed (§1 decision)."""
    catalog = COMPRESSIBLE_FLOW_CATALOG         # the built-in element/BC specs (§9.2)
    def __init__(self, topology, node_params, edge_params, thermo_inputs):
        # in : topology / *_params / thermo_inputs  → the §7.0 raw input
        self._compiled = None                   # cached CompiledProblem (lazy)

    def add_node(self, name, type, **params): ...  # build the graph by string type name
    def connect(self, a, b, element, **params): ...# edge a→b carrying an element type
    #   (BCs are node types — a mass-flow-inlet / pressure-outlet is just add_node(...))

    def compile(self):
        """Run §7.1 once, freeze the immutable bundle. Topology vs parameter split: a
        structural edit invalidates this; a parameter-only edit patches it."""
        # out : CompiledProblem = (conn, layout, thermo_params, state_template)
        conn              = build_connectivity(self._raw_topology)
        tparams, manifest = thermo_configure(self._thermo_inputs)   # resolves n_elem/n_species
        layout            = build_registry(*self._manifests(manifest))  # registry needs the thermo manifest first
        self._compiled    = CompiledProblem(conn, layout, tparams, ...)
        return self._compiled

    def solve(self, solver=None, initial=None, **knobs):
        """Everyday mean-flow entry point. Compile (cached) → run a Solver → return a
        Solution. `solver=None` builds a default Newton solver from knobs; `initial`
        overrides the boundary-seeded guess (see below)."""
        # out : Solution
        prob   = self._compiled or self.compile()
        solver = solver or Solver(**knobs)
        return solver.solve(prob, initial=initial)

    def acoustics(self, solution, acoustic_params=None):
        """The second consumer (§8): build an Acoustics study on a converged Solution,
        sharing the same CompiledProblem."""
        # out : Acoustics
        return Acoustics(self._compiled or self.compile(), solution, acoustic_params)


class Solver:
    """The optional second object: the mean-flow engine. Owns the control loop and the
    SolverState lifecycle; selects a strategy (Newton / pseudo-transient / continuation).
    Implements nothing itself — it threads the §7.1 solver bundle and calls the §7.4
    `solve` driver."""
    def __init__(self, method="newton", tol=1e-8, max_iter=50, **knobs): ...
    def solve(self, prob, initial=None):
        # out : Solution
        sparams, controls = solver_configure(self._inputs, prob.layout)
        state = allocate_state(prob.conn, prob.layout)         # the SolverState bag
        seed_initial(state, prob, initial)                     # boundary seed + user override
        state, status = solve(state, prob.conn, prob.layout,
                              prob.thermo_params, sparams, controls)
        return Solution(state, prob.layout, status)


class Solution:
    """The return value, never constructed by the user. A lazy, read-only view over the
    converged flat arrays through the registry — copies/restructures nothing."""
    def __getitem__(self, name): ...        # field by registry name → the flat column
    def edge(self, name): ...               # named edge → its row of every field
    def node(self, name): ...
    @property
    def history(self): ...                  # residual-norm trace, convergence status


class Acoustics:
    """The acoustic consumer (§8): the optional fourth object. Wraps a converged
    Solution plus the shared CompiledProblem and exposes the three frequency-domain
    analyses. Like Solver it implements no numerics — it threads acoustic_params and
    calls the §8.6 drivers; nothing here crosses the @njit line (there is no acoustic
    kernel)."""
    def __init__(self, problem, solution, acoustic_params=None): ...
    def scattering(self, ports, omegas): ...          # out : S/T-matrix spectrum
    def modes(self, region): ...                      # out : (frequency, growth, mode shape) per mode
    def identify(self, block, measured, omegas): ...  # out : B(ω), the inverse problem
```

Everyday mean-flow use is one object plus its return: `Network` → `solve()` →
`Solution`. The `Solver` surfaces only to tune strategy, sweep parameters, or
warm-restart. The **acoustic** study is the second consumer: `Acoustics(problem,
solution)` exposes `.scattering(...)`, `.modes(...)` and `.identify(...)` — it
threads the §8.4 `acoustic_params` and calls the §8.6 drivers, exactly as `Solver`
threads `solver_params` and calls the §7 loop. Both read the one `CompiledProblem`
that `compile()` froze; the mean solve produces the `Solution` the acoustic study
consumes.

### 9.2 The element catalog and acoustic faces

With the model framework dropped, the per-physics content that used to live in an
injected `NetworkModel` collapses into one thing the concrete `Network` owns
directly: the **element catalog** — the legal node / edge / BC type names and what
each declares. Each entry is an `ElementSpec`, declared explicitly (a registry entry
or decorator, never the old import-a-class-by-string-name trick of the earlier framework):

```python
catalog : dict[str, ElementSpec]      # legal type name → spec  (owned by Network)

class ElementSpec:
    """What one element/BC type declares. Validated at build time; only the integer
    ids and the unknown count ever reach the kernels."""
    params      : dict   # name → (default, required?, units) — build-time validation, never in a kernel
    n_unknowns  : int    # band-1 contribution
    residual_id : int    # dispatch tag for the @njit steady residual (§7.2)
    acoustic_id : int = DEFAULT_FACE   # dispatch tag for the acoustic stamp (§8.3) — optional
```

The single addition for the perturbation network is **`acoustic_id`**, the element's
*acoustic face* (§8.3). Its default value selects the generic face — "my acoustic
block is the CSD linearization of my steady rows," i.e. the $J_{\mathrm{alg}}$ path —
so **every catalogued element is acoustically live with no extra declaration**. Only
three specs override it: the `Duct` (propagation $P$), a finite-volume element
(storage $M$), and a `Flame` (source $S$). Acoustic-only parameters (`length`,
`volume`, `effective_length`, `n`, `tau`, `reflection`) are declared in the *same*
`params` schema as the mean-flow ones (§8.4); they are validated identically and
simply never read by a steady residual.

This is the structural form of the project's founding goal: **one catalog in which
each element type carries both a mean face and an acoustic face**, over a single
shared connectivity and registry. A genuinely different physics is out of scope by
the §1 decision — the catalog, not an injected model object, is where this one
physics is defined.

### 9.3 User-supplied initial conditions

This changes **no structure** — it adds one optional input on the path that already
exists. By default `initial_guess` seeds `state.x` from the imposed boundary node
values (§7.1). The user override is layered *on top* of that seed and resolved through
the registry by field name, so it speaks the same string vocabulary as everything else:

```python
def seed_initial(state, prob, initial=None):
    """Boundary-seed the state, then apply any user-supplied initial conditions on top.
    `initial` is None (BC seed only), a {field_name: value | array} dict, or a prior
    Solution to replay (warm restart)."""
    # mod : state.x, state.edge_data
    initial_guess(state, prob.conn, prob.layout, prob.thermo_params)   # default: from BCs
    if initial is not None:
        for name, value in resolve_fields(initial, prob.layout):       # registry name→index
            state.x[index_of(name)] = value                            # override the seed
```

Two consequences fall out for free: a partial spec is fine (only the named fields are
pinned; the rest keep their boundary seed), and **passing a previous `Solution`
back in as `initial` is warm restart** — because `Solution` is already a view over the
same packed arrays the next solve reads, no conversion is needed. Surface form:
`net.solve(initial={"pressure": 2e5})` or `net.solve(initial=prev_solution)`.

