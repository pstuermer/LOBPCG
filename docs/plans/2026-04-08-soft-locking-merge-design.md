# Soft-Locking Merge: profiling + main

## Goal
Port main's soft-locking (skip converged eigenvectors in P/W) onto profiling's optimized codebase. Both lobpcg and ilobpcg.

## Branch Strategy
New branch off `profiling`. Cherry-pick/port main's features.

## Phase 1: Soft-Locking (correctness, Duersch interaction gated off)

### Core Compaction Logic (both lobpcg + ilobpcg)
At iteration start, when `nconv > 0 && iter > 0`:
```
nconv = alg->converged
n_active = sizeSub - nconv
sizeW = (iter == 0) ? sizeSub : n_active   // before ortho
```

Compact S in-place:
```
P_base = S + sizeSub * size
memmove(P_base, P_base + nconv*size, n_active * size)   // shift P
memcpy(P_base + n_active*size,                            // copy W
       S + (2*sizeSub + nconv)*size, n_active * size)
```

After compaction: `S = [X(sizeSub) | P_active(n_active) | W(sizeW)]`

W pointer: `iter==0 ? S + 2*sizeSub*size : S + (sizeSub + n_active)*size`

### Edge Case Guards

**Underflow guard** (before sizeSub computation in both modified RR):
```c
if (sizeW + n_active <= nx) {
  // degenerate: fall back to mult=2 behavior
  // skip Cp, solve only [X, W_remaining]
}
```

**sizeW == 0 fallback** (in main loops, after ortho):
```c
if (0 == sizeW && alg->iter > 0) {
  mult = 2;  // no search direction, degrade gracefully
}
```

### rayleigh_ritz_modified (definite)

sizeSub formula: `sizeSub_local = (mult-1)*nx + sizeW - (3==mult ? nconv : 0)`

New parameter: `use_cache` (int8_t). Controls whether analytical Gram blocks + PAP_cache are READ. PAP_cache pointer stays non-NULL so it's still WRITTEN at end of RR.
- Phase 1: callers pass `use_cache = (0 == nconv)`
- Phase 2: always pass `use_cache = 1` after implementing full compaction

Existing analytical block code gated:
```c
if (use_cache && PAP_cache && 3 == mult) {
  // overwrite GA/GB with analytical blocks
}
```

### indefinite_rr_modified

Same sizeSub formula. Additional fixes:
- W block width: replace hardcoded `nx` with `sizeW` in B*W application and Gram assembly
- P block width: replace `(mult-1)*nx` with `sizeSub_local - sizeW` for n_cached (the [X,P] portion)
- GB diagonal: `blkdiag(Jx[0:nx], Jp[0:n_active], Jw[0:sizeW])`
- Cp: n_remainder = sizeSub_local - nx. Guard: if n_remainder < 1, skip Cp

### ilobpcg-specific: sig_cached + Jw

sig_cached Gram: `n_v = (iter==0) ? sizeSub : sizeSub + n_active` (not `2*sizeSub`)

Jw: ortho_indefinite writes sizeW entries. Trailing entries are stale. Pass sizeW to indefinite_rr_modified so it reads only valid Jw entries.

### Projection (project_back)

Use wrk4 as separate output buffer (NOT in-place rowpanel GEMM):
```c
ms = sizeSub + (iter==0 ? 0 : n_active) + sizeW;
FN(gemm_nn)(size, 2*sizeSub, ms, 1, S, wrk1, 0, wrk4);
memcpy(S, wrk4, size * 2 * sizeSub * sizeof(CTYPE));
```

project_back writes full-width `[X_new(sizeSub), P_new(sizeSub)]`. Next iteration's compaction extracts active columns.

### Implicit Product Gate

When `nconv > 0`: always use explicit `apply_block_op` for AX. Do NOT use AS/BS implicit path. Do NOT refresh AS/BS (avoid the P-width mismatch in Phase 1).

```c
if (alg->AS && 0 == nconv && !(1 == useOrtho && NULL != alg->B)) {
  // implicit path (existing profiling code)
} else {
  // explicit path
  FN(apply_block_op)(alg->A, X, alg->AX, size, sizeSub);
  // skip AS/BS refresh entirely in Phase 1
}
```

### Ortho

lobpcg (definite):
- iter==0: ortho_drop against X (sizeSub cols)
- iter>=1: ortho_drop against [X, P_active] = sizeSub + n_active cols
- sizeW = return value (may shrink further from dropping)

ilobpcg (indefinite):
- iter==0: ortho_indefinite against X, sig=sig_cached, BV_cache, Jw output
- iter>=1: ortho_indefinite against [X, P_active] = sizeSub + n_active cols
- sizeW = return value
- Jw valid for sizeW entries only

### lobpcg.h Struct Merge

Keep profiling fields: sig_cached, Jx, Jp, Jw, PAP_cache, lambda0, AP, rr_tau
Remove (profiling already removed): rr_VR, rr_ggev
Add from main: verbosity

### Function Signature Changes
- rayleigh_ritz_modified: add `use_cache` param, rename ndrop -> sizeW
- indefinite_rr_modified: rename ndrop -> sizeW (already nretain on profiling)
- Both: already accept nconv

## Phase 2: Full AS/BS/PAP_cache Compaction

After Phase 1 is correct and tested:
- Compact AS/BS with same memmove/memcpy as S
- Recompute PAP_cache at n_active x n_active after RR
- Adjust analytical Gram block indices: P block is n_active wide
- Re-enable implicit product path: pass `use_cache = 1`
- Fix AS P-block refresh to use n_active (not sizeSub)

## Cleanup
- Unify ndrop -> sizeW across all call sites
- Apply type_dispatch.h to all .inc files (replace inline CABS/CREAL)
- Port ortho_indefinite early return fix from main
- Remove restrict from wrk1 in RR (aliasing fix from main)
- Gate diagnostic printf behind verbosity

## Testing
- Deterministic seeds, compare iteration counts
- Port main's soft-locking integration tests (test_lobpcg.c, test_ilobpcg.c)
- Port test_svqb_drop.c, test_ortho_drop.c, test_ortho_indefinite.c
- Verify: no iteration count regression vs main's soft-locking results
- Verify: no iteration count regression vs profiling's Duersch results (when nconv==0)
