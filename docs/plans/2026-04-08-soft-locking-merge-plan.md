# Soft-Locking Merge Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Port soft-locking from main onto profiling branch, gating off Duersch interaction for correctness first.

**Architecture:** New branch `merge-softlock` off `profiling`. Modify function signatures to accept `sizeW` + `use_cache`. Add compaction logic to both main loops. Gate implicit products when `nconv > 0`.

**Tech Stack:** C, X-macros, MKL BLAS/LAPACK

**Design doc:** `docs/plans/2026-04-08-soft-locking-merge-design.md`

---

### Task 1: Create branch and update lobpcg.h struct

**Files:**
- Modify: `lobpcg.h:14-59` (struct definition)

**Step 1: Create branch**

```bash
cd /storage/users/philst/LOBPCG
git checkout profiling
git checkout -b merge-softlock
```

**Step 2: Add `verbosity` field to struct**

In the `LOBPCG_STRUCT` macro, add after `int8_t implicit_product_update;`:
```c
    int8_t verbosity;                            \
```

**Step 3: Commit**

```bash
git add lobpcg.h
git commit -m "added verbosity field to lobpcg_t struct"
```

---

### Task 2: Update rayleigh_ritz_modified signature and implementation

**Files:**
- Modify: `lobpcg.h:185-206` (DECLARE_RR_MODIFIED)
- Modify: `src/rayleigh/rayleigh_ritz_modified_impl.inc`

**Step 1: Rename `ndrop` -> `sizeW` and add `use_cache` in declaration (lobpcg.h)**

Change parameter name `ndrop` -> `sizeW` in `DECLARE_RR_MODIFIED` (line 190).
Add `const int8_t use_cache,` parameter after `uint8_t *useOrtho,` (after line 191).
Update the `_Generic` macro to match.

**Step 2: Update rayleigh_ritz_modified_impl.inc**

Current profiling code (lines 76-79):
```c
(void)nconv;
(void)ndrop;
const uint64_t sizeSub = mult * nx;
```

Replace with:
```c
const uint64_t sizeSub = (mult - 1) * nx + sizeW - (3 == mult ? nconv : 0);
const uint64_t n_remainder = (sizeSub > nx) ? sizeSub - nx : 0;
```

Gate analytical Gram blocks. Current code checks `if (PAP_cache && 3 == mult)`. Change to:
```c
if (use_cache && PAP_cache && 3 == mult) {
```

Do the same for the cached_AS/cached_BS Gram paths. Anywhere that reads PAP_cache or uses analytical overwrites, gate with `use_cache`.

PAP_cache WRITE at end of function: leave ungated (always compute for future use).

Add `n_remainder == 0` guard before QR/Cp section: if `n_remainder < 1`, memset Cp to zero and skip QR.

**Step 3: Update all callers of rayleigh_ritz_modified**

In `lobpcg_impl.inc` (profiling lines 192-196), add `use_cache` argument. For now pass `1` (no soft-locking yet — that comes in Task 5).

**Step 4: Build and verify no regression**

```bash
source /storage/share/intel/ubuntu/setvars.sh
make clean && make -j test_lobpcg && ./test_lobpcg
```

**Step 5: Commit**

```bash
git commit -am "changed rayleigh_ritz_modified: sizeW param, use_cache gate for analytical blocks"
```

---

### Task 3: Update indefinite_rr_modified signature and fix W/P dimensions

**Files:**
- Modify: `lobpcg.h` (DECLARE_INDEF_RR_MOD)
- Modify: `src/rayleigh/indefinite_rr_modified_impl.inc`

**Step 1: Rename `nretain` -> `sizeW` in declaration if not already**

Check current profiling signature — it uses `nretain`. Keep the name but ensure semantics match: sizeW = number of retained W columns after ortho.

**Step 2: Fix hardcoded `nx` for W block width**

In profiling's `indefinite_rr_modified_impl.inc`, the GB_cached assembly applies B to W using hardcoded `nx`:
```c
FN(apply_block_op)(B, &S[size * w_off], wrk2, size, nx);
```
Change `nx` -> `sizeW` (the parameter, which is `nretain`).

Similarly fix `n_cached = (mult-1)*nx` -> use actual P width. After compaction, [X,P] has `sizeSub - sizeW` columns (i.e., `nx + n_active` where `n_active = nx - nconv`). So `n_cached = sizeSub - sizeW`.

Wait — need to be precise. The S layout after compaction is `[X(nx), P(n_active), W(sizeW)]` where `n_active = nx - nconv`. So:
- `sizeSub_local = (mult-1)*nx + sizeW - (3==mult ? nconv : 0)`
- `n_cached = sizeSub_local - sizeW` (the [X,P] portion)
- `w_off = n_cached` (W starts after [X,P])

Update sizeSub computation (same as definite case):
```c
const uint64_t sizeSub = (mult - 1) * nx + nretain - (3 == mult ? nconv : 0);
```

**Step 3: Fix GB diagonal assembly for Jw truncation**

The `blkdiag(Jx, Jp, Jw)` assembly must use `sizeW` (not `nx`) for Jw block:
- Jx block: indices `[0, nx)`
- Jp block: indices `[nx, nx + n_active)` where `n_active = nx - nconv` (only if mult==3)
- Jw block: indices `[nx + n_active, sizeSub)` which is `sizeW` entries

Verify the loop bounds match these dimensions.

**Step 4: Add n_remainder guard**

Same as definite: if `n_remainder = sizeSub - nx < 1`, memset Cp/Jp and return early.

**Step 5: Add underflow guard**

Before sizeSub computation:
```c
if (3 == mult && nretain + nx <= nconv + nx) {
  /* degenerate: no search direction, skip */
  memset(Cp, 0, ...);
  return;
}
```

Actually simpler: just check `if (nretain <= nconv && 3 == mult)` — this means W has fewer columns than converged eigenvectors, so subspace is too small.

**Step 6: Build and test**

```bash
make clean && make -j test_indefinite_rr && ./test_indefinite_rr
```

**Step 7: Commit**

```bash
git commit -am "fixed indefinite_rr_modified: sizeW dimensions for W/P blocks, underflow guard"
```

---

### Task 4: Update ortho_drop to return sizeW

**Files:**
- Modify: `src/ortho/ortho_drop_impl.inc`

**Step 1: Port main's sizeW return**

On profiling, `ortho_drop` already returns `uint64_t`. Check if it returns the retained column count. If it returns 0 or void-equivalent, update it to return the number of columns retained after svqb dropping.

Main's approach: after the inner svqb loop, `colsretain` from svqb is the number of retained W columns. Return this value.

**Step 2: Build and test**

```bash
make clean && make -j test_ortho_drop && ./test_ortho_drop 2>/dev/null || echo "test not yet ported"
```

**Step 3: Commit**

```bash
git commit -am "changed ortho_drop to return retained column count (sizeW)"
```

---

### Task 5: Update ortho_indefinite to return sizeW

**Files:**
- Modify: `src/ortho/ortho_indefinite_impl.inc`

Same as Task 4 but for the indefinite variant. Profiling already has `BV_cache` and `Jw` params. Ensure the return value is the number of retained columns after svqb.

Also: port main's early return fix for ortho_indefinite (commit `915f4cd`).

**Step 1: Check and fix return value**
**Step 2: Build and test**
**Step 3: Commit**

```bash
git commit -am "changed ortho_indefinite to return retained column count"
```

---

### Task 6: Add soft-locking to lobpcg_impl.inc

**Files:**
- Modify: `src/core/lobpcg_impl.inc`

This is the main integration task for the definite solver.

**Step 1: Add compaction at iteration start (after line 149, inside while loop)**

After `uint64_t mult = ...;`:
```c
const uint64_t nconv = alg->converged;
const uint64_t n_active = sizeSub - nconv;
uint64_t sizeW = (0 == alg->iter) ? sizeSub : n_active;

/* Compact P and W: remove converged columns */
if (nconv > 0 && alg->iter > 0) {
  CTYPE *P_base = alg->S + sizeSub * size;
  memmove(P_base, P_base + nconv * size, n_active * size * sizeof(CTYPE));
  memcpy(P_base + n_active * size,
         alg->S + (2 * sizeSub + nconv) * size,
         n_active * size * sizeof(CTYPE));
}

/* W pointer: dynamic based on compaction */
W = (0 == alg->iter) ? alg->S + size * 2 * sizeSub
                     : alg->S + size * (sizeSub + n_active);
```

**Step 2: Update ortho calls to use n_active**

Profiling line 164: `FN(ortho_drop)(size, sizeSub, 2 * sizeSub, ...)`.
Change to:
```c
if (0 == alg->iter) {
  FN(ortho_drop)(size, sizeSub, sizeSub, eps_ortho, eps_drop,
                 W, X, alg->wrk1, alg->wrk2, alg->wrk3, alg->B);
} else {
  sizeW = FN(ortho_drop)(size, n_active, sizeSub + n_active, eps_ortho, eps_drop,
                          W, alg->S, alg->wrk1, alg->wrk2, alg->wrk3, alg->B);
}
```

Note: at iter>=1, W has `n_active` columns (not sizeSub). Ortho against `[X, P_active]` = `sizeSub + n_active` columns.

**Step 3: Add sizeW==0 fallback**

After ortho:
```c
if (0 == sizeW && alg->iter > 0) {
  mult = 2;
}
```

**Step 4: Update AW computation for AS cache**

Profiling lines 174-189 compute AW into AS. With compaction, W has `sizeW` columns. Change:
```c
if (alg->AS) {
  if (0 == alg->iter) {
    FN(apply_block_op)(alg->A, W, alg->AS + size * sizeSub, size, sizeSub);
  } else {
    /* W has sizeW columns, placed after [X, P_active] in S */
    FN(apply_block_op)(alg->A, W, alg->AS + size * 2 * sizeSub, size, sizeW);
  }
  /* ... same for BS ... */
}
```

Wait — but Phase 1 gates off implicit products when nconv > 0. So the AS/BS computation for AW doesn't matter when nconv > 0 because it won't be read. However, the AW computation also feeds rayleigh_ritz_modified's cached_AS path. Since we gate that with `use_cache = (0 == nconv)`, the AW-in-AS is only used when nconv==0, i.e., sizeW==sizeSub==n_active. So no dimension issue in Phase 1.

Keep the profiling code as-is for the AS/AW computation. The gate handles correctness.

**Step 5: Update rayleigh_ritz_modified call**

Profiling line 192: `FN(rayleigh_ritz_modified)(size, sizeSub, mult, alg->converged, 0, &useOrtho, ...)`

Change 5th arg from `0` to `sizeW`. Add `use_cache` arg:
```c
const int8_t use_cache = (0 == nconv);
FN(rayleigh_ritz_modified)(size, sizeSub, mult, nconv, sizeW, &useOrtho,
                           use_cache,
                           alg->S, alg->AX, alg->wrk1, alg->wrk2, alg->wrk3,
                           alg->Cx, alg->Cp, alg->eigVals,
                           alg->rr_eigvals, alg->rr_tau, alg->rr_D,
                           alg->PAP_cache, alg->AS, alg->BS,
                           alg->A, alg->B);
```

Do the same for the retry path (useOrtho==2 block, profiling lines 227-231).

**Step 6: Update project_back to use wrk4 (not in-place)**

Profiling lines 239-243 already use wrk4:
```c
FN(gemm_nn)(size, 2 * sizeSub, ms, (CTYPE)1, alg->S, alg->wrk1, (CTYPE)0, alg->wrk4);
FN(copy)(size * 2 * sizeSub, alg->wrk4, alg->S);
```

Good — already safe. Just update `ms`:
```c
uint64_t ms = sizeSub + (0 == alg->iter ? 0 : n_active) + sizeW;
```

**Step 7: Gate implicit product updates**

Profiling line 248: `if (alg->AS && !(1 == useOrtho && NULL != alg->B))`.

Add nconv gate:
```c
if (alg->AS && 0 == nconv && !(1 == useOrtho && NULL != alg->B)) {
  /* implicit path — only when no soft-locking active */
  ...
} else {
  /* explicit path */
  if (alg->AX)
    FN(apply_block_op)(alg->A, X, alg->AX, size, sizeSub);
  /* Skip AS/BS refresh in Phase 1 when nconv > 0 */
}
```

**Step 8: Reset P and W pointers after project_back**

Profiling lines 235-236 redefine P and W to fixed offsets. Keep this — project_back writes full-width `[X, P]`, so after project_back:
```c
P = alg->S + size * sizeSub;
W = alg->S + size * 2 * sizeSub;
```

The next iteration's compaction will handle the shrinking.

**Step 9: Build and test**

```bash
make clean && make -j test_lobpcg && ./test_lobpcg
```

**Step 10: Commit**

```bash
git commit -am "added soft-locking to lobpcg: P/W compaction, use_cache gate, sizeW propagation"
```

---

### Task 7: Add soft-locking to ilobpcg_impl.inc

**Files:**
- Modify: `src/core/ilobpcg_impl.inc`

Same compaction pattern as Task 6, plus indefinite-specific changes.

**Step 1: Add compaction at iteration start**

Same memmove/memcpy as lobpcg. Insert after `uint64_t mult = ...;` (profiling line 142).

**Step 2: Update sig_cached computation**

Profiling lines 153-155:
```c
const uint64_t n_v = (0 == alg->iter) ? sizeSub : 2 * sizeSub;
```
Change to:
```c
const uint64_t n_v = (0 == alg->iter) ? sizeSub : sizeSub + n_active;
```

And V_ortho should be [X, P_active] at iter>=1. Since after compaction S = [X, P_active, W_active], and we want [X, P_active]:
```c
CTYPE *V_ortho = (0 == alg->iter) ? X : alg->S;  /* S starts with [X, P_active] */
```
This is already correct — S starts with X, and the next `n_active` columns are P_active.

**Step 3: Update ortho_indefinite calls**

Profiling line 164: `FN(ortho_indefinite)(size, sizeSub, 2 * sizeSub, ...)`
Change to use n_active for W width and sizeSub + n_active for V width:
```c
if (0 == alg->iter) {
  FN(ortho_indefinite)(size, sizeSub, sizeSub, eps_ortho, eps_drop,
                       W, X, alg->sig_cached,
                       alg->wrk1, alg->wrk2, alg->wrk3, alg->B,
                       alg->wrk4, alg->Jw);
} else {
  sizeW = FN(ortho_indefinite)(size, n_active, sizeSub + n_active,
                                eps_ortho, eps_drop,
                                W, alg->S, alg->sig_cached,
                                alg->wrk1, alg->wrk2, alg->wrk3, alg->B,
                                alg->wrk4, alg->Jw);
}
```

**Step 4: Update indefinite_rr_modified call**

Profiling line 173: change 5th arg from `sizeSub` to `sizeW`:
```c
FN(indefinite_rayleigh_ritz_modified)(size, sizeSub, mult, alg->converged, sizeW,
                                      alg->S, alg->AX, NULL,
                                      alg->lambda0,
                                      alg->wrk1, alg->wrk2, alg->wrk3, alg->wrk4,
                                      alg->rr_tau,
                                      alg->Cx, alg->Cp,
                                      alg->eigVals, alg->signature,
                                      alg->rr_eigvals, alg->rr_sig,
                                      alg->rr_indices,
                                      alg->Jx, Jp_rr, alg->Jw,
                                      alg->sig_cached,
                                      alg->PAP_cache,
                                      alg->A, alg->B);
```

**Step 5: Replace in-place rowpanel GEMM with wrk4 buffer**

Profiling lines 192-193:
```c
FN(gemm_nn_rowpanel)(size, 2 * sizeSub, ms, (CTYPE)1, alg->S,
                     alg->wrk2, (CTYPE)0, alg->S);
```

Replace with safe version:
```c
uint64_t ms = sizeSub + (0 == alg->iter ? 0 : n_active) + sizeW;
FN(copy)(ms * sizeSub, alg->Cx, alg->wrk2);
FN(copy)(ms * sizeSub, alg->Cp, alg->wrk2 + ms * sizeSub);
FN(gemm_nn)(size, 2 * sizeSub, ms, (CTYPE)1, alg->S,
            alg->wrk2, (CTYPE)0, alg->wrk4);
FN(copy)(size * 2 * sizeSub, alg->wrk4, alg->S);
```

**Step 6: sizeW==0 fallback**

Same as lobpcg: `if (0 == sizeW && alg->iter > 0) mult = 2;`

**Step 7: Build and test**

```bash
make clean && make -j test_ilobpcg && ./test_ilobpcg
```

**Step 8: Commit**

```bash
git commit -am "added soft-locking to ilobpcg: P/W compaction, sig_cached fix, sizeW propagation"
```

---

### Task 8: Port soft-locking tests from main

**Files:**
- Modify: `tests/test_lobpcg.c`
- Modify: `tests/test_ilobpcg.c`

**Step 1: Port `diag_matvec_d` helper and `d_lobpcg_softlock` test from main**

Copy `diag_ctx_d_t`, `diag_matvec_d`, and `TEST(d_lobpcg_softlock)` from main's `tests/test_lobpcg.c` (lines 440-500). Adapt linop_ctx setup to match profiling's pattern (profiling uses direct `op->ctx` assignment, not intermediate `lctx` struct — check existing profiling tests for the pattern).

Add `RUN(d_lobpcg_softlock)` to main().

**Step 2: Port `d_ilobpcg_softlock` test from main**

Copy `TEST(d_ilobpcg_softlock)` from main's `tests/test_ilobpcg.c` (lines 438-495). Same linop_ctx adaptation.

Add `RUN(d_ilobpcg_softlock)` to main().

**Step 3: Build and run all tests**

```bash
make clean && make -j test_lobpcg test_ilobpcg
./test_lobpcg
./test_ilobpcg
```

Both soft-locking tests must PASS. Existing tests must not regress.

**Step 4: Commit**

```bash
git commit -am "added soft-locking integration tests for lobpcg and ilobpcg"
```

---

### Task 9: Port ortho tests from main

**Files:**
- Create: `tests/test_svqb_drop.c` (from main)
- Create: `tests/test_ortho_drop.c` (from main)

**Step 1: Cherry-pick or copy test files from main**

These are new files on main. Copy them and adapt function signatures to match profiling's API (e.g., ortho_indefinite has BV_cache and Jw params).

**Step 2: Update Makefile if needed**

Add test targets for new test files.

**Step 3: Build and run**

```bash
make clean && make -j test_svqb_drop test_ortho_drop
./test_svqb_drop
./test_ortho_drop
```

**Step 4: Commit**

```bash
git commit -am "added ortho drop and svqb drop tests"
```

---

### Task 10: Cleanup — type_dispatch.h and misc fixes

**Files:**
- Create: `include/lobpcg/type_dispatch.h` (from main)
- Modify: all `.inc` files with inline CABS/CREAL macros

**Step 1: Copy type_dispatch.h from main**

```bash
git show main:include/lobpcg/type_dispatch.h > include/lobpcg/type_dispatch.h
```

**Step 2: Replace inline CABS/CREAL blocks in each .inc file**

In each file that has the `#ifdef TYPE_IS_FLOAT / CABS / CREAL` block (lobpcg_impl.inc, ilobpcg_impl.inc, rayleigh_ritz_modified_impl.inc, indefinite_rr_modified_impl.inc, svqb_impl.inc, etc.):

Replace the ~15-line macro block with:
```c
#include "lobpcg/type_dispatch.h"
```

**Step 3: Port restrict removal from wrk1 in RR**

From main commit `2baa54a`: remove `restrict` from wrk1 parameter in rayleigh_ritz functions where it aliases with Cx_ortho.

**Step 4: Gate diagnostic printf behind verbosity**

In indefinite_rr_modified, wrap `printf("RR: kappa(R)=...")` with:
```c
if (alg && alg->verbosity > 0) { ... }
```

Actually, the RR functions don't receive the alg struct. Skip this — diagnostic prints can be cleaned up separately.

**Step 5: Build full test suite**

```bash
make clean && make -j && make test
```

All tests must pass.

**Step 6: Commit**

```bash
git commit -am "added type_dispatch.h, replaced inline CABS/CREAL macros, removed restrict alias"
```

---

### Task 11: Final validation

**Step 1: Run full test suite with deterministic seeds**

```bash
make clean && make -j
./test_lobpcg
./test_ilobpcg
./test_rayleigh_ritz
./test_indefinite_rr
./test_svqb_drop
./test_ortho_drop
```

**Step 2: Verify iteration counts**

Compare soft-locking test iteration counts with main branch results. They should match exactly (same seeds, same algorithm).

**Step 3: Verify Duersch path (nconv==0) iteration counts**

Run a test where nothing converges early — iteration count should match profiling branch exactly.

**Step 4: Update TODO.md**

Check off completed items, note Phase 2 as pending.
