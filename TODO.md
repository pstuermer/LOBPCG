# LOBPCG Implementation TODO

## Phase 1: Foundation

### Header Fixes
- [x] `types.h:14`: Fix `complex` to `float complex`
- [x] `lobpcg.h:2`: Fix typo `LOPBCG_H` to `LOBPCG_H`
- [x] `lobpcg.h`: Fix all `#undef` without identifier (lines 50, 118, 149, 183, 213, 243)
- [x] `lobpcg.h:95`: Move `#endif` to end of file
- [x] `lobpcg.h:179`: Fix `rtype` to `ctype` for Cp parameter
- [x] `lobpcg.h:192`: Fix macro args mismatch (uses `sizeSub, mult` but params are `nx, nconv, ndrop`)
- [x] `lobpcg.h:231`: Fix typo `uin64_t` to `uint64_t`
- [x] `lobpcg.h:234-235`: Add missing comma after `eps_drop`

**Verify:** Headers compile without warnings: `gcc -c -Wall -std=c11 lobpcg.h` ✓ PASSED

### Build System
- [x] Create `Makefile` with MKL/OpenBLAS/BLIS backend support
- [x] Create `examples/Makefile`
- [x] Clean up Makefile: wildcard-based test discovery, all tests link against `liblobpcg.a`; moved `linop_test.c` → `tests/linop_test.c`
- [ ] (Deferred) CMake build system

**Verify:** `make tests && make run-tests`

### BLAS Wrappers
- [x] Create `include/lobpcg/blas_wrapper.h`
- [x] Implement GEMM wrappers (s, d, c, z) - variants: NN, NT/NH, TN/HN
- [x] Implement NRM2 wrappers
- [x] Implement DOT (real) / DOTC (complex) wrappers
- [x] Implement AXPY, SCAL, COPY wrappers
- [x] Implement SYRK (real) / HERK (complex) for Gram matrices
- [x] Implement POTRF (Cholesky) — upper triangular R^H*R
- [x] Implement TRSM (triangular solve) - LLN, LLT/LLH, and RUN (right-upper-notrans) variants
- [x] Implement SYEV (real) / HEEV (complex) eigensolve
- [x] Implement GEEV (general eigensolve)
- [x] Implement GGEV (generalized eigensolve for indefinite)
- [x] Implement TRCON (triangular condition number)
- [x] Add type-generic macros: `nrm2`, `axpy`, `scal`, `copy`, `dot`, `gram`, `potrf`, `eig`
- [ ] ~~Implement GEMV wrappers~~ (not needed - all matvecs via LinearOperator)

**Verify:** `test_blas.c` passes for all types

---

## Phase 2: Orthogonalization

### SVQB (Standard) - for lobpcg, klobpcg
- [x] Create `src/ortho/svqb_impl.inc`
- [x] Implement Gram matrix computation: `G = U^H * B * U`
- [x] Implement diagonal scaling
- [x] Implement eigendecomposition of scaled Gram matrix
- [x] Implement column transformation
- [x] Create instantiation files `svqb_{s,d,c,z}.c`
- [x] Create `tests/test_svqb.c`
- [ ] (Deferred) Implement drop='y' mode (randomize weak columns)

**Verify:** `test_svqb.c` - `||U^H * B * U - I||_F < 1e-14` (double) ✓ PASSED
**Reference:** `lobpcg.c:598-751` (zsvqb)

---

### ortho_randomize - REMOVED (superseded by ortho_drop)
Identical algorithm; deleted impl, instantiation files, test, and lobpcg.h declarations.

---

### svqb_mat - for ilobpcg
- [x] Create `src/ortho/svqb_mat_impl.inc`
- [x] Implement matrix-based SVQB: `G = U^H * mat * U`
- [x] Takes explicit matrix instead of LinearOperator
- [x] Diagonal scaling and eigendecomposition
- [x] Reuse scaling/eigensolve logic from svqb
- [x] Create instantiation files `svqb_mat_{s,d,c,z}.c`
- [x] Add declarations to `lobpcg.h`
- [x] Create `tests/test_svqb_mat.c`

**Verify:** `||U^H*mat*U - I||_F < tol` for indefinite mat ✓ PASSED
- [x] Fixed undefined `info` variable in eig error path
- [x] Added 2x2 permutation matrix tests (d/z)
**Reference:** `ilobpcg.c:73-126` (zsvqb_mat)
**Priority:** Implement before ortho_drop_mat (dependency)

---

### ortho_drop_mat - for ilobpcg
- [x] Create `src/ortho/ortho_drop_mat_impl.inc`
- [x] Implement matrix-based orthogonalization against V
- [x] Uses svqb_mat internally (not svqb)
- [x] Double projection to handle indefinite metric
- [x] Create instantiation files `ortho_drop_mat_{s,d,c,z}.c`
- [x] Add declarations to `lobpcg.h`
- [x] Create `tests/test_ortho_drop_mat.c`

**Verify:** `||V^H*mat*U||_F < tol` ✓ PASSED; fixed wrk3 underallocation (was max_n*max_n, needs m*max_n)
**Reference:** `ilobpcg.c:128-183` (ortho_drop_mat)
**Note:** Was MISSING from original TODO

---

### ortho_indefinite - for ilobpcg soft-locking
- [x] Create `src/ortho/ortho_indefinite_impl.inc`
- [x] Implement B-orthogonalization with signature tracking
- [x] Signature-weighted projection: `U = U - V*sig*(V^H*B*U)`
- [x] Handle positive/negative signature separately
- [x] Create instantiation files `ortho_indefinite_{s,d,c,z}.c`
- [x] Add declaration to `lobpcg.h`
- [x] Create `tests/test_ortho_indefinite.c`

- [x] Add overdetermined guard (`m < n_u + n_v`)
- [x] Replace fill_lower + full-matrix norm with upper-triangle-only `ortho_err_upper` helper

**Verify:** `test_ortho_indefinite.c` ✓ PASSED (4 diagonal-B + 4 permutation-B + 2 B=NULL tests)
**Reference:** `ilobpcg.c:185-350` (zortho_randomize_indefinite)

---

## Phase 3: Rayleigh-Ritz

### Standard Rayleigh-Ritz
- [x] Create `src/rayleigh/rayleigh_ritz_impl.inc`
- [x] Build `G_A = S^H * A * S`
- [x] Build `G_B = S^H * B * S`
- [x] Cholesky factorization of G_B
- [x] Transform to standard eigenvalue problem
- [x] Eigensolve
- [x] Back-transform eigenvectors
- [x] Switch to upper Cholesky (R^H*R) matching herk/syrk upper-triangle output
- [x] Fix trsm bug: use right-side trsm_run for D*inv(R) instead of left-side trsm_llh
- [x] Remove unnecessary fill-lower loops (herk→potrf→eig all use upper triangle)
- [x] Change eigVal parameter from CTYPE* to RTYPE* (eigenvalues are always real)

**Verify:** `test_rayleigh_ritz.c` - 8 tests: 4x4/6x6 dense matrices, d/z types, standard/modified/ortho/chol branches ✓ PASSED

### Modified Rayleigh-Ritz
- [x] Create `src/rayleigh/rayleigh_ritz_modified_impl.inc`
- [x] Handle search space `S = [X_active | P | W]`
- [x] Extract Cx coefficients for X update
- [x] Extract Cp coefficients for P update
- [x] Switch to upper Cholesky + trsm_run (same fixes as standard RR)
- [x] Add useOrtho==1 branch: direct eigensolve on S^H*A*S (no Gram/Cholesky/D_inv_R)
- [x] Fix Cp QR dimensions: store Z2 (not Z2^T), QR on tall matrix, fix K in GEMM (both branches)

**Verify:** Covered by standard RR tests (ortho + chol branches, mult=2 and mult=3) ✓ PASSED

### Indefinite Rayleigh-Ritz
- [x] Create `src/rayleigh/indefinite_rr_impl.inc`
- [x] Use GGEV for generalized problem
- [x] Track signature (+1/-1) of each eigenpair
- [x] Sort: positive ascending, negative descending
- [x] Implement `bubble_sort_sig()` - sort by signature

**Verify:** `test_indefinite_rr.c` - 17 tests: diag/dense A × diag/perm B, basic/modified, d/s/z/c types ✓ PASSED

### K-based Rayleigh-Ritz (for klobpcg)
- [x] K-based RR aliased to standard RR in `lobpcg.h` (algorithmically identical)

**Verify:** Uses same code as standard RR ✓

---

## Phase 4: Residual & Main Loop

### Memory Management
- [x] Create `include/lobpcg/memory.h` with aligned allocation utilities
  - `xmalloc(size)` - 64-byte aligned malloc
  - `xcalloc(num, size)` - 64-byte aligned calloc with overflow check
  - `safe_free(void**)` - properly nullifying free (fixes reference bug)
- [x] Remove memory functions from `linop.h` (lines 25-39)
- [x] Implement `lobpcg_alloc()`/`lobpcg_free()`/`ilobpcg_alloc()` as inline X-macros in lobpcg.h
- [x] Fix lobpcg_alloc: removed duplicate `sizeSub = nev` overwrite, fixed Cx/Cp/wrk buffer sizes for small problems (need `(3*sizeSub)^2` not `3*sizeSub^2`)
- [x] Add parameter validation in lobpcg: `nev <= sizeSub` and `3*sizeSub <= size`
- [ ] Implement `lobpcg_setup()` - init params, allocate X,W,P,S,Cx,Cp
- [ ] Support optional `cache_products` flag for AX,AW,AP
- [ ] (Future) Portability: Add aligned_alloc fallbacks
  - posix_memalign (POSIX systems)
  - _aligned_malloc + _aligned_free (MSVC)
  - CMake detection logic for available functions
- [ ] (Future) Custom debug mode for allocation tracking
  - Rationale: valgrind/ASan better for safety, but custom tracking useful for:
    * LOBPCG-specific context (e.g., "X matrix: 1.2GB")
    * Live memory statistics during execution
    * Zero overhead when disabled (vs valgrind 10-50x, ASan 2x)
  - Implementation: linked list registry, track ptr/size/file/line
  - API: xmalloc_debug, xcalloc_debug, safe_free_debug, memory_report()

**Verify:** No memory leaks with valgrind

### Initialization & Support
- [x] Implement `reset_eigenvectors()` - fill X with Gaussian random
- [x] ~~Implement `setup_S()`~~ — removed; S layout [X,P,W] makes it unnecessary
- [x] Implement `project_back()` - X = S*Cx, P = S*Cp
- [x] Implement `estimate_norm()` - power iteration for ||A||
- [x] Extract `fill_random` and `estimate_norm` to standalone `src/residual/estimate_norm_impl.inc` + instantiation files
- [x] Remove static duplicates from `lobpcg_impl.inc` and `ilobpcg_impl.inc`
- [x] Add `DECLARE_FILL_RANDOM` / `DECLARE_ESTIMATE_NORM` + `_Generic` macros to `lobpcg.h`
- [x] Create `tests/test_estimate_norm.c` (real 3x3 + complex 3x3, both pass)

### Residual Computation
- [x] Create `src/residual/residual_impl.inc`
- [x] Implement `R = A*X - B*X*diag(lambda)`
- [x] Implement relative norm computation: ||R_i|| / (λ_i * ||A||)
- [x] Add complex eigvec test, non-eigvec tests (real/complex), B-operator tests, analytical residual norm tests to `test_residual.c` (11 tests, all pass)

**Verify:** `test_residual.c` - 11/11 tests pass ✓ PASSED
**Verify:** `test_estimate_norm.c` - 2/2 tests pass ✓ PASSED

### Preconditioner
- [x] Implement `apply_precond()` - call preconditioner LinearOperator
- [x] Handle NULL preconditioner (no-op)

### LOBPCG Main Loop
- [x] Create `src/core/lobpcg_impl.inc`
- [x] Implement initialization (random X, initial Rayleigh-Ritz)
- [x] Implement main iteration loop
- [x] Fix S pointer layout: [X,W,P] → [X,P,W], removed setup_S
- [x] Fix ortho_drop iter check: `if(1==iter)` → `if(0==iter)`
- [ ] Implement soft-locking (skip converged columns)
- [x] Implement convergence checking
- [ ] Support `cache_products` option for implicit product update
- [x] Create instantiation files

**Verify:** `test_lobpcg.c` - 5 tests: d/z 4x4 dense (nev=1), d 6x6 dense (nev=1,2), Laplacian n=100 nev=3 sizeSub=5 ✓ PASSED

### K-based LOBPCG (klobpcg)
- [ ] Create `src/core/klobpcg_impl.inc`
- [ ] Simpler variant using krayleigh_ritz (no B operator)
- [ ] Create instantiation files

**Verify:** `test_klobpcg.c` - K-matrix eigenvalue problem

### Indefinite LOBPCG (ilobpcg)
- [x] Create `src/core/ilobpcg_impl.inc`
- [x] Use indefinite Rayleigh-Ritz with GGEV
- [x] Implement signature-aware convergence
- [x] Handle orthogonalization in indefinite metric
- [x] Use ortho_indefinite for B-orthogonalization
- [x] Create instantiation files

**Verify:** `test_ilobpcg.c` - diagonal indefinite test problem

---

## Phase 5: Physics Layer (Separate Library)

### BdG Operators
- [ ] Create `physics/CMakeLists.txt`
- [ ] Create `physics/include/lobpcg_bdg/bdg.h`
- [ ] Implement matmulK (kinetic + trap + interactions)
- [ ] Implement matmulM (kinetic + 3*interactions + dipolar)
- [ ] Implement preconditioner

**Verify:** Matrix-vector products match reference `~/LREP_post`

---

## Refactoring

### Gram matrix helpers
- [x] Create `src/gram/gram_impl.inc` with `apply_block_op`, `gram_self`, `gram_cross`
- [x] Create instantiation files `gram_{s,d,c,z}.c`
- [x] Add declarations + `_Generic` macros to `lobpcg.h`
- [x] ~~Add `GRAM_SRC` to Makefile~~ (obsolete: tests link against `liblobpcg.a`)
- [x] Create `tests/test_gram.c` (11 tests, all pass)
- [x] Replace Gram patterns in: svqb, ortho_drop, ortho_randomize, ortho_indefinite, rayleigh_ritz, rayleigh_ritz_modified, indefinite_rr, indefinite_rr_modified, residual, lobpcg, ilobpcg
- [x] Fix pre-existing wrk3 buffer underallocation in test_ortho_indefinite (was `coef_size`, needs `wrk_size`)
- [ ] (Skipped) svqb_mat, ortho_drop_mat — only 2 files, left as-is

**Verify:** All tests pass after replacements ✓

---

## Phase 6: Documentation & Examples

- [x] Create CLAUDE.md
- [x] Create `examples/Makefile`
- [ ] Create `examples/simple_laplacian.c`
- [ ] Create `examples/custom_operator.c`
- [ ] Add API documentation in headers

---

## Verification Summary

| Component | Test | Tolerance (double) | Tolerance (float) | Status |
|-----------|------|-------------------|-------------------|--------|
| BLAS wrappers | GEMM correctness | 1e-14 | 1e-6 | ✓ |
| SVQB | `\|\|U^H*B*U - I\|\|_F` | 1e-14 | 1e-6 | ✓ |
| ortho_drop | `\|\|V^H*B*U\|\|_F` | 1e-14 | 1e-6 | ✓ |
| svqb_mat | `\|\|U^H*mat*U - I\|\|_F` | 1e-14 | 1e-6 | ✓ |
| ortho_drop_mat | `\|\|V^H*mat*U\|\|_F` | 1e-14 | 1e-6 | ✓ |
| ortho_indefinite | ortho & norm | 1e-14 | 1e-6 | ✓ |
| Rayleigh-Ritz | Eigenvalue error | 1e-12 | 1e-5 | ✓ |
| Residual | Exact eigenvector | 1e-14 | 1e-6 | - |
| LOBPCG | Laplacian eigenvalues | 1e-4 | 1e-4 | - |
