# LOBPCG Codebase Review

## Grade A — Critical Bugs (will crash or produce wrong results)

### A1. `alg->AS` and `alg->BS` used but never allocated — NULL dereference
**`lobpcg.h:578-579`, `ilobpcg_impl.inc:159,172,219`**

The allocations in `ilobpcg_alloc` are **commented out**:
```c
/*    alg->AS        = xcalloc(3*n*sizeSub, sizeof(ctype));
alg->BS        = xcalloc(3*sizeSub*sizeSub, sizeof(ctype));*/
```
But `ilobpcg_impl.inc` writes to `alg->AS` at line 159 (`memcpy(S_asm, X, ...)` where `S_asm = alg->AS = NULL`) and passes `alg->BS` as `Cx_ortho` at line 172 and uses it in a GEMM at line 219. This is a guaranteed segfault on the first iteration of any `ilobpcg` call.

### A2. Complex GGEV eigenvalue — unchecked division by `beta`
**`indefinite_rr_impl.inc:104`, `indefinite_rr_modified_impl.inc:121`**

The complex path does `eigVal[i] = CREAL(alpha[i] / beta[i])` with no check for `beta[i] == 0`. The real path correctly guards with `if (CABS(beta[i]) > 1e-30)` and assigns `+/-1e30` for the degenerate case. The complex path will produce NaN/Inf for infinite generalized eigenvalues.

---

## Grade B — High Severity (memory leaks, broken API, wrong guard)

### B1. `lobpcg_free` does not free `AX`
**`lobpcg.h:542-558`**

`ilobpcg_alloc` allocates `alg->AX = xcalloc(n*sizeSub, sizeof(ctype))` at line 580, but tkhe `DEFINE_LOBPCG_FREE` macro never calls `safe_free((void**)&(*alg)->AX)`. Every ilobpcg run leaks this buffer.

### B2. `ilobpcg_alloc` generic macro drops `sizeSub` argument
**`lobpcg.h:588`**

```c
#define ilobpcg_alloc(n, nev, prefix) prefix##_ilobpcg_alloc(n, nev)
```
The function takes 3 parameters `(n, nev, sizeSub)` but the macro forwards only 2. Any use of the generic macro would fail to compile. Tests bypass it by calling `d_ilobpcg_alloc(...)` directly.

### B3. `creal()` used instead of `CREAL()` macro on `RTYPE` values
**`ilobpcg_impl.inc:124,212`**

`alg->eigVals` is `RTYPE*` (real). Calling `creal()` on a `float` or `double` is a type mismatch that produces warnings/errors with strict compilers. The `CREAL()` macro exists and handles this correctly.

### B4. Wrong guard for `AX` computation: checks `AS` instead of `AX`
**`ilobpcg_impl.inc:190`**

```c
if (alg->AS)   // <-- should be alg->AX
    FN(apply_block_op)(alg->A, X, alg->AX, size, sizeSub);
```
Since `AS` is NULL (bug A1), the `AX` computation is **never executed** after iteration 0. The residual then uses a stale `AX` that doesn't correspond to the current `X`. (Downstream of A1 — if A1 is fixed by restoring the AS allocation, this bug becomes independently active.)

### B5. Test `RUN` macro prints `[PASS]` even after assertion failure
**`test_blas.c:21-26` (and all 11+ test files)**

```c
#define RUN(name) do { \
    printf("  %-40s ", #name); \
    test_##name(); \
    printf("[PASS]\n"); \  // <-- always reached after return
    tests_passed++; \
} while(0)
```
If `ASSERT` fails inside the test function, it does an early `return`, but the macro unconditionally prints `[PASS]` and increments `tests_passed`. Failed tests appear as passed in the output.

--------------------------------------------

## Grade C — Medium Severity (incorrect convergence checks, latent UB, potential overflow)

### C1. `ortho_drop` Frobenius norm reads half-filled Gram matrix
**`ortho_drop_impl.inc:87-99`**

When `B==NULL`, `gram_self` uses `syrk`/`herk` which only fills the **upper triangle**. The code then computes `nrm2(n_u * n_u, wrk2)` over the entire buffer. The lower triangle contains stale data, making the orthonormality convergence check unreliable. `ortho_indefinite_impl.inc` correctly uses `ortho_err_upper()` for this — the same should be done here.

### C2. `ortho_drop_mat` workspace docstring: `wrk3` size is wrong
**`ortho_drop_mat_impl.inc:45`**

Documentation says `wrk3 >= max(n_v * n_u, n_u * n_u)`. But `svqb_mat` (called at line 118) uses `wrk3` for a `gemm_nn` result of size `m * n_u`. Since `m >> n_u`, a caller following the docstring would have a buffer overflow. The tests happen to allocate `m * max_n` which is correct.

### C3. `safe_free` strict aliasing violation
**`memory.h:91-96`**

Every call site casts `T**` to `void**`: `safe_free((void**)&some_typed_ptr)`. Writing `NULL` through `void**` to a location that holds a `T*` is a strict aliasing violation under C11 6.5p7. Works universally in practice, but is technically UB. A `SAFE_FREE(ptr)` macro would avoid this.

### C4. `_Generic` in `lobpcg_free` evaluates `*alg` at runtime
**`lobpcg.h:563`**

```c
#define lobpcg_free(alg) _Generic((*alg), ...)(alg)
```
The controlling expression `*alg` is evaluated even though `_Generic` only needs the type at compile time. If `alg` is NULL, this is UB before the function's null check can run.

### C5. `linop_apply` `_Generic` missing `const` pointer variants
**`linop.h:69-74`**

The macro only matches `LinearOperator_s_t*` etc. (non-const). Passing a `const LinearOperator_s_t*` will fail to compile. Need to add `const` variants.

### C6. Two incompatible `linop_ctx_t` usage patterns across tests
**Multiple test files**

Pattern A (direct cast): `ctx = (linop_ctx_t*)my_ctx` — type-punning, strict aliasing UB.
Pattern B (data indirection): `linop_ctx.data = my_data` — correct usage.
Both patterns coexist across the test suite. Pattern A is technically undefined behavior.

### C7. `matvec` function signature mismatch in tests
**`test_lobpcg.c:72-73`, `test_estimate_norm.c:22,39`**

Test matvec functions declare `const f64 *x` but the `matvec_func_d_t` typedef has `f64 *restrict x`. The cast at assignment silences warnings but is technically UB (calling through incompatible function pointer type).

### C8. Rayleigh-Ritz functions return `void` — no error propagation
**`rayleigh_ritz_impl.inc`, `indefinite_rr_impl.inc`, `indefinite_rr_modified_impl.inc`**

When Cholesky, eigensolve, or GGEV fails, the function prints to stderr and returns early. The caller has no way to detect failure — `eigVal` and `Cx` may be uninitialized or partially filled.

### C9. `uint64_t` to `int` truncation in all BLAS wrappers
**`blas_wrapper.h` (throughout)**

Every wrapper casts `uint64_t` parameters to `(int)` for CBLAS/LAPACK calls. If `n > INT_MAX`, this silently truncates, causing incorrect BLAS calls.

---

## Grade D — Low Severity (wrong tolerances, missing coverage, minor inconsistencies)

### D1. Hardcoded double-precision tolerances for all types
**`lobpcg_impl.inc:77-78`, `ilobpcg_impl.inc:63-64`**

```c
const RTYPE eps_ortho = 1e-12;
const RTYPE eps_drop = 1e-12;
```
For `f32` (machine epsilon ~1.2e-7), these are below machine precision and will never be achievable. Should scale with type.

### D2. `x_norm < 1e-14` threshold not precision-dependent
**`lobpcg_impl.inc:93`, `ilobpcg_impl.inc:87`**

For `f32`, no computation can produce a norm this small. For `f64`, it's reasonable.

### D3. `printf` format `%ld` for `uint64_t`
**`lobpcg_impl.inc:113,201-205`, `ilobpcg_impl.inc:208-212`**

Technically UB. Should use `PRIu64` or `%lu` with `(unsigned long)` cast.

### D4. Inconsistent `LINOP` definitions (`struct` prefix vs typedef)
**Various `src/**/*.c` files**

Some use `#define LINOP struct LinearOperator_d_t`, others use `#define LINOP LinearOperator_d_t`. Both work but inconsistent.

### D5. Inconsistent include path conventions
**`lobpcg.h:4-5`, various `.c` files**

Mix of `"types.h"`, `"lobpcg/types.h"`, and `"../../include/lobpcg/types.h"` throughout.

### D6. SVQB division by zero for all-zero input
**`svqb_impl.inc:85-90`, `svqb_mat_impl.inc:93-98`**

When all Gram eigenvalues are zero, `1/sqrt(MAX(0, 0))` = inf. Should guard with early return.

### D7. `gram_self` ignores `ldg` parameter silently
**`gram_impl.inc:66`**

`(void)ldg;` — accepted but unused. Passing `ldg != k` would silently produce wrong results.

### D8. No test coverage for key paths
**Various test files**

Missing tests for: preconditioner (`T != NULL`), generalized eigenproblem (`B != NULL`) in LOBPCG integration tests, single precision (`s_lobpcg`, `c_lobpcg`), SVQB column dropping (rank-deficient input), `AX`-precomputed path in `get_residual`.

### D9. `linop_test.c` never asserts correctness
**`linop_test.c:57-73`**

The identity operator test prints `x` and `I*x` but never checks `y[i] == x[i]`. The test always passes.

### D10. BLIS backend documented but not implemented
**`Makefile:5`**

Comment says `BLAS_BACKEND ?= MKL, OPENBLAS, or BLIS` but there's no `ifeq ($(BLAS_BACKEND),BLIS)` block. Setting `BLIS` silently falls through with no BLAS flags.

### D11. `svqb_mat` error message hardcodes `LAPACKE_zheev` for all types
**`svqb_mat_impl.inc:85`**

Misleading for s/d/c instantiations.

---

## Grade E — Cleanup / Cosmetic

### E1. Emacs backup/autosave files tracked in git
`tests/#test_residual.c#`, `tests/#test_rayleigh_ritz.c#`, `tests/test_memory.c~`, `tests/test_svqb.c~`, `src/rayleigh/#indefinite_rr_modified_impl.inc#`, `src/ortho/#ortho_drop_impl.inc#`

### E2. Stale `.o` files with no source
`src/ortho/ortho_randomize_{s,d,c,z}.o`, `src/ortho/ortho_randomized_mat_{s,d,c,z}.o`

### E3. `CTYPE_IS_COMPLEX` never `#undef`'d in complex instantiation files
Works because each `.c` is a separate translation unit, but poor hygiene.

### E4. `data_size` field in `linop_ctx_t` is dead
Set in some tests, never read by library code.

### E5. Generic include guard names (`TYPES_H`, `LINOP_H`, `BLAS_WRAPPER_H`)
Should be `LOBPCG_TYPES_H` etc. to avoid collisions.

### E6. BLAS wrapper macros pollute global namespace
`nrm2`, `axpy`, `scal`, `copy`, `dot`, `gram`, `potrf`, `eig`, `geqrf`, `ungqr` — all short, generic names in a header. Should be prefixed `lobpcg_`.

### E7. `cabs()` used for single-precision complex instead of `cabsf()`
All `.inc` files define `#define CABS(x) cabs(x)`. For `c32`, this promotes to double unnecessarily.

### E8. `linop_test.c` uses `free()` instead of `safe_free()`
Violates project convention. Also `ctx = NULL;` on a local is a no-op.

### E9. Commented-out dead code in various places
`lobpcg.h:578-579`, `rayleigh_ritz_impl.inc:86-92`, backup `.svqb_impl.inc.full`.

---

## Grade F — Curiosities / Design observations

### F1. `lobpcg.h` lives at root, not `include/lobpcg/` as CLAUDE.md says
Works via `-I.` in Makefile but contradicts documented structure.

### F2. No `cmake/` directory or `CMakeLists.txt` despite documentation
Build system is Makefile-only. CLAUDE.md mentions CMake modules that don't exist.

### F3. `Makefile` uses `src/**/*.c` wildcard
GNU Make's `wildcard` doesn't support `**` for recursive glob. Works by coincidence with one level of nesting.

### F4. `gram_self` fills upper-only (B==NULL) vs full (B!=NULL)
Callers must know this. Not buggy per se, but a subtle API asymmetry that has already caused C1.

### F5. Standard LOBPCG doesn't use AX caching (TODO in code)
`lobpcg_impl.inc:110` acknowledges this — always passes `NULL` for AX, forcing recomputation.

---

## Summary

| Grade | Count | Key Themes |
|-------|-------|-----------|
| **A** | 2 | NULL deref in ilobpcg, unchecked complex division |
| **B** | 5 | Memory leak (AX), broken macro, wrong guard, test harness lies |
| **C** | 9 | Half-filled Gram check, strict aliasing UB, no error propagation |
| **D** | 11 | Wrong tolerances for f32, missing test coverage, dead API |
| **E** | 9 | Stale files, namespace pollution, cosmetic |
| **F** | 5 | Documentation drift, design asymmetries |
