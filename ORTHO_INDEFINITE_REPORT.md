# ortho_indefinite — Implementation Status Report

**Date:** 2026-02-23
**File:** `src/ortho/ortho_indefinite_impl.inc` (195 lines)
**Reference:** `~/LREP_post/src/algorithms/ilobpcg.c:185-326` (`zortho_randomize_indefinite`)
**Tests:** `tests/test_ortho_indefinite.c` (514 lines, 6 tests, all pass)

## Purpose

B-orthogonalizes a candidate basis U against an external basis V when B is indefinite (eigenvalues can be +1 or -1). Used during ilobpcg soft-locking to orthogonalize new search directions against already-converged eigenvectors.

The key difference from standard `ortho_drop` is the signature-weighted projection:
```
U = U - V * sig * (V^H * B * U)
```
where `sig = V^H * B * V` is a diagonal matrix with entries in {+1, -1}.

## Algorithm

```
1. If sig not provided: compute sig = V^H * B * V (via gram_self)
2. Compute ||B*V||_F for relative error normalization
3. For outer = 0..2:
   a. coef = V^H * B * U            (gram_cross)
   b. tmp  = sig * coef             (gemm_nn)
   c. U    = U - V * tmp            (gemm_nn + axpy loop)
   d. For inner = 0..2:
      i.  SVQB(U)                   (B-orthonormalize)
      ii. Check ||U^H*B*U - I||_F   (gram_self + ortho_err_upper)
      iii. Break if < eps_ortho
   e. Check ||V^H*B*U||_F / (||B*V||_F * ||U||_F) < eps_ortho
      Break if converged
4. Free sig if locally allocated
```

## Differences from Reference

| Aspect | Reference (`zortho_randomize_indefinite`) | New (`ortho_indefinite`) |
|--------|-------------------------------------------|--------------------------|
| Layout | Row-major (transposes everywhere) | Column-major (no transposes) |
| Types | `double complex` only | Type-generic (s/d/c/z) |
| B operator | `externalBasis` function pointer + raw `matmul_ctx_t` | `LinearOperator` interface |
| Inner check | Reuses svqb's output Gram in `work3` | Computes fresh `gram_self` + `ortho_err_upper` |
| Inner metric | `svqb_err / (U_norm * U_norm)` | `||U^H*B*U - I||_F` directly |
| BV_norm | Computed lazily on iteration 1 | Computed once before loop |
| Outer check | Skipped on iteration 0 | Checked every iteration |
| Overdetermined guard | `fprintf(stderr, ...)` + return | Silent return 0 |
| Column dropping | Not implemented | Not implemented (returns 0) |
| Sig handling | Always copies into local buffer | Uses pointer directly if provided, allocates only if NULL |
| Debug prints | `printf` of errors each iteration | None (silent) |

### Inner Check Detail

The reference reuses the Gram matrix that `zsvqb` leaves in `work3` after orthogonalization. Our `svqb` doesn't preserve this (wrk3 is used as scratch). Instead of modifying svqb's interface, we recompute `U^H*B*U` via `gram_self`. This costs one extra syrk/herk call per inner iteration, but:
- `n_u` is small (typically 3-15), so the `O(m * n_u^2)` syrk is negligible vs the `O(m * n_u^2)` work svqb already does
- Keeps svqb's interface clean (no optional output parameter)
- The `ortho_err_upper` helper avoids filling the lower triangle and computes the Frobenius norm directly from the upper triangle

### ortho_err_upper

Computes `||G - I||_F` for Hermitian G using only upper triangle entries:
```
||G - I||_F^2 = sum_i |G_ii - 1|^2 + 2 * sum_{i<j} |G_ij|^2
```
Replaces the previous 3-step approach (fill_lower + subtract identity + full matrix norm).

## API

```c
uint64_t PREFIX_ortho_indefinite(
    const uint64_t m,        // problem size (rows)
    const uint64_t n_u,      // columns of U
    const uint64_t n_v,      // columns of V
    const RTYPE eps_ortho,   // orthogonality tolerance
    const RTYPE eps_drop,    // drop tolerance for svqb
    CTYPE *restrict U,       // [in/out] candidate basis
    CTYPE *restrict V,       // [in] external basis (already B-ortho)
    CTYPE *restrict sig,     // [in] signature matrix (NULL = compute internally)
    CTYPE *restrict wrk1,    // workspace: m * max(n_u, n_v)
    CTYPE *restrict wrk2,    // workspace: m * max(n_u, n_v)
    CTYPE *restrict wrk3,    // workspace: max(n_v*n_u, n_v*n_v, n_u*n_u)
    LINOP *B);               // B operator (NULL = identity)
// Returns: number of columns dropped (always 0 currently)
```

## Test Coverage

| Test | Type | m | n_u | n_v | B | sig | Checks |
|------|------|--:|----:|----:|---|-----|--------|
| `d_ortho_indefinite_basic` | f64 | 100 | 8 | 5 | diag(+1...-1), 60 pos | provided | cross < 8e-9, norm < 8e-9 |
| `d_ortho_indefinite_no_sig` | f64 | 80 | 6 | 4 | diag(+1...-1), 50 pos | NULL | cross < 6e-9, norm < 6e-9 |
| `d_ortho_indefinite_no_B` | f64 | 80 | 6 | 4 | NULL (identity) | NULL | cross < 6e-9, norm < 6e-9 |
| `z_ortho_indefinite_basic` | c64 | 100 | 8 | 5 | diag(+1...-1), 60 pos | NULL | cross < 8e-9, norm < 8e-9 |
| `z_ortho_indefinite_larger` | c64 | 500 | 15 | 10 | diag(+1...-1), 300 pos | NULL | cross < 1.5e-8, norm < 1.5e-8 |
| `z_ortho_indefinite_no_B` | c64 | 80 | 6 | 4 | NULL (identity) | NULL | cross < 6e-9, norm < 6e-9 |

Typical achieved errors are O(1e-14) for double, well within the `TOL_F64 * n_u` bounds.

The B-orthonormality check in the tests verifies the indefinite condition properly: diagonal entries should be +/-1 (not just +1), and off-diagonal entries should be zero. This is handled by `B_norm_error_d/z` which checks `max(||off-diag||_F, max_i(||G_ii| - 1|))`.

## Known Limitations

1. **No column dropping** — always returns 0. The reference also doesn't implement this. When a column becomes linearly dependent during orthogonalization, svqb handles it via eigenvalue thresholding, but the column count isn't reduced.

2. **No s/c type tests** — only d and z are tested. The instantiation files exist (`ortho_indefinite_s.c`, `ortho_indefinite_c.c`) and compile, but no single-precision tests. This is acceptable since ilobpcg only uses d/z in practice.

3. **Inner check cost** — one extra `gram_self` per inner iteration vs the reference. Negligible for small n_u but could be eliminated by having svqb output its post-ortho Gram.

4. **Outer iteration 0** — we compute the outer convergence check (`||V^H*B*U||_F`) on the first iteration, while the reference skips it. This adds one extra `gram_cross` but catches already-orthogonal inputs immediately.

## Integration

Used by:
- `src/core/ilobpcg_impl.inc` — called during the main loop to orthogonalize the search direction against locked eigenvectors

Depends on:
- `gram_self`, `gram_cross`, `apply_block_op` (from `src/gram/gram_impl.inc`)
- `svqb` (from `src/ortho/svqb_impl.inc`)
- `gemm_nn` (from `include/lobpcg/blas_wrapper.h`)
