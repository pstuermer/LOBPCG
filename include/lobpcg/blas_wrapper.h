/**
 * @file blas_wrapper.h
 * @brief Type-generic BLAS/LAPACK wrappers for LOBPCG
 *
 * Simplified interface assuming:
 *   - Contiguous storage (inc = 1)
 *   - Packed matrices (ld = number of rows)
 *
 * Naming convention: {prefix}_{operation}[_{variant}]
 */

#ifndef BLAS_WRAPPER_H
#define BLAS_WRAPPER_H

#include "types.h"

/* --------------------------------------------------------------------
 * Backend selection
 * ------------------------------------------------------------------ */
#if defined(USE_MKL)
  #include <mkl.h>
#elif defined(USE_OPENBLAS)
  #include <cblas.h>
  #include <lapacke.h>
#elif defined(USE_BLIS)
  #include <blis.h>
  #include <lapacke.h>
#else
  /* Default: assume OpenBLAS-style interface */
  #include <cblas.h>
  #include <lapacke.h>
#endif

/* ====================================================================
 * BLAS Level 1: Vector operations
 * ==================================================================== */

/* --------------------------------------------------------------------
 * nrm2: ||x||_2
 * ------------------------------------------------------------------ */
static inline f32 s_nrm2(uint64_t n, const f32 *x) {
    return cblas_snrm2((int)n, x, 1);
}

static inline f64 d_nrm2(uint64_t n, const f64 *x) {
    return cblas_dnrm2((int)n, x, 1);
}

static inline f32 c_nrm2(uint64_t n, const c32 *x) {
    return cblas_scnrm2((int)n, x, 1);
}

static inline f64 z_nrm2(uint64_t n, const c64 *x) {
    return cblas_dznrm2((int)n, x, 1);
}

/* --------------------------------------------------------------------
 * axpy: y = alpha*x + y
 * ------------------------------------------------------------------ */
static inline void s_axpy(uint64_t n, f32 alpha, const f32 *x, f32 *y) {
    cblas_saxpy((int)n, alpha, x, 1, y, 1);
}

static inline void d_axpy(uint64_t n, f64 alpha, const f64 *x, f64 *y) {
    cblas_daxpy((int)n, alpha, x, 1, y, 1);
}

static inline void c_axpy(uint64_t n, c32 alpha, const c32 *x, c32 *y) {
    cblas_caxpy((int)n, &alpha, x, 1, y, 1);
}

static inline void z_axpy(uint64_t n, c64 alpha, const c64 *x, c64 *y) {
    cblas_zaxpy((int)n, &alpha, x, 1, y, 1);
}

/* --------------------------------------------------------------------
 * scal: x = alpha*x
 * ------------------------------------------------------------------ */
static inline void s_scal(uint64_t n, f32 alpha, f32 *x) {
    cblas_sscal((int)n, alpha, x, 1);
}

static inline void d_scal(uint64_t n, f64 alpha, f64 *x) {
    cblas_dscal((int)n, alpha, x, 1);
}

static inline void c_scal(uint64_t n, c32 alpha, c32 *x) {
    cblas_cscal((int)n, &alpha, x, 1);
}

static inline void z_scal(uint64_t n, c64 alpha, c64 *x) {
    cblas_zscal((int)n, &alpha, x, 1);
}

/* --------------------------------------------------------------------
 * copy: y = x
 * ------------------------------------------------------------------ */
static inline void s_copy(uint64_t n, const f32 *x, f32 *y) {
    cblas_scopy((int)n, x, 1, y, 1);
}

static inline void d_copy(uint64_t n, const f64 *x, f64 *y) {
    cblas_dcopy((int)n, x, 1, y, 1);
}

static inline void c_copy(uint64_t n, const c32 *x, c32 *y) {
    cblas_ccopy((int)n, x, 1, y, 1);
}

static inline void z_copy(uint64_t n, const c64 *x, c64 *y) {
    cblas_zcopy((int)n, x, 1, y, 1);
}

/* --------------------------------------------------------------------
 * dot/dotc: Inner product
 *   Real: x^T * y
 *   Complex: x^H * y (conjugate)
 * ------------------------------------------------------------------ */
static inline f32 s_dot(uint64_t n, const f32 *x, const f32 *y) {
    return cblas_sdot((int)n, x, 1, y, 1);
}

static inline f64 d_dot(uint64_t n, const f64 *x, const f64 *y) {
    return cblas_ddot((int)n, x, 1, y, 1);
}

static inline c32 c_dotc(uint64_t n, const c32 *x, const c32 *y) {
    c32 result;
    cblas_cdotc_sub((int)n, x, 1, y, 1, &result);
    return result;
}

static inline c64 z_dotc(uint64_t n, const c64 *x, const c64 *y) {
    c64 result;
    cblas_zdotc_sub((int)n, x, 1, y, 1, &result);
    return result;
}

/* ====================================================================
 * BLAS Level 3: Matrix-matrix operations
 * ==================================================================== */

/* --------------------------------------------------------------------
 * gemm: C = alpha*op(A)*op(B) + beta*C
 *
 * Parameters (column-major, packed storage):
 *   nrows_c: rows of C (and rows of op(A))
 *   ncols_c: cols of C (and cols of op(B))
 *   ninner:  inner dimension (cols of op(A) = rows of op(B))
 *
 * Variants:
 *   _nn: C = A * B
 *   _nt: C = A * B^T  (real)
 *   _tn: C = A^T * B  (real)
 *   _nh: C = A * B^H  (complex)
 *   _hn: C = A^H * B  (complex)
 * ------------------------------------------------------------------ */

/* s_gemm variants */
static inline void s_gemm_nn(uint64_t nrows_c, uint64_t ncols_c, uint64_t ninner,
                             f32 alpha, const f32 *A, const f32 *B,
                             f32 beta, f32 *C) {
    /* A: nrows_c x ninner, B: ninner x ncols_c, C: nrows_c x ncols_c */
    cblas_sgemm(CblasColMajor, CblasNoTrans, CblasNoTrans,
                (int)nrows_c, (int)ncols_c, (int)ninner,
                alpha, A, (int)nrows_c, B, (int)ninner, beta, C, (int)nrows_c);
}

static inline void s_gemm_nt(uint64_t nrows_c, uint64_t ncols_c, uint64_t ninner,
                             f32 alpha, const f32 *A, const f32 *B,
                             f32 beta, f32 *C) {
    /* A: nrows_c x ninner, B: ncols_c x ninner (B^T: ninner x ncols_c) */
    cblas_sgemm(CblasColMajor, CblasNoTrans, CblasTrans,
                (int)nrows_c, (int)ncols_c, (int)ninner,
                alpha, A, (int)nrows_c, B, (int)ncols_c, beta, C, (int)nrows_c);
}

static inline void s_gemm_tn(uint64_t nrows_c, uint64_t ncols_c, uint64_t ninner,
                             f32 alpha, const f32 *A, const f32 *B,
                             f32 beta, f32 *C) {
    /* A: ninner x nrows_c (A^T: nrows_c x ninner), B: ninner x ncols_c */
    cblas_sgemm(CblasColMajor, CblasTrans, CblasNoTrans,
                (int)nrows_c, (int)ncols_c, (int)ninner,
                alpha, A, (int)ninner, B, (int)ninner, beta, C, (int)nrows_c);
}


/* d_gemm variants */
static inline void d_gemm_nn(uint64_t nrows_c, uint64_t ncols_c, uint64_t ninner,
                             f64 alpha, const f64 *A, const f64 *B,
                             f64 beta, f64 *C) {
    cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans,
                (int)nrows_c, (int)ncols_c, (int)ninner,
                alpha, A, (int)nrows_c, B, (int)ninner, beta, C, (int)nrows_c);
}

static inline void d_gemm_nt(uint64_t nrows_c, uint64_t ncols_c, uint64_t ninner,
                             f64 alpha, const f64 *A, const f64 *B,
                             f64 beta, f64 *C) {
    cblas_dgemm(CblasColMajor, CblasNoTrans, CblasTrans,
                (int)nrows_c, (int)ncols_c, (int)ninner,
                alpha, A, (int)nrows_c, B, (int)ncols_c, beta, C, (int)nrows_c);
}

static inline void d_gemm_tn(uint64_t nrows_c, uint64_t ncols_c, uint64_t ninner,
                             f64 alpha, const f64 *A, const f64 *B,
                             f64 beta, f64 *C) {
    cblas_dgemm(CblasColMajor, CblasTrans, CblasNoTrans,
                (int)nrows_c, (int)ncols_c, (int)ninner,
                alpha, A, (int)ninner, B, (int)ninner, beta, C, (int)nrows_c);
}


/* c_gemm variants */
static inline void c_gemm_nn(uint64_t nrows_c, uint64_t ncols_c, uint64_t ninner,
                             c32 alpha, const c32 *A, const c32 *B,
                             c32 beta, c32 *C) {
    cblas_cgemm(CblasColMajor, CblasNoTrans, CblasNoTrans,
                (int)nrows_c, (int)ncols_c, (int)ninner,
                &alpha, A, (int)nrows_c, B, (int)ninner, &beta, C, (int)nrows_c);
}

static inline void c_gemm_nh(uint64_t nrows_c, uint64_t ncols_c, uint64_t ninner,
                             c32 alpha, const c32 *A, const c32 *B,
                             c32 beta, c32 *C) {
    /* A: nrows_c x ninner, B: ncols_c x ninner (B^H: ninner x ncols_c) */
    cblas_cgemm(CblasColMajor, CblasNoTrans, CblasConjTrans,
                (int)nrows_c, (int)ncols_c, (int)ninner,
                &alpha, A, (int)nrows_c, B, (int)ncols_c, &beta, C, (int)nrows_c);
}

static inline void c_gemm_hn(uint64_t nrows_c, uint64_t ncols_c, uint64_t ninner,
                             c32 alpha, const c32 *A, const c32 *B,
                             c32 beta, c32 *C) {
    /* A: ninner x nrows_c (A^H: nrows_c x ninner), B: ninner x ncols_c */
    cblas_cgemm(CblasColMajor, CblasConjTrans, CblasNoTrans,
                (int)nrows_c, (int)ncols_c, (int)ninner,
                &alpha, A, (int)ninner, B, (int)ninner, &beta, C, (int)nrows_c);
}

/* z_gemm variants */
static inline void z_gemm_nn(uint64_t nrows_c, uint64_t ncols_c, uint64_t ninner,
                             c64 alpha, const c64 *A, const c64 *B,
                             c64 beta, c64 *C) {
    cblas_zgemm(CblasColMajor, CblasNoTrans, CblasNoTrans,
                (int)nrows_c, (int)ncols_c, (int)ninner,
                &alpha, A, (int)nrows_c, B, (int)ninner, &beta, C, (int)nrows_c);
}

static inline void z_gemm_nh(uint64_t nrows_c, uint64_t ncols_c, uint64_t ninner,
                             c64 alpha, const c64 *A, const c64 *B,
                             c64 beta, c64 *C) {
    cblas_zgemm(CblasColMajor, CblasNoTrans, CblasConjTrans,
                (int)nrows_c, (int)ncols_c, (int)ninner,
                &alpha, A, (int)nrows_c, B, (int)ncols_c, &beta, C, (int)nrows_c);
}

static inline void z_gemm_hn(uint64_t nrows_c, uint64_t ncols_c, uint64_t ninner,
                             c64 alpha, const c64 *A, const c64 *B,
                             c64 beta, c64 *C) {
    cblas_zgemm(CblasColMajor, CblasConjTrans, CblasNoTrans,
                (int)nrows_c, (int)ncols_c, (int)ninner,
                &alpha, A, (int)ninner, B, (int)ninner, &beta, C, (int)nrows_c);
}

/* --------------------------------------------------------------------
 * syrk/herk: Gram matrix C = alpha*A^T*A + beta*C (real)
 *                        C = alpha*A^H*A + beta*C (complex)
 *
 * Parameters:
 *   nrows_a: rows of A
 *   ncols_a: cols of A (= rows and cols of C)
 *
 * Result: C is ncols_a x ncols_a (upper triangle filled)
 * ------------------------------------------------------------------ */
static inline void s_syrk(uint64_t nrows_a, uint64_t ncols_a,
                          f32 alpha, const f32 *A, f32 beta, f32 *C) {
    cblas_ssyrk(CblasColMajor, CblasUpper, CblasTrans,
                (int)ncols_a, (int)nrows_a, alpha, A, (int)nrows_a, beta, C, (int)ncols_a);
}

static inline void d_syrk(uint64_t nrows_a, uint64_t ncols_a,
                          f64 alpha, const f64 *A, f64 beta, f64 *C) {
    cblas_dsyrk(CblasColMajor, CblasUpper, CblasTrans,
                (int)ncols_a, (int)nrows_a, alpha, A, (int)nrows_a, beta, C, (int)ncols_a);
}

static inline void c_herk(uint64_t nrows_a, uint64_t ncols_a,
                          f32 alpha, const c32 *A, f32 beta, c32 *C) {
    cblas_cherk(CblasColMajor, CblasUpper, CblasConjTrans,
                (int)ncols_a, (int)nrows_a, alpha, A, (int)nrows_a, beta, C, (int)ncols_a);
}

static inline void z_herk(uint64_t nrows_a, uint64_t ncols_a,
                          f64 alpha, const c64 *A, f64 beta, c64 *C) {
    cblas_zherk(CblasColMajor, CblasUpper, CblasConjTrans,
                (int)ncols_a, (int)nrows_a, alpha, A, (int)nrows_a, beta, C, (int)ncols_a);
}

/* --------------------------------------------------------------------
 * trsm: Triangular solve variants
 *
 * Parameters:
 *   n:    system size (triangular matrix is n x n)
 *   nrhs: number of right-hand sides (B is n x nrhs for left, m x n for right)
 *
 * Variants:
 *   _lln: L * X = B      (Left, Lower, NoTrans)
 *   _llt: L^T * X = B    (Left, Lower, Trans, real)
 *   _llh: L^H * X = B    (Left, Lower, ConjTrans, complex)
 *   _run: X * R = B      (Right, Upper, NoTrans)
 *         m: rows of B, n: cols of B (= size of R)
 * ------------------------------------------------------------------ */
static inline void s_trsm_lln(uint64_t n, uint64_t nrhs, f32 alpha,
                              const f32 *L, f32 *B) {
    cblas_strsm(CblasColMajor, CblasLeft, CblasLower, CblasNoTrans, CblasNonUnit,
                (int)n, (int)nrhs, alpha, L, (int)n, B, (int)n);
}

static inline void d_trsm_lln(uint64_t n, uint64_t nrhs, f64 alpha,
                              const f64 *L, f64 *B) {
    cblas_dtrsm(CblasColMajor, CblasLeft, CblasLower, CblasNoTrans, CblasNonUnit,
                (int)n, (int)nrhs, alpha, L, (int)n, B, (int)n);
}

static inline void c_trsm_lln(uint64_t n, uint64_t nrhs, c32 alpha,
                              const c32 *L, c32 *B) {
    cblas_ctrsm(CblasColMajor, CblasLeft, CblasLower, CblasNoTrans, CblasNonUnit,
                (int)n, (int)nrhs, &alpha, L, (int)n, B, (int)n);
}

static inline void z_trsm_lln(uint64_t n, uint64_t nrhs, c64 alpha,
                              const c64 *L, c64 *B) {
    cblas_ztrsm(CblasColMajor, CblasLeft, CblasLower, CblasNoTrans, CblasNonUnit,
                (int)n, (int)nrhs, &alpha, L, (int)n, B, (int)n);
}

static inline void s_trsm_llt(uint64_t n, uint64_t nrhs, f32 alpha,
                              const f32 *L, f32 *B) {
    cblas_strsm(CblasColMajor, CblasLeft, CblasLower, CblasTrans, CblasNonUnit,
                (int)n, (int)nrhs, alpha, L, (int)n, B, (int)n);
}

static inline void d_trsm_llt(uint64_t n, uint64_t nrhs, f64 alpha,
                              const f64 *L, f64 *B) {
    cblas_dtrsm(CblasColMajor, CblasLeft, CblasLower, CblasTrans, CblasNonUnit,
                (int)n, (int)nrhs, alpha, L, (int)n, B, (int)n);
}

static inline void c_trsm_llh(uint64_t n, uint64_t nrhs, c32 alpha,
                              const c32 *L, c32 *B) {
    cblas_ctrsm(CblasColMajor, CblasLeft, CblasLower, CblasConjTrans, CblasNonUnit,
                (int)n, (int)nrhs, &alpha, L, (int)n, B, (int)n);
}

static inline void z_trsm_llh(uint64_t n, uint64_t nrhs, c64 alpha,
                              const c64 *L, c64 *B) {
    cblas_ztrsm(CblasColMajor, CblasLeft, CblasLower, CblasConjTrans, CblasNonUnit,
                (int)n, (int)nrhs, &alpha, L, (int)n, B, (int)n);
}

/* trsm_run: solve X * R = alpha * B  (Right, Upper, NoTrans)
 *   m: rows of B, n: cols of B (= order of R) */
static inline void s_trsm_run(uint64_t m, uint64_t n, f32 alpha,
                               const f32 *R, f32 *B) {
    cblas_strsm(CblasColMajor, CblasRight, CblasUpper, CblasNoTrans, CblasNonUnit,
                (int)m, (int)n, alpha, R, (int)n, B, (int)m);
}

static inline void d_trsm_run(uint64_t m, uint64_t n, f64 alpha,
                               const f64 *R, f64 *B) {
    cblas_dtrsm(CblasColMajor, CblasRight, CblasUpper, CblasNoTrans, CblasNonUnit,
                (int)m, (int)n, alpha, R, (int)n, B, (int)m);
}

static inline void c_trsm_run(uint64_t m, uint64_t n, c32 alpha,
                               const c32 *R, c32 *B) {
    cblas_ctrsm(CblasColMajor, CblasRight, CblasUpper, CblasNoTrans, CblasNonUnit,
                (int)m, (int)n, &alpha, R, (int)n, B, (int)m);
}

static inline void z_trsm_run(uint64_t m, uint64_t n, c64 alpha,
                               const c64 *R, c64 *B) {
    cblas_ztrsm(CblasColMajor, CblasRight, CblasUpper, CblasNoTrans, CblasNonUnit,
                (int)m, (int)n, &alpha, R, (int)n, B, (int)m);
}

/* ====================================================================
 * LAPACK: Linear algebra decompositions and solvers
 * ==================================================================== */

/* --------------------------------------------------------------------
 * potrf: Cholesky A = R^H * R (upper triangular)
 *   n: matrix dimension (A is n x n)
 * ------------------------------------------------------------------ */
static inline int s_potrf(uint64_t n, f32 *A) {
    return LAPACKE_spotrf(LAPACK_COL_MAJOR, 'U', (int)n, A, (int)n);
}

static inline int d_potrf(uint64_t n, f64 *A) {
    return LAPACKE_dpotrf(LAPACK_COL_MAJOR, 'U', (int)n, A, (int)n);
}

static inline int c_potrf(uint64_t n, c32 *A) {
    return LAPACKE_cpotrf(LAPACK_COL_MAJOR, 'U', (int)n, A, (int)n);
}

static inline int z_potrf(uint64_t n, c64 *A) {
    return LAPACKE_zpotrf(LAPACK_COL_MAJOR, 'U', (int)n, A, (int)n);
}

/* --------------------------------------------------------------------
 * syev/heev: Hermitian eigenvalue A*V = V*diag(w)
 *   n: matrix dimension
 *   A: overwritten with eigenvectors
 *   w: eigenvalues (ascending order)
 * ------------------------------------------------------------------ */
static inline int s_syev(uint64_t n, f32 *A, f32 *w) {
    return LAPACKE_ssyev(LAPACK_COL_MAJOR, 'V', 'U', (int)n, A, (int)n, w);
}

static inline int d_syev(uint64_t n, f64 *A, f64 *w) {
    return LAPACKE_dsyev(LAPACK_COL_MAJOR, 'V', 'U', (int)n, A, (int)n, w);
}

static inline int c_heev(uint64_t n, c32 *A, f32 *w) {
    return LAPACKE_cheev(LAPACK_COL_MAJOR, 'V', 'U', (int)n, A, (int)n, w);
}

static inline int z_heev(uint64_t n, c64 *A, f64 *w) {
    return LAPACKE_zheev(LAPACK_COL_MAJOR, 'V', 'U', (int)n, A, (int)n, w);
}

/* Alias eig functions for use with FN() macro */
static inline int s_eig(uint64_t n, f32 *A, f32 *w) {
    return s_syev(n, A, w);
}

static inline int d_eig(uint64_t n, f64 *A, f64 *w) {
    return d_syev(n, A, w);
}

static inline int c_eig(uint64_t n, c32 *A, f32 *w) {
    return c_heev(n, A, w);
}

static inline int z_eig(uint64_t n, c64 *A, f64 *w) {
    return z_heev(n, A, w);
}

/* --------------------------------------------------------------------
 * geev: General eigenvalue A*VR = VR*diag(w)
 *   n:  matrix dimension
 *   A:  destroyed on output
 *   VR: right eigenvectors (n x n)
 *
 * Real: eigenvalues as (wr, wi) pairs
 * Complex: eigenvalues as single array w
 * ------------------------------------------------------------------ */
static inline int s_geev(uint64_t n, f32 *A, f32 *wr, f32 *wi, f32 *VR) {
    return LAPACKE_sgeev(LAPACK_COL_MAJOR, 'N', 'V', (int)n, A, (int)n,
                         wr, wi, NULL, 1, VR, (int)n);
}

static inline int d_geev(uint64_t n, f64 *A, f64 *wr, f64 *wi, f64 *VR) {
    return LAPACKE_dgeev(LAPACK_COL_MAJOR, 'N', 'V', (int)n, A, (int)n,
                         wr, wi, NULL, 1, VR, (int)n);
}

static inline int c_geev(uint64_t n, c32 *A, c32 *w, c32 *VR) {
    return LAPACKE_cgeev(LAPACK_COL_MAJOR, 'N', 'V', (int)n, A, (int)n,
                         w, NULL, 1, VR, (int)n);
}

static inline int z_geev(uint64_t n, c64 *A, c64 *w, c64 *VR) {
    return LAPACKE_zgeev(LAPACK_COL_MAJOR, 'N', 'V', (int)n, A, (int)n,
                         w, NULL, 1, VR, (int)n);
}

/* --------------------------------------------------------------------
 * ggev: Generalized eigenvalue A*VR = B*VR*diag(alpha/beta)
 *   n:  matrix dimension
 *   A, B: destroyed on output
 *   VR: right eigenvectors (n x n)
 *
 * Real: alpha as (alphar, alphai) pairs
 * Complex: alpha as single array
 * ------------------------------------------------------------------ */
static inline int s_ggev(uint64_t n, f32 *A, f32 *B,
                         f32 *alphar, f32 *alphai, f32 *beta, f32 *VR) {
    return LAPACKE_sggev(LAPACK_COL_MAJOR, 'N', 'V', (int)n,
                         A, (int)n, B, (int)n,
                         alphar, alphai, beta,
                         NULL, 1, VR, (int)n);
}

static inline int d_ggev(uint64_t n, f64 *A, f64 *B,
                         f64 *alphar, f64 *alphai, f64 *beta, f64 *VR) {
    return LAPACKE_dggev(LAPACK_COL_MAJOR, 'N', 'V', (int)n,
                         A, (int)n, B, (int)n,
                         alphar, alphai, beta,
                         NULL, 1, VR, (int)n);
}

static inline int c_ggev(uint64_t n, c32 *A, c32 *B,
                         c32 *alpha, c32 *beta, c32 *VR) {
    return LAPACKE_cggev(LAPACK_COL_MAJOR, 'N', 'V', (int)n,
                         A, (int)n, B, (int)n,
                         alpha, beta,
                         NULL, 1, VR, (int)n);
}

static inline int z_ggev(uint64_t n, c64 *A, c64 *B,
                         c64 *alpha, c64 *beta, c64 *VR) {
    return LAPACKE_zggev(LAPACK_COL_MAJOR, 'N', 'V', (int)n,
                         A, (int)n, B, (int)n,
                         alpha, beta,
                         NULL, 1, VR, (int)n);
}

/* --------------------------------------------------------------------
 * trcon: Reciprocal condition number of triangular matrix
 *   norm: '1' or 'I'
 *   n: matrix dimension
 * ------------------------------------------------------------------ */
static inline int s_trcon(char norm, uint64_t n, const f32 *A, f32 *rcond) {
    return LAPACKE_strcon(LAPACK_COL_MAJOR, norm, 'U', 'N', (int)n, A, (int)n, rcond);
}

static inline int d_trcon(char norm, uint64_t n, const f64 *A, f64 *rcond) {
    return LAPACKE_dtrcon(LAPACK_COL_MAJOR, norm, 'U', 'N', (int)n, A, (int)n, rcond);
}

static inline int c_trcon(char norm, uint64_t n, const c32 *A, f32 *rcond) {
    return LAPACKE_ctrcon(LAPACK_COL_MAJOR, norm, 'U', 'N', (int)n, A, (int)n, rcond);
}

static inline int z_trcon(char norm, uint64_t n, const c64 *A, f64 *rcond) {
    return LAPACKE_ztrcon(LAPACK_COL_MAJOR, norm, 'U', 'N', (int)n, A, (int)n, rcond);
}

/* ====================================================================
 * Type-generic macros (using _Generic)
 * ==================================================================== */

#define nrm2(n, x) _Generic((x), \
    const f32*: s_nrm2, f32*: s_nrm2, \
    const f64*: d_nrm2, f64*: d_nrm2, \
    const c32*: c_nrm2, c32*: c_nrm2, \
    const c64*: z_nrm2, c64*: z_nrm2)(n, x)

#define axpy(n, alpha, x, y) _Generic((x), \
    const f32*: s_axpy, f32*: s_axpy, \
    const f64*: d_axpy, f64*: d_axpy, \
    const c32*: c_axpy, c32*: c_axpy, \
    const c64*: z_axpy, c64*: z_axpy)(n, alpha, x, y)

#define scal(n, alpha, x) _Generic((x), \
    f32*: s_scal, \
    f64*: d_scal, \
    c32*: c_scal, \
    c64*: z_scal)(n, alpha, x)

#define copy(n, x, y) _Generic((x), \
    const f32*: s_copy, f32*: s_copy, \
    const f64*: d_copy, f64*: d_copy, \
    const c32*: c_copy, c32*: c_copy, \
    const c64*: z_copy, c64*: z_copy)(n, x, y)

#define dot(n, x, y) _Generic((x), \
    const f32*: s_dot, f32*: s_dot, \
    const f64*: d_dot, f64*: d_dot, \
    const c32*: c_dotc, c32*: c_dotc, \
    const c64*: z_dotc, c64*: z_dotc)(n, x, y)

#define gram(nrows, ncols, alpha, A, beta, C) _Generic((A), \
    const f32*: s_syrk, f32*: s_syrk, \
    const f64*: d_syrk, f64*: d_syrk, \
    const c32*: c_herk, c32*: c_herk, \
    const c64*: z_herk, c64*: z_herk)(nrows, ncols, alpha, A, beta, C)

#define potrf(n, A) _Generic((A), \
    f32*: s_potrf, \
    f64*: d_potrf, \
    c32*: c_potrf, \
    c64*: z_potrf)(n, A)

#define eig(n, A, w) _Generic((A), \
    f32*: s_syev, \
    f64*: d_syev, \
    c32*: c_heev, \
    c64*: z_heev)(n, A, w)

/* -------------------------------------------------------------------
 * QR factorization (geqrf + orgqr/ungqr)
 * ------------------------------------------------------------------ */

/* geqrf - QR factorization via Householder reflectors */
static inline int s_geqrf(uint64_t m, uint64_t n, f32 *A, f32 *tau) {
    return LAPACKE_sgeqrf(LAPACK_COL_MAJOR, (lapack_int)m, (lapack_int)n, A, (lapack_int)m, tau);
}

static inline int d_geqrf(uint64_t m, uint64_t n, f64 *A, f64 *tau) {
    return LAPACKE_dgeqrf(LAPACK_COL_MAJOR, (lapack_int)m, (lapack_int)n, A, (lapack_int)m, tau);
}

static inline int c_geqrf(uint64_t m, uint64_t n, c32 *A, c32 *tau) {
    return LAPACKE_cgeqrf(LAPACK_COL_MAJOR, (lapack_int)m, (lapack_int)n, (lapack_complex_float*)A,
                          (lapack_int)m, (lapack_complex_float*)tau);
}

static inline int z_geqrf(uint64_t m, uint64_t n, c64 *A, c64 *tau) {
    return LAPACKE_zgeqrf(LAPACK_COL_MAJOR, (lapack_int)m, (lapack_int)n, (lapack_complex_double*)A,
                          (lapack_int)m, (lapack_complex_double*)tau);
}

/* orgqr/ungqr - Generate Q from Householder reflectors */
static inline int s_orgqr(uint64_t m, uint64_t n, uint64_t k, f32 *A, f32 *tau) {
    return LAPACKE_sorgqr(LAPACK_COL_MAJOR, (lapack_int)m, (lapack_int)n, (lapack_int)k,
                          A, (lapack_int)m, tau);
}

static inline int d_orgqr(uint64_t m, uint64_t n, uint64_t k, f64 *A, f64 *tau) {
    return LAPACKE_dorgqr(LAPACK_COL_MAJOR, (lapack_int)m, (lapack_int)n, (lapack_int)k,
                          A, (lapack_int)m, tau);
}

static inline int c_ungqr(uint64_t m, uint64_t n, uint64_t k, c32 *A, c32 *tau) {
    return LAPACKE_cungqr(LAPACK_COL_MAJOR, (lapack_int)m, (lapack_int)n, (lapack_int)k,
                          (lapack_complex_float*)A, (lapack_int)m, (lapack_complex_float*)tau);
}

static inline int z_ungqr(uint64_t m, uint64_t n, uint64_t k, c64 *A, c64 *tau) {
    return LAPACKE_zungqr(LAPACK_COL_MAJOR, (lapack_int)m, (lapack_int)n, (lapack_int)k,
                          (lapack_complex_double*)A, (lapack_int)m, (lapack_complex_double*)tau);
}

#define geqrf(m, n, A, tau) _Generic((A), \
    f32*: s_geqrf, \
    f64*: d_geqrf, \
    c32*: c_geqrf, \
    c64*: z_geqrf)(m, n, A, tau)

#define ungqr(m, n, k, A, tau) _Generic((A), \
    f32*: s_orgqr, \
    f64*: d_orgqr, \
    c32*: c_ungqr, \
    c64*: z_ungqr)(m, n, k, A, tau)

#endif /* BLAS_WRAPPER_H */
