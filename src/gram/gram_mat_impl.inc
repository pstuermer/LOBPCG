/**
 * @file gram_mat_impl.inc
 * @brief Type-generic Gram matrix helpers for dense metric matrices
 *
 * Like gram_impl.inc, but takes an explicit dense matrix instead of
 * a LinearOperator. Used in ilobpcg where the indefinite metric is
 * stored as a dense matrix.
 *
 * Macros required: PREFIX, CTYPE, RTYPE
 */

#include <string.h>
#include "lobpcg/memory.h"

#define CONCAT2(a, b) a##_##b
#define CONCAT(a, b) CONCAT2(a, b)
#define FN(name) CONCAT(PREFIX, name)

/**
 * gram_self_mat - Compute G = U^H * mat * U (upper triangle)
 *
 * If mat == NULL, computes G = U^H * U directly via herk/syrk.
 * Otherwise computes wrk = mat*U, then G = U^H * wrk via gemm.
 *
 * @param U    Input matrix (n x k)
 * @param n    Number of rows
 * @param k    Number of columns
 * @param mat  Dense metric matrix (n x n), or NULL for identity
 * @param G    Output Gram matrix (k x k, upper triangle filled)
 * @param ldg  Leading dimension of G (must be >= k)
 * @param wrk  Workspace (n x k), unused when mat == NULL
 */
void FN(gram_self_mat)(CTYPE *restrict U, const uint64_t n, const uint64_t k,
                       const CTYPE *mat, CTYPE *restrict G, const uint64_t ldg,
                       CTYPE *restrict wrk) {
  if (NULL == mat) {
#ifdef CTYPE_IS_COMPLEX
    FN(herk)(n, k, (RTYPE)1, U, (RTYPE)0, G);
#else
    FN(syrk)(n, k, (RTYPE)1, U, (RTYPE)0, G);
#endif
  } else {
    FN(gemm_nn)(n, k, n, (CTYPE)1, (CTYPE *)mat, U, (CTYPE)0, wrk);
#ifdef CTYPE_IS_COMPLEX
    FN(gemm_hn)(k, k, n, (CTYPE)1, U, wrk, (CTYPE)0, G);
#else
    FN(gemm_tn)(k, k, n, (CTYPE)1, U, wrk, (CTYPE)0, G);
#endif
  }
  (void)ldg; /* packed storage: ld == k */
}

/**
 * gram_cross_mat - Compute G = V^H * mat * U
 *
 * If mat == NULL, computes G = V^H * U directly via gemm.
 * Otherwise computes wrk = mat*U, then G = V^H * wrk via gemm.
 *
 * @param V    Left matrix (n x nv)
 * @param nv   Number of columns of V
 * @param U    Right matrix (n x nu)
 * @param nu   Number of columns of U
 * @param n    Number of rows
 * @param mat  Dense metric matrix (n x n), or NULL for identity
 * @param G    Output matrix (nv x nu)
 * @param ldg  Leading dimension of G (must be >= nv)
 * @param wrk  Workspace (n x nu), unused when mat == NULL
 */
void FN(gram_cross_mat)(CTYPE *restrict V, const uint64_t nv,
                        CTYPE *restrict U, const uint64_t nu,
                        const uint64_t n, const CTYPE *mat,
                        CTYPE *restrict G, const uint64_t ldg,
                        CTYPE *restrict wrk) {
  if (NULL == mat) {
#ifdef CTYPE_IS_COMPLEX
    FN(gemm_hn)(nv, nu, n, (CTYPE)1, V, U, (CTYPE)0, G);
#else
    FN(gemm_tn)(nv, nu, n, (CTYPE)1, V, U, (CTYPE)0, G);
#endif
  } else {
    FN(gemm_nn)(n, nu, n, (CTYPE)1, (CTYPE *)mat, U, (CTYPE)0, wrk);
#ifdef CTYPE_IS_COMPLEX
    FN(gemm_hn)(nv, nu, n, (CTYPE)1, V, wrk, (CTYPE)0, G);
#else
    FN(gemm_tn)(nv, nu, n, (CTYPE)1, V, wrk, (CTYPE)0, G);
#endif
  }
  (void)ldg; /* packed storage: ld == nv */
}

#undef CONCAT2
#undef CONCAT
#undef FN
