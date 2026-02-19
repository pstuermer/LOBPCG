/**
 * @file gram_impl.inc
 * @brief Type-generic Gram matrix helpers
 *
 * Centralizes the repeated patterns of:
 *   1. Applying a block operator column-by-column
 *   2. Forming self Gram G = U^H * B * U (upper triangle)
 *   3. Forming cross Gram G = V^H * B * U
 *
 * Macros required: PREFIX, CTYPE, RTYPE, LINOP
 */

#include <string.h>
#include "lobpcg/memory.h"

#define CONCAT2(a, b) a##_##b
#define CONCAT(a, b) CONCAT2(a, b)
#define FN(name) CONCAT(PREFIX, name)

/**
 * apply_block_op - Apply operator column-by-column: Y[:,j] = Op * X[:,j]
 *
 * @param Op   Linear operator
 * @param X    Input matrix (n x k)
 * @param Y    Output matrix (n x k)
 * @param n    Number of rows
 * @param k    Number of columns
 */
void FN(apply_block_op)(const LINOP *Op, CTYPE *restrict X,
                        CTYPE *restrict Y, const uint64_t n, const uint64_t k) {
  for (uint64_t j = 0; j < k; j++)
    Op->matvec(Op, &X[j * n], &Y[j * n]);
}

/**
 * gram_self - Compute G = U^H * B * U (upper triangle via herk/syrk)
 *
 * If B == NULL, computes G = U^H * U directly via herk/syrk.
 * Otherwise computes wrk = B*U, then G = U^H * wrk via gemm.
 *
 * @param U    Input matrix (n x k)
 * @param n    Number of rows
 * @param k    Number of columns
 * @param B    B-operator (NULL = identity)
 * @param G    Output Gram matrix (k x k, upper triangle filled)
 * @param ldg  Leading dimension of G (must be >= k)
 * @param wrk  Workspace (n x k), unused when B == NULL
 */
void FN(gram_self)(CTYPE *restrict U, const uint64_t n, const uint64_t k,
                   const LINOP *B, CTYPE *restrict G, const uint64_t ldg,
                   CTYPE *restrict wrk) {
  if (NULL == B) {
#ifdef CTYPE_IS_COMPLEX
    FN(herk)(n, k, (RTYPE)1, U, (RTYPE)0, G);
#else
    FN(syrk)(n, k, (RTYPE)1, U, (RTYPE)0, G);
#endif
  } else {
    FN(apply_block_op)(B, U, wrk, n, k);
#ifdef CTYPE_IS_COMPLEX
    FN(gemm_hn)(k, k, n, (CTYPE)1, U, wrk, (CTYPE)0, G);
#else
    FN(gemm_tn)(k, k, n, (CTYPE)1, U, wrk, (CTYPE)0, G);
#endif
  }
  (void)ldg; /* packed storage: ld == k */
}

/**
 * gram_cross - Compute G = V^H * B * U
 *
 * If B == NULL, computes G = V^H * U directly via gemm.
 * Otherwise computes wrk = B*U, then G = V^H * wrk via gemm.
 *
 * @param V    Left matrix (n x nv)
 * @param nv   Number of columns of V
 * @param U    Right matrix (n x nu)
 * @param nu   Number of columns of U
 * @param n    Number of rows
 * @param B    B-operator (NULL = identity)
 * @param G    Output matrix (nv x nu)
 * @param ldg  Leading dimension of G (must be >= nv)
 * @param wrk  Workspace (n x nu), unused when B == NULL
 */
void FN(gram_cross)(CTYPE *restrict V, const uint64_t nv,
                    CTYPE *restrict U, const uint64_t nu,
                    const uint64_t n, const LINOP *B,
                    CTYPE *restrict G, const uint64_t ldg,
                    CTYPE *restrict wrk) {
  if (NULL == B) {
#ifdef CTYPE_IS_COMPLEX
    FN(gemm_hn)(nv, nu, n, (CTYPE)1, V, U, (CTYPE)0, G);
#else
    FN(gemm_tn)(nv, nu, n, (CTYPE)1, V, U, (CTYPE)0, G);
#endif
  } else {
    FN(apply_block_op)(B, U, wrk, n, nu);
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
