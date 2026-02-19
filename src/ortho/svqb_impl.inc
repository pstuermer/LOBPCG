/**
 * @file svqb_impl.inc
 * @brief Type-generic SVQB orthogonalization
 *
 * Reference: Duersch & Shao, "Simultaneous block diagonalization of Gram
 * matrices for orthonormalization", SIAM J. Matrix Anal. Appl. 39(3), 2018
 * (Algorithm 4)
 *
 * Macros required: PREFIX, CTYPE, RTYPE, LINOP
 */

#include <string.h>
#include <stdlib.h>
#include <math.h>
#include "lobpcg/memory.h"

#define CONCAT2(a, b) a##_##b
#define CONCAT(a, b) CONCAT2(a, b)
#define FN(name) CONCAT(PREFIX, name)

#ifndef MAX
#define MAX(a, b) ((a) > (b) ? (a) : (b))
#endif

#ifdef CTYPE_IS_COMPLEX
  #define CABS(x) cabs(x)
#else
  #define CABS(x) fabs(x)
#endif

/**
 * svqb - SVQB orthogonalization via eigendecomposition of scaled Gram matrix
 *
 * Orthonormalizes U w.r.t. B-inner product (B=NULL → standard inner product).
 * Small eigenvalues are clamped to tau*max(|λ|) for numerical stability.
 * Column dropping (drop='y') is not implemented; randomization is handled
 * separately in ortho_randomize.
 *
 * @param rows   Number of rows of U (problem size)
 * @param cols   Number of columns of U (block size)
 * @param tau    Regularization threshold for small eigenvalues
 * @param drop   Drop mode (unused, reserved for 'y' = drop small columns)
 * @param U      [in/out] rows x cols matrix, B-orthonormalized on output
 * @param wrk1   Workspace, cols x cols elements
 * @param wrk2   Workspace, max(rows*cols, cols*cols) elements
 * @param wrk3   Workspace, rows x cols elements
 * @param B      B-operator for generalized inner product, or NULL for identity
 *
 * @return Number of columns retained (always cols, no dropping)
 */
uint64_t FN(svqb)(const uint64_t rows, const uint64_t cols, const RTYPE tau,
                  const char drop, CTYPE *restrict U,
                  CTYPE *restrict wrk1, CTYPE *restrict wrk2, CTYPE *restrict wrk3,
                  LINOP *B) {
  if (0 == cols) return 0;
  (void)drop;  /* drop='y' not implemented */

  RTYPE *eigvals = xcalloc(cols, sizeof(RTYPE));
  RTYPE *D = xcalloc(cols, sizeof(RTYPE));

  /* 1. Gram matrix G = U^H*B*U in wrk1 (upper triangle) */
  memset(wrk1, 0, cols*cols*sizeof(CTYPE));
  FN(gram_self)(U, rows, cols, B, wrk1, cols, wrk2);

  /* 2. Diagonal scaling D_ii = 1/sqrt(|G_ii|) */
  for (uint64_t i = 0; i < cols; i++) {
    RTYPE gii = CABS(wrk1[i + i*cols]);
    D[i] = (gii > 0) ? (RTYPE)1/sqrt(gii) : (RTYPE)1;
  }

  /* 3. Scale upper triangle: G̃ = D*G*D */
  for (uint64_t j = 0; j < cols; j++)
    for (uint64_t i = 0; i <= j; i++)
      wrk1[i + j*cols] *= D[i] * D[j];

  /* 4. Eigensolve G̃ = V Λ V^H (reads upper triangle) */
  if (0 != FN(eig)(cols, wrk1, eigvals)) {
    safe_free((void**)&eigvals); safe_free((void**)&D); return cols;
  }

  /* 5. Threshold */
  RTYPE maxeig = 0;
  for (uint64_t i = 0; i < cols; i++)
    if ((RTYPE)fabs(eigvals[i]) > maxeig) maxeig = (RTYPE)fabs(eigvals[i]);
  RTYPE thresh = tau * maxeig;

  /* 6. Fused transform: T = D * V * D_final in wrk2 (cols x cols)
   *    where D_final_jj = 1/sqrt(max(|λ_j|, thresh)) */
  for (uint64_t j = 0; j < cols; j++) {
    RTYPE df = (RTYPE)1/sqrt(MAX((RTYPE)fabs(eigvals[j]), thresh));
    for (uint64_t i = 0; i < cols; i++)
      wrk2[i + j*cols] = D[i] * wrk1[i + j*cols] * df;
  }

  /* U_new = U * T */
  FN(gemm_nn)(rows, cols, cols, (CTYPE)1, U, wrk2, (CTYPE)0, wrk3);
  memcpy(U, wrk3, rows*cols*sizeof(CTYPE));

  safe_free((void**)&eigvals); safe_free((void**)&D);
  return cols;
}

#undef CONCAT2
#undef CONCAT
#undef FN
#undef MAX
#undef CABS
