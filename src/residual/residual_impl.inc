/**
 * @file residual_impl.inc
 * @brief Type-generic residual computation functions
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

#ifdef CTYPE_IS_COMPLEX
  #define CABS(x) cabs(x)
  #define CREAL(x) creal(x)
#else
  #define CABS(x) fabs(x)
  #define CREAL(x) (x)
#endif
#define RABS(x) ((x) >= 0 ? (x) : -(x))

/* Compute residual R = A*X - B*X*eigVal
 *
 * Inputs:
 *   X: eigenvector approximations (size x sizeSub)
 *   AX: cached A*X (if NULL, will compute)
 *   eigVal: eigenvalues (sizeSub, stored as diagonal or vector)
 *   A, B: linear operators
 *
 * Output:
 *   R: residual (size x sizeSub)
 *
 * Uses wrk as workspace (size x sizeSub)
 */
void FN(get_residual)(const uint64_t size, const uint64_t sizeSub,
                      CTYPE *restrict X, CTYPE *restrict AX, CTYPE *restrict R,
                      RTYPE *restrict eigVal, CTYPE *restrict wrk,
                      LINOP *A, LINOP *B) {

  /* 1. Compute R = A*X or copy from AX */
  if (!AX) {
    for (uint64_t j = 0; j < sizeSub; j++)
      A->matvec(A, &X[j*size], &R[j*size]);
  } else memcpy(R, AX, size * sizeSub * sizeof(CTYPE));

  /* 2. Compute R = R - B*X*eigVal */
  if (B) {
    /* wrk = B*X */
    for (uint64_t j = 0; j < sizeSub; j++)
      B->matvec(B, &X[j*size], &wrk[j*size]);
    /* R[:,j] = R[:,j] - eigVal[j] * wrk[:,j] */
    for (uint64_t j = 0; j < sizeSub; j++) {
      const CTYPE lambda = eigVal[j];
      FN(axpy)(size, -lambda, &wrk[j*size], &R[j*size]);
    }
  } else {
    /* R[:,j] = R[:,j] - eigVal[j] * X[:,j] */
    for (uint64_t j = 0; j < sizeSub; j++) {
      const CTYPE lambda = eigVal[j];
      FN(axpy)(size, -lambda, &X[j*size], &R[j*size]);
    }
  }
}

/* Compute relative residual norms
 *
 * Formula: resNorm[i] = ||W[:,i]||_B / (ANorm + |eigVal[i]| * BNorm)
 *
 * Inputs:
 *   W: residual vectors (size x sizeSub)
 *   eigVals: eigenvalue estimates (sizeSub)
 *   ANorm, BNorm: operator norms
 *   B: mass matrix operator (or NULL)
 *
 * Output:
 *   resNorm: relative residual norms (nev)
 *
 * Uses wrk1, wrk2, wrk3 as workspace
 */
void FN(get_residual_norm)(const uint64_t size, const uint64_t sizeSub,
                           const uint64_t nev, CTYPE *restrict W,
                           RTYPE *restrict eigVals, RTYPE *restrict resNorm,
                           CTYPE *restrict wrk1, CTYPE *restrict wrk2, CTYPE *restrict wrk3,
                           const RTYPE ANorm, const RTYPE BNorm, LINOP *B) {

  (void)wrk2;  /* Unused for now */
  (void)wrk3;

  for (uint64_t i = 0; i < nev; i++) {
    RTYPE nom;

    if (B) {
      /* BW_i = B * W[:,i] in wrk1 */
      B->matvec(B, &W[i*size], wrk1);
      /* nom = sqrt(|W[:,i]^H * BW_i|) */
#ifdef CTYPE_IS_COMPLEX
      const CTYPE inner = FN(dotc)(size, &W[i*size], wrk1);
#else
      const CTYPE inner = FN(dot)(size, &W[i*size], wrk1);
#endif
      nom = sqrt(CABS(inner));
    } else {
      /* nom = ||W[:,i]||_2 */
      nom = FN(nrm2)(size, &W[i*size]);
    }

    /* denom = ANorm + |eigVal[i]| * BNorm */
    const RTYPE lambda_abs = RABS(eigVals[i]);
    const RTYPE denom = ANorm + lambda_abs * (BNorm > 0 ? BNorm : (RTYPE)1);

    resNorm[i] = nom / denom;
  }
}

#undef CONCAT2
#undef CONCAT
#undef FN
#undef CABS
#undef CREAL
#undef RABS
