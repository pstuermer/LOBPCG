/**
 * @file indefinite_rr_impl.inc
 * @brief Type-generic indefinite Rayleigh-Ritz procedure (for ilobpcg)
 *
 * Uses GGEV to solve the generalized eigenvalue problem
 * GA * v = lambda * GB * v  where GA = S^H*A*S, GB = S^H*B*S
 * with indefinite B.
 *
 * Macros required: PREFIX, CTYPE, RTYPE, LINOP
 */

#include <string.h>
#include <stdlib.h>
#include <math.h>
#include <stdio.h>
#include "lobpcg/memory.h"

#define CONCAT2(a, b) a##_##b
#define CONCAT(a, b) CONCAT2(a, b)
#define FN(name) CONCAT(PREFIX, name)

#ifdef TYPE_IS_FLOAT
  #ifdef CTYPE_IS_COMPLEX
  #define CABS(x) cabsf(x)
  #define CREAL(x) creal(x)
  #else
  #define CABS(x) fabsf(x)
  #define CREAL(x) (x)
  #endif
#else
  #ifdef CTYPE_IS_COMPLEX
  #define CABS(x) cabs(x)
  #define CREAL(x) creal(x)
  #else
  #define CABS(x) fabs(x)
  #define CREAL(x) (x)
  #endif
#endif

#include "bubble_sort_sig_impl.inc"

/* Indefinite Rayleigh-Ritz procedure
 *
 * Solves the projected generalized eigenvalue problem:
 *   (S^H * A * S) * v = lambda * (S^H * B * S) * v
 *
 * using GGEV. Eigenvalues are sorted by signature (positive ascending,
 * negative descending). Eigenvectors are B-normalized.
 *
 * Inputs:
 *   size:      problem dimension (rows of S)
 *   sizeSub:   number of columns of S
 *   S:         search space (size x sizeSub)
 *   A, B:      operators (B is indefinite, must not be NULL)
 *
 * Outputs:
 *   Cx:        eigenvector coefficients (sizeSub x sizeSub)
 *   eigVal:    eigenvalues (array of CTYPE, length sizeSub)
 *   signature: +1/-1 per eigenpair (length sizeSub)
 *
 * Workspace:
 *   wrk1: sizeSub x sizeSub
 *   wrk2: size x sizeSub
 *   wrk3: sizeSub x sizeSub
 *   wrk4: sizeSub x sizeSub (GGEV destroys both inputs)
 */
void FN(indefinite_rayleigh_ritz)(const uint64_t size, const uint64_t sizeSub,
                                  CTYPE *restrict S, CTYPE *restrict Cx, RTYPE *restrict eigVal,
                                  int8_t *restrict signature,
                                  CTYPE *restrict wrk1, CTYPE *restrict wrk2,
                                  CTYPE *restrict wrk3, CTYPE *restrict wrk4,
                                  uint64_t *restrict rr_indices,
                                  CTYPE *restrict rr_ggev,
                                  LINOP *A, LINOP *B) {

  const uint64_t ss = sizeSub * sizeSub;

  /* Pre-allocated buffers from caller */
  uint64_t *indices = rr_indices;
#ifdef CTYPE_IS_COMPLEX
  CTYPE *alpha = rr_ggev;
  CTYPE *beta  = rr_ggev + sizeSub;
#else
  RTYPE *alphar = rr_ggev;
  RTYPE *alphai = rr_ggev + sizeSub;
  RTYPE *beta   = rr_ggev + 2 * sizeSub;
#endif

  /* 1. GA = S^H * A * S in wrk1 */
  FN(gram_self)(S, size, sizeSub, A, wrk1, sizeSub, wrk2);

  /* 2. GB = S^H * B * S in wrk3 */
  FN(gram_self)(S, size, sizeSub, B, wrk3, sizeSub, wrk2);

  /* 3. Copy GA -> wrk4, GB -> wrk2 (GGEV destroys both) */
  memcpy(wrk4, wrk1, ss * sizeof(CTYPE));
  memcpy(wrk2, wrk3, ss * sizeof(CTYPE));

  /* 4. GGEV: wrk4 (GA), wrk2 (GB) -> eigenvectors in Cx */
#ifdef CTYPE_IS_COMPLEX
  if (0 != FN(ggev)(sizeSub, wrk4, wrk2, alpha, beta, Cx)) {
    fprintf(stderr, "indefinite_rayleigh_ritz: GGEV failed\n");
    goto cleanup;
  }
#else
  if (0 != FN(ggev)(sizeSub, wrk4, wrk2, alphar, alphai, beta, Cx)) {
    fprintf(stderr, "indefinite_rayleigh_ritz: GGEV failed\n");
    goto cleanup;
  }
#endif

  /* 5. Compute eigenvalues: eigVal[i] = Re(alpha[i] / beta[i]) */
  for (uint64_t i = 0; i < sizeSub; i++) {
#ifdef CTYPE_IS_COMPLEX
    if (CABS(beta[i]) > (RTYPE)1e-30)
      eigVal[i] = CREAL(alpha[i] / beta[i]);
    else
      eigVal[i] = (CREAL(alpha[i]) >= 0) ? (RTYPE)1e30 : (RTYPE)-1e30;
#else
    if (CABS(beta[i]) > (RTYPE)1e-30)
      eigVal[i] = alphar[i] / beta[i];
    else
      eigVal[i] = (alphar[i] >= 0) ? (RTYPE)1e30 : (RTYPE)-1e30;
#endif
  }

  /* 6. Normalize eigenvectors by B-inner product
   *    Compute V^H * GB * V using preserved GB in wrk3
   *    wrk1 = GB * V, then wrk4 = V^H * wrk1 */
  FN(gemm_nn)(sizeSub, sizeSub, sizeSub, (CTYPE)1, wrk3, Cx, (CTYPE)0, wrk1);
#ifdef CTYPE_IS_COMPLEX
  FN(gemm_hn)(sizeSub, sizeSub, sizeSub, (CTYPE)1, Cx, wrk1, (CTYPE)0, wrk4);
#else
  FN(gemm_tn)(sizeSub, sizeSub, sizeSub, (CTYPE)1, Cx, wrk1, (CTYPE)0, wrk4);
#endif

  /* Scale each column: v_i /= sqrt(|diag_i|), extract signature */
  for (uint64_t i = 0; i < sizeSub; i++) {
    const CTYPE diag_val = wrk4[i + i * sizeSub];
    const RTYPE abs_diag = CABS(diag_val);

    if (abs_diag > (RTYPE)1e-30) {
      const RTYPE scale = (RTYPE)1 / sqrt(abs_diag);
      for (uint64_t k = 0; k < sizeSub; k++)
        Cx[k + i * sizeSub] *= (CTYPE)scale;
    }

    /* 7. Extract signature from sign of B-inner product diagonal */
    signature[i] = (CREAL(diag_val) >= 0) ? 1 : -1;
  }

  /* 8. Sort: positive ascending, negative descending */
  FN(bubble_sort_sig)(eigVal, signature, indices, sizeSub);

  /* 9. Reorder columns of Cx according to sort permutation.
   *    Use wrk1 as temp to hold sorted columns. */
  memcpy(wrk1, Cx, ss * sizeof(CTYPE));
  for (uint64_t j = 0; j < sizeSub; j++) {
    memcpy(&Cx[j * sizeSub], &wrk1[indices[j] * sizeSub],
           sizeSub * sizeof(CTYPE));
  }

cleanup:
  (void)0;
}

#undef CONCAT2
#undef CONCAT
#undef FN
#undef CABS
#undef CREAL
