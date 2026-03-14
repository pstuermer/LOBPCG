/**
 * @file indefinite_rr_modified_impl.inc
 * @brief Type-generic modified indefinite Rayleigh-Ritz with Cx/Cp extraction
 *
 * Same as indefinite_rr but for the iterative LOBPCG loop:
 * - S has mult*nx columns (mult=2 for [X,W], mult=3 for [X,P,W])
 * - Extracts Cx (first nx eigenvectors) and Cp = [0; Z2] (direct lower block)
 * - Cx_ortho is B-orthogonalized copy of Cx when GGEV quality is poor
 * - Returns RTYPE eigenvalues (real part)
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

#include "lobpcg/type_dispatch.h"

#ifndef MIN
#define MIN(a, b) ((a) < (b) ? (a) : (b))
#endif
/*
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
*/
#include "bubble_sort_sig_impl.inc"

/* Modified indefinite Rayleigh-Ritz
 *
 * Inputs:
 *   size:      problem dimension
 *   nx:        block size (nev)
 *   mult:      block multiplier (2 or 3)
 *   nconv:     converged count (unused, for soft-locking)
 *   nretain:   number of retained W columns after ortho_drop
 *   S:         search space (size x sizeSub)
 *   A, B:      operators (B indefinite, must not be NULL)
 *
 * Outputs:
 *   Cx:        coefficients for X update (sizeSub x nx)
 *   Cp:        coefficients for P update (sizeSub x nx): top nx rows = 0,
 *              rows [nx:sizeSub] copied from Cx lower block
 *   Cx_ortho:  B-orthogonalized Cx (sizeSub x nx); equals Cx if quality good
 *   eigVal:    first nx eigenvalues (RTYPE, real part)
 *   signature: +1/-1 per eigenpair (length sizeSub, sorted)
 *   quality_flag: 1 = good GGEV quality, 5 = poor (svqb_mat applied to Cx_ortho)
 *
 * Workspace:
 *   wrk1: sizeSub x sizeSub
 *   wrk2: size x sizeSub
 *   wrk3: sizeSub x sizeSub
 *   wrk4: sizeSub x sizeSub
 */
void FN(indefinite_rayleigh_ritz_modified)(const uint64_t size, const uint64_t nx,
                                           const uint64_t mult, const uint64_t nconv,
                                           const uint64_t nretain, CTYPE *restrict S,
                                           const CTYPE *restrict AX,
                                           CTYPE *restrict wrk1, CTYPE *restrict wrk2,
                                           CTYPE *restrict wrk3, CTYPE *restrict wrk4,
                                           CTYPE *Cx, CTYPE *restrict Cp,
                                           CTYPE *Cx_ortho,
                                           RTYPE *restrict eigVal,
                                           int8_t *restrict signature,
                                           int *restrict quality_flag,
                                           RTYPE *restrict rr_eigvals,
                                           int8_t *restrict rr_sig,
                                           uint64_t *restrict rr_indices,
                                           CTYPE *restrict rr_VR,
                                           CTYPE *restrict rr_ggev,
                                           LINOP *A, LINOP *B) {

  const uint64_t sizeSub = (mult - 1) * nx + nretain - (3 == mult ? nconv : 0);
  const uint64_t n_remainder = sizeSub - nx;
  const uint64_t ss = sizeSub * sizeSub;

  /* Pre-allocated buffers from caller */
  RTYPE *eigvals_all = rr_eigvals;
  int8_t *sig_all = rr_sig;
  uint64_t *indices = rr_indices;
  CTYPE *VR = rr_VR;
#ifdef CTYPE_IS_COMPLEX
  CTYPE *alpha = rr_ggev;
  CTYPE *beta  = rr_ggev + sizeSub;
#else
  RTYPE *alphar = rr_ggev;
  RTYPE *alphai = rr_ggev + sizeSub;
  RTYPE *beta   = rr_ggev + 2 * sizeSub;
#endif

  /* 1. GA = S^H * A * S in wrk1 */
  if (AX) {
    FN(apply_block_op)(A, &S[size*nx], &wrk2[size*nx], size, n_remainder);
    memcpy(wrk2, AX, size * nx * sizeof(CTYPE));
#ifdef CTYPE_IS_COMPLEX
    FN(gemm_hn)(sizeSub, sizeSub, size, (CTYPE)1, S, wrk2, (CTYPE)0, wrk1);
#else
    FN(gemm_tn)(sizeSub, sizeSub, size, (CTYPE)1, S, wrk2, (CTYPE)0, wrk1);
#endif
  } else
    FN(gram_self)(S, size, sizeSub, A, wrk1, sizeSub, wrk2);

  /* 2. GB = S^H * B * S in wrk3 */
  FN(gram_self)(S, size, sizeSub, B, wrk3, sizeSub, wrk2);

  /* 3. Copy GA -> wrk4, GB -> wrk2 (GGEV destroys both) */
  memcpy(wrk4, wrk1, ss * sizeof(CTYPE));
  memcpy(wrk2, wrk3, ss * sizeof(CTYPE));

  /* 4. GGEV */
#ifdef CTYPE_IS_COMPLEX
  if (0 != FN(ggev)(sizeSub, wrk4, wrk2, alpha, beta, VR)) {
    fprintf(stderr, "indefinite_rayleigh_ritz_modified: GGEV failed\n");
    goto cleanup;
  }
#else
  if (0 != FN(ggev)(sizeSub, wrk4, wrk2, alphar, alphai, beta, VR)) {
    fprintf(stderr, "indefinite_rayleigh_ritz_modified: GGEV failed\n");
    goto cleanup;
  }
#endif

  /* 5. Compute eigenvalues */
  for (uint64_t i = 0; i < sizeSub; i++) {
#ifdef CTYPE_IS_COMPLEX
    if (CABS(beta[i]) > (RTYPE)1e-30)
      eigvals_all[i] = CREAL(alpha[i] / beta[i]);
    else
      eigvals_all[i] = (CREAL(alpha[i]) >= 0) ? (RTYPE)1e30 : (RTYPE)-1e30;
#else
    if (CABS(beta[i]) > (RTYPE)1e-30)
      eigvals_all[i] = alphar[i] / beta[i];
    else
      eigvals_all[i] = (alphar[i] >= 0) ? (RTYPE)1e30 : (RTYPE)-1e30;
#endif
  }

  /* 6a. B-normalize eigenvectors (first pass): V^H * GB * V
   *     wrk3 still holds GB, VR holds eigenvectors */
  FN(gemm_nn)(sizeSub, sizeSub, sizeSub, (CTYPE)1, wrk3, VR, (CTYPE)0, wrk1);
#ifdef CTYPE_IS_COMPLEX
  FN(gemm_hn)(sizeSub, sizeSub, sizeSub, (CTYPE)1, VR, wrk1, (CTYPE)0, wrk4);
#else
  FN(gemm_tn)(sizeSub, sizeSub, sizeSub, (CTYPE)1, VR, wrk1, (CTYPE)0, wrk4);
#endif

  for (uint64_t i = 0; i < sizeSub; i++) {
    const RTYPE abs_diag = CABS(wrk4[i + i * sizeSub]);
    if (abs_diag > (RTYPE)1e-30) {
      const RTYPE scale = (RTYPE)1 / sqrt(abs_diag);
      for (uint64_t k = 0; k < sizeSub; k++)
        VR[k + i * sizeSub] *= (CTYPE)scale;
    }
  }

  /* 6b. B-normalize eigenvectors (second pass) + extract signatures
   *     Double normalization improves B-orthogonality (ref ilobpcg.c:608-622)
   *     TODO: measure whether this reduces quality=5 fallback frequency */
  FN(gemm_nn)(sizeSub, sizeSub, sizeSub, (CTYPE)1, wrk3, VR, (CTYPE)0, wrk1);
#ifdef CTYPE_IS_COMPLEX
  FN(gemm_hn)(sizeSub, sizeSub, sizeSub, (CTYPE)1, VR, wrk1, (CTYPE)0, wrk4);
#else
  FN(gemm_tn)(sizeSub, sizeSub, sizeSub, (CTYPE)1, VR, wrk1, (CTYPE)0, wrk4);
#endif



  for (uint64_t i = 0; i < sizeSub; i++) {
    const CTYPE diag_val = wrk4[i + i * sizeSub];
    const RTYPE abs_diag = CABS(diag_val);
    if (abs_diag > (RTYPE)1e-30) {
      const RTYPE scale = (RTYPE)1 / sqrt(abs_diag);
      for (uint64_t k = 0; k < sizeSub; k++)
        VR[k + i * sizeSub] *= (CTYPE)scale;
    }
    sig_all[i] = (CREAL(diag_val) >= 0) ? 1 : -1;
  }
  

  /* Quality check: relative B-orthogonality after double normalization
   * Recompute V^H * GB * V; wrk3 = GB still valid */
  FN(gemm_nn)(sizeSub, sizeSub, sizeSub, (CTYPE)1, wrk3, VR, (CTYPE)0, wrk1);
#ifdef CTYPE_IS_COMPLEX
  FN(gemm_hn)(sizeSub, sizeSub, sizeSub, (CTYPE)1, VR, wrk1, (CTYPE)0, wrk4);
#else
  FN(gemm_tn)(sizeSub, sizeSub, sizeSub, (CTYPE)1, VR, wrk1, (CTYPE)0, wrk4);
#endif
  for (uint64_t i = 0; i < sizeSub; i++)
    wrk4[i + i * sizeSub] = (CTYPE)(CABS(wrk4[i + i * sizeSub]) - (RTYPE)1);
  const RTYPE eerr  = FN(nrm2)(ss, wrk4);
  const RTYPE cerr  = FN(nrm2)(ss, VR);
  const RTYPE Bcerr = FN(nrm2)(ss, wrk1);
  const int quality_ok = (Bcerr < (RTYPE)1e-30 || eerr <= (RTYPE)1e-12 * cerr * Bcerr);

  /* 7. Sort by signature */
  FN(bubble_sort_sig)(eigvals_all, sig_all, indices, sizeSub);

  /* 8. Reorder VR columns according to sort permutation -> wrk1 as temp */
  memcpy(wrk1, VR, ss * sizeof(CTYPE));
  for (uint64_t j = 0; j < sizeSub; j++) {
    memcpy(&VR[j * sizeSub], &wrk1[indices[j] * sizeSub],
           sizeSub * sizeof(CTYPE));
  }

  /* 9. Extract first nx eigenvalues and signature */
  for (uint64_t i = 0; i < nx; i++)
    eigVal[i] = eigvals_all[i];
  for (uint64_t i = 0; i < sizeSub; i++)
    signature[i] = sig_all[i];

  /* 10. Extract Cx = first nx columns of sorted VR */
  memset(Cx, 0, sizeSub * nx * sizeof(CTYPE));
  for (uint64_t j = 0; j < nx; j++) {
    for (uint64_t i = 0; i < sizeSub; i++) {
      Cx[i + j * sizeSub] = VR[i + j * sizeSub];
    }
  }

  /* 11. Extract Cp = [0; Z2]: top nx rows zero, rows [nx:sizeSub] from Cx */
  {
    memset(Cp, 0, sizeSub * nx * sizeof(CTYPE));
    for (uint64_t j = 0; j < nx; j++)
      memcpy(&Cp[nx + j * sizeSub], &Cx[nx + j * sizeSub],
             n_remainder * sizeof(CTYPE));
  }

  /* 12. Cx_ortho + quality_flag + Cp orthogonalization:
   *     Ortho Cp against Cx (matching reference ilobpcg.c:655,696).
   *     If GGEV eigenvectors are well-conditioned, Cx_ortho = Cx.
   *     Otherwise B-orthogonalize Cx in-place via svqb_mat, ortho Cp against
   *     the well-conditioned Cx, save result to Cx_ortho, restore accurate Cx. */
  if (quality_ok) {
    FN(ortho_indefinite_mat)(sizeSub, nx, nx,
                             (RTYPE)EPS_TOL, (RTYPE)EPS_TOL,
                             Cp, Cx, wrk3, wrk1, wrk2, wrk4);
    memcpy(Cx_ortho, Cx, sizeSub * nx * sizeof(CTYPE));
    *quality_flag = 1;
  } else {
    memcpy(VR, Cx, sizeSub * nx * sizeof(CTYPE));           /* save accurate Cx */
    FN(svqb_mat)(sizeSub, nx, (RTYPE)EPS_TOL, 'n',
                 Cx, wrk3, wrk1, wrk2, wrk4);               /* ortho Cx in-place */
    FN(ortho_indefinite_mat)(sizeSub, nx, nx,
                             (RTYPE)EPS_TOL, (RTYPE)EPS_TOL,
                             Cp, Cx, wrk3, wrk1, wrk2, wrk4);
    memcpy(Cx_ortho, Cx, sizeSub * nx * sizeof(CTYPE));     /* result -> Cx_ortho */
    memcpy(Cx, VR, sizeSub * nx * sizeof(CTYPE));           /* restore accurate Cx */
    *quality_flag = 5;
  }

cleanup:
  (void)0;
}

#undef CONCAT2
#undef CONCAT
#undef FN
#undef EPS_TOL
#undef MIN
#undef CABS
#undef CREAL
