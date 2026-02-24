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
#include "lobpcg/memory.h"

#define CONCAT2(a, b) a##_##b
#define CONCAT(a, b) CONCAT2(a, b)
#define FN(name) CONCAT(PREFIX, name)

#ifndef MIN
#define MIN(a, b) ((a) < (b) ? (a) : (b))
#endif

#ifdef CTYPE_IS_COMPLEX
#define CABS(x) cabs(x)
#define CREAL(x) creal(x)
#else
#define CABS(x) fabs(x)
#define CREAL(x) (x)
#endif

#include "bubble_sort_sig_impl.inc"

/* Modified indefinite Rayleigh-Ritz
 *
 * Inputs:
 *   size:      problem dimension
 *   nx:        block size (nev)
 *   mult:      block multiplier (2 or 3), sizeSub = mult*nx
 *   nconv:     converged count (unused, for soft-locking)
 *   ndrop:     dropped columns (unused)
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
					   const uint64_t ndrop, CTYPE *restrict S,
					   CTYPE *restrict wrk1, CTYPE *restrict wrk2,
					   CTYPE *restrict wrk3, CTYPE *restrict wrk4,
					   CTYPE *restrict Cx, CTYPE *restrict Cp,
					   CTYPE *restrict Cx_ortho,
					   RTYPE *restrict eigVal,
					   int8_t *restrict signature,
					   int *restrict quality_flag,
					   LINOP *A, LINOP *B) {

  (void)nconv;
  (void)ndrop;

  const uint64_t sizeSub = mult * nx;
  const uint64_t ss = sizeSub * sizeSub;

  /* Internal allocations */
  RTYPE *eigvals_all = xcalloc(sizeSub, sizeof(RTYPE));
  int8_t *sig_all = xcalloc(sizeSub, sizeof(int8_t));
  uint64_t *indices = xcalloc(sizeSub, sizeof(uint64_t));
  CTYPE *VR = xcalloc(ss, sizeof(CTYPE)); /* GGEV right eigenvectors */
#ifdef CTYPE_IS_COMPLEX
  CTYPE *alpha = xcalloc(sizeSub, sizeof(CTYPE));
  CTYPE *beta  = xcalloc(sizeSub, sizeof(CTYPE));
#else
  RTYPE *alphar = xcalloc(sizeSub, sizeof(RTYPE));
  RTYPE *alphai = xcalloc(sizeSub, sizeof(RTYPE));
  RTYPE *beta   = xcalloc(sizeSub, sizeof(RTYPE));
#endif

  /* 1. GA = S^H * A * S in wrk1 */
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
    CTYPE ev = alpha[i] / beta[i];
    eigvals_all[i] = CREAL(ev);
#else
    if (CABS(beta[i]) > (RTYPE)1e-30)
      eigvals_all[i] = alphar[i] / beta[i];
    else
      eigvals_all[i] = (alphar[i] >= 0) ? (RTYPE)1e30 : (RTYPE)-1e30;
#endif
  }

  /* 6. B-normalize eigenvectors: V^H * GB * V
   *    wrk3 still holds GB, VR holds eigenvectors */
  FN(gemm_nn)(sizeSub, sizeSub, sizeSub, (CTYPE)1, wrk3, VR, (CTYPE)0, wrk1);
#ifdef CTYPE_IS_COMPLEX
  FN(gemm_hn)(sizeSub, sizeSub, sizeSub, (CTYPE)1, VR, wrk1, (CTYPE)0, wrk4);
#else
  FN(gemm_tn)(sizeSub, sizeSub, sizeSub, (CTYPE)1, VR, wrk1, (CTYPE)0, wrk4);
#endif

  for (uint64_t i = 0; i < sizeSub; i++) {
    CTYPE diag_val = wrk4[i + i * sizeSub];
    RTYPE abs_diag = CABS(diag_val);

    if (abs_diag > (RTYPE)1e-30) {
      RTYPE scale = (RTYPE)1 / sqrt(abs_diag);
      for (uint64_t k = 0; k < sizeSub; k++)
	VR[k + i * sizeSub] *= (CTYPE)scale;
    }

    sig_all[i] = (CREAL(diag_val) >= 0) ? 1 : -1;
  }

  /* Quality check: relative B-orthogonality of normalized GGEV eigenvectors
   * Recompute V^H * GB * V after normalization; wrk3 = GB still valid */
  FN(gemm_nn)(sizeSub, sizeSub, sizeSub, (CTYPE)1, wrk3, VR, (CTYPE)0, wrk1);
#ifdef CTYPE_IS_COMPLEX
  FN(gemm_hn)(sizeSub, sizeSub, sizeSub, (CTYPE)1, VR, wrk1, (CTYPE)0, wrk4);
#else
  FN(gemm_tn)(sizeSub, sizeSub, sizeSub, (CTYPE)1, VR, wrk1, (CTYPE)0, wrk4);
#endif
  for (uint64_t i = 0; i < sizeSub; i++)
    wrk4[i + i * sizeSub] = (CTYPE)(CABS(wrk4[i + i * sizeSub]) - (RTYPE)1);
  RTYPE eerr  = FN(nrm2)(ss, wrk4);
  RTYPE cerr  = FN(nrm2)(ss, VR);
  RTYPE Bcerr = FN(nrm2)(ss, wrk1);
  const int quality_ok = (Bcerr < (RTYPE)1e-30 || eerr <= (RTYPE)1e-10 * cerr * Bcerr);

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
    const uint64_t n_remainder = (mult - 1) * nx;
    memset(Cp, 0, sizeSub * nx * sizeof(CTYPE));
    for (uint64_t j = 0; j < nx; j++)
      memcpy(&Cp[nx + j * sizeSub], &Cx[nx + j * sizeSub],
             n_remainder * sizeof(CTYPE));
  }

  /* 12. Cx_ortho + quality_flag:
   *     If GGEV eigenvectors are well-conditioned, Cx_ortho = Cx.
   *     Otherwise B-orthogonalize via svqb_mat (wrk3=GB still valid). */
  memcpy(Cx_ortho, Cx, sizeSub * nx * sizeof(CTYPE));
  if (quality_ok) {
    *quality_flag = 1;
  } else {
    /* wrk3=GB (read-only in svqb_mat); wrk1, wrk2, wrk4 free to clobber */
    FN(svqb_mat)(sizeSub, nx, (RTYPE)1e-12, 'n', Cx_ortho, wrk3, wrk1, wrk2, wrk4);
    *quality_flag = 5;
  }

cleanup:
  safe_free((void**)&eigvals_all);
  safe_free((void**)&sig_all);
  safe_free((void**)&indices);
  safe_free((void**)&VR);
#ifdef CTYPE_IS_COMPLEX
  safe_free((void**)&alpha);
  safe_free((void**)&beta);
#else
  safe_free((void**)&alphar);
  safe_free((void**)&alphai);
  safe_free((void**)&beta);
#endif
}

#undef CONCAT2
#undef CONCAT
#undef FN
#undef MIN
#undef CABS
#undef CREAL
