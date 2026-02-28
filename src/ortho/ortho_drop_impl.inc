/**
 * @file ortho_drop_impl.inc
 * @brief Type-generic B-orthogonalization of U against V (positive definite B)
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

/* Compute ||G-I_sig||_F from upper triangle of Hermitian G */
static RTYPE FN(ortho_err_upper)(uint64_t n, const CTYPE *G) {
  RTYPE sum = 0;
  for (uint64_t j = 0; j < n; j++) {
    const RTYPE d = CABS(G[j + j*n]) - (RTYPE)1;
    sum += d*d;
    for (uint64_t i = 0; i < j; i++) {
      const RTYPE v = CABS(G[i + j*n]);
      sum += 2*v*v;
    }
  }
  return sqrt(sum);
}

/* B-orthogonalize U against V with positive definite B
 *
 * Workspace layout:
 *   wrk1: m*max(n_u,n_v) — gram scratch (holds B*V or B*U after gram calls)
 *   wrk2: max(n_v*n_u, n_u*n_u) — gram result storage
 *   wrk3: m*max(n_u,n_v) — svqb scratch
 *
 * Algorithm:
 * 1. Outer loop (max 3 iterations):
 *    a. Project: U = U - V * (V^H * B * U)
 *    b. Inner loop (max 3 iterations):
 *       - svqb to B-orthonormalize U
 *       - Check ||U^H * B * U - I||_F < eps_ortho
 *    c. Check ||V^H * B * U||_F < eps_ortho
 *
 * Returns: number of columns retained in U
 */
uint64_t FN(ortho_drop)(const uint64_t m, const uint64_t n_u, const uint64_t n_v,
                        const RTYPE eps_ortho, const RTYPE eps_drop,
                        CTYPE *restrict U, CTYPE *restrict V,
                        CTYPE *restrict wrk1, CTYPE *restrict wrk2, CTYPE *restrict wrk3,
                        LINOP *B) {
  if (0 == n_u || 0 == n_v) return n_u;

  /* Check dimension consistency */
  if (m < n_u + n_v) {
    fprintf(stderr, "ortho_drop: overdetermined shape m=%lu < n_u+n_v=%lu+%lu\n",
	    (unsigned long)m, (unsigned long)n_u, (unsigned long)n_v);
    return n_u;
  }

  /* Compute ||B*V||_F for relative tolerance */
  RTYPE BV_norm = 0;
  if (NULL == B) {
    BV_norm = FN(nrm2)(m * n_v, V);
  } else {
    FN(apply_block_op)(B, V, wrk1, m, n_v);
    BV_norm = FN(nrm2)(m * n_v, wrk1);
  }

  if (BV_norm < eps_ortho) BV_norm = (RTYPE)1;

  const uint64_t max_outer = 3;
  const uint64_t max_inner = 3;

  /* Outer loop: orthogonalization iterations */
  for (uint64_t outer = 0; outer < max_outer; outer++) {
    /* Compute coef = V^H * B * U  (n_v x n_u) stored in wrk2 */
    FN(gram_cross)(V, n_v, U, n_u, m, B, wrk2, n_v, wrk1);

    /* U = U - V * coef */
    FN(gemm_nn)(m, n_u, n_v, (CTYPE)(-1), V, wrk2, (CTYPE)1, U);

    /* Inner loop: svqb iterations */
    for (uint64_t inner = 0; inner < max_inner; inner++) {
      /* svqb to B-orthonormalize U */
      uint64_t n_ret = FN(svqb)(m, n_u, eps_drop, 'n', U, wrk1, wrk2, wrk3, B);

      if (n_ret != n_u) {
	fprintf(stderr, "ortho_drop: svqb dropped columns (%lu -> %lu)\n",
		(unsigned long)n_u, (unsigned long)n_ret);
	return n_ret;
      }

      /* Check ||U^H * B * U - I||_F < eps_ortho */
      /* NOTE: denominator choice differs between ortho_drop (||B*U||*||U||)
       * and ortho_indefinite (||U||^2). For indefinite B, ||B*U|| can vanish
       * near null-space of B, making ||B*U||*||U|| unsuitable. Investigate
       * whether ortho_drop should also use ||U||^2 for consistency. */
      RTYPE U_norm = FN(nrm2)(m * n_u, U);
      if (U_norm < eps_ortho) U_norm = (RTYPE)1;
      FN(gram_self)(U, m, n_u, B, wrk2, n_u, wrk1);
      RTYPE BU_norm;
      if (NULL == B) {
	BU_norm = U_norm;
      } else {
	BU_norm = FN(nrm2)(m * n_u, wrk1);
      }

      RTYPE R_norm = FN(ortho_err_upper)(n_u, wrk2);
      RTYPE rerr = R_norm / (BU_norm * U_norm);

      if (rerr < eps_ortho) break;
    }

    /* Check ||V^H * B * U||_F < eps_ortho */
    FN(gram_cross)(V, n_v, U, n_u, m, B, wrk2, n_v, wrk1);

    RTYPE VU_norm = FN(nrm2)(n_v * n_u, wrk2);
    RTYPE U_norm = FN(nrm2)(m * n_u, U);
    if (U_norm < eps_ortho) U_norm = (RTYPE)1;
    RTYPE rerr = VU_norm / (BV_norm * U_norm);

    if (rerr < eps_ortho) break;
  }

  return n_u;
}

#undef CONCAT2
#undef CONCAT
#undef FN
#undef MAX
#undef CABS
