/**
 * @file ilobpcg_impl.inc
 * @brief Type-generic indefinite LOBPCG solver implementation
 *
 * Solves A*X = lambda*B*X where B is indefinite.
 * Uses GGEV-based Rayleigh-Ritz with signature tracking.
 *
 * Macros required: PREFIX, CTYPE, RTYPE, LINOP
 */

#include <string.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
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

/* Helper: Fill array with Gaussian random values */
// TODO: how is this different from the other one, besides
// it being static?
static void FN(ifill_random)(uint64_t n, CTYPE *x) {
  static int seed_initialized = 0;
  if (0 == seed_initialized) {
    srand((unsigned int)time(NULL));
    seed_initialized = 1;
  }

  for (uint64_t i = 0; i < n; i++) {
#ifdef CTYPE_IS_COMPLEX
    RTYPE re = (RTYPE)rand() / RAND_MAX - 0.5;
    RTYPE im = (RTYPE)rand() / RAND_MAX - 0.5;
    x[i] = re + im * I;
#else
    x[i] = (RTYPE)rand() / RAND_MAX - 0.5;
#endif
  }
}

/* Helper: Estimate ||A|| via power iteration */
static RTYPE FN(iestimate_norm)(uint64_t size, LINOP *A, CTYPE *wrk1, CTYPE *wrk2) {
  const uint64_t max_iter = 10;

  FN(ifill_random)(size, wrk1);
  RTYPE nrm = FN(nrm2)(size, wrk1);
  FN(scal)(size, (CTYPE)((RTYPE)1 / nrm), wrk1);

  for (uint64_t iter = 0; iter < max_iter; iter++) {
    A->matvec(A, wrk1, wrk2);
    nrm = FN(nrm2)(size, wrk2);
    if (nrm > 0) {
      FN(scal)(size, (CTYPE)((RTYPE)1 / nrm), wrk2);
    }
    memcpy(wrk1, wrk2, size * sizeof(CTYPE));
  }

  return nrm;
}

/* Indefinite LOBPCG solver
 *
 * Solves A*X = lambda*B*X for the smallest eigenvalues with indefinite B.
 *
 * On entry:
 *   alg->S[0 : size*sizeSub] may contain initial guess (or zero for random)
 *   alg->A, alg->B: operators (B must not be NULL and is indefinite)
 *   alg->T: preconditioner (NULL = no preconditioning)
 *
 * On exit:
 *   alg->S[0 : size*sizeSub] contains converged eigenvectors
 *   alg->eigVals: eigenvalues
 *   alg->resNorm: residual norms
 *   alg->signature: eigenpair signatures (+1/-1)
 *   alg->converged: number of converged eigenpairs
 *
 * Workspace layout:
 *   alg->S:    size * 3*sizeSub  (X, W, P stored in-place)
 *   alg->AS:   size * 3*sizeSub  (assembled search space, avoids aliasing)
 *   alg->BS:   size * sizeSub    (temp for project_back)
 *   alg->Cx:   3*sizeSub * sizeSub
 *   alg->Cp:   3*sizeSub * 2*sizeSub
 *   alg->wrk1: size * 3*sizeSub
 *   alg->wrk2: size * 3*sizeSub
 *   alg->wrk3: 3*sizeSub * 3*sizeSub
 *   alg->wrk4: size * 3*sizeSub
 */
void FN(ilobpcg)(FN(lobpcg_t) *alg) {
  const uint64_t size = alg->size;
  const uint64_t nev = alg->nev;
  const uint64_t sizeSub = alg->sizeSub;
    
  // TODO: these need to be machine-precision dependent
  const RTYPE eps_ortho = 1e-11;
  const RTYPE eps_drop = 1e-12;

  /* B must not be NULL for indefinite problem */
  if (NULL == alg->B) {
    fprintf(stderr, "ilobpcg: B operator must not be NULL\n");
    return;
  }

  /* X, W, P live in alg->S but we use AS for assembled search space
   * to avoid aliasing issues when X/W/P overlap with S */
  CTYPE *X = alg->S;
  CTYPE *W = alg->S + size * sizeSub;
  CTYPE *P = alg->S + size * 2 * sizeSub;

  /* Estimate operator norms */
  RTYPE ANorm = FN(iestimate_norm)(size, alg->A, alg->wrk1, alg->wrk2);
  RTYPE BNorm = FN(iestimate_norm)(size, alg->B, alg->wrk1, alg->wrk2);

  /* Initialize eigenvectors if needed */
  // TODO: here a custom initialization would be good
  // also need to pay attention to inertia of these
  // vectors (see reference implementation)
  RTYPE x_norm = FN(nrm2)(size * sizeSub, X);
  if (x_norm < 1e-14) {
    FN(ifill_random)(size * sizeSub, X);
  }

  /* B-orthonormalize X via svqb */
  FN(svqb)(size, sizeSub, (RTYPE)1e-12, 'n', X,
	   alg->wrk1, alg->wrk2, alg->wrk3, alg->B);

  /* Initial indefinite Rayleigh-Ritz on X */
  FN(indefinite_rayleigh_ritz)(size, sizeSub, X, alg->Cx, alg->eigVals,
			       alg->signature,
			       alg->wrk1, alg->wrk2, alg->wrk3, alg->wrk4,
			       alg->A, alg->B);

  /* Project back: X = X * Cx (use wrk4 as temp) */
  FN(gemm_nn)(size, sizeSub, sizeSub, (CTYPE)1, X, alg->Cx, (CTYPE)0, alg->wrk4);
  memcpy(X, alg->wrk4, size * sizeSub * sizeof(CTYPE));

  /* Compute initial residual W = A*X - B*X*eigVal */
  FN(get_residual)(size, sizeSub, X, NULL, W, alg->eigVals, alg->wrk1,
		   alg->A, alg->B);

  /* Compute initial residual norms */
  FN(get_residual_norm)(size, sizeSub, nev, W, alg->eigVals, alg->resNorm,
			alg->wrk1, alg->wrk2, alg->wrk3, ANorm, BNorm, alg->B);

  /* Main iteration loop */
  alg->converged = 0;
  alg->iter = 0;

  while (alg->iter < alg->maxIter) {
    uint64_t mult = (0 == alg->iter) ? 2 : 3;

    /* Apply preconditioner T to W (if provided) */
    if (NULL != alg->T) {
      for (uint64_t j = 0; j < sizeSub; j++) {
	alg->T->matvec(alg->T, &W[j * size], &alg->wrk1[j * size]);
      }
      memcpy(W, alg->wrk1, size * sizeSub * sizeof(CTYPE));
    }

    /* Orthogonalize W against [X] or [X, P] using indefinite ortho */
    if (0 == alg->iter) {
      FN(ortho_indefinite)(size, sizeSub, sizeSub,
			   eps_ortho, eps_drop,
			   W, X, NULL,
			   alg->wrk1, alg->wrk2, alg->wrk3, alg->B);
    } else {
      /* Assemble [X, P] in wrk4 for ortho reference basis */
      memcpy(alg->wrk4, X, size * sizeSub * sizeof(CTYPE));
      memcpy(alg->wrk4 + size * sizeSub, P, size * sizeSub * sizeof(CTYPE));
      FN(ortho_indefinite)(size, sizeSub, 2 * sizeSub,
			   eps_ortho, eps_drop,
			   W, alg->wrk4, NULL,
			   alg->wrk1, alg->wrk2, alg->wrk3, alg->B);
    }

    /* Assemble search space S into AS buffer to avoid aliasing.
     * X, W, P point into alg->S, so we cannot assemble into alg->S directly
     * when the layout changes (iter>0 needs [X, P, W] but memory is [X, W, P]).
     * Use alg->AS as the assembled search space. */
    CTYPE *S_asm = alg->AS;
    memcpy(S_asm, X, size * sizeSub * sizeof(CTYPE));
    if (0 == alg->iter) {
      memcpy(S_asm + size * sizeSub, W, size * sizeSub * sizeof(CTYPE));
    } else {
      memcpy(S_asm + size * sizeSub, P, size * sizeSub * sizeof(CTYPE));
      memcpy(S_asm + size * 2 * sizeSub, W, size * sizeSub * sizeof(CTYPE));
    }

    /* Modified indefinite Rayleigh-Ritz */
    FN(indefinite_rayleigh_ritz_modified)(
					  size, sizeSub, mult, alg->converged, 0,
					  S_asm, alg->wrk1, alg->wrk2, alg->wrk3, alg->wrk4,
					  alg->Cx, alg->Cp, alg->eigVals, alg->signature,
					  alg->A, alg->B);

    /* Project back from assembled S (not alg->S which has wrong layout).
     * Compute X_new and P_new into separate buffers before writing back,
     * so both use the same unmodified S_asm. */

    /* X_new = S_asm * Cx -> wrk4, then copy to X */
    FN(gemm_nn)(size, sizeSub, mult * sizeSub, (CTYPE)1, S_asm, alg->Cx,
		(CTYPE)0, alg->wrk4);
    memcpy(X, alg->wrk4, size * sizeSub * sizeof(CTYPE));

    /* P_new = S_asm * Cp -> wrk4, then copy to P */
    FN(gemm_nn)(size, sizeSub, mult * sizeSub, (CTYPE)1, S_asm, alg->Cp,
		(CTYPE)0, alg->wrk4);
    memcpy(P, alg->wrk4, size * sizeSub * sizeof(CTYPE));

    /* Compute residual W = A*X - B*X*eigVal */
    FN(get_residual)(size, sizeSub, X, NULL, W, alg->eigVals, alg->wrk1,
		     alg->A, alg->B);

    /* Compute residual norms */
    FN(get_residual_norm)(size, sizeSub, nev, W, alg->eigVals, alg->resNorm,
			  alg->wrk1, alg->wrk2, alg->wrk3, ANorm, BNorm, alg->B);

    /* Check convergence */
    alg->converged = 0;
    for (uint64_t i = 0; i < nev; i++) {
      if (alg->resNorm[i] > alg->tol) break;
      alg->converged++;
    }

    if (alg->converged == nev) {
      break;
    }

    alg->iter++;
  }

}

#undef CONCAT2
#undef CONCAT
#undef FN
#undef CABS
#undef CREAL
