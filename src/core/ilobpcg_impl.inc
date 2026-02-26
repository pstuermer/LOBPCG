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

/* fill_random and estimate_norm are linked from src/residual/estimate_norm_*.c */

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
 *   alg->S:    size * 3*sizeSub  (X, P, W stored contiguously)
 *   alg->AX:   size * sizeSub    (precomputed A*X, optional)
 *   alg->Cx:   3*sizeSub * sizeSub
 *   alg->Cp:   3*sizeSub * 2*sizeSub
 *   alg->wrk1: size * 3*sizeSub  (holds Cx_ortho after modified RR)
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

  /* X, P, W live contiguously in alg->S: [X | P | W]
   * iter==0 uses [X, W] (mult=2), iter>0 uses [X, P, W] (mult=3) */
  CTYPE *restrict X = alg->S;
  CTYPE *restrict P = alg->S + size * sizeSub;
  CTYPE *restrict W = alg->S + size * 2 * sizeSub;

  /* Estimate operator norms */
  const RTYPE ANorm = FN(estimate_norm)(size, alg->A, alg->wrk1, alg->wrk2);
  const RTYPE BNorm = FN(estimate_norm)(size, alg->B, alg->wrk1, alg->wrk2);

  /* Initialize eigenvectors if needed */
  // TODO: here a custom initialization would be good
  // also need to pay attention to inertia of these
  // vectors (see reference implementation)
  const RTYPE x_norm = FN(nrm2)(size * sizeSub, X);
  if (x_norm < 1e-14) {
    FN(fill_random)(size * sizeSub, X);
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

  /* set AX (and later XAX) */
  if (alg->AX)
    FN(apply_block_op)(alg->A, X, alg->AX, size, sizeSub);

  /* Compute initial residual W = A*X - B*X*eigVal */
  FN(get_residual)(size, sizeSub, X, alg->AX, W, alg->eigVals, alg->wrk1,
		   alg->A, alg->B);

  /* Compute initial residual norms */
  FN(get_residual_norm)(size, nev, W, alg->eigVals, alg->resNorm,
			alg->wrk1, alg->wrk2, alg->wrk3, ANorm, BNorm, alg->B);

  for (uint64_t i = 0; i < nev; i++) {
    printf("Eigenvalue %ld: %.9e\t Residual Norm: %.5e\n",
	     i, creal(alg->eigVals[i]), alg->resNorm[i]);
  }
  printf("\n");

  /* Main iteration loop */
  alg->converged = 0;
  alg->iter = 0;

  while (alg->iter < alg->maxIter) {
    uint64_t mult = (0 == alg->iter) ? 2 : 3;

    /* Apply preconditioner T to W (if provided) */
    if (NULL != alg->T) {
      FN(apply_block_op)(alg->T, W, alg->wrk1, size, sizeSub);
      memcpy(W, alg->wrk1, size * sizeSub * sizeof(CTYPE));
    }

    /* Orthogonalize W against [X] or [X, P] using indefinite ortho
     * TODO: cache signature matrix from RR for ortho_indefinite reuse */
    if (0 == alg->iter) {
      FN(ortho_indefinite)(size, sizeSub, sizeSub,
			   eps_ortho, eps_drop,
			   W, X, NULL,
			   alg->wrk1, alg->wrk2, alg->wrk3, alg->B);
    } else {
      FN(ortho_indefinite)(size, sizeSub, 2 * sizeSub,
			   eps_ortho, eps_drop,
			   W, alg->S, NULL,
			   alg->wrk1, alg->wrk2, alg->wrk3, alg->B);
    }

    /* At iter==0, copy W into P slot so S[0:2*sizeSub] = [X,W] for mult=2 RR */
    if (0 == alg->iter)
      memcpy(P, W, size*sizeSub * sizeof(CTYPE));

    /* Modified indefinite Rayleigh-Ritz
     * Cx_ortho reuses wrk1 storage (pure output, written only after wrk1 is dead) */
    CTYPE *Cx_ortho = alg->wrk1;
    int quality_flag = 1;
    FN(indefinite_rayleigh_ritz_modified)(size, sizeSub, mult, alg->converged, 0,
					  alg->S, alg->wrk1, alg->wrk2, alg->wrk3, alg->wrk4,
					  alg->Cx, alg->Cp, Cx_ortho,
					  alg->eigVals, alg->signature,
					  &quality_flag, alg->A, alg->B);

    /* redefine P and W after first iteration */
    P = alg->S + size*sizeSub;
    W = alg->S + size*2*sizeSub;

    /* Two-path projection depending on eigenvector quality.
     *
     * quality_flag == 1 (good): Cx is well-conditioned, use directly.
     * quality_flag == 5 (poor): Cx has accurate eigenvalue correspondence
     *   but may be poorly conditioned. Use Cx for residual (needs accurate λ),
     *   then Cx_ortho (= wrk1) for the iteration basis (needs stability). */

    if (1 == quality_flag) {
      const uint64_t ms = mult*sizeSub;
      memcpy(alg->wrk2, alg->Cx, ms*sizeSub * sizeof(CTYPE));
      memcpy(alg->wrk2 + ms*sizeSub, alg->Cp, ms*sizeSub * sizeof(CTYPE));
      FN(gemm_nn)(size, 2*sizeSub, ms, (CTYPE)1, alg->S,
		  alg->wrk2, (CTYPE)0, alg->wrk4);
      memcpy(alg->S, alg->wrk4, 2*size*sizeSub * sizeof(CTYPE));

      if (alg->AX)
	FN(apply_block_op)(alg->A, X, alg->AX, size, sizeSub);

      FN(get_residual)(size, sizeSub, X, alg->AX, W, alg->eigVals,
		       alg->wrk1, alg->A, alg->B);

      FN(get_residual_norm)(size, nev, W, alg->eigVals, alg->resNorm,
			    alg->wrk1, alg->wrk2, alg->wrk3,
			    ANorm, BNorm, alg->B);
    } else if (5 == quality_flag) {
      const uint64_t ms = mult*sizeSub;
      FN(gemm_nn)(size, sizeSub, ms, (CTYPE)1, alg->S,
		  alg->Cx, (CTYPE)0, alg->wrk4);

      FN(get_residual)(size, sizeSub, alg->wrk4, NULL, W, alg->eigVals,
		       alg->wrk2, alg->A, alg->B);

      FN(get_residual_norm)(size, nev, W, alg->eigVals, alg->resNorm,
			    alg->wrk2, alg->wrk1, alg->wrk3,
			    ANorm, BNorm, alg->B);
    }

    /* Check convergence */
    alg->converged = 0;
    for (uint64_t i = 0; i < nev; i++) {
      if (alg->resNorm[i] > alg->tol) break;
      alg->converged++;
    }

    printf("Iteration %ld\t Converged Eigenpairs %ld/%ld\n",
	   alg->iter, alg->converged, nev);
    for (uint64_t i = 0; i < nev; i++) {
      printf("Eigenvalue %ld: %.9e\t Residual Norm: %.5e\n",
	     i, creal(alg->eigVals[i]), alg->resNorm[i]);
    }
    printf("\n");
	     

    if (5 == quality_flag) {
      /* Overwrite X with well-conditioned basis for next iteration */
      const uint64_t ms = mult*sizeSub;
      memcpy(alg->wrk2, alg->wrk1, ms*sizeSub * sizeof(CTYPE));
      memcpy(alg->wrk2 + ms*sizeSub, alg->Cp, ms*sizeSub * sizeof(CTYPE));
      FN(gemm_nn)(size, 2*sizeSub, ms, (CTYPE)1, alg->S,
		  alg->wrk2, (CTYPE)0, alg->wrk4);
      memcpy(alg->S, alg->wrk4, 2*size*sizeSub * sizeof(CTYPE));

      if (alg->AX)
	FN(apply_block_op)(alg->A, X, alg->AX, size, sizeSub);

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
