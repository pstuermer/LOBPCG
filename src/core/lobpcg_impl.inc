/**
 * @file lobpcg_impl.inc
 * @brief Type-generic main LOBPCG solver implementation
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

/* Note: S layout is [X, P, W] with X at offset 0, P at offset sizeSub, W at offset 2*sizeSub.
 * At iter==0, P does not exist yet, so we copy W into the P slot to get S = [X, W_copy]
 * for mult=2 Rayleigh-Ritz. At iter>=1, S = [X, P, W] is already contiguous for mult=3. */

/* Helper: Project back X = S * C */
// TODO: in the long term, get rid of this
static void FN(project_back)(uint64_t size, uint64_t sizeSub, uint64_t mult,
                             CTYPE *S, CTYPE *C, CTYPE *X, CTYPE *wrk) {
    /* X = S * C where S is (size x mult*sizeSub), C is (mult*sizeSub x sizeSub) */
    FN(gemm_nn)(size, sizeSub, mult * sizeSub, (CTYPE)1, S, C, (CTYPE)0, wrk);
    memcpy(X, wrk, size * sizeSub * sizeof(CTYPE));
}

/* Main LOBPCG solver
 *
 * Solves A*X = λ*B*X for the smallest eigenvalues
 *
 * Algorithm state is in alg structure:
 *   alg->A, alg->B: operators
 *   alg->size, alg->nev: problem dimensions
 *   alg->maxIter, alg->tol: convergence parameters
 *   alg->S: workspace for [X, P, W]
 *   alg->Cx, alg->Cp: projection coefficients
 *   alg->eigVals, alg->resNorm: output arrays
 *   alg->wrk1, wrk2, wrk3, wrk4: workspaces
 *
 * On entry: alg->S should contain initial eigenvector guess in first nev columns
 *           (or will be filled with random if S is zero)
 *
 * On exit: alg->S contains converged eigenvectors in first nev columns
 *          alg->eigVals contains eigenvalues
 *          alg->converged contains number of converged eigenpairs
 */
void FN(lobpcg)(FN(lobpcg_t) *alg) {
    const uint64_t size = alg->size;
    const uint64_t nev = alg->nev;
    const uint64_t sizeSub = alg->sizeSub;
    const RTYPE eps_ortho = 1e-12;
    const RTYPE eps_drop = 1e-12;

    /* Pointers into S = [X, P, W] */
    CTYPE *X = alg->S;
    CTYPE *P = alg->S + size * sizeSub;
    CTYPE *W = alg->S + size * 2 * sizeSub;

    /* Estimate operator norms */
    RTYPE ANorm = FN(estimate_norm)(size, alg->A, alg->wrk1, alg->wrk2);
    RTYPE BNorm = (NULL != alg->B) ? FN(estimate_norm)(size, alg->B, alg->wrk1, alg->wrk2) : (RTYPE)1;

    /* Initialize eigenvectors if needed (check if S is zero) */
    RTYPE x_norm = FN(nrm2)(size * sizeSub, X);
    if (x_norm < 1e-14) {
        FN(fill_random)(size * sizeSub, X);
    }

    /* Initial Rayleigh-Ritz on X */
    FN(rayleigh_ritz)(size, sizeSub, X, alg->Cx, alg->eigVals,
                      alg->wrk1, alg->wrk2, alg->wrk3, alg->A, alg->B);

    /* Project back: X = X * Cx */
    FN(project_back)(size, sizeSub, 1, X, alg->Cx, X, alg->wrk4);

    /* Compute initial residual W = A*X - B*X*eigVal */
    FN(get_residual)(size, sizeSub, X, NULL, W, alg->eigVals, alg->wrk1, alg->A, alg->B);

    /* Compute initial residual norms */
    FN(get_residual_norm)(size, nev, W, alg->eigVals, alg->resNorm,
                          alg->wrk1, alg->wrk2, alg->wrk3, ANorm, BNorm, alg->B);
    // TODO: print current eigenvalues and residual norm
    uint8_t useOrtho = 0;
    alg->converged = 0;
    alg->iter = 0;

    while (alg->iter < alg->maxIter) {
      uint64_t mult = (0 == alg->iter) ? 2 : 3;

      /* Apply preconditioner T to W (if provided) */
      if (NULL != alg->T) {
	FN(apply_block_op)(alg->T, W, alg->wrk1, size, sizeSub);
	memcpy(W, alg->wrk1, size * sizeSub * sizeof(CTYPE));
      }

      /* If useOrtho: orthogonalize W against [X] (iter==0) or [X, P] (iter>=1) */
      if (useOrtho) {
	if (0 == alg->iter) {
	  FN(ortho_drop)(size, sizeSub, sizeSub, eps_ortho, eps_drop,
			 W, X, alg->wrk1, alg->wrk2, alg->wrk3, alg->B);
	} else {
	  FN(ortho_drop)(size, sizeSub, 2 * sizeSub, eps_ortho, eps_drop,
			 W, alg->S, alg->wrk1, alg->wrk2, alg->wrk3, alg->B);
	}
      }

      /* At iter==0, copy W into P slot so S[0:2*sizeSub] = [X, W] for mult=2 RR */
      if (0 == alg->iter) {
	memcpy(P, W, size * sizeSub * sizeof(CTYPE));
      }

      /* Modified Rayleigh-Ritz */
      FN(rayleigh_ritz_modified)(size, sizeSub, mult, alg->converged, 0, &useOrtho,
				 alg->S, alg->wrk1, alg->wrk2, alg->wrk3,
				 alg->Cx, alg->Cp, alg->eigVals, alg->A, alg->B);

      /* If condition check failed: set useOrtho and retry */
      if (2 == useOrtho) {
	useOrtho = 1;
	/* Re-orthogonalize W */
	if (0 == alg->iter) {
	  FN(ortho_drop)(size, sizeSub, sizeSub, eps_ortho, eps_drop,
			 W, X, alg->wrk1, alg->wrk2, alg->wrk3, alg->B);
	} else {
	  FN(ortho_drop)(size, sizeSub, 2 * sizeSub, eps_ortho, eps_drop,
			 W, alg->S, alg->wrk1, alg->wrk2, alg->wrk3, alg->B);
	}
	/* Re-copy W into P slot at iter==0 */
	if (0 == alg->iter) {
	  memcpy(P, W, size * sizeSub * sizeof(CTYPE));
	}
	/* Retry RR with useOrtho=1 */
	FN(rayleigh_ritz_modified)(size, sizeSub, mult, alg->converged, 0, &useOrtho,
				   alg->S, alg->wrk1, alg->wrk2, alg->wrk3,
				   alg->Cx, alg->Cp, alg->eigVals, alg->A, alg->B);
      }

      /* redefine P and W after first iteration */
      P = alg->S + size * sizeSub;
      W = alg->S + size * 2 * sizeSub;

      /* Project back: X = S * Cx, P = S * Cp */
      FN(project_back)(size, sizeSub, mult, alg->S, alg->Cx, X, alg->wrk4);
      FN(project_back)(size, sizeSub, mult, alg->S, alg->Cp, P, alg->wrk4);

      /* Compute residual W = A*X - B*X*eigVal */
      FN(get_residual)(size, sizeSub, X, NULL, W, alg->eigVals, alg->wrk1, alg->A, alg->B);

      /* Compute residual norms */
      FN(get_residual_norm)(size, nev, W, alg->eigVals, alg->resNorm,
			    alg->wrk1, alg->wrk2, alg->wrk3, ANorm, BNorm, alg->B);
	
      /* Check convergence */
      alg->converged = 0;
      for (uint64_t i = 0; i < nev; i++) {
	if (alg->resNorm[i] > alg->tol) break;
	alg->converged++;
      }

      if (alg->converged == nev) {
	break;  /* All converged */
      }

      alg->iter++;
    }

    /* Final eigenvectors are in X = alg->S */
}

#undef CONCAT2
#undef CONCAT
#undef FN
#undef CABS
#undef CREAL
