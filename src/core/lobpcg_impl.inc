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

#include "lobpcg/type_dispatch.h"
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
/* fill_random and estimate_norm are linked from src/residual/estimate_norm_*.c */

/* Note: S layout is [X, P, W] with X at offset 0, P at offset sizeSub, W at offset 2*sizeSub.
 * At iter==0, P does not exist yet, so we copy W into the P slot to get S = [X, W_copy]
 * for mult=2 Rayleigh-Ritz. At iter>=1, S = [X, P, W] is already contiguous for mult=3.
 *
 * Soft-locking: when nconv eigenvectors converge, P and W shrink from sizeSub to
 * n_active = sizeSub - nconv. X stays full width. Project_back always writes all
 * sizeSub columns of P, so compaction at top of each iteration is stateless:
 * just memmove P and memcpy W to remove converged columns. */

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

  /* Parameter validation */
  if (nev > sizeSub) {
    fprintf(stderr, "lobpcg: nev (%lu) > sizeSub (%lu)\n",
            (unsigned long)nev, (unsigned long)sizeSub);
    return;
  }
  if (3 * sizeSub > size) {
    fprintf(stderr, "lobpcg: 3*sizeSub (%lu) > size (%lu)\n",
            (unsigned long)(3 * sizeSub), (unsigned long)size);
    return;
  }

  const RTYPE eps_ortho = (RTYPE)EPS_TOL;
  const RTYPE eps_drop = (RTYPE)EPS_TOL;

  /* Pointers into S = [X, P, W] */
  CTYPE *restrict X = alg->S;
  CTYPE *restrict P = alg->S + size * sizeSub;
  CTYPE *restrict W = alg->S + size * 2 * sizeSub;

  /* Estimate operator norms */
  const RTYPE ANorm = FN(estimate_norm)(size, alg->A, alg->wrk1, alg->wrk2);
  const RTYPE BNorm = (NULL != alg->B) ? FN(estimate_norm)(size, alg->B, alg->wrk1, alg->wrk2) : (RTYPE)1;

  printf("F-Norm: %.5e %.5e\n", ANorm, BNorm);

  /* Initialize eigenvectors if needed (check if S is zero) */
  RTYPE x_norm = FN(nrm2)(size * sizeSub, X);
  if (x_norm < (RTYPE)EPS_TOL) {
    FN(fill_random)(size * sizeSub, X);
  }

  /* Initial Rayleigh-Ritz on X */
  FN(rayleigh_ritz)(size, sizeSub, X, alg->Cx, alg->eigVals,
                    alg->wrk1, alg->wrk2, alg->wrk3,
                    alg->rr_D, alg->A, alg->B);

  /* Project back: X = X * Cx */
  FN(project_back)(size, sizeSub, 1, X, alg->Cx, X, alg->wrk4);

  /* set AX (and later XAX) */
  if (alg->AX)
    FN(apply_block_op)(alg->A, X, alg->AX, size, sizeSub);

  /* Compute initial residual W = A*X - B*X*eigVal */
  FN(get_residual)(size, sizeSub, X, alg->AX, W,
                   alg->eigVals, alg->wrk1, alg->A, alg->B);

  /* Compute initial residual norms */
  FN(get_residual_norm)(size, nev, W, alg->eigVals, alg->resNorm,
                        alg->wrk1, alg->wrk2, alg->wrk3, ANorm, BNorm, alg->B);

  for (uint64_t i = 0; i < nev; i++) {
    printf("Eigenvalue %ld: %.9e\t Residual Norm: %.5e\n",
           i, CREAL(alg->eigVals[i]), alg->resNorm[i]);
  }
  printf("\n");

  uint8_t useOrtho = 0;
  alg->converged = 0;
  alg->iter = 0;

  while (alg->iter < alg->maxIter) {
    const uint64_t nconv = alg->converged;
    const uint64_t n_active = sizeSub - nconv;
    uint64_t mult = (0 == alg->iter) ? 2 : 3;
    uint64_t sizeW = (0 == alg->iter) ? sizeSub : n_active;

    /* Compact P and W: remove converged columns from full-width layout.
     * After project_back, S = [X(sizeSub) | P(sizeSub) | W(sizeSub)].
     * Shift active cols of P to front, copy active cols of W right after. */
    if (nconv > 0 && alg->iter > 0) {
      CTYPE *P_base = alg->S + sizeSub * size;
      memmove(P_base, P_base + nconv * size, n_active * size * sizeof(CTYPE));
      memcpy(P_base + n_active * size,
             alg->S + (2 * sizeSub + nconv) * size,
             n_active * size * sizeof(CTYPE));
    }

    /* W pointer: iter 0 at fixed offset, iter >= 1 right after P_active */
    W = (0 == alg->iter) ? alg->S + size * 2 * sizeSub
                         : alg->S + size * (sizeSub + n_active);

    /* Apply preconditioner T to W (if provided) */
    if (NULL != alg->T) {
      FN(apply_block_op)(alg->T, W, alg->wrk1, size, sizeW);
      memcpy(W, alg->wrk1, size * sizeW * sizeof(CTYPE));
    }

    /* Orthogonalize W against [X] (iter==0) or [X, P_active] (iter>=1) */
    if (useOrtho) {
      if (0 == alg->iter) {
        FN(ortho_drop)(size, sizeSub, sizeSub, eps_ortho, eps_drop,
                       W, X, alg->wrk1, alg->wrk2, alg->wrk3, alg->B);
      } else {
        sizeW = FN(ortho_drop)(size, n_active, sizeSub + n_active, eps_ortho, eps_drop,
                               W, alg->S, alg->wrk1, alg->wrk2, alg->wrk3, alg->B);
      }
    }

    /* At iter==0, copy W into P slot so S = [X, W_copy] for mult=2 RR */
    if (0 == alg->iter)
      memcpy(P, W, size * sizeW * sizeof(CTYPE));

    /* Modified Rayleigh-Ritz */
    FN(rayleigh_ritz_modified)(size, sizeSub, mult, alg->converged, sizeW, &useOrtho,
                               alg->S, alg->AX, alg->wrk1, alg->wrk2, alg->wrk3,
                               alg->Cx, alg->Cp, alg->eigVals,
                               alg->rr_eigvals, alg->rr_tau, alg->rr_D,
                               alg->A, alg->B);

    /* If condition check failed: set useOrtho and retry */
    if (2 == useOrtho) {
      useOrtho = 1;
      if (0 == alg->iter) {
        FN(ortho_drop)(size, sizeSub, sizeSub, eps_ortho, eps_drop,
                       W, X, alg->wrk1, alg->wrk2, alg->wrk3, alg->B);
      } else {
        sizeW = FN(ortho_drop)(size, n_active, sizeSub + n_active, eps_ortho, eps_drop,
                               W, alg->S, alg->wrk1, alg->wrk2, alg->wrk3, alg->B);
      }
      if (0 == alg->iter)
        memcpy(P, W, size * sizeW * sizeof(CTYPE));
      FN(rayleigh_ritz_modified)(size, sizeSub, mult, alg->converged, sizeW, &useOrtho,
                                 alg->S, alg->AX, alg->wrk1, alg->wrk2, alg->wrk3,
                                 alg->Cx, alg->Cp, alg->eigVals,
                                 alg->rr_eigvals, alg->rr_tau, alg->rr_D,
                                 alg->A, alg->B);
    }

    /* Redefine P (always right after X) */
    P = alg->S + size * sizeSub;

    /* Subspace size for this iteration's projection */
    const uint64_t ms = sizeSub + (0 == alg->iter ? 0 : n_active) + sizeW;

    /* Project back: [X_new, P_new] = S * [Cx | Cp] — full sizeSub columns each */
    memcpy(alg->wrk1, alg->Cx, ms * sizeSub * sizeof(CTYPE));
    memcpy(alg->wrk1 + ms * sizeSub, alg->Cp, ms * sizeSub * sizeof(CTYPE));
    FN(gemm_nn)(size, 2 * sizeSub, ms, (CTYPE)1, alg->S, alg->wrk1, (CTYPE)0, alg->wrk4);
    memcpy(alg->S, alg->wrk4, size * 2 * sizeSub * sizeof(CTYPE));

    /* Compute AX for all X */
    if (alg->AX)
      FN(apply_block_op)(alg->A, X, alg->AX, size, sizeSub);

    /* Residual at fixed offset for ALL sizeSub vectors */
    W = alg->S + size * 2 * sizeSub;
    FN(get_residual)(size, sizeSub, X, alg->AX, W, alg->eigVals,
                     alg->wrk1, alg->A, alg->B);

    /* Residual norms for all nev vectors */
    FN(get_residual_norm)(size, nev, W, alg->eigVals, alg->resNorm,
                          alg->wrk1, alg->wrk2, alg->wrk3, ANorm, BNorm, alg->B);

    /* Check convergence */
    alg->converged = 0;
    for (uint64_t i = 0; i < nev; i++) {
      if (alg->resNorm[i] > alg->tol) break;
      alg->converged++;
    }

    printf("Iteration %ld\t Converged Eigenpairs: %ld/%ld\n",
           alg->iter, alg->converged, nev);
    for (uint64_t i = 0; i < nev; i++) {
      printf("Eigenvalue %ld: %.9e\t Residual Norm: %.5e\n",
             i, CREAL(alg->eigVals[i]), alg->resNorm[i]);
    }
    printf("\n");

    if (alg->converged == nev) {
      break;
    }

    alg->iter++;
  }

  /* Final eigenvectors are in X = alg->S */
}

#undef CONCAT2
#undef CONCAT
#undef FN
#undef EPS_TOL
#undef CABS
#undef CREAL
