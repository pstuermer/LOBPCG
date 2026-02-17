/**
 * @file lobpcg_impl.inc
 * @brief Type-generic main LOBPCG solver implementation
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
static void FN(fill_random)(uint64_t n, CTYPE *x) {
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
static RTYPE FN(estimate_norm)(uint64_t size, LINOP *A, CTYPE *wrk1, CTYPE *wrk2) {
    const uint64_t max_iter = 10;

    /* Initialize with random vector */
    FN(fill_random)(size, wrk1);
    RTYPE nrm = FN(nrm2)(size, wrk1);
    FN(scal)(size, (CTYPE)((RTYPE)1 / nrm), wrk1);

    /* Power iteration */
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

/* Helper: Assemble S = [X, P, W] or [X, W] */
static void FN(setup_S)(uint64_t size, uint64_t sizeSub, uint64_t iter,
                        CTYPE *X, CTYPE *P, CTYPE *W, CTYPE *S) {
    /* X is first sizeSub columns */
    memcpy(S, X, size * sizeSub * sizeof(CTYPE));

    if (0 == iter) {
        /* S = [X, W] */
        memcpy(S + size * sizeSub, W, size * sizeSub * sizeof(CTYPE));
    } else {
        /* S = [X, P, W] */
        memcpy(S + size * sizeSub, P, size * sizeSub * sizeof(CTYPE));
        memcpy(S + size * 2 * sizeSub, W, size * sizeSub * sizeof(CTYPE));
    }
}

/* Helper: Project back X = S * C */
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
    const uint64_t sizeSub = alg->sizeSub;  /* Usually == nev */
    const RTYPE eps_ortho = 1e-12;
    const RTYPE eps_drop = 1e-12;

    /* Pointers into S for X, P, W */
    CTYPE *X = alg->S;
    CTYPE *W = alg->S + size * sizeSub;
    CTYPE *P = alg->S + size * 2 * sizeSub;  /* Only used after first iteration */

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
    FN(get_residual_norm)(size, sizeSub, nev, W, alg->eigVals, alg->resNorm,
                          alg->wrk1, alg->wrk2, alg->wrk3, ANorm, BNorm, alg->B);

    /* Main iteration loop */
    RTYPE *eigvals_r = xcalloc(sizeSub, sizeof(RTYPE));
    uint8_t useOrtho = 0;
    alg->converged = 0;
    alg->iter = 0;

    while (alg->iter < alg->maxIter) {
        uint64_t mult = (0 == alg->iter) ? 2 : 3;

        /* Apply preconditioner T to W (if provided) */
        if (NULL != alg->T) {
            for (uint64_t j = 0; j < sizeSub; j++) {
                alg->T->matvec(alg->T, &W[j*size], &alg->wrk1[j*size]);
            }
            memcpy(W, alg->wrk1, size * sizeSub * sizeof(CTYPE));
        }

        /* If useOrtho: orthogonalize W against [X, P] */
        if (useOrtho) {
            if (1 == alg->iter) {
                /* Only X exists */
                FN(ortho_drop)(size, sizeSub, sizeSub, eps_ortho, eps_drop,
                               W, X, alg->wrk1, alg->wrk2, alg->wrk3, alg->B);
            } else {
                /* Need to orthogonalize against [X, P] - use wrk4 to assemble */
                memcpy(alg->wrk4, X, size * sizeSub * sizeof(CTYPE));
                memcpy(alg->wrk4 + size * sizeSub, P, size * sizeSub * sizeof(CTYPE));
                FN(ortho_drop)(size, sizeSub, 2 * sizeSub, eps_ortho, eps_drop,
                               W, alg->wrk4, alg->wrk1, alg->wrk2, alg->wrk3, alg->B);
            }
        }

        /* Assemble S = [X, W] or [X, P, W] */
        FN(setup_S)(size, sizeSub, alg->iter, X, P, W, alg->S);

        /* Modified Rayleigh-Ritz */
        FN(rayleigh_ritz_modified)(size, sizeSub, mult, alg->converged, 0, &useOrtho,
                                   alg->S, alg->wrk1, alg->wrk2, alg->wrk3,
                                   alg->Cx, alg->Cp, eigvals_r, alg->A, alg->B);

        /* If condition check failed: set useOrtho and retry */
        if (2 == useOrtho) {
            useOrtho = 1;
            /* Re-orthogonalize W */
            if (1 == alg->iter) {
                FN(ortho_drop)(size, sizeSub, sizeSub, eps_ortho, eps_drop,
                               W, X, alg->wrk1, alg->wrk2, alg->wrk3, alg->B);
            } else {
                memcpy(alg->wrk4, X, size * sizeSub * sizeof(CTYPE));
                memcpy(alg->wrk4 + size * sizeSub, P, size * sizeSub * sizeof(CTYPE));
                FN(ortho_drop)(size, sizeSub, 2 * sizeSub, eps_ortho, eps_drop,
                               W, alg->wrk4, alg->wrk1, alg->wrk2, alg->wrk3, alg->B);
            }
            /* Reassemble S */
            FN(setup_S)(size, sizeSub, alg->iter, X, P, W, alg->S);
            /* Retry RR with useOrtho=1 */
            FN(rayleigh_ritz_modified)(size, sizeSub, mult, alg->converged, 0, &useOrtho,
                                       alg->S, alg->wrk1, alg->wrk2, alg->wrk3,
                                       alg->Cx, alg->Cp, eigvals_r, alg->A, alg->B);
        }

        /* Copy real eigenvalues into CTYPE eigVals array */
        for (uint64_t i = 0; i < sizeSub; i++) {
#ifdef CTYPE_IS_COMPLEX
            alg->eigVals[i] = eigvals_r[i] + 0*I;
#else
            alg->eigVals[i] = eigvals_r[i];
#endif
        }

        /* Project back: X = S * Cx, P = S * Cp */
        FN(project_back)(size, sizeSub, mult, alg->S, alg->Cx, X, alg->wrk4);
        FN(project_back)(size, sizeSub, mult, alg->S, alg->Cp, P, alg->wrk4);

        /* Compute residual W = A*X - B*X*eigVal */
        FN(get_residual)(size, sizeSub, X, NULL, W, alg->eigVals, alg->wrk1, alg->A, alg->B);

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
            break;  /* All converged */
        }

        alg->iter++;
    }

    safe_free((void**)&eigvals_r);

    /* Final eigenvectors are in X = alg->S */
}

#undef CONCAT2
#undef CONCAT
#undef FN
#undef CABS
#undef CREAL
