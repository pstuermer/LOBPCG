/**
 * @file svqb_impl.inc
 * @brief Type-generic SVQB orthogonalization
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

/* Fill lower triangle from upper for symmetric/Hermitian matrix */
static void FN(fill_lower)(uint64_t n, CTYPE *A) {
    for (uint64_t j = 0; j < n; j++)
        for (uint64_t i = j + 1; i < n; i++)
#ifdef CTYPE_IS_COMPLEX
            A[i + j*n] = conj(A[j + i*n]);
#else
            A[i + j*n] = A[j + i*n];
#endif
}

uint64_t FN(svqb)(const uint64_t m, const uint64_t n, const RTYPE tau,
                  const char drop, CTYPE *restrict U,
                  CTYPE *restrict wrk1, CTYPE *restrict wrk2, CTYPE *restrict wrk3,
                  LINOP *B) {
    if (n == 0) return 0;
    (void)drop;  /* drop='y' not implemented */

    RTYPE *eigvals = xcalloc(n, sizeof(RTYPE));
    RTYPE *D = xcalloc(n, sizeof(RTYPE));

    /* 1. Gram matrix G = U^H*B*U in wrk1 */
    memset(wrk1, 0, n*n*sizeof(CTYPE));
    if (B == NULL) {
#ifdef CTYPE_IS_COMPLEX
        FN(herk)(m, n, (RTYPE)1, U, (RTYPE)0, wrk1);
#else
        FN(syrk)(m, n, (RTYPE)1, U, (RTYPE)0, wrk1);
#endif
    } else {
        for (uint64_t j = 0; j < n; j++)
            B->matvec(B, &U[j*m], &wrk2[j*m]);
#ifdef CTYPE_IS_COMPLEX
        FN(gemm_hn)(n, n, m, (CTYPE)1, U, wrk2, (CTYPE)0, wrk1);
#else
	FN(gemm_tn)(n, n, m, (CTYPE)1, U, wrk2, (CTYPE)0, wrk1);
#endif
    }
    FN(fill_lower)(n, wrk1);

    /* 2. Diagonal scaling D_ii = 1/sqrt(|G_ii|) */
    for (uint64_t i = 0; i < n; i++) {
        RTYPE gii = CABS(wrk1[i + i*n]);
        D[i] = (gii > 0) ? (RTYPE)1/sqrt(gii) : (RTYPE)1;
    }

    /* 3. Scale: G_scaled = D*G*D */
    for (uint64_t j = 0; j < n; j++)
        for (uint64_t i = 0; i < n; i++)
            wrk1[i + j*n] *= D[i] * D[j];

    /* 4. Eigensolve */
    if (FN(eig)(n, wrk1, eigvals) != 0) {
        safe_free((void**)&eigvals); safe_free((void**)&D); return n;
    }

    /* 5. Threshold */
    RTYPE maxeig = 0;
    for (uint64_t i = 0; i < n; i++)
        if (fabs(eigvals[i]) > maxeig) maxeig = fabs(eigvals[i]);
    RTYPE thresh = tau * maxeig;

    /* 6. Transform: U_new = U * D * V * D_final */
    /*    D_final_ii = 1/sqrt(max(|λ_i|, thresh)) */
    /* T = D*V in wrk2 */
    for (uint64_t j = 0; j < n; j++)
        for (uint64_t i = 0; i < n; i++)
            wrk2[i + j*n] = D[i] * wrk1[i + j*n];

    /* T2 = T * D_final in wrk3 */
    for (uint64_t j = 0; j < n; j++) {
        RTYPE df = (RTYPE)1/sqrt(MAX(fabs(eigvals[j]), thresh));
        for (uint64_t i = 0; i < n; i++)
            wrk3[i + j*n] = wrk2[i + j*n] * df;
    }

    /* U = U * T2 */
    FN(gemm_nn)(m, n, n, (CTYPE)1, U, wrk3, (CTYPE)0, wrk2);
    memcpy(U, wrk2, m*n*sizeof(CTYPE));

    safe_free((void**)&eigvals); safe_free((void**)&D);
    return n;
}

#undef CONCAT2
#undef CONCAT
#undef FN
#undef MAX
#undef CABS
