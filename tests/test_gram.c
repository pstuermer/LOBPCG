/**
 * @file test_gram.c
 * @brief Unit tests for Gram matrix helpers (apply_block_op, gram_self, gram_cross)
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <complex.h>
#include "lobpcg/types.h"
#include "lobpcg.h"
#include "lobpcg/blas_wrapper.h"
#include "lobpcg/linop.h"
#include "lobpcg/memory.h"

#define TOL_F64 1e-12
#define TOL_F32 1e-5

#include "test_macros.h"

/* --- Diagonal operator for testing B != NULL --- */
typedef struct {
    uint64_t n;
    f64 *diag;
} diag_ctx_d;

static void diag_matvec_d(const struct LinearOperator_d_t *op, f64 *restrict x, f64 *restrict y) {
    diag_ctx_d *ctx = (diag_ctx_d *)op->ctx->data;
    for (uint64_t i = 0; i < ctx->n; i++)
        y[i] = ctx->diag[i] * x[i];
}

typedef struct {
    uint64_t n;
    c64 *diag;
} diag_ctx_z;

static void diag_matvec_z(const struct LinearOperator_z_t *op, c64 *restrict x, c64 *restrict y) {
    diag_ctx_z *ctx = (diag_ctx_z *)op->ctx->data;
    for (uint64_t i = 0; i < ctx->n; i++)
        y[i] = ctx->diag[i] * x[i];
}

/* ============================================================
 * Tests: apply_block_op
 * ============================================================ */

TEST(apply_block_op_d_identity) {
    /* Test apply_block_op with a 2x diagonal operator (all ones = identity) */
    const uint64_t n = 4, k = 2;
    f64 *X = xcalloc(n * k, sizeof(f64));
    f64 *Y = xcalloc(n * k, sizeof(f64));

    /* X = [[1,2],[3,4],[5,6],[7,8]] column-major */
    for (uint64_t i = 0; i < n * k; i++) X[i] = (f64)(i + 1);

    f64 diag_vals[] = {1.0, 1.0, 1.0, 1.0};
    diag_ctx_d dctx = { .n = n, .diag = diag_vals };
    linop_ctx_t lctx = { .data = &dctx, .data_size = sizeof(dctx) };
    LinearOperator_d_t op = { .rows = n, .cols = n, .matvec = diag_matvec_d,
                              .cleanup = NULL, .ctx = &lctx };

    d_apply_block_op(&op, X, Y, n, k);

    for (uint64_t i = 0; i < n * k; i++)
        ASSERT_NEAR(Y[i], X[i], TOL_F64);

    safe_free((void**)&X);
    safe_free((void**)&Y);
}

TEST(apply_block_op_d_scaling) {
    const uint64_t n = 3, k = 2;
    f64 *X = xcalloc(n * k, sizeof(f64));
    f64 *Y = xcalloc(n * k, sizeof(f64));

    for (uint64_t i = 0; i < n * k; i++) X[i] = 1.0;

    f64 diag_vals[] = {2.0, 3.0, 4.0};
    diag_ctx_d dctx = { .n = n, .diag = diag_vals };
    linop_ctx_t lctx = { .data = &dctx, .data_size = sizeof(dctx) };
    LinearOperator_d_t op = { .rows = n, .cols = n, .matvec = diag_matvec_d,
                              .cleanup = NULL, .ctx = &lctx };

    d_apply_block_op(&op, X, Y, n, k);

    /* Each column should be [2,3,4] */
    for (uint64_t j = 0; j < k; j++) {
        ASSERT_NEAR(Y[0 + j*n], 2.0, TOL_F64);
        ASSERT_NEAR(Y[1 + j*n], 3.0, TOL_F64);
        ASSERT_NEAR(Y[2 + j*n], 4.0, TOL_F64);
    }

    safe_free((void**)&X);
    safe_free((void**)&Y);
}

/* ============================================================
 * Tests: gram_self
 * ============================================================ */

TEST(gram_self_d_identity_B) {
    /* G = U^T * U with B=NULL, U = orthonormal columns */
    const uint64_t n = 4, k = 2;
    f64 *U = xcalloc(n * k, sizeof(f64));
    f64 *G = xcalloc(k * k, sizeof(f64));
    f64 *wrk = xcalloc(n * k, sizeof(f64));

    /* U = [e1, e2] (orthonormal) */
    U[0] = 1.0; /* col 0 */
    U[1 + 1*n] = 1.0; /* col 1 */

    d_gram_self(U, n, k, NULL, G, k, wrk);

    /* Should be identity (upper triangle) */
    ASSERT_NEAR(G[0], 1.0, TOL_F64); /* G[0,0] */
    ASSERT_NEAR(G[1], 0.0, TOL_F64); /* G[1,0] - lower, may not be set */
    ASSERT_NEAR(G[0 + 1*k], 0.0, TOL_F64); /* G[0,1] */
    ASSERT_NEAR(G[1 + 1*k], 1.0, TOL_F64); /* G[1,1] */

    safe_free((void**)&U);
    safe_free((void**)&G);
    safe_free((void**)&wrk);
}

TEST(gram_self_d_with_B) {
    /* G = U^T * B * U with diagonal B = diag(2,3,4,5) */
    const uint64_t n = 4, k = 2;
    f64 *U = xcalloc(n * k, sizeof(f64));
    f64 *G = xcalloc(k * k, sizeof(f64));
    f64 *wrk = xcalloc(n * k, sizeof(f64));

    /* U = [e1, e2] */
    U[0] = 1.0;
    U[1 + 1*n] = 1.0;

    f64 diag_vals[] = {2.0, 3.0, 4.0, 5.0};
    diag_ctx_d dctx = { .n = n, .diag = diag_vals };
    linop_ctx_t lctx = { .data = &dctx, .data_size = sizeof(dctx) };
    LinearOperator_d_t B = { .rows = n, .cols = n, .matvec = diag_matvec_d,
                             .cleanup = NULL, .ctx = &lctx };

    d_gram_self(U, n, k, &B, G, k, wrk);

    /* G[0,0] = e1^T * B * e1 = 2, G[1,1] = e2^T * B * e2 = 3 */
    ASSERT_NEAR(G[0], 2.0, TOL_F64);
    ASSERT_NEAR(G[0 + 1*k], 0.0, TOL_F64);
    ASSERT_NEAR(G[1 + 1*k], 3.0, TOL_F64);

    safe_free((void**)&U);
    safe_free((void**)&G);
    safe_free((void**)&wrk);
}

TEST(gram_self_d_general) {
    /* Dense matrix, verify G = U^T * U manually */
    const uint64_t n = 3, k = 2;
    f64 *U = xcalloc(n * k, sizeof(f64));
    f64 *G = xcalloc(k * k, sizeof(f64));
    f64 *wrk = xcalloc(n * k, sizeof(f64));

    /* U = [[1,0],[0,1],[1,1]] column-major */
    U[0] = 1.0; U[1] = 0.0; U[2] = 1.0; /* col 0 */
    U[3] = 0.0; U[4] = 1.0; U[5] = 1.0; /* col 1 */

    d_gram_self(U, n, k, NULL, G, k, wrk);

    /* G[0,0] = 1^2 + 0^2 + 1^2 = 2 */
    ASSERT_NEAR(G[0], 2.0, TOL_F64);
    /* G[0,1] = 1*0 + 0*1 + 1*1 = 1 */
    ASSERT_NEAR(G[0 + 1*k], 1.0, TOL_F64);
    /* G[1,1] = 0^2 + 1^2 + 1^2 = 2 */
    ASSERT_NEAR(G[1 + 1*k], 2.0, TOL_F64);

    safe_free((void**)&U);
    safe_free((void**)&G);
    safe_free((void**)&wrk);
}

TEST(gram_self_z_identity_B) {
    /* G = U^H * U with B=NULL */
    const uint64_t n = 3, k = 2;
    c64 *U = xcalloc(n * k, sizeof(c64));
    c64 *G = xcalloc(k * k, sizeof(c64));
    c64 *wrk = xcalloc(n * k, sizeof(c64));

    /* U = [e1, e2] orthonormal */
    U[0] = 1.0;
    U[1 + 1*n] = 1.0;

    z_gram_self(U, n, k, NULL, G, k, wrk);

    ASSERT_NEAR(cabs(G[0] - 1.0), 0.0, TOL_F64);
    ASSERT_NEAR(cabs(G[0 + 1*k]), 0.0, TOL_F64);
    ASSERT_NEAR(cabs(G[1 + 1*k] - 1.0), 0.0, TOL_F64);

    safe_free((void**)&U);
    safe_free((void**)&G);
    safe_free((void**)&wrk);
}

TEST(gram_self_z_with_B) {
    /* G = U^H * B * U with diagonal B */
    const uint64_t n = 3, k = 2;
    c64 *U = xcalloc(n * k, sizeof(c64));
    c64 *G = xcalloc(k * k, sizeof(c64));
    c64 *wrk = xcalloc(n * k, sizeof(c64));

    /* U = [[1+i, 0], [0, 1-i], [0, 0]] */
    U[0] = 1.0 + 1.0*I;
    U[1 + 1*n] = 1.0 - 1.0*I;

    c64 diag_vals[] = {2.0, 3.0, 4.0};
    diag_ctx_z dctx = { .n = n, .diag = diag_vals };
    linop_ctx_t lctx = { .data = &dctx, .data_size = sizeof(dctx) };
    LinearOperator_z_t B = { .rows = n, .cols = n, .matvec = diag_matvec_z,
                             .cleanup = NULL, .ctx = &lctx };

    z_gram_self(U, n, k, &B, G, k, wrk);

    /* G[0,0] = conj(1+i)*2*(1+i) = (1-i)*2*(1+i) = 2*(1+1) = 4 */
    ASSERT_NEAR(cabs(G[0] - 4.0), 0.0, TOL_F64);
    /* G[0,1] = 0 (orthogonal columns) */
    ASSERT_NEAR(cabs(G[0 + 1*k]), 0.0, TOL_F64);
    /* G[1,1] = conj(1-i)*3*(1-i) = (1+i)*3*(1-i) = 3*2 = 6 */
    ASSERT_NEAR(cabs(G[1 + 1*k] - 6.0), 0.0, TOL_F64);

    safe_free((void**)&U);
    safe_free((void**)&G);
    safe_free((void**)&wrk);
}

/* ============================================================
 * Tests: gram_cross
 * ============================================================ */

TEST(gram_cross_d_identity_B) {
    /* G = V^T * U with B=NULL */
    const uint64_t n = 4, nv = 2, nu = 3;
    f64 *V = xcalloc(n * nv, sizeof(f64));
    f64 *U = xcalloc(n * nu, sizeof(f64));
    f64 *G = xcalloc(nv * nu, sizeof(f64));
    f64 *wrk = xcalloc(n * nu, sizeof(f64));

    /* V = [e1, e2], U = [e1, e3, e4] */
    V[0] = 1.0;
    V[1 + 1*n] = 1.0;
    U[0] = 1.0;
    U[2 + 1*n] = 1.0;
    U[3 + 2*n] = 1.0;

    d_gram_cross(V, nv, U, nu, n, NULL, G, nv, wrk);

    /* G = [[1,0,0],[0,0,0]] */
    ASSERT_NEAR(G[0], 1.0, TOL_F64);
    ASSERT_NEAR(G[1], 0.0, TOL_F64);
    ASSERT_NEAR(G[0 + 1*nv], 0.0, TOL_F64);
    ASSERT_NEAR(G[1 + 1*nv], 0.0, TOL_F64);
    ASSERT_NEAR(G[0 + 2*nv], 0.0, TOL_F64);
    ASSERT_NEAR(G[1 + 2*nv], 0.0, TOL_F64);

    safe_free((void**)&V);
    safe_free((void**)&U);
    safe_free((void**)&G);
    safe_free((void**)&wrk);
}

TEST(gram_cross_d_with_B) {
    /* G = V^T * B * U with diagonal B */
    const uint64_t n = 3, nv = 2, nu = 2;
    f64 *V = xcalloc(n * nv, sizeof(f64));
    f64 *U = xcalloc(n * nu, sizeof(f64));
    f64 *G = xcalloc(nv * nu, sizeof(f64));
    f64 *wrk = xcalloc(n * nu, sizeof(f64));

    /* V = [e1, e2], U = [e1, e2] */
    V[0] = 1.0; V[1 + 1*n] = 1.0;
    U[0] = 1.0; U[1 + 1*n] = 1.0;

    f64 diag_vals[] = {2.0, 3.0, 4.0};
    diag_ctx_d dctx = { .n = n, .diag = diag_vals };
    linop_ctx_t lctx = { .data = &dctx, .data_size = sizeof(dctx) };
    LinearOperator_d_t B = { .rows = n, .cols = n, .matvec = diag_matvec_d,
                             .cleanup = NULL, .ctx = &lctx };

    d_gram_cross(V, nv, U, nu, n, &B, G, nv, wrk);

    /* G = diag(2,3) */
    ASSERT_NEAR(G[0], 2.0, TOL_F64);
    ASSERT_NEAR(G[1], 0.0, TOL_F64);
    ASSERT_NEAR(G[0 + 1*nv], 0.0, TOL_F64);
    ASSERT_NEAR(G[1 + 1*nv], 3.0, TOL_F64);

    safe_free((void**)&V);
    safe_free((void**)&U);
    safe_free((void**)&G);
    safe_free((void**)&wrk);
}

TEST(gram_cross_d_rectangular) {
    /* Test non-square cross-Gram: V^T * U with dense matrices */
    const uint64_t n = 3, nv = 1, nu = 2;
    f64 *V = xcalloc(n * nv, sizeof(f64));
    f64 *U = xcalloc(n * nu, sizeof(f64));
    f64 *G = xcalloc(nv * nu, sizeof(f64));
    f64 *wrk = xcalloc(n * nu, sizeof(f64));

    /* V = [1,1,1]^T, U = [[1,0],[0,1],[0,0]] */
    V[0] = 1.0; V[1] = 1.0; V[2] = 1.0;
    U[0] = 1.0; U[1 + 1*n] = 1.0;

    d_gram_cross(V, nv, U, nu, n, NULL, G, nv, wrk);

    /* G = [1, 1] (1x2) */
    ASSERT_NEAR(G[0], 1.0, TOL_F64);
    ASSERT_NEAR(G[0 + 1*nv], 1.0, TOL_F64);

    safe_free((void**)&V);
    safe_free((void**)&U);
    safe_free((void**)&G);
    safe_free((void**)&wrk);
}

TEST(gram_cross_z_identity_B) {
    /* G = V^H * U with B=NULL, complex case */
    const uint64_t n = 3, nv = 2, nu = 2;
    c64 *V = xcalloc(n * nv, sizeof(c64));
    c64 *U = xcalloc(n * nu, sizeof(c64));
    c64 *G = xcalloc(nv * nu, sizeof(c64));
    c64 *wrk = xcalloc(n * nu, sizeof(c64));

    /* V = [[1+i, 0], [0, 1], [0, 0]], U = [[1, 0], [0, 1-i], [0, 0]] */
    V[0] = 1.0 + 1.0*I;
    V[1 + 1*n] = 1.0;
    U[0] = 1.0;
    U[1 + 1*n] = 1.0 - 1.0*I;

    z_gram_cross(V, nv, U, nu, n, NULL, G, nv, wrk);

    /* G[0,0] = conj(1+i)*1 = 1-i */
    ASSERT_NEAR(cabs(G[0] - (1.0 - 1.0*I)), 0.0, TOL_F64);
    /* G[0,1] = 0 */
    ASSERT_NEAR(cabs(G[0 + 1*nv]), 0.0, TOL_F64);
    /* G[1,0] = 0 */
    ASSERT_NEAR(cabs(G[1]), 0.0, TOL_F64);
    /* G[1,1] = conj(1)*(1-i) = 1-i */
    ASSERT_NEAR(cabs(G[1 + 1*nv] - (1.0 - 1.0*I)), 0.0, TOL_F64);

    safe_free((void**)&V);
    safe_free((void**)&U);
    safe_free((void**)&G);
    safe_free((void**)&wrk);
}

int main(void) {
    printf("=== Gram helper tests ===\n");

    printf("\napply_block_op:\n");
    RUN(apply_block_op_d_identity);
    RUN(apply_block_op_d_scaling);

    printf("\ngram_self:\n");
    RUN(gram_self_d_identity_B);
    RUN(gram_self_d_with_B);
    RUN(gram_self_d_general);
    RUN(gram_self_z_identity_B);
    RUN(gram_self_z_with_B);

    printf("\ngram_cross:\n");
    RUN(gram_cross_d_identity_B);
    RUN(gram_cross_d_with_B);
    RUN(gram_cross_d_rectangular);
    RUN(gram_cross_z_identity_B);

    printf("\n%d passed, %d failed\n", tests_passed, tests_failed);
    return tests_failed ? 1 : 0;
}
