/**
 * @file estimate_norm_impl.inc
 * @brief Type-generic fill_random and estimate_norm (power iteration)
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

/* Fill array with uniform random values in [-0.5, 0.5] */
void FN(fill_random)(uint64_t n, CTYPE *x) {
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

/* Estimate ||A|| via power iteration (spectral radius approximation) */
RTYPE FN(estimate_norm)(uint64_t size, LINOP *A, CTYPE *wrk1, CTYPE *wrk2) {
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

#undef CONCAT2
#undef CONCAT
#undef FN
