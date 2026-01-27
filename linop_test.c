#include "linop.h"

// do the other test cases as well for complex and float

typedef struct {
  uint64_t n;
} IdentityOpData_d;

void cleanup_identity_d(linop_ctx_t *ctx) {
  if (ctx && ctx->data) {
    free( ctx->data );
    ctx->data = NULL;
  }
  free( ctx );
  ctx = NULL;
}

void identity_matvec_d(const LinearOperator_d_t *op,
				  f64 *restrict x,
				  f64 *restrict y) {
  IdentityOpData_d *data = (IdentityOpData_d *)op->ctx->data;
  if (y) {
    for (uint64_t i = 0; i < data->n; i++)
      y[i] = x[i];
  
  } else {
    printf("In-place identity operation (y=NULL)\n");
  }
}

LinearOperator_d_t *create_identity_d(uint64_t n) {
  linop_ctx_t *ctx = xcalloc(1, sizeof(linop_ctx_t));
  IdentityOpData_d *data = xcalloc(1, sizeof(IdentityOpData_d));
  data->n = n;

  ctx->data = data;
  ctx->data_size = sizeof(IdentityOpData_d);

  return linop_create(n, n,
		      (matvec_func_d_t)identity_matvec_d,
		      (cleanup_func_d_t)cleanup_identity_d,
		      ctx);
}

// Utility function declarations
void print_vector_d(const f64 *v, uint64_t n, const char *name) {
  printf("%s = [", name);
  for (uint64_t i = 0; i < (n>6 ? 4 : n); i++) {
    printf("%.3f", v[i]);
    if (i < n - 1) printf(", ");
  }
  if (n > 6) printf("...]\n"); else printf("]\n");
}

// --- Test Functions ---

void test_identity_operator() {
  printf("\n--- Testing Identity Operator (double) ---\n");

  uint64_t n = 5;
  LinearOperator_d_t *ident = create_identity_d(n);

  f64 x[] = {1.0, 2.0, 3.0, 4.0, 5.0};
  f64 y[5];

  printf("Operator dimensions %zu x %zu\n", ident->rows, ident->cols);

  linop_apply(ident, x, y);

  print_vector_d(x, n, "x");
  print_vector_d(y, n, "I*x");

  linop_destroy( ident );
}


int main() {
  test_identity_operator();

  return 0;
}
