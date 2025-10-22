#ifndef LINOP_H
#define LINOP_H

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <complex.h>
#include <stdint.h>
//#include <cblas.h>

typedef float f32;
typedef double f64;
typedef float complex c32;
typedef double complex c64;

// forward declare
//struct linop_ctx_t;

typedef struct {
  void *data;
  size_t data_size;
} linop_ctx_t;

void *xcalloc(size_t size) {
  void *ptr = calloc(1, size);
  if (!ptr) {
    fprintf(stderr, "Memory allocation failed.\n");
    exit(1);
  }
  return ptr;
}

void safe_free(void *ptr) {
  if (ptr) {
    free(*(void**)ptr);
    *(void**)ptr = NULL;
  }
}

// Generic LinearOperator structure
#define DEFINE_LINEAR_OPERATOR(type, suffix)				\
  typedef struct LinearOperator_##suffix LinearOperator_##suffix;	\
  typedef void (*matvec_func_##suffix##_t)(const LinearOperator_##suffix *op, \
					   type *restrict x,	\
					   type *restrict y);		\
  typedef void (*cleanup_func_##suffix##_t)(linop_ctx_t *ctx);	\
									\
  struct LinearOperator_##suffix {					\
    uint64_t rows;							\
    uint64_t cols;							\
    matvec_func_##suffix##_t matvec;					\
    cleanup_func_##suffix##_t cleanup;					\
    linop_ctx_t *ctx;						\
  };									\
									\
  LinearOperator_##suffix *linop_create_##suffix(uint64_t rows, uint64_t cols, \
						 matvec_func_##suffix##_t matvec, \
						 cleanup_func_##suffix##_t cleanup, \
						 linop_ctx_t *ctx) { \
									\
    LinearOperator_##suffix *op = calloc(1, sizeof(LinearOperator_##suffix)); \
    op->rows = rows;							\
    op->cols = cols;							\
    op->matvec = matvec;						\
    op->cleanup = cleanup;						\
    op->ctx = ctx;							\
    return op;								\
  }									\
									\
  void linop_destroy_##suffix(LinearOperator_##suffix* op) {		\
    if (op->cleanup && op->ctx) op->cleanup(op->ctx);			\
    free( op );								\
  }									\
									\
  void linop_apply_##suffix(const LinearOperator_##suffix *op,		\
			    type *restrict x,				\
			    type *restrict y) {				\
    op->matvec(op, x, y);						\
  }

// Create instances for each type
// maybe later use as conditional compliation
DEFINE_LINEAR_OPERATOR(f32, s)
DEFINE_LINEAR_OPERATOR(f64, d)
DEFINE_LINEAR_OPERATOR(c32, c)
DEFINE_LINEAR_OPERATOR(c64, z)

// Generic interface using C11
#define linop_create(rows, cols, matvec, cleanup, ctx) _Generic((matvec),\
    matvec_func_s_t: linop_create_s, \
    matvec_func_d_t: linop_create_d, \
    matvec_func_c_t: linop_create_c, \
    matvec_func_z_t: linop_create_z \
								)(rows, cols, matvec, cleanup, ctx)

#define linop_apply(op, x, y) _Generic((op), \
    LinearOperator_s*: linop_apply_s, \
    LinearOperator_d*: linop_apply_d, \
    LinearOperator_c*: linop_apply_c, \
    LinearOperator_z*: linop_apply_z \
				       )(op, x, y)

#define linop_destroy(op) _Generic((op), \
    LinearOperator_s*: linop_destroy_s, \
    LinearOperator_d*: linop_destroy_d, \
    LinearOperator_c*: linop_destroy_c, \
    LinearOperator_z*: linop_destroy_z \
				   )(op)

#endif // LINOP_H
