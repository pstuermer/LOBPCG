#ifndef LINOP_H
#define LINOP_H

#include "lobpcg/types.h"
#include "lobpcg/memory.h"

typedef struct {
  void *data;
  size_t data_size;
} linop_ctx_t;

// Generic LinearOperator structure
#define DEFINE_LINEAR_OPERATOR(type, suffix)				\
  typedef struct LinearOperator_##suffix##_t LinearOperator_##suffix##_t; \
  typedef void (*matvec_func_##suffix##_t)(const LinearOperator_##suffix##_t *op, \
					   type *restrict x,	\
					   type *restrict y);		\
  typedef void (*cleanup_func_##suffix##_t)(linop_ctx_t *ctx);	\
									\
  struct LinearOperator_##suffix##_t {					\
    uint64_t rows;							\
    uint64_t cols;							\
    matvec_func_##suffix##_t matvec;					\
    cleanup_func_##suffix##_t cleanup;					\
    linop_ctx_t *ctx;						\
  };									\
									\
  static inline LinearOperator_##suffix##_t *linop_create_##suffix(uint64_t rows, uint64_t cols, \
						 matvec_func_##suffix##_t matvec, \
						 cleanup_func_##suffix##_t cleanup, \
						 linop_ctx_t *ctx) { \
									\
    LinearOperator_##suffix##_t *op = xcalloc(1, sizeof(LinearOperator_##suffix##_t)); \
    op->rows = rows;							\
    op->cols = cols;							\
    op->matvec = matvec;						\
    op->cleanup = cleanup;						\
    op->ctx = ctx;							\
    return op;								\
  }									\
									\
  static inline void linop_destroy_##suffix(LinearOperator_##suffix##_t **op) { \
    if (op && *op) {							\
      if ((*op)->cleanup && (*op)->ctx) (*op)->cleanup((*op)->ctx);	\
      safe_free((void**)op);						\
    }									\
  }									\
									\
  static inline void linop_apply_##suffix(const LinearOperator_##suffix##_t *op,	\
			    type *restrict x,				\
			    type *restrict y) {				\
    op->matvec(op, x, y);						\
  }

// Create instances for each type
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
    LinearOperator_s_t*: linop_apply_s, \
    LinearOperator_d_t*: linop_apply_d, \
    LinearOperator_c_t*: linop_apply_c, \
    LinearOperator_z_t*: linop_apply_z \
				       )(op, x, y)

#define linop_destroy(op) _Generic((op), \
    LinearOperator_s_t**: linop_destroy_s, \
    LinearOperator_d_t**: linop_destroy_d, \
    LinearOperator_c_t**: linop_destroy_c, \
    LinearOperator_z_t**: linop_destroy_z \
				   )(op)

#endif // LINOP_H
