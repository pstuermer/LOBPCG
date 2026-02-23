#ifndef LOBPCG_MEMORY_H
#define LOBPCG_MEMORY_H

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>

/* Memory Management Utilities
 *
 * 64-byte aligned allocation functions for SIMD and cache line optimization.
 * All functions check for allocation failure and exit on error.
 *
 */

/**
 * xmalloc - 64-byte aligned malloc
 * @size: Number of bytes to allocate
 *
 * Returns pointer to allocated memory, aligned to 64-byte boundary.
 * Exits with error message on failure.
 *
 * Alignment rationale:
 *  - 64 bytes = cache line size on most modern CPUs
 *  - AVX-512 requires 64-byte alignment for optimal performance
 *  - Prevents cache line splits in BLAS operations
 */
static inline void *xmalloc(size_t size) {
  if (size == 0) {
    fprintf(stderr, "xmalloc: Cannot allocate 0 bytes\n");
    exit(1);
  }

  // Round up to multiple of 64
  size_t aligned_size = (size + 63) & ~(size_t)63;

  void *ptr = aligned_alloc(64, aligned_size);
  if (!ptr) {
    fprintf(stderr, "xmalloc: Allocation of %zu bytes failed\n", size);
    exit(1);
  }

  return ptr;
}

/**
 * xcalloc - 64-byte aligned calloc with overflow checking
 * @num: Number of elements
 * @size: Size of each element in bytes
 *
 * Allocates memory for array of num elements of size bytes each,
 * initialized to zero. Memory is aligned to 64-byte boundary.
 * Exits with error message on failure or overflow.
 *
 * Matches standard calloc(num, size) signature (unlike reference implementation).
 */
static inline void *xcalloc(size_t num, size_t size) {
  if (num == 0 || size == 0) {
    fprintf(stderr, "xcalloc: Cannot allocate 0 elements or 0-sized elements\n");
    exit(1);
  }

  // Check for overflow in multiplication
  if (num > SIZE_MAX / size) {
    fprintf(stderr, "xcalloc: Overflow in size calculation (%zu * %zu)\n", num, size);
    exit(1);
  }

  size_t total_size = num * size;
  void *ptr = xmalloc(total_size);

  // Zero-initialize the memory
  memset(ptr, 0, total_size);

  return ptr;
}

/**
 * safe_free - Free memory and nullify pointer
 * @ptr: Pointer to pointer to free
 *
 * Frees memory pointed to by *ptr and sets *ptr to NULL to prevent
 * use-after-free and double-free bugs.
 *
 * CRITICAL: Takes void** (pointer-to-pointer), not void*.
 *
 * Usage:
 *   double *data = xmalloc(1024);
 *   safe_free((void**)&data);  // data is now NULL
 */
static inline void safe_free(void **ptr) {
  if (ptr && *ptr) {
    free(*ptr);
    *ptr = NULL;
  }
}

#endif // LOBPCG_MEMORY_H
