#include <stdio.h>
#include <stdint.h>
#include <assert.h>
#include "lobpcg/memory.h"

int main(void) {
  printf("Testing memory management utilities...\n\n");

  // Test 1: xmalloc alignment
  printf("Test 1: xmalloc 64-byte alignment\n");
  for (size_t size = 1; size <= 1024; size *= 2) {
    void *ptr = xmalloc(size);
    uintptr_t addr = (uintptr_t)ptr;
    assert(addr % 64 == 0);
    printf("  xmalloc(%zu) -> %p (aligned: %s)\n",
           size, ptr, (addr % 64 == 0) ? "YES" : "NO");
    free(ptr);
  }
  printf("  All allocations properly aligned\t\t\t[PASS]\n\n");

  // Test 2: xcalloc alignment and zero-initialization
  printf("Test 2: xcalloc alignment and zero-initialization\n");
  size_t num = 128;
  size_t elem_size = 8;
  double *arr = (double*)xcalloc(num, elem_size);
  uintptr_t addr = (uintptr_t)arr;
  assert(addr % 64 == 0);
  printf("  xcalloc(%zu, %zu) -> %p (aligned: %s)\n",
         num, elem_size, (void*)arr, (addr % 64 == 0) ? "YES" : "NO");

  // Verify zero-initialization
  int all_zero = 1;
  for (size_t i = 0; i < num; i++) {
    if (arr[i] != 0.0) {
      all_zero = 0;
      break;
    }
  }
  printf("  Zero-initialization: %s\n", all_zero ? "YES" : "NO");
  assert(all_zero);
  free(arr);
  printf("  Allocation aligned and zero-initialized\t\t\t[PASS]\n\n");

  // Test 3: safe_free nullification
  printf("Test 3: safe_free pointer nullification\n");
  double *data = (double*)xmalloc(1024);
  printf("  Before safe_free: data = %p\n", (void*)data);
  safe_free((void**)&data);
  printf("  After safe_free:  data = %p\n", (void*)data);
  assert(data == NULL);
  printf("  Pointer properly nullified\t\t\t[PASS]\n\n");

  // Test 4: safe_free on NULL pointer (should not crash)
  printf("Test 4: safe_free on NULL pointer\n");
  double *null_ptr = NULL;
  safe_free((void**)&null_ptr);
  printf("  safe_free(NULL) does not crash\t\t\t[PASS]\n\n");

  // Test 5: xcalloc signature matches standard calloc
  printf("Test 5: xcalloc signature (num, size)\n");
  float *matrix = (float*)xcalloc(100, sizeof(float));
  uintptr_t mat_addr = (uintptr_t)matrix;
  assert(mat_addr % 64 == 0);
  printf("  xcalloc(100, sizeof(float)) -> %p (aligned: YES)\n", (void*)matrix);
  safe_free((void**)&matrix);
  printf("  Two-argument signature works correctly\t\t\t[PASS]\n\n");

  printf("All memory management tests passed! \n");
  return 0;
}
