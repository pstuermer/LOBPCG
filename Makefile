# LOBPCG Makefile
CC = gcc
CFLAGS = -std=c11 -O3 -march=native -fopenmp -fPIC -Wall -Wextra

# BLAS backend: MKL (default), OPENBLAS, or BLIS
BLAS_BACKEND ?= MKL

ifeq ($(BLAS_BACKEND),MKL)
  BLAS_INC = $(MKLROOT)/include
  BLAS_LIB = $(MKLROOT)/lib/intel64
  BLAS_LINK = -L$(BLAS_LIB) -Wl,-rpath,$(BLAS_LIB) \
              -lmkl_intel_lp64 -lmkl_gnu_thread -lmkl_core -lgomp -lpthread -lm
  CFLAGS += -DUSE_MKL -Wno-incompatible-pointer-types
else ifeq ($(BLAS_BACKEND),OPENBLAS)
  BLAS_INC = /usr/include
  BLAS_LIB = /usr/lib
  BLAS_LINK = -lopenblas -lpthread -lm
endif

INCLUDES = -I. -Iinclude -Iinclude/lobpcg -I$(BLAS_INC)
LDFLAGS = $(BLAS_LINK)

# Sources
SRC = $(wildcard src/*.c src/**/*.c)
OBJ = $(patsubst %.c,%.o,$(SRC))

# Auto-discover tests
TEST_SRC = $(wildcard tests/test_*.c) tests/linop_test.c
TESTS = $(patsubst tests/%.c,build/%.ex,$(TEST_SRC))

.PHONY: all lib tests run-tests clean

all: lib tests

lib: build build/liblobpcg.a

tests: build $(TESTS)

build:
	@mkdir -p build

%.o: %.c
	$(CC) $(CFLAGS) $(INCLUDES) -c $< -o $@

build/liblobpcg.a: $(OBJ)
	ar rcs $@ $(OBJ) 2>/dev/null || ar rcs $@

# Generic test rule: compile test .c and link against library
build/%.ex: tests/%.c build/liblobpcg.a
	$(CC) $(CFLAGS) $(INCLUDES) $< build/liblobpcg.a -o $@ $(LDFLAGS)

run-tests: tests
	@for t in $(TESTS); do echo ">>> $$t"; $$t && echo "[PASS]" || echo "[FAIL]"; done

clean:
	rm -rf build $(OBJ) *~
