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

# Tests
TESTS = build/test_blas.ex build/test_memory.ex build/linop_test.ex build/test_svqb.ex build/test_ortho_indefinite.ex

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

build/test_blas.ex: tests/test_blas.c
	$(CC) $(CFLAGS) $(INCLUDES) $< -o $@ $(LDFLAGS)

build/test_memory.ex: tests/test_memory.c
	$(CC) $(CFLAGS) $(INCLUDES) $< -o $@

build/linop_test.ex: linop_test.c
	$(CC) $(CFLAGS) $(INCLUDES) $< -o $@ $(LDFLAGS)

build/test_svqb.ex: tests/test_svqb.c src/ortho/svqb_s.c src/ortho/svqb_d.c src/ortho/svqb_c.c src/ortho/svqb_z.c
	$(CC) $(CFLAGS) $(INCLUDES) $^ -o $@ $(LDFLAGS)

# SVQB sources needed for ortho_indefinite
SVQB_SRC = src/ortho/svqb_s.c src/ortho/svqb_d.c src/ortho/svqb_c.c src/ortho/svqb_z.c
ORTHO_INDEF_SRC = src/ortho/ortho_indefinite_s.c src/ortho/ortho_indefinite_d.c \
                  src/ortho/ortho_indefinite_c.c src/ortho/ortho_indefinite_z.c

build/test_ortho_indefinite.ex: tests/test_ortho_indefinite.c $(SVQB_SRC) $(ORTHO_INDEF_SRC)
	$(CC) $(CFLAGS) $(INCLUDES) $^ -o $@ $(LDFLAGS)

run-tests: tests
	@for t in $(TESTS); do echo ">>> $$t"; $$t && echo "[PASS]" || echo "[FAIL]"; done

clean:
	rm -rf build $(OBJ) *~
