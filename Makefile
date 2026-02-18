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
TESTS = build/test_blas.ex build/test_memory.ex build/linop_test.ex build/test_svqb.ex build/test_ortho_indefinite.ex \
        build/test_ortho_drop.ex build/test_ortho_randomize.ex build/test_svqb_mat.ex build/test_ortho_randomized_mat.ex \
        build/test_rayleigh_ritz.ex build/test_residual.ex build/test_lobpcg.ex build/test_indefinite_rr.ex \
        build/test_ilobpcg.ex

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

# ortho_drop test
ORTHO_DROP_SRC = src/ortho/ortho_drop_s.c src/ortho/ortho_drop_d.c \
                 src/ortho/ortho_drop_c.c src/ortho/ortho_drop_z.c

build/test_ortho_drop.ex: tests/test_ortho_drop.c $(SVQB_SRC) $(ORTHO_DROP_SRC)
	$(CC) $(CFLAGS) $(INCLUDES) $^ -o $@ $(LDFLAGS)

# ortho_randomize test
ORTHO_RAND_SRC = src/ortho/ortho_randomize_s.c src/ortho/ortho_randomize_d.c \
                 src/ortho/ortho_randomize_c.c src/ortho/ortho_randomize_z.c

build/test_ortho_randomize.ex: tests/test_ortho_randomize.c $(SVQB_SRC) $(ORTHO_RAND_SRC)
	$(CC) $(CFLAGS) $(INCLUDES) $^ -o $@ $(LDFLAGS)

# svqb_mat test
SVQB_MAT_SRC = src/ortho/svqb_mat_s.c src/ortho/svqb_mat_d.c \
               src/ortho/svqb_mat_c.c src/ortho/svqb_mat_z.c

build/test_svqb_mat.ex: tests/test_svqb_mat.c $(SVQB_MAT_SRC)
	$(CC) $(CFLAGS) $(INCLUDES) $^ -o $@ $(LDFLAGS)

# ortho_randomized_mat test
ORTHO_RMAT_SRC = src/ortho/ortho_randomized_mat_s.c src/ortho/ortho_randomized_mat_d.c \
                 src/ortho/ortho_randomized_mat_c.c src/ortho/ortho_randomized_mat_z.c

build/test_ortho_randomized_mat.ex: tests/test_ortho_randomized_mat.c $(SVQB_MAT_SRC) $(ORTHO_RMAT_SRC)
	$(CC) $(CFLAGS) $(INCLUDES) $^ -o $@ $(LDFLAGS)

# rayleigh_ritz test
RAYLEIGH_SRC = src/rayleigh/rayleigh_ritz_s.c src/rayleigh/rayleigh_ritz_d.c \
               src/rayleigh/rayleigh_ritz_c.c src/rayleigh/rayleigh_ritz_z.c
RAYLEIGH_MOD_SRC = src/rayleigh/rayleigh_ritz_modified_s.c src/rayleigh/rayleigh_ritz_modified_d.c \
                   src/rayleigh/rayleigh_ritz_modified_c.c src/rayleigh/rayleigh_ritz_modified_z.c

INDEF_RR_SRC = src/rayleigh/indefinite_rr_s.c src/rayleigh/indefinite_rr_d.c \
               src/rayleigh/indefinite_rr_c.c src/rayleigh/indefinite_rr_z.c
INDEF_RR_MOD_SRC = src/rayleigh/indefinite_rr_modified_s.c src/rayleigh/indefinite_rr_modified_d.c \
                   src/rayleigh/indefinite_rr_modified_c.c src/rayleigh/indefinite_rr_modified_z.c

build/test_rayleigh_ritz.ex: tests/test_rayleigh_ritz.c $(RAYLEIGH_SRC) $(RAYLEIGH_MOD_SRC)
	$(CC) $(CFLAGS) $(INCLUDES) $^ -o $@ $(LDFLAGS)

# indefinite rayleigh_ritz test
build/test_indefinite_rr.ex: tests/test_indefinite_rr.c $(INDEF_RR_SRC) $(INDEF_RR_MOD_SRC)
	$(CC) $(CFLAGS) $(INCLUDES) $^ -o $@ $(LDFLAGS)

# residual test
RESIDUAL_SRC = src/residual/residual_s.c src/residual/residual_d.c \
               src/residual/residual_c.c src/residual/residual_z.c

build/test_residual.ex: tests/test_residual.c $(RESIDUAL_SRC)
	$(CC) $(CFLAGS) $(INCLUDES) $^ -o $@ $(LDFLAGS)

# LOBPCG integration test
LOBPCG_CORE_SRC = src/core/lobpcg_s.c src/core/lobpcg_d.c \
                  src/core/lobpcg_c.c src/core/lobpcg_z.c

build/test_lobpcg.ex: tests/test_lobpcg.c $(LOBPCG_CORE_SRC) $(RAYLEIGH_SRC) $(RAYLEIGH_MOD_SRC) \
                      $(RESIDUAL_SRC) $(ORTHO_DROP_SRC) $(SVQB_SRC)
	$(CC) $(CFLAGS) $(INCLUDES) $^ -o $@ $(LDFLAGS)

# iLOBPCG integration test
ILOBPCG_CORE_SRC = src/core/ilobpcg_s.c src/core/ilobpcg_d.c \
                   src/core/ilobpcg_c.c src/core/ilobpcg_z.c

build/test_ilobpcg.ex: tests/test_ilobpcg.c $(ILOBPCG_CORE_SRC) $(INDEF_RR_SRC) $(INDEF_RR_MOD_SRC) \
                       $(RESIDUAL_SRC) $(ORTHO_INDEF_SRC) $(SVQB_SRC)
	$(CC) $(CFLAGS) $(INCLUDES) $^ -o $@ $(LDFLAGS)

run-tests: tests
	@for t in $(TESTS); do echo ">>> $$t"; $$t && echo "[PASS]" || echo "[FAIL]"; done

clean:
	rm -rf build $(OBJ) *~
