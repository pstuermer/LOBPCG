# Examples Makefile
# Build examples against liblobpcg

CC = gcc
CFLAGS = -std=c11 -O3 -march=native -fopenmp -Wall

# BLAS backend (must match main library)
BLAS_BACKEND ?= MKL

ifeq ($(BLAS_BACKEND),MKL)
  BLAS_LIB = $(MKLROOT)/lib/intel64
  BLAS_LINK = -L$(BLAS_LIB) -Wl,-rpath,$(BLAS_LIB) \
              -lmkl_intel_lp64 -lmkl_gnu_thread -lmkl_core -lgomp -lpthread -lm
else ifeq ($(BLAS_BACKEND),OPENBLAS)
  BLAS_LINK = -lopenblas -lpthread -lm
endif

LOBPCG_INC = ..
LOBPCG_LIB = ../build

INCLUDES = -I$(LOBPCG_INC) -I$(LOBPCG_INC)/include -I$(LOBPCG_INC)/include/lobpcg
LDFLAGS = -L$(LOBPCG_LIB) -Wl,-rpath,$(LOBPCG_LIB) -llobpcg $(BLAS_LINK)

SRC = $(wildcard *.c)
EXE = $(patsubst %.c,%.ex,$(SRC))

.PHONY: all clean

all: $(EXE)

%.ex: %.c
	$(CC) $(CFLAGS) $(INCLUDES) $< -o $@ $(LDFLAGS)

clean:
	rm -f $(EXE)
