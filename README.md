 LOBPCG

Robust and efficient implementations of the Locally Optimal Block Preconditioned Conjugate Gradient (LOBPCG) method and its variations for solving large-scale eigenvalue problems in C.

## Overview

This library provides type-generic, type-safe implementations of several LOBPCG variants for symmetric eigenvalue problems. It is designed for large-scale problems where only a few extremal eigenvalues and eigenvectors are needed, and has been tested on systems with matrix dimensions up to 4 million, computing 150+ eigenvalues.

On problems arising in quantum many-body physics (Bogoliubov-de Gennes equations for dipolar supersolids), it achieves speedups of up to 100x compared to ARPACK/Implicit Restarted Arnoldi.

### Features

- **Type-generic:** Supports `float`, `double`, `complex float`, and `complex double` through C11 `_Generic` selection. The compiler selects the correct specialization automatically.
- **Flexible operator interface:** The solver takes up to three user-supplied linear operators: the system matrix A, an optional stiffness matrix B, and an optional preconditioner T. Dense, sparse, and matrix-free operators are all supported through the same interface.
- **Preconditioning support:** User-supplied preconditioners via the linear operator interface. Built-in preconditioning schemes (randomised Nystrom and others) are planned.

## Dependencies

- C11 compiler with OpenMP support
- BLAS and LAPACK (currently Intel MKL; OpenBLAS and BLIS support planned)

## Building

```bash
make
```

This builds the library and the unit test suite. Test binaries are placed in `build/`.

## Running Tests

```bash
./build/
```

The test suite includes standard validation problems such as Laplacian eigenvalues without preconditioning.

## Project Structure

```
├── src/        # Source files
├── include/    # Header files
├── tests/      # Unit tests
├── Makefile    # Build system (CMake migration planned)

## Theoretical Background

The implementation is based on:

- J. A. Duersch and M. Ye, [A Robust and Efficient Implementation of LOBPCG](https://epubs.siam.org/doi/10.1137/17M1129830), SIAM J. Sci. Comput. 40(5), 2018
- D. Kressner, M. M. Pandur, M. Shao [An indefinite variant of LOBPCG for definite matrix pencils](https://doi.org/10.1007/s11075-013-9754-3), Numer. Algor. 66, 681–703 (2014)

## Related Publications

This solver was developed for computing excitation spectra of dipolar supersolids. A separate physics interface for the Bogoliubov-de Gennes equations in ultracold Bose gases is under development.

## Citing

If you use this software, please cite:

> Paper in preparation. Citation information will be updated upon publication.

## License

Licensed under the BSD 3-Clause License. See [LICENSE](LICENSE) for details.
