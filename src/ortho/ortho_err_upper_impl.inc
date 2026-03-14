/* Compute ||G-I_sig||_F from upper triangle of Hermitian G */
static RTYPE FN(ortho_err_upper)(uint64_t n, const CTYPE *G) {
  RTYPE sum = 0;
  for (uint64_t j = 0; j < n; j++) {
    RTYPE d = CABS(G[j + j*n]) - (RTYPE)1;
    sum += d*d;
    for (uint64_t i = 0; i < j; i++) {
      RTYPE v = CABS(G[i + j*n]);
      sum += 2*v*v;
    }
  }
  return RSQRT(sum);
}
