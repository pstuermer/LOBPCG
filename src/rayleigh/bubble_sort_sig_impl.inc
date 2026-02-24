/**
 * @file bubble_sort_sig_impl.inc
 * @brief Shared bubble sort for indefinite Rayleigh-Ritz eigenvalue ordering
 *
 * Sorts: positive-signature eigenvalues ascending, then negative descending.
 * Relies on FN, RTYPE macros from the including file.
 */

static void FN(bubble_sort_sig)(RTYPE *eigVal, int8_t *signature,
				uint64_t *indices, uint64_t n) {
  for (uint64_t i = 0; i < n; i++)
    indices[i] = i;

  for (uint64_t i = 0; i < n; i++) {
    int8_t swapped = 0;
    for (uint64_t j = 0; j + 1 < n - i; j++) {
      int8_t siga = signature[j];
      int8_t sigb = signature[j + 1];
      RTYPE reala = eigVal[j];
      RTYPE realb = eigVal[j + 1];
      int8_t should_swap = 0;

      if (siga == 0 && sigb != 0) should_swap = 1;
      if (siga != 0 && sigb == 0) should_swap = 0;
      if (siga > 0 && sigb > 0)  should_swap = (reala > realb);
      if (siga < 0 && sigb < 0)  should_swap = (reala < realb);
      if (siga > 0 && sigb < 0)  should_swap = 0;
      if (siga < 0 && sigb > 0)  should_swap = 1;

      if (should_swap) {
	RTYPE tmp_e = eigVal[j];
	eigVal[j] = eigVal[j + 1];
	eigVal[j + 1] = tmp_e;

	int8_t tmp_s = signature[j];
	signature[j] = signature[j + 1];
	signature[j + 1] = tmp_s;

	uint64_t tmp_i = indices[j];
	indices[j] = indices[j + 1];
	indices[j + 1] = tmp_i;

	swapped = 1;
      }
    }
    if (!swapped) break;
  }
}
