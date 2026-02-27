/**
 * @file test_macros.h
 * @brief Shared test macros for the LOBPCG test suite
 */
#ifndef TEST_MACROS_H
#define TEST_MACROS_H

#include <stdio.h>
#include <math.h>

static int tests_passed = 0;
static int tests_failed = 0;

#define TEST(name) static void test_##name(void)

#define RUN(name) do { \
    printf("  %-50s ", #name); \
    const int prev_failed = tests_failed; \
    test_##name(); \
    if (prev_failed == tests_failed) { \
        printf("[PASS]\n"); \
        tests_passed++; \
    } \
} while(0)

#define ASSERT(cond) do { \
    if (!(cond)) { \
        printf("[FAIL] line %d: %s\n", __LINE__, #cond); \
        tests_failed++; \
        return; \
    } \
} while(0)

#define ASSERT_NEAR(a, b, tol) ASSERT(fabs((a) - (b)) < (tol))

#endif
