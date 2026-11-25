#pragma once
// A minimal test framework without external dependencies.
//   TEST(name) { CHECK(condition); CHECK_NEAR(a, b, tol); CHECK_THROWS(expression); }
#include <cmath>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

struct TestFailure : std::runtime_error {
    using std::runtime_error::runtime_error;
};

struct TestCase {
    const char* name;
    void (*fn)();
};

inline std::vector<TestCase>& test_registry() {
    static std::vector<TestCase> r;
    return r;
}

struct TestRegistrar {
    TestRegistrar(const char* name, void (*fn)()) { test_registry().push_back({name, fn}); }
};

#define TEST(name)                                      \
    static void name();                                 \
    static TestRegistrar registrar_##name(#name, name); \
    static void name()

#define CHECK(cond)                                                          \
    do {                                                                     \
        if (!(cond)) {                                                       \
            std::ostringstream os__;                                         \
            os__ << __FILE__ << ":" << __LINE__ << ": CHECK failed: " #cond; \
            throw TestFailure(os__.str());                                   \
        }                                                                    \
    } while (0)

#define CHECK_NEAR(a, b, tol)                                                                                      \
    do {                                                                                                           \
        double a__ = (a), b__ = (b), t__ = (tol);                                                                  \
        if (!(std::fabs(a__ - b__) <= t__)) {                                                                      \
            std::ostringstream os__;                                                                               \
            os__ << __FILE__ << ":" << __LINE__ << ": " #a " = " << a__ << ", " #b " = " << b__ << " (tol " << t__ \
                 << ")";                                                                                           \
            throw TestFailure(os__.str());                                                                         \
        }                                                                                                          \
    } while (0)

#define CHECK_THROWS(expr)                                                         \
    do {                                                                           \
        bool threw__ = false;                                                      \
        try {                                                                      \
            (void)(expr);                                                          \
        } catch (const std::exception&) {                                          \
            threw__ = true;                                                        \
        }                                                                          \
        if (!threw__) {                                                            \
            std::ostringstream os__;                                               \
            os__ << __FILE__ << ":" << __LINE__ << ": expected exception: " #expr; \
            throw TestFailure(os__.str());                                         \
        }                                                                          \
    } while (0)
