#include <cstdio>
#include <cstring>
#include "test_framework.h"

// Usage: tests.exe               - all tests
//        tests.exe <name part>   - only tests whose name contains this part
int main(int argc, char** argv) {
    const char* filter = argc > 1 ? argv[1] : nullptr;
    int passed = 0, failed = 0;
    for (const TestCase& t : test_registry()) {
        if (filter && !std::strstr(t.name, filter)) continue;
        try {
            t.fn();
            std::printf("[ OK ] %s\n", t.name);
            ++passed;
        } catch (const std::exception& e) {
            std::printf("[FAIL] %s\n       %s\n", t.name, e.what());
            ++failed;
        }
    }
    std::printf("\n%d passed, %d failed\n", passed, failed);
    return failed == 0 ? 0 : 1;
}
