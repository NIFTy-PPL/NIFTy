#include <nanobind/nanobind.h>

namespace nb = nanobind;

NB_MODULE(_nifty_cl_core, m) {
    m.def("myadd", [](int a, int b) { return a + b; });
}
