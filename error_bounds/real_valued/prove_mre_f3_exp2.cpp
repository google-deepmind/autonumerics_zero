// Prove maximum relative error (MRE) bound for the evolved 2^x approximation f3.
//
// f3 is the best 3-operation program found by evolutionary search:
//   def f3(x):
//     c1 = -8.387819235563974
//     c2 = 1.4427239805682266
//     c3 = -3.4355225277901402
//     x1 = c3 + x
//     x2 = c1 / x1
//     x3 = x2 - c2
//     return x3
//
// i.e., f3(x) = -8.387819235563974 / (-3.4355225277901402 + x) - 1.4427239805682266
//
// This program uses the IBEX interval arithmetic library to rigorously bound
// |(2^x - f3(x)) / 2^x| on [0, 1] via interval subdivision.
//
// f3 has no 1/x terms, so direct evaluation on [0, 1] suffices.
//
// Expected result: ~0.00123 (paper reports 0.00123).

#include "ibex.h"
#include <cstddef>
#include <iostream>

using namespace ibex;

double compute_abs_bound(const Function& f, const Interval& z, size_t subdivisions) {
    double step = z.diam() / subdivisions;
    double max_abs_value = std::numeric_limits<double>::lowest();

    for (size_t i = 0; i < subdivisions; ++i) {
        Interval subInterval(z.lb() + i * step, z.lb() + (i + 1) * step);
        Interval result = f.eval(subInterval);

        double abs_upper = std::abs(result.ub());
        double abs_lower = std::abs(result.lb());
        double local_max_abs = std::max(abs_upper, abs_lower);

        if (local_max_abs > max_abs_value) {
            max_abs_value = local_max_abs;
        }
    }
    return max_abs_value;
}

// f3(x) = c1 / (c3 + x) - c2
// c1 = -8.387819235563974, c2 = 1.4427239805682266, c3 = -3.4355225277901402
int main() {
    Variable x;
    Function f(x, (pow(2, x) - ((-8.387819235563974) / ((-3.4355225277901402) + x) - 1.4427239805682266)) / pow(2, x));

    Interval z(0, 1);
    std::cout << "The bound on the maximum relative error of f3 on [" << z.lb() << "," << z.ub() << "] is: " << compute_abs_bound(f, z, 2<<25) << std::endl;

    return 0;
}
