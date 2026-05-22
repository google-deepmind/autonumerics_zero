// Prove maximum relative error (MRE) bound for the evolved 2^x approximation f4.
//
// f4 is the best 4-operation program found by evolutionary search:
//   def f4(x):
//     c1 = -13.889185227358549
//     c2 = -6.3103598040432605
//     x1 = c2 + x
//     c3 = 1.2011665727304095
//     x2 = c1 / x1
//     x3 = x2 - c3
//     x4 = x3 * x3
//     return x4
//
// i.e., f4(x) = (-13.889185227358549 / (-6.3103598040432605 + x) - 1.2011665727304095)^2
//
// This program uses the IBEX interval arithmetic library to rigorously bound
// |(2^x - f4(x)) / 2^x| on [0, 1] via interval subdivision.
//
// f4 has no 1/x terms, so direct evaluation on [0, 1] suffices.
//
// Expected result: ~0.000307 (paper reports 0.0003072).

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

// f4(x) = (c1 / (c2 + x) - c3)^2
// c1 = -13.889185227358549, c2 = -6.3103598040432605, c3 = 1.2011665727304095
int main() {
    Variable x;
    Function f(x, (pow(2, x) - pow((-13.889185227358549) / ((-6.3103598040432605) + x) - 1.2011665727304095, 2)) / pow(2, x));

    Interval z(0, 1);
    std::cout << "The bound on the maximum relative error of f4 on [" << z.lb() << "," << z.ub() << "] is: " << compute_abs_bound(f, z, 2<<25) << std::endl;

    return 0;
}
