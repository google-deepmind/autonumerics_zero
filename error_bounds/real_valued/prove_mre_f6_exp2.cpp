// Prove maximum relative error (MRE) bound for the evolved 2^x approximation f6.
//
// f6 is the best 6-operation program found by evolutionary search:
//   def f6(x):
//     c1 = 34.839796464204852
//     c2 = -17.414314151974395
//     c3 = 100.52501140663516
//     x1 = c3 / x
//     x2 = x + x1
//     x3 = c2 + x2
//     x4 = c1 / x3
//     c4 = -0.99999980025884683
//     x5 = x4 - c4
//     x6 = x5 * x5
//     return x6
//
// i.e., f6(x) = (34.839796464204852 / (-17.414314151974395 + x + 100.52501140663516/x)
//               + 0.99999980025884683)^2
//
// This program uses the IBEX interval arithmetic library to rigorously bound
// |(2^x - f6(x)) / 2^x| on (0, 1].
//
// f6 has a 1/x term, so the domain is split at x=0.0001:
//   [0, 0.0001]:   direct interval evaluation (2^26 subdivisions)
//   [0.0001, 1]:   Lipschitz-based bound (100M subdivisions)
//
// Expected result: ~4.02e-7 (paper reports 4.016 × 10^-7).

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

Interval compute_lipschitz_bounds(const Function& f, const Interval& z, double lipschitz_L) {
  Interval f_left = f.eval(Interval(z.lb(), z.lb()));
  Interval f_right = f.eval(Interval(z.ub(), z.ub()));
  double max_left = std::max(f_left.lb(), f_left.ub());
  double max_right = std::max(f_right.lb(), f_right.ub());
  double upper = std::max(max_left, max_right) + lipschitz_L*(z.diam());
  double min_left = std::min(f_left.lb(), f_left.ub());
  double min_right = std::min(f_right.lb(), f_right.ub());
  double lower = std::min(min_left, min_right) - lipschitz_L*(z.diam());
  return Interval(lower, upper);
}

// f6(x) = (c1 / (c2 + x + c3/x) - c4)^2
// c1 = 34.839796464204852, c2 = -17.414314151974395,
// c3 = 100.52501140663516, c4 = -0.99999980025884683
int main() {
    Variable x;
    Function f(x, (pow(2, x) - pow(34.839796464204852 / ((-17.414314151974395) + x + 100.52501140663516 / x) + 0.99999980025884683, 2)) / pow(2, x));
    Function df(f, Function::DIFF);

    double small_separator = 0.0001;

    Interval z_f(0, small_separator);
    std::cout << "The bound on the absolute value of f on [" << z_f.lb() << "," << z_f.ub() << "] is: " << compute_abs_bound(f, z_f, 2<<25) << std::endl;

    Interval z_df(small_separator, 1);

    double L = compute_abs_bound(df, z_df, 2<<25);
    std::cout << "The bound on the absolute value of df on [" << z_df.lb() << "," << z_df.ub() << "] is: " << L << std::endl;

    Interval z = z_df;
    size_t subdivisions = 100000000;
    double step = z.diam() / subdivisions;

    double max_abs_value = std::numeric_limits<double>::lowest();

    for (size_t i = 0; i < subdivisions; ++i) {
        Interval subInterval(z.lb() + i * step, z.lb() + (i + 1) * step);
        Interval result = compute_lipschitz_bounds(f, subInterval, L);

        double abs_upper = std::abs(result.ub());
        double abs_lower = std::abs(result.lb());
        double local_max_abs = std::max(abs_upper, abs_lower);

        if (local_max_abs > max_abs_value) {
            max_abs_value = local_max_abs;
        }
    }

    std::cout << "The bound of the absolute value of f, estimated by Lipschitz, on [" << z.lb() << "," << z.ub() << "] is: " << max_abs_value << std::endl;

    return 0;
}
