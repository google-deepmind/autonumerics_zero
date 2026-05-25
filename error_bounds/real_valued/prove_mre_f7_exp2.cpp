// Prove maximum relative error (MRE) bound for the evolved 2^x approximation f7.
//
// f7 is the best 7-operation program found by evolutionary search:
//   def f7(x):
//     c0 = -31.36503547149854
//     c1 = 237.4188094069423
//     c2 = 62.73011283808836
//     c3 = -0.9999999991613342
//     c4 = 90.500418999822784
//     x1 = c4 / x
//     x2 = x + x1
//     x3 = c1 / x2
//     x4 = x3 + x2
//     x5 = c0 + x4
//     x6 = c2 / x5
//     x7 = x6 - c3
//     return x7
//
// i.e., f7(x) = 62.73011283808836 / (-31.36503547149854 + 237.4188094069423/(x + 90.500418999822784/x)
//               + x + 90.500418999822784/x) + 0.9999999991613342
//
// This program uses the IBEX interval arithmetic library to rigorously bound
// |(2^x - f7(x)) / 2^x| on (0, 1].
//
// f7 has a 1/x term, so the domain is split at x=0.0001:
//   [0, 0.0001]:   direct interval evaluation (2^26 subdivisions)
//   [0.0001, 1]:   Lipschitz-based bound (100M subdivisions)
//
// Expected result: ~8.40e-10 (paper reports 8.417 × 10^-10).

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

int main() {
    Variable x;
    Function f(x,(pow(2,x) + -0.9999999991613342 - 62.73011283808836 / (( -31.36503547149854) + x + (237.4188094069423) / (x +   90.500418999822784/ x) + 90.500418999822784 / x)) / pow(2,x));
    Function df(f,Function::DIFF);

    double small_separator = 0.0001;

    Interval z_f(0, small_separator);
    // make subdivision argument to compte_abs_bound 2<<25 to get 8.39e-10 in 2min*/
    std::cout << "The bound on the absolute value of f on [" << z_f.lb() << ","<< z_f.ub() << "] is: " << compute_abs_bound(f, z_f, 2<<25) << std::endl;

    Interval z_df(small_separator, 1);

    double L = compute_abs_bound(df, z_df, 2<<25);
    std::cout << "The bound on the absolute value of df on [" << z_df.lb() << ","<< z_df.ub() << "] is: " << L << std::endl;

    Interval z = z_df;
    size_t subdivisions = 100000000;
    double step = z.diam()/subdivisions;

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

    std::cout << "The bound of the absolute value of f, estimated by Lipschitz, on [" << z.lb()<< ","<< z.ub() << "] is: "<< max_abs_value << std::endl;

    return 0;
}
