// Prove maximum relative error (MRE) bound for the evolved 2^x approximation f9.
//
// f9 is the best 9-operation program found by evolutionary search:
//   def f9(x):
//     c1 = -125.28498998901401
//     c2 = 3786.1251186399709
//     c3 = -250.56998014383566
//     c4 = 1445.9842709817003
//     c5 = 0.99999999999994771
//     x1 = c4 / x
//     x2 = x + x1
//     x3 = c2 / x2
//     x4 = x3 + x2
//     x5 = c1 + x4
//     x6 = c3 / x5
//     x7 = x6 - c5
//     x8 = x7 * x7
//     x9 = x8 * x8
//     return x9
//
// i.e., f9(x) = (-250.56998014383566 / (-125.28498998901401 + 3786.1251186399709/(x + 1445.9842709817003/x)
//               + x + 1445.9842709817003/x) - 0.99999999999994771)^4
//
// This program uses a unified adaptive recursive prover with the IBEX interval
// arithmetic library to rigorously prove the MRE bound over the entire [0, 1].
//
// Near the singularity at x=0, it uses direct interval evaluation.
// Away from zero (x >= 0.0001), it uses the local Lipschitz bound.
//
// Expected result: ~2.15e-13 (paper reports 2.15 × 10^-13).

#include "ibex.h"
#include <cstddef>
#include <iostream>
#include <iomanip>

using namespace ibex;

bool prove_unified_recursive(const Function& f, const Function& df, const Interval& z, double epsilon, size_t& subdivisions_count) {
    subdivisions_count++;
    
    double diam = z.diam();
    double m = (z.lb() + z.ub()) / 2.0;
    
    double eta = 0.0;
    
    // Near the singularity (x < 1e-9), use direct interval evaluation
    if (z.lb() < 1e-9) {
        Interval f_val = f.eval(z);
        eta = std::max(std::abs(f_val.lb()), std::abs(f_val.ub()));
    } else {
        // Away from zero, use the Lipschitz bound
        Interval df_val = df.eval(z);
        double L = std::max(std::abs(df_val.lb()), std::abs(df_val.ub()));
        
        Interval E_m = f.eval(Interval(m, m));
        double E_m_abs = std::max(std::abs(E_m.lb()), std::abs(E_m.ub()));
        
        eta = E_m_abs + L * (diam / 2.0);
    }
    
    if (eta <= epsilon) {
        return true;
    }
    
    if (diam < 1e-16) {
        std::cout << "Failed at interval [" << std::scientific << std::setprecision(18) 
                  << z.lb() << ", " << z.ub() << "] with eta = " << eta << " (epsilon = " << epsilon << ")" << std::endl;
        return false;
    }
    
    // Subdivide and recurse
    Interval left(z.lb(), m);
    Interval right(m, z.ub());
    
    return prove_unified_recursive(f, df, left, epsilon, subdivisions_count) &&
           prove_unified_recursive(f, df, right, epsilon, subdivisions_count);
}

int main() {
    Variable x;
    Function f(x, (pow(2, x) - pow((-250.56998014383566) / ((-125.28498998901401) + 3786.1251186399709 / (x + 1445.9842709817003 / x) + x + 1445.9842709817003 / x) - 0.99999999999994771, 4)) / pow(2, x));
    Function df(f, Function::DIFF);

    Interval z(0, 1);
    double epsilon = 2.15e-13;
    size_t subdivisions_count = 0;

    std::cout << "Proving f9 MRE <= " << epsilon << " on [0, 1] using unified recursive prover..." << std::endl;
    bool success = prove_unified_recursive(f, df, z, epsilon, subdivisions_count);
    
    if (success) {
        std::cout << "SUCCESS! Proved f9 MRE <= " << epsilon << " on [0, 1] in " << subdivisions_count << " steps." << std::endl;
    } else {
        std::cout << "FAILED to prove MRE <= " << epsilon << std::endl;
    }

    return 0;
}
