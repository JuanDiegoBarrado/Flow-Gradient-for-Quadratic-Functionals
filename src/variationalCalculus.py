import numpy as np
from numpy.polynomial import Polynomial
from scipy.integrate import quad, quad_vec

def compute_A(p, q, base, t0=0, T=1):
    n = len(base)
    dbase = [phi.deriv() for phi in base]

    A = np.zeros((n, n))
    for i in range(n):
        for j in range(i, n):
            integrand = lambda t, i=i, j=j: (
                p(t) * dbase[i](t) * dbase[j](t)
                + q(t) * base[i](t) * base[j](t)
            )
            a_ij, _ = quad(integrand, t0, T)
            A[i, j] = a_ij
            A[j, i] = a_ij

    return A

def compute_B(f, base, t0=0, T=1):
    n = len(base)

    B = np.zeros(n)
    for i in range(n):
        integrand = lambda t, i=i: f(t) * base[i](t)
        b_i, _ = quad(integrand, t0, T)
        B[i] = b_i

    return B

def Delta_J(A, x, b):
    return A @ x - b

def J(x, dx, T, p, q, f, t0=0):
    def j(t):
        return 0.5 * (p(t) * dx(t) ** 2 + q(t) * x(t) ** 2) - f(t) * x(t)

    ans, _ = quad(j, t0, T)
    return ans