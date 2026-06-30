import os
import json
import argparse
import textwrap
import numpy as np
from scipy.integrate import solve_bvp, solve_ivp, simpson
from variationalCalculus import *
from numpy.polynomial import Polynomial
import matplotlib.pyplot as plt

# =====================================================================
# --- RUTAS Y PARAMETROS ---
# =====================================================================

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(ROOT_DIR, "data")

# Valores por defecto (se pueden sobrescribir por argumentos de línea de comandos).
DEFAULT_N = 8
DEFAULT_T0 = 0.0
DEFAULT_T = 3.0
DEFAULT_TIME_LIMIT = 200.0
DEFAULT_CASE = "poly"


# =====================================================================
# --- CARGA DE CASOS (data/<caso>.json) ---
# =====================================================================
# Cada caso de prueba vive en data/<nombre>.json y se selecciona con --case
# usando el nombre del fichero (sin la extensión .json). Dentro del JSON, las
# funciones p, q, f, dp, exact y dexact se guardan como expresiones de texto
# en función de t (y de los extremos t0, T) y se evalúan con numpy disponible
# como np; el campo "labels" da la versión legible de p, q y f para los
# títulos. Recuerda: dp debe ser EXACTAMENTE la derivada de p (lo usa el BVP)
# y las soluciones exactas deben verificar x(t0) = x(T) = 0 (lo impone la base).

def available_cases():
    """Casos disponibles = ficheros .json dentro de data/ (sin extensión)."""
    if not os.path.isdir(DATA_DIR):
        return []
    return sorted(os.path.splitext(name)[0]
                  for name in os.listdir(DATA_DIR)
                  if name.endswith(".json"))


def make_func(expr, t0, T):
    """Compila una expresión de texto en una función de t (con t0, T y np)."""
    code = compile(expr, "<caso>", "eval")
    return lambda t: eval(code, {"np": np}, {"t": t, "t0": t0, "T": T})


def load_case(case_name, t0, T):
    """Lee data/<case_name>.json y devuelve (funciones, etiquetas) del caso."""
    path = os.path.join(DATA_DIR, case_name + ".json")
    with open(path, encoding="utf-8") as fh:
        spec = json.load(fh)
    funcs = {key: make_func(spec[key], t0, T)
             for key in ("p", "q", "f", "dp", "exact", "dexact")}
    return funcs, spec["labels"]


# --- Argumentos de línea de comandos (si no se pasan, se usan los por defecto) ---
parser = argparse.ArgumentParser(
    description="Flujo gradiente para el problema variacional 1D."
)
parser.add_argument("--case", choices=available_cases(), default=DEFAULT_CASE,
                    help=f"Caso a resolver (por defecto: {DEFAULT_CASE}).")
parser.add_argument("-n", type=int, default=DEFAULT_N,
                    help=f"Número de funciones base (por defecto: {DEFAULT_N}).")
parser.add_argument("--t0", type=float, default=DEFAULT_T0,
                    help=f"Extremo inicial del intervalo (por defecto: {DEFAULT_T0}).")
parser.add_argument("-T", type=float, default=DEFAULT_T,
                    help=f"Extremo final del intervalo (por defecto: {DEFAULT_T}).")
parser.add_argument("--time-limit", type=float, default=DEFAULT_TIME_LIMIT,
                    help=f"Tiempo final del flujo gradiente (por defecto: {DEFAULT_TIME_LIMIT}).")
args = parser.parse_args()

n = args.n
t0 = args.t0
T = args.T
TIME_LIMIT = args.time_limit
CASE = args.case


# =====================================================================
# --- CÁLCULO POR CASO ---
# =====================================================================

def solve_case(case_name):
    """Resuelve un caso por el flujo gradiente y devuelve mallas, errores y
    valores del funcional necesarios para el informe y las figuras."""
    case, labels = load_case(case_name, t0, T)
    p, q, f, dp = case["p"], case["q"], case["f"], case["dp"]
    exact, dexact = case["exact"], case["dexact"]

    # --- Solución por el método del flujo gradiente ---
    # Base polinómica que se anula en t0 y T: phi_i(t) = (t-t0)(t-T) t^i.
    base = [Polynomial([-t0, 1]) * Polynomial([-T, 1]) * Polynomial([0]*i + [1]) for i in range(n + 1)]
    A = compute_A(p, q, base, t0, T)
    b = compute_B(f, base, t0, T)

    # Flujo gradiente X'(s) = -(A X - b), integrado hasta s = TIME_LIMIT.
    func = lambda s, X: (-1) * Delta_J(A, X, b)
    X0 = np.array([0.0] * (n + 1))
    sol = solve_ivp(func, (0, TIME_LIMIT), X0, method='BDF')
    coef = sol.y.T[-1, :]
    xn = sum((c * phi for c, phi in zip(coef, base)), Polynomial([0.0]))

    # --- Solución de referencia del problema de contorno (BVP) ---
    ts = np.linspace(t0, T, 100)
    f_edif = lambda t, x: [x[1], (-dp(t) * x[1] + q(t) * x[0] - f(t)) / p(t)]
    bc = lambda ya, yb: np.array([ya[0], yb[0]])
    y = solve_bvp(f_edif, bc, ts, np.zeros((2, ts.size)))

    # --- Errores frente a la solución exacta ---
    ts = np.linspace(t0, T, 100)
    xs, dxs = xn(ts), xn.deriv()(ts)
    ys, dys = exact(ts), dexact(ts)

    error_L2 = np.sqrt(simpson(y=(xs - ys) ** 2, x=ts))
    error_puntual_H1 = (xs - ys) ** 2 + (dxs - dys) ** 2
    error_H1 = np.sqrt(simpson(y=error_puntual_H1, x=ts))

    return {
        "ts": ts, "xs": xs, "ys": ys,
        "error_puntual_H1": error_puntual_H1,
        "error_L2": error_L2, "error_H1": error_H1,
        "J_flow": J(xn, xn.deriv(), T, p, q, f, t0),
        "J_bvp": J(lambda t: y.sol(t)[0], lambda t: y.sol(t)[1], T, p, q, f, t0),
        "labels": labels,
    }


def print_report(case_name, r):
    """Imprime por consola un informe legible del caso resuelto."""
    lab = r["labels"]
    line, sub = "=" * 70, "-" * 70
    print(line)
    print("  MÉTODO DEL FLUJO GRADIENTE  ·  INFORME")
    print(line)
    print(f"  Caso          : {case_name}")
    print(f"  Datos         : p(t) = {lab['p']}")
    print(f"                  q(t) = {lab['q']}")
    print(f"                  f(t) = {lab['f']}")
    print(f"  Problema      : -(p x')' + q x = f   en [{t0:g}, {T:g}],  x({t0:g}) = x({T:g}) = 0")
    print(f"  Parámetros    : n = {n}    t_max = {TIME_LIMIT:g}")
    print(sub)
    print("  Error frente a la solución exacta")
    print(f"    Norma L2    : {r['error_L2']:.6e}")
    print(f"    Norma H1    : {r['error_H1']:.6e}")
    print(sub)
    print("  Valor del funcional J")
    print(f"    Flujo grad. : {r['J_flow']:.6e}")
    print(f"    BVP (ref.)  : {r['J_bvp']:.6e}")
    print(line)


# =====================================================================
# --- GRÁFICOS ---
# =====================================================================

r = solve_case(CASE)
lab = r["labels"]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# Subplot 1: solución del método frente a la solución exacta.
ax1.plot(r["ts"], r["xs"], label=r'$x_{método}(t)$', color='blue', linestyle='--')
ax1.plot(r["ts"], r["ys"], label=r'$x_{exacta}(t)$', color='red', alpha=0.7)
ax1.set_xlabel('t')
ax1.set_ylabel('x')
title = (f"Comparación de soluciones para el problema "
         f"p(t) = {lab['p']}, q(t) = {lab['q']}, f(t) = {lab['f']} "
         f"y n = {n} y t_max = {TIME_LIMIT:g}")
ax1.set_title(textwrap.fill(title, width=60), fontsize=10)
ax1.legend()
ax1.grid(True)

# Subplot 2: densidad del error H1 (dónde se acumula el error total).
ax2.plot(r["ts"], r["error_puntual_H1"], label=r'Densidad de error $H^1$', color='darkgreen')
ax2.set_xlabel('t')
ax2.set_ylabel('Error combinado')
# Eje Y en notación científica: el orden de magnitud aparece arriba del eje.
ax2.ticklabel_format(axis='y', style='sci', scilimits=(0, 0), useMathText=True)
ax2.set_title(f'Distribución del Error en el Tiempo\n'
              f'Norma $L^2$: {r["error_L2"]:.4e}  |  Norma $H^1$: {r["error_H1"]:.4e}')
ax2.legend()
ax2.grid(True)

plt.tight_layout()
out_path = os.path.join(ROOT_DIR, "flow_gradient.png")
plt.savefig(out_path, dpi=150)

print_report(CASE, r)
print(f"  Figura guardada en: {out_path}")
print("=" * 70)
plt.show()