# Método del flujo gradiente — implementación numérica

Código del **trabajo práctico** asociado al Trabajo de Fin de Grado
**«Cálculo de variaciones y optimización»** (Juan Diego Barrado Daganzo,
Doble Grado en Ingeniería Informática y Matemáticas, Universidad Complutense
de Madrid; director: Aníbal Rodríguez Bernal).

Documento del TFG: <https://github.com/JuanDiegoBarrado/Bachelor-s-Thesis-in-Mathematics>

Este repositorio contiene la implementación del **método del flujo gradiente**
propuesto en la memoria y reproduce las **Figuras 3.1 y 3.2** de la Sección 3.2.2.

> [!NOTE]
> Este `README.md` ha sido generado con ayuda de IA.

## Dónde encaja en el TFG

El TFG estudia, de forma autocontenida, el cálculo de variaciones como parte de
la optimización matemática. Su Capítulo 3 analiza una familia concreta de
**problemas de contorno variacionales** y, como aportación propia, propone
resolverlos mediante un **flujo gradiente** sobre el funcional de energía. La
memoria desarrolla la teoría (formulación débil, existencia y unicidad vía
Lax-Milgram, método de Galerkin y convergencia del flujo gradiente); **este
código es la implementación numérica** que ilustra el procedimiento, compara la
solución obtenida con la solución exacta y produce las figuras que cierran el
capítulo.

## El problema

Se resuelven los *problemas de contorno variacionales* (PCLV) de la forma

$$
\begin{cases}
-\dfrac{d}{dt} \left(p(t) \dfrac{d}{dt} x(t) \right) + q(t) x(t) = f(t), & t \in [t_0, T]\\
x(t_0) = 0, \quad x(T) = 0,
\end{cases}
$$

con $p \ge p_0 > 0$ y $q \ge 0$ (coeficientes que pueden ser continuos a trozos).
En **formulación débil**, se busca $x \in H_0^1(t_0, T)$ tal que

$$
\int_{t_0}^{T} p(t) x'(t)\varphi'(t) \ dt + \int_{t_0}^{T} q(t) x(t) \varphi(t) \ dt = \int_{t_0}^{T} f(t)\varphi(t) \ dt \qquad \forall \varphi \in H_0^1(t_0, T).
$$

El Teorema de Lax-Milgram garantiza existencia y unicidad de la solución débil, y
(al ser la forma bilineal simétrica) esa solución coincide con el **único
minimizador global** del funcional de energía

$$
J(x) = \frac{1}{2}\int_{t_0}^{T} \left( p(t) x'(t)^2 + q(t) x(t)^2 \right) \ dt - \int_{t_0}^{T} f(t) x(t) \ dt.
$$

## El método

**1. Discretización de Galerkin.** Se aproxima la solución en un subespacio de
dimensión finita $V_h \subset H_0^1(t_0, T)$ generado por la base polinómica

$$
\varphi_i(t) = (t - t_0)(t - T) t^{i}, \qquad i = 0, \dots, n,
$$

cuyos elementos se anulan en $t_0$ y en $T$ (cumplen las condiciones de contorno
por construcción). Escribiendo $x \approx \sum_i X_i\,\varphi_i$, el funcional
restringido a $V_h$ es la forma cuadrática

$$
J|_{V_h}(X) = \frac{1}{2} X^{\mathsf T} A X - b^{\mathsf T} X,
$$

con

$$
A_{ij} = \int_{t_0}^{T} \left(p(t) \varphi_i(t) \varphi_j(t) + q(t) \varphi_i(t) \varphi_j(t) \right) \ dt,
\qquad
b_i = \int_{t_0}^{T} f(t) \varphi_i(t) \ dt.
$$

Las integrales se calculan por cuadratura (`scipy.integrate.quad`); $A$ es
simétrica y definida positiva.

**2. Flujo gradiente.** En lugar de resolver el sistema lineal
$\nabla J|_{V_h}(X) = AX - b = 0$ directamente (como harían los métodos de
elementos finitos) o de resolver el problema de contorno, se hace **evolucionar
una función inicial en la dirección de máximo descenso** del funcional,
integrando el sistema de EDOs

$$
\frac{dX}{dt} = -\nabla J|_{V_h}(X) = -(A X - b), \qquad X(0) = X_0.
$$

Como $J|_{V_h}$ es coercivo y tiene un único punto crítico, la solución converge
al minimizador $X_\ast = A^{-1}b$ cuando $t \to \infty$ (Teorema 3.23 de la
memoria), **para cualquier** $X_0$.

**3. Referencia y error.** Se resuelve también el problema de contorno con
`scipy.integrate.solve_bvp` y se compara la solución del flujo gradiente con la
**solución exacta** conocida del caso, midiendo el error en las normas $L^2$ y
$H^1$ (esta última recoge también el error en la derivada, acorde a la norma
natural de $H_0^1$).

### Sobre la convergencia (rigidez y condicionamiento)

El sistema de EDOs es lineal, $X'(t) = -A (X - X_\ast)$, con solución
$X(t) = X_\ast + e^{-At}(X_0 - X_\ast)$. La velocidad de convergencia la fija el menor
autovalor de $A$: la base de monomios convierte a $A$ en (esencialmente) una
matriz de Gram tipo **Hilbert**, mal condicionada, cuyo coeficiente de rigidez
$\mathrm{cond}(A) = \lambda_{\max}/\lambda_{\min}$ crece exponencialmente
con $n$. Por eso el sistema es **rígido** y se integra con el método implícito
**BDF** (A-estable), que no impone restricciones sobre el paso. Esto explica que
los casos regulares converjan muy bien con $n$ moderado, mientras que el caso de
capa límite (con transición brusca) tenga un error mucho mayor que no mejora —e
incluso empeora el condicionamiento— al aumentar $n$.

## Estructura del proyecto

| Fichero | Contenido |
|---|---|
| `flowGradient.py` | Script principal: CLI, ensamblado del problema, integración del flujo, comparación y figura. |
| `variationalCalculus.py` | Funciones de apoyo: ensamblado de $A$ (`compute_A`) y $b$ (`compute_B`), gradiente $\nabla J$ (`Delta_J`) y funcional $J$ (`J`). |
| `data/*.json` | Un fichero por caso de prueba (ver más abajo). |
| `requirements.txt` | Dependencias directas (numpy, scipy, matplotlib). |

## Instalación

Requiere **Python ≥ 3.11**.

```bash
python -m venv .venv
# Windows (PowerShell):
.\.venv\Scripts\Activate.ps1
# Linux/macOS:
source .venv/bin/activate

pip install -r requirements.txt
```

## Uso

```bash
python flowGradient.py [--case CASO] [-n N] [--t0 T0] [-T T] [--time-limit TMAX]
```

| Argumento | Descripción | Por defecto |
|---|---|---|
| `--case` | Caso a resolver (= nombre de un fichero en `data/`). | `poly` |
| `-n` | Número de funciones base. | `8` |
| `--t0` | Extremo inicial $t_0$ del intervalo. | `0.0` |
| `-T` | Extremo final $T$ del intervalo. | `3.0` |
| `--time-limit` | Tiempo final $t_{\max}$ del flujo gradiente. | `200.0` |

Ejemplos:

```bash
python flowGradient.py                          # caso por defecto
python flowGradient.py --case layer -n 12       # capa límite con 12 funciones base
python flowGradient.py --case wave --t0 0 -T 6  # onda en [0, 6]
```

Cada ejecución imprime por consola un **informe** (datos del problema, error en
normas $L^2$ y $H^1$, valor del funcional $J$ del flujo gradiente vs. el del BVP
de referencia) y guarda una figura **`flow_gradient.png`** con dos paneles: la
solución del método frente a la exacta, y la densidad del error $H^1$ a lo largo
de $t$ —es el formato de las Figuras 3.1 y 3.2 de la memoria.

## Casos de prueba (`data/`)

Cada caso vive en `data/<nombre>.json` y se selecciona con `--case <nombre>` (el
nombre del fichero sin la extensión). Los casos `sinh`, `wave` y `layer`
reproducen las figuras del TFG:

| Caso | Problema | Solución exacta | En el TFG |
|---|---|---|---|
| `poly` | $-x'' = 1$ | $(t-t_0)(T-t)/2$ — polinómica, está en la base. | — |
| `sinh` | $-x'' + x = t$ | $t + A\cosh t + B\sinh t$. | Figura 3.1(a) |
| `variable` | coef. variable $p(t)=1+t^2$ | $\sin\!\big(k(t-t_0)\big)$, $k=\pi/(T-t_0)$. | — |
| `wave` | oscilante, $-x'' = (2\pi/(T-t_0))^2\sin(\cdots)$ | $\sin\!\big(2\pi(t-t_0)/(T-t_0)\big)$. | Figura 3.1(b) |
| `layer` | capa límite, $-\varepsilon x'' + x = 1$, $\varepsilon=0.01$ | meseta plana que cae en los bordes. | Figura 3.2 |

### Formato de cada caso

```json
{
  "name": "poly",
  "description": "Texto explicativo del caso.",
  "labels": { "p": "1", "q": "0", "f": "1" },
  "p": "1.0 + 0*t",
  "q": "0.0 + 0*t",
  "f": "1.0 + 0*t",
  "dp": "0.0 + 0*t",
  "exact": "(t - t0) * (T - t) / 2",
  "dexact": "(t0 + T - 2*t) / 2"
}
```

- `p`, `q`, `f`, `dp`, `exact`, `dexact` son **expresiones de texto** en función
  de `t` (y de los extremos `t0`, `T`), evaluadas con NumPy disponible como `np`
  (p. ej. `np.sin`, `np.cosh`, `np.pi`).
- `dp` debe ser **exactamente la derivada de `p`** (la usa el solver de contorno
  de referencia).
- `exact` debe **anularse en `t0` y en `T`** (coherente con la base $H_0^1$) y
  `dexact` es su derivada (se usa para el error en norma $H^1$).
- `labels` da la versión legible de $p$, $q$ y $f$ para los títulos de la figura.

**Añadir un caso nuevo** consiste únicamente en dejar un `.json` en `data/`; el
script lo detecta automáticamente y lo ofrece en `--case`, sin tocar el código.
