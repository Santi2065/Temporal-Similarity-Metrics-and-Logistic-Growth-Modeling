import numpy as np
import matplotlib.pyplot as plt

# Definir las ecuaciones diferenciales
def logistic_model(t, N, r, K):
    return r * N * (1 - N / K)

def allee_effect_model(t, N, r, K, A):
    return r * N * (1 - N / K) * (N / A - 1)

# Método de Euler
def euler_step(f, t, N, h, *args):
    return N + h * f(t, N, *args)

# Método RK4
def rk4_step(f, t, N, h, *args):
    k1 = f(t, N, *args)
    k2 = f(t + h / 2, N + h / 2 * k1, *args)
    k3 = f(t + h / 2, N + h / 2 * k2, *args)
    k4 = f(t + h, N + h * k3, *args)
    return N + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)

# Diferentes combinaciones de condiciones iniciales y parámetros
combinations = [
    {"N0": 5, "r": 0.1, "K": 100, "A": 10},
    {"N0": 20, "r": 0.05, "K": 80, "A": 15},
    {"N0": 50, "r": 0.2, "K": 150, "A": 20},
    {"N0": 80, "r": 0.15, "K": 120, "A": 25},
]

# Preparar las figuras
fig_logistic, ax_logistic = plt.subplots(figsize=(10, 6))
fig_allee, ax_allee = plt.subplots(figsize=(10, 6))

for idx, params in enumerate(combinations):
    N0, r, K, A = params["N0"], params["r"], params["K"], params["A"]
    
    # Preparar los arreglos para almacenar los resultados
    t0, tf = 0, 100
    h = 0.5
    t_values = np.arange(t0, tf + h, h)
    N_euler_logistic = np.zeros(len(t_values))  # Inicializar el arreglo con ceros
    N_rk4_logistic = np.zeros(len(t_values))  # Inicializar el arreglo con ceros
    N_euler_allee = np.zeros(len(t_values))  # Inicializar el arreglo con ceros
    N_rk4_allee = np.zeros(len(t_values))  # Inicializar el arreglo con ceros

    # Condiciones iniciales
    N_euler_logistic[0] = N0
    N_rk4_logistic[0] = N0
    N_euler_allee[0] = N0
    N_rk4_allee[0] = N0

    # Solución numérica
    for i in range(1, len(t_values)):
        t = t_values[i - 1]
        N_euler_logistic[i] = euler_step(logistic_model, t, N_euler_logistic[i - 1], h, r, K)
        N_rk4_logistic[i] = rk4_step(logistic_model, t, N_rk4_logistic[i - 1], h, r, K)
        N_euler_allee[i] = euler_step(allee_effect_model, t, N_euler_allee[i - 1], h, r, K, A)
        N_rk4_allee[i] = rk4_step(allee_effect_model, t, N_rk4_allee[i - 1], h, r, K, A)

    # Graficar las soluciones para el modelo logístico
    ax_logistic.plot(t_values, N_euler_logistic, label=f'Euler (Logístico) N0={N0}, r={r}, K={K}', linestyle='--')
    ax_logistic.plot(t_values, N_rk4_logistic, label=f'RK4 (Logístico) N0={N0}, r={r}, K={K}', linestyle='-')

    # Graficar las soluciones para el modelo con efecto Allee
    ax_allee.plot(t_values, N_euler_allee, label=f'Euler (Allee) N0={N0}, r={r}, K={K}, A={A}', linestyle='--')
    ax_allee.plot(t_values, N_rk4_allee, label=f'RK4 (Allee) N0={N0}, r={r}, K={K}, A={A}', linestyle='-')

# Configurar el gráfico del modelo logístico
ax_logistic.set_xlabel('Tiempo t')
ax_logistic.set_ylabel('Población N(t)')
ax_logistic.set_title('Modelo Logístico')
ax_logistic.legend()

# Configurar el gráfico del modelo con efecto Allee
ax_allee.set_xlabel('Tiempo t')
ax_allee.set_ylabel('Población N(t)')
ax_allee.set_title('Modelo con Efecto Allee')
ax_allee.legend()

plt.show()

# Cerrar las figuras después de mostrarlas
plt.close(fig_logistic)
plt.close(fig_allee)
