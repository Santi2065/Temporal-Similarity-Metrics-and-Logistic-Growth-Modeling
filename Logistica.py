#%%
import numpy as np
import matplotlib.pyplot as plt

def fDinamicaPoblacional(N, r, K):
    return r * N * (K - N) / K

def runge_kutta(f, N0, r, K, h, num_steps):
    N_values = [N0]
    t_values = [0]
    for i in range(1, num_steps + 1):
        k1 = h * f(N0, r, K)
        k2 = h * f(N0 + 0.5 * k1, r, K)
        k3 = h * f(N0 + 0.5 * k2, r, K)
        k4 = h * f(N0 + k3, r, K)
        N0 = N0 + (k1 + 2 * k2 + 2 * k3 + k4) / 6
        t_new = i * h
        N_values.append(N0)
        t_values.append(t_new)
    return t_values, N_values

def euler(f, N0, r, K, h, num_steps):
    N_values = [N0]
    t_values = [0]
    for i in range(1, num_steps + 1):
        N0 = N0 + h * f(N0, r, K)
        t_new = i * h
        N_values.append(N0)
        t_values.append(t_new)
    return t_values, N_values

def solucion_analitica(N0, r, K, t):
    return K * N0 * np.exp(r * t) / (K + N0 * (np.exp(r * t) - 1))

# Parámetros
N0 = 500
r = 0.5
K = 1000
h = 0.05
num_steps = 100

# Solución numérica con Runge-Kutta
t_rk, N_rk = runge_kutta(fDinamicaPoblacional, N0, r, K, h, num_steps)

# Solución numérica con Euler
t_euler, N_euler = euler(fDinamicaPoblacional, N0, r, K, h, num_steps)

# Solución analítica
t_analitico = np.linspace(0, num_steps * h, num_steps + 1)
N_analitico = solucion_analitica(N0, r, K, t_analitico)

# Gráfico de la dinámica poblacional
plt.figure(figsize=(10, 6))
plt.plot(t_rk, N_rk, label='Solución Numérica (RK4)')
plt.plot(t_euler, N_euler, label='Solución Numérica (Euler)')
plt.plot(t_analitico, N_analitico, label='Solución Analítica', linestyle='dashed')
plt.xlabel('Tiempo')
plt.ylabel('Población')
plt.title('Evolución de la población a lo largo del tiempo (Ecuación logística)')
plt.legend()
plt.grid(True)
plt.show()

# Gráfico del error para ambos metodos
error_rk = np.abs(np.array(N_rk) - N_analitico)
error_euler = np.abs(np.array(N_euler) - N_analitico)

plt.figure(figsize=(10, 6))
plt.plot(t_rk, error_rk, label='Error Absoluto (RK4)')
plt.plot(t_euler, error_euler, label='Error Absoluto (Euler)')
plt.xlabel('Tiempo')
plt.ylabel('Error')
plt.title('Error entre las aproximaciones nuemericas y la solución analítica')
plt.legend()
plt.grid(True)
plt.show()

# Gráfico del error para el metodo de Runge-Kutta 4
plt.figure(figsize=(10, 6))
plt.plot(t_rk, error_rk, label='Error Absoluto (RK4)')
plt.xlabel('Tiempo')
plt.ylabel('Error')
plt.title('Error entre la solución numérica (RK4) y la solución analítica')
plt.legend()
plt.grid(True)
plt.show()
# %%
