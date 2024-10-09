import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

ciudad_1 = "Montvideo"
ciudad_2 = "Montvideo"

# Cargar el dataset
file_path = 'city_temperature.csv'
df = pd.read_csv(file_path, dtype={'Region': str, 'Country': str, 'State': str, 'City': str, 'Month': int, 'Day': int, 'Year': int, 'AvgTemperature': float})

# Filtrar datos por ciudad
def filtrar_ciudad_por_año(df, ciudad, año):
    datos_ciudad = df[(df['City'] == ciudad) & (df['Year'] == año)].copy()
    datos_ciudad['Fecha'] = pd.to_datetime(datos_ciudad[['Year', 'Month', 'Day']])
    return datos_ciudad[['Fecha', 'AvgTemperature']]

# Calcular tasas de variación Δx/Δt
def tasa_cambio(x, t):
    dif_t = np.diff(t.astype('int64') // 1e9)  # Convertir tiempo a segundos y calcular diferencias
    dif_t[dif_t == 0] = np.nan  # Evitar divisiones por cero
    return np.diff(x) / dif_t  # Retornar tasa de cambio

# Métrica de similaridad
epsilon = 1e-10
def similaridad(delta_x_i, delta_x_j):
    valid_idx = ~np.isnan(delta_x_i) & ~np.isnan(delta_x_j)
    delta_x_i = delta_x_i[valid_idx]
    delta_x_j = delta_x_j[valid_idx]
    
    if len(delta_x_i) == 0:  # Si no hay suficientes datos válidos, retornar NaN
        return np.nan
    
    n = len(delta_x_i)
    denominador = np.abs(delta_x_i) + np.abs(delta_x_j) + epsilon
    similitud = 1 - np.sum(np.abs(delta_x_i - delta_x_j) / denominador) / n
    
    return similitud

# Lista para almacenar los resultados de similaridad por año
años = range(1995, 2021)
similaridades = []

# Calcular similaridad año por año
for año in años:
    datos_ciudad_1 = filtrar_ciudad_por_año(df, ciudad_1, año)
    datos_ciudad_2 = filtrar_ciudad_por_año(df, ciudad_2, año)
    
    # Encontrar las fechas comunes entre ambas ciudades
    fechas_comun = set(datos_ciudad_1['Fecha']).intersection(set(datos_ciudad_2['Fecha']))
    datos_ciudad_1 = datos_ciudad_1[datos_ciudad_1['Fecha'].isin(fechas_comun)].copy()
    datos_ciudad_2 = datos_ciudad_2[datos_ciudad_2['Fecha'].isin(fechas_comun)].copy()
    
    # Extraer las series de tiempo y fechas
    x_i = datos_ciudad_1['AvgTemperature'].values
    x_j = datos_ciudad_2['AvgTemperature'].values
    t = datos_ciudad_1['Fecha'].values  # Usamos las fechas como base de tiempo
    
    # Calcular las tasas de cambio para ambas ciudades
    delta_x_i = tasa_cambio(x_i, t)
    delta_x_j = tasa_cambio(x_j, t)
    
    # Calcular la similaridad entre las dos series temporales para el año
    S_ij = similaridad(delta_x_i, delta_x_j)
    similaridades.append(S_ij)

# Graficar la evolución de la similaridad entre las dos ciudades de 1995 a 2020
plt.figure(figsize=(10, 6))
plt.plot(años, similaridades, marker='o', linestyle='-', color='b')
plt.title(f"Evolución de la Similaridad entre {ciudad_1} y {ciudad_2} (1995-2020)")
plt.xlabel("Año")
plt.ylabel("Similaridad")
plt.ylim(0, 1)
plt.grid(True)
plt.show()