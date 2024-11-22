import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Carga de datos
df_temp = pd.read_csv('city_temperature.csv', dtype={'Region': str, 'Country': str, 'State': str, 'City': str, 'Month': int, 'Day': int, 'Year': int, 'AvgTemperature': float})

# Selección de ciudades y período de tiempo
ciudades = ['Lisbon', 'Madrid', 'Montvideo', 'Buenos Aires']
inicio_año = 1995
fin_año = 2010
df_temp_filtrado = df_temp[(df_temp['Year'] >= inicio_año) & (df_temp['Year'] < fin_año) & (df_temp['City'].isin(ciudades))]

# Cálculo de la derivada con la fórmula de cinco puntos centrada
def derivada_cinco_puntos(serie_temp, h):
    derivada = (-serie_temp[4:] + 8*serie_temp[3:-1] - 8*serie_temp[1:-3] + serie_temp[:-4]) / (12 * h)
    return derivada

#calculo de la derivada con la formula de tres puntos centrada
def derivada_tres_puntos(serie_temp, h):
    derivada = (serie_temp[2:] - serie_temp[:-2]) / (2 * h)
    return derivada

# Métrica de similaridad (Correlación de Pearson)
def similaridad(x_i, x_j):
    x_i_mean = np.mean(x_i)
    x_j_mean = np.mean(x_j)
    numerador = np.sum((x_i - x_i_mean) * (x_j - x_j_mean))
    denominador = np.sqrt(np.sum((x_i - x_i_mean)**2) * np.sum((x_j - x_j_mean)**2))
    return numerador / denominador

# Diccionario para almacenar las series temporales de cada ciudad
series_temp_ciudades = {}
for ciudad in ciudades:
    serie_temp = df_temp_filtrado[df_temp_filtrado['City'] == ciudad]['AvgTemperature'].values
    series_temp_ciudades[ciudad] = serie_temp

# Cálculo de similaridades entre todas las combinaciones de ciudades con la formula de cinco puntos
h = 4  # Intervalo de tiempo para la diferenciación numérica
años = range(inicio_año, fin_año)
similaridades = np.zeros((len(ciudades), len(ciudades)))
for i, ciudad_i in enumerate(ciudades):
    for j, ciudad_j in enumerate(ciudades):
        x_i = series_temp_ciudades[ciudad_i]
        x_j = series_temp_ciudades[ciudad_j]
        delta_x_i = derivada_cinco_puntos(x_i,h)
        delta_x_j = derivada_cinco_puntos(x_j,h)

        # Calcular similaridad año por año (ajustar índices)
        S_ij = []
        for k in range(len(años)):
            S_ij.append(similaridad(delta_x_i[k:k+10], delta_x_j[k:k+10]))
        similaridades[i, j] = np.mean(S_ij)  # Promedio de las similaridades

# Gráfico de mapa de calor
plt.figure(figsize=(10, 6))
sns.heatmap(similaridades, annot=True, cmap='coolwarm', xticklabels=ciudades, yticklabels=ciudades)
plt.xlabel('Ciudades')
plt.ylabel('Ciudades')
plt.title('Mapa de calor de similaridad entre ciudades')
plt.show()

# Asegúrate de que la derivada tenga la misma longitud que los años
derivada_buenos_aires_5 = derivada_cinco_puntos(series_temp_ciudades['Buenos Aires'], h)
derivada_Madrid_5 = derivada_cinco_puntos(series_temp_ciudades['Madrid'], h)
derivada_montvideo_5 = derivada_cinco_puntos(series_temp_ciudades['Montvideo'], h)
derivada_Lisbon_5 = derivada_cinco_puntos(series_temp_ciudades['Lisbon'], h)

derivada_buenos_aires_3 = derivada_tres_puntos(series_temp_ciudades['Buenos Aires'], h)
derivada_Madrid_3 = derivada_tres_puntos(series_temp_ciudades['Madrid'], h)
derivada_montvideo_3 = derivada_tres_puntos(series_temp_ciudades['Montvideo'], h)
derivada_Lisbon_3 = derivada_tres_puntos(series_temp_ciudades['Lisbon'], h)

# Ajusta la longitud de las derivadas si es necesario
if len(derivada_buenos_aires_5) > len(años):
    derivada_buenos_aires_5 = derivada_buenos_aires_5[:len(años)]
if len(derivada_Madrid_5) > len(años):
    derivada_Madrid_5 = derivada_Madrid_5[:len(años)]
if len(derivada_montvideo_5) > len(años):
    derivada_montvideo_5 = derivada_montvideo_5[:len(años)]
if len(derivada_Lisbon_5) > len(años):
    derivada_Lisbon_5 = derivada_Lisbon_5[:len(años)]

if len(derivada_buenos_aires_3) > len(años):
    derivada_buenos_aires_3 = derivada_buenos_aires_3[:len(años)]
if len(derivada_Madrid_3) > len(años):
    derivada_Madrid_3 = derivada_Madrid_3[:len(años)]
if len(derivada_montvideo_3) > len(años):
    derivada_montvideo_3 = derivada_montvideo_3[:len(años)]
if len(derivada_Lisbon_3) > len(años):
    derivada_Lisbon_3 = derivada_Lisbon_3[:len(años)]

# Gráfico que compara las derivadas de las ciudades con la técnica de 5 puntos
plt.figure(figsize=(10, 6))
plt.plot(años, derivada_buenos_aires_5, label='Buenos Aires (5 puntos)', linestyle='-', color='blue')
plt.plot(años, derivada_Madrid_5, label='Madrid (5 puntos)', linestyle='-', color='green')
plt.plot(años, derivada_montvideo_5, label='Montvideo (5 puntos)', linestyle='-', color='red')
plt.plot(años, derivada_Lisbon_5, label='Lisbon (5 puntos)', linestyle='-', color='purple')

plt.xlabel('Año')
plt.ylabel('Derivada')
plt.title('Comparación de derivadas de Buenos Aires, Madrid, Monte video y Lisbon 5 puntos')
plt.legend()
plt.show()

# Grafico que compara las derivadas de 3 y 5 puntos de Buenos Aires y Madrid
plt.figure(figsize=(10, 6))
plt.plot(años, derivada_buenos_aires_5, label='Buenos Aires (5 puntos)', linestyle='-', color='blue')
plt.plot(años, derivada_buenos_aires_3, label='Buenos Aires (3 puntos)', linestyle='--', color='blue')
plt.plot(años, derivada_Madrid_5, label='Madrid (5 puntos)', linestyle='-', color='red')
plt.plot(años, derivada_Madrid_3, label='Madrid (3 puntos)', linestyle='--', color='red')
plt.xlabel('Año')
plt.ylabel('Derivada')
plt.title('Comparación de derivadas de Buenos Aires (3 y 5 puntos)')
plt.legend()
plt.show()
