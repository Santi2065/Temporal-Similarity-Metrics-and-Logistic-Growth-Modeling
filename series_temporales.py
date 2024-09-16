import pandas as pd
from numpy import linalg as LA
import seaborn as sns
import matplotlib.pyplot as plt

# Cargar el dataset
file_path = 'city_temperature.csv'
df = pd.read_csv(file_path)

# Convertir la temperatura de Fahrenheit a Celsius
df['AvgTemperature_Celsius'] = (df['AvgTemperature'] - 32) * 5/9

# Función para calcular la similaridad entre las tasas de cambio de dos ciudades
def calcular_similaridad(df_ciudad1, df_ciudad2):
    # Asegurarse de que las dos series tengan las mismas fechas
    merged_df = pd.merge(df_ciudad1[['AvgTemperature_Celsius']], df_ciudad2[['AvgTemperature_Celsius']], 
                         left_index=True, right_index=True, suffixes=('_ciudad1', '_ciudad2')).dropna()

    # Calcular la distancia euclidiana entre las tasas de cambio
    distancia = LA.norm(merged_df['AvgTemperature_Celsius_ciudad1'] - merged_df['AvgTemperature_Celsius_ciudad2'])
    
    # Definir la similaridad como inversa de la distancia
    similaridad = 1 / (1 + distancia)
    
    return similaridad
# Filtrar una ciudad en particular, por ejemplo, 'Buenos Aires'
df_paris = df[df['City'] == 'Buenos Aires']

# Calcular la tasa de variación de temperatura usando diferencias finitas
df_paris['Temp_Change'] = df_paris['AvgTemperature_Celsius'].diff() / df_paris.index.to_series().diff()

# Filtrar otra ciudad, por ejemplo 'New York'
df_ny = df[df['City'] == 'New York']

# Calcular la tasa de variación de temperatura usando diferencias finitas
df_ny['Temp_Change'] = df_ny['AvgTemperature_Celsius'].diff() / df_ny.index.to_series().diff()

# Calcular la similaridad entre Paris y New York
similaridad_paris_ny = calcular_similaridad(df_paris, df_ny)
print(f"Similaridad entre Buenos Aires y New York: {similaridad_paris_ny}")

# Graficar la tasa de cambio de temperatura para Paris y New York
plt.figure(figsize=(10, 6))
plt.plot(df_paris.index, df_paris['Temp_Change'], label='Buenos Aires')
plt.plot(df_ny.index, df_ny['Temp_Change'], label='New York')
plt.title('Tasa de Cambio de Temperatura - Paris vs New York')
plt.xlabel('Fecha')
plt.ylabel('Tasa de Cambio (°C/día)')
plt.legend()
plt.show()

# Filtrar algunas ciudades, por ejemplo, 'Paris', 'New York', 'Moscow'
ciudades = ['Buenos Aires', 'Paris', 'Moscow']
similaridades = pd.DataFrame(index=ciudades, columns=ciudades)

# Calcular similaridades entre cada par de ciudades
for city1 in ciudades:
    df_city1 = df[df['City'] == city1]
    df_city1['Temp_Change'] = df_city1['AvgTemperature_Celsius'].diff() / df_city1.index.to_series().diff()
    
    for city2 in ciudades:
        df_city2 = df[df['City'] == city2]
        df_city2['Temp_Change'] = df_city2['AvgTemperature_Celsius'].diff() / df_city2.index.to_series().diff()
        
        # Calcular similaridad
        similaridades.loc[city1, city2] = calcular_similaridad(df_city1, df_city2)

# Visualizar el heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(similaridades.astype(float), annot=True, cmap='coolwarm', cbar=True)
plt.title('Heatmap de Similaridad entre Ciudades')
plt.show()