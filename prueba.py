import pandas as pd
import numpy as np
import matplotlib.pyplot as plt  # Corregido: Importar matplotlib.pyplot

"""
*********************************************
1. SELECCION DEL CONJUNTO DE DATOS
*********************************************
"""
# Se tomó data real del SENAMHI sobre Contaminantes del Aire

#https://repositorio.senamhi.gob.pe/handle/20.500.12542/2467





"""
*********************************************
2. CARGA Y PROCESAMIENTO DE DATOS
*********************************************
"""
# Lectura y Carga del archivo CSV original
df = pd.read_csv('datos_Ate.csv', sep=';')
print(f"\n{'='*130}\nHacemos la carga del archivo csv original datos_Ate.csv\n")

# Información general del dataset original numero de columnas ,numero de filas y tipo de datos
print(f"\n{'='*130}\nInformación general del dataset original numero de columnas ,numero de filas y tipo de datos:\n")
print(df.info())

#Manejo de valores nulos y datos faltantes.
df_nulos = df.copy()

# Llenar nulos en 'PM 10' con la media
df_nulos['PM 10'].fillna(df_nulos['PM 10'].mean(), inplace=True)

# Llenar nulos en 'NO2' con la mediana
df_nulos['NO2'].fillna(df_nulos['NO2'].median(), inplace=True)

# Llenar nulos en 'O3' con valores aleatorios entre 200 y 500
nulos_o3 = df_nulos['O3'].isnull().sum()
df_nulos['O3'] = np.random.randint(200, 500, size=len(df_nulos))

# Llenar nulos en 'CO' con valores aleatorios entre 360 y 700
nulos_co = df_nulos['CO'].isnull().sum()
df_nulos['CO'] = np.random.randint(360, 700, size=len(df_nulos))

# Eliminar la columna 'PM 2.5' ya que todos sus valores son nulos
df_nulos.drop('PM 2.5', axis=1, inplace=True)

# Guardar el DataFrame con nulos manejados
df_nulos.to_csv('datos_nulos_agregados_Ate.csv', index=False)
print("DataFrame guardado en datos_nulos_agregados_Ate.csv")

# Carga del DataFrame modificado (con O3 y CO rellenados)
df_modificado = pd.read_csv('datos_nulos_agregados_Ate.csv')
print(f"{'='*130}\n")

# Operaciones con NumPy y creación de nuevas columnas en el DataFrame modificado
print(f"{'='*130}\n\nOperaciones con NumPy y creación de nuevas columnas\n\n{'='*130}\n")
df_modificado['ICA'] = np.sqrt(df_modificado['PM 10']**2 + df_modificado['NO2']**2)  # Ejemplo simplificado de ICA
df_modificado['Promedio_Gaseosos'] = df_modificado[['NO2', 'O3', 'CO', 'SO2']].mean(axis=1)
print(df_modificado)



"""
*********************************************
3. ANALISIS DE DATOS
*********************************************
"""
# Análisis descriptivo
print(f"\n{'='*130}\nAnálisis descriptivo:\n")
print(df_modificado.describe())

# Identificación de patrones y correlaciones
print(f"\n{'='*130}\nCorrelaciones:\n")
#para las columnas de contaminantes para calcular las correlaciones
correlaciones_df = df_modificado[['DIA','HORA','PM 10', 'SO2', 'NO2', 'O3', 'CO','ICA','Promedio_Gaseosos',]]
print(correlaciones_df.corr())


# Aplicación de filtros y segmentaciones

#primer filtro
#indicador mayor a 100 en PM 10
print(f"\n{'='*130}\nDatos con PM 10 mayor a 100:\n")
print(df_modificado[df_modificado['PM 10'] > 100])


#falta una segmentacion
#datos del dia lunes
print(f"\n{'='*130}\nDatos del día Lunes:\n")
print(df_modificado[df_modificado['DIA'] == 'Lunes'])


#Segundo filtro 
#indicador SO2 menor a 50
print(f"\n{'='*130}\nDatos con SO2 menor a 50:\n")
print(df_modificado[df_modificado['SO2'] < 50])



"""
*********************************************
4. VISUALIZACION DE DATOS
*********************************************
"""

# Visualización de datos
# Gráfico de líneas para PM 10
plt.figure(figsize=(10, 5))
plt.plot(df_modificado['HORA'], df_modificado['PM 10'])
plt.title('PM 10 por Hora')
plt.xlabel('Hora')
plt.ylabel('PM 10')
plt.grid(True)
plt.show()

# Gráfico de barras para SO2
plt.figure(figsize=(10, 5))
plt.bar(df_modificado['HORA'], df_modificado['SO2'])
plt.title('SO2 por Hora')
plt.xlabel('Hora')
plt.ylabel('SO2')
plt.grid(True)
plt.show()

# Histograma para NO2
plt.figure(figsize=(8, 5))
plt.hist(df_modificado['NO2'], bins=10)
plt.title('Histograma de NO2')
plt.xlabel('NO2')
plt.ylabel('Frecuencia')
plt.grid(True)
plt.show()

"""
GRAFICO DE TORTA
"""
# Definir los rangos de horas y sus etiquetas
bins = [0, 6, 12, 18, 24]
labels = ['Madrugada', 'Mañana', 'Tarde', 'Noche']

# Crear una nueva columna 'Rango_Hora' basada en los rangos
df_modificado['Rango_Hora'] = pd.cut(df_modificado['HORA'], bins=bins, right=False, labels=labels)

# Calcular el promedio de ICA para cada rango de hora
ica_por_rango = df_modificado.groupby('Rango_Hora')['ICA'].mean()

# Crear el gráfico de torta
plt.figure(figsize=(8, 8))
ica_por_rango.plot(kind='pie', autopct='%1.1f%%', startangle=140)
plt.title('Distribución del ICA por Rango de Hora')
plt.ylabel('')  # No queremos la etiqueta del eje y en un gráfico de torta
plt.show()



# Boxplot del Índice de Riesgo de Calidad del Aire
plt.figure(figsize=(8, 5))
plt.boxplot(df_modificado['ICA']) # Cambiado a 'ICA' para que coincida con el DataFrame
plt.ylabel('Índice de Riesgo')
plt.title('Distribución del Índice de Riesgo de Calidad del Aire')
plt.grid()
plt.show()

# Gráfico de Índice de Contaminación
plt.figure(figsize=(10, 5))
plt.plot(df_modificado['HORA'], df_modificado['ICA'], label='Índice de Contaminación') # Cambiado a 'ICA'
plt.xlabel('Hora')
plt.ylabel('Índice de Contaminación')
plt.title('Evolución del Índice de Contaminación en el Tiempo')
plt.legend()
plt.grid()
plt.show()


# Gráfico de dispersión de Hora vs. ICA
plt.figure(figsize=(10, 6))
plt.scatter(df_modificado['HORA'], df_modificado['ICA'], alpha=0.5)  # alpha para transparencia
plt.title('Gráfico de Dispersión: Hora vs. Índice de Calidad del Aire (ICA)')
plt.xlabel('Hora')
plt.ylabel('Índice de Calidad del Aire (ICA)')
plt.grid(True)
plt.show()



