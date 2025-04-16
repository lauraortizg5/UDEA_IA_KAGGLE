import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#ruta del archivo
ruta_archivo = r"C:\Users\laura\OneDrive\Escritorio\Universidad\2025-1\IA\train.csv"

# Leer el archivo CSV
df = pd.read_csv(ruta_archivo)

# Mostrar las primeras filas para ver la estructura básica de los datos
print("Primeras filas del DataFrame:")
print(df.head())

#Tabla cruzada con los conteos de estudiantes según "FAMI_EDUCACIONPADRE" y "FAMI_TIENEINTERNET"
ct_counts = pd.crosstab(df["FAMI_EDUCACIONPADRE"], df["FAMI_TIENEINTERNET"])
print("\nTabla cruzada (conteos absolutos):")
print(ct_counts)

#Gráfico de Barras Agrupadas

plt.figure(figsize=(10, 6))
ct_counts.plot(kind='bar', stacked=False, color=['steelblue', 'salmon'])
plt.title("Educación del Padre vs. Tener Internet (Conteos absolutos)")
plt.xlabel("FAMI_EDUCACIONPADRE")
plt.ylabel("Cantidad de estudiantes")
plt.xticks(rotation=45, ha='right')
plt.legend(title="FAMI_TIENEINTERNET", loc='upper right')
plt.tight_layout()
plt.show()

# Normalizar la tabla cruzada por filas para obtener proporciones
ct_props = ct_counts.div(ct_counts.sum(axis=1), axis=0)
print("\nTabla cruzada (proporciones):")
print(ct_props)

plt.figure(figsize=(10, 6))
ct_props.plot(kind='bar', stacked=False, color=['steelblue', 'salmon'])
plt.title("Proporción de estudiantes con/sin Internet según Educación del Padre")
plt.xlabel("FAMI_EDUCACIONPADRE")
plt.ylabel("Proporción")
plt.xticks(rotation=45, ha='right')
plt.legend(title="FAMI_TIENEINTERNET", loc='upper right')
plt.tight_layout()
plt.show()


if "ESTU_HORASEMANATRABAJA" in df.columns:
    print("\nDistribución de ESTU_HORASEMANATRABAJA:")
    counts_trabajo = df["ESTU_HORASEMANATRABAJA"].value_counts(dropna=False).sort_index()
    print(counts_trabajo)
    
    # Gráfico de barras para la distribución de horas de trabajo semanal
    plt.figure(figsize=(8, 5))
    plt.bar(counts_trabajo.index.astype(str), counts_trabajo.values, color='teal')
    plt.title("Distribución de Horas Semanales de Trabajo del Estudiante")
    plt.xlabel("Horas de Trabajo")
    plt.ylabel("Cantidad de estudiantes")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()
else:
    print("\nLa columna 'ESTU_HORASEMANATRABAJA' no se encuentra en el DataFrame.")

#variables numéricas: Matriz de correlación

# Seleccionar solo las columnas numéricas
numeric_vars = df.select_dtypes(include=[np.number])
print("\nMatriz de correlación (Variables Numéricas):")
print(numeric_vars.corr())

# Visualizar la matriz de correlación con un mapa de calor usando Matplotlib
plt.figure(figsize=(10, 8))
corr = numeric_vars.corr()
plt.imshow(corr, cmap='coolwarm', interpolation='none')
plt.colorbar()
plt.xticks(range(len(corr)), corr.columns, rotation=90)
plt.yticks(range(len(corr)), corr.index)
plt.title("Matriz de Correlación de Variables Numéricas")
plt.tight_layout()
plt.show()



