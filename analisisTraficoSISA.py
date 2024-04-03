import pandas as pd
import numpy as np

# Lectura del dataset de entrenamiento
df = pd.read_csv('SISA Trafic/prb.csv', header=None)

# Obtenemos datos de la primera fila y lo guardamos em headers


df.to_csv('TraficoSISA.csv', index=False, header=True)