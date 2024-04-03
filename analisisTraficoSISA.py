import pandas as pd
import numpy as np

# Lectura del dataset de entrenamiento
df = pd.read_csv('SISA Trafic/prb.csv', header=None)

# Obtenemos datos de la primera fila y lo guardamos en headers
headers = df.iloc[0]

# Asignamos los headers a las columnas
df.columns = headers

# Se elimina la primera fila de datos
df = df.iloc[1:]

# Eliminamos la columna 'No.'
df = df.drop(columns=['No.'])

# Convertimos la columna 'Time' a tipo float64
df['Time'] = df['Time'].astype('float64')

# Restamos los valores de tiempo siguiente con el tiempo actual de la columna 'Time'
# y lo guardamos en una nueva columna llamada 'Duration' y lo recorremos una posición hacia arriba
df['Duration'] = df['Time'].diff().shift(-1)

# Eliminanos la ultima fila
df = df.iloc[:-1]

# Movemos la columna 'Duration' a la primera posición
df = df[['Duration'] + [col for col in df.columns if col != 'Duration']]
df = df.reset_index(drop=True)

# Redondeamos la columna 'Duration' a enteros
df['Duration'] = df['Duration'].round(0).astype('int')

# Cambiamos los valores 'HTTP', 'TLSv1.2', 'TLSv1.3', 'DB-LSP-DISC', 'BROWSER', 'DHCP' por 'TCP'
df['Protocol'] = df['Protocol'].replace(['HTTP', 'TLSv1.2', 'TLSv1.3', 'DB-LSP-DISC', 'BROWSER', 'DHCP'], 'TCP')

# Cambiamos los valores 'QUIC', 'SSDP', 'MDNS', 'DNS', 'NBNS', 'LLMNR' por 'UDP'
df['Protocol'] = df['Protocol'].replace(['QUIC', 'SSDP', 'MDNS', 'DNS', 'NBNS', 'LLMNR'], 'UDP')

# Cambiamos los valores 'ICMPv6' por 'ICMP'
df['Protocol'] = df['Protocol'].replace('ICMPv6', 'ICMP')

# Cambiamos los valores 'DTLS', 'OCSP', '0x99ea' por 'OTRO'
df['Protocol'] = df['Protocol'].replace(['DTLS', 'OCSP', '0x99ea', 'ARP', 'IGMPv2', 'DB-LSP-DISC/JSON'], 'OTRO')


df.to_csv('TraficoSISA.csv', index=False, header=True)