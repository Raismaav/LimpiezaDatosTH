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

# Movemos la columna 'Protocol' a la segunda posición
df = df[['Duration', 'Protocol'] + [col for col in df.columns if col != 'Duration' and col != 'Protocol']]
df = df.reset_index(drop=True)

# Cambiamos los valores 'TCP', 'UDP', 'ICMP', 'OTRO' por 0, 1, 2, 3
df['Protocol'] = df['Protocol'].replace(['TCP', 'UDP', 'ICMP', 'OTRO'], [0, 1, 2, 3])

# Creamos dos columnas 'Service' y 'Flag' con valores de 0 después de la columna 'Protocol'
df.insert(2, 'Service', 0)
df.insert(3, 'Flag', 0)

# Creamos dos columnas de tipo int64 'Str_bytes' y 'Dts_bytes' con valores de 0 y las ponemos en la posición 5 y 6
df.insert(4, 'Scr_bytes', 0)
df.insert(5, 'Dts_bytes', 0)

# Convertir la columna 'Length' a tipo int64
df['Length'] = df['Length'].astype('int64')

previous_destination = df.at[0, 'Destination']
previous_source = df.at[0, 'Source']
column = 'Scr_bytes'
df.at[0, column] += df.at[0, 'Length']
for i in range(1, len(df)):
    current_destination = df.at[i, 'Destination']
    current_source = df.at[i, 'Source']
    length = df.at[i, 'Length']

    if current_destination != previous_destination and current_source != previous_source:
        if column == 'Scr_bytes':
            column = 'Dts_bytes'
        else:
            column = 'Scr_bytes'
        if current_source != previous_source and current_source != previous_destination:
            column = 'Scr_bytes'

    df.at[i, column] += length
    previous_destination = current_destination
    previous_source = current_source

df.to_csv('SISA Trafic/TraficoSISA.csv', index=False, header=True)