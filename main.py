import pandas as pd
import numpy as np

# Lectura del dataset de entrenamiento
df = pd.read_csv('Data_train.csv', header=None)

# Se elimina la primera fila de datos
df = df.iloc[1:]

# Se reemplazan los headers por sus nombres
headers = ["duration", "protocol_type", "service", "flag", "src_bytes", "dst_bytes", "land",
           "wrong_fragment", "urgent", "hot", "num_failed_logins", "logged_in",
           "num_compromised", "root_shell", "su_attempted", "num_root", "num_file_creations",
           "num_shells", "num_access_files", "num_outbound_cmds", "is_host_login",
           "is_guest_login", "count", "srv_count", "serror_rate", "srv_serror_rate",
           "rerror_rate", "srv_rerror_rate", "same_srv_rate", "diff_srv_rate",
           "srv_diff_host_rate", "dst_host_count", "dst_host_srv_count", "dst_host_same_srv_rate",
           "dst_host_diff_srv_rate", "dst_host_same_src_port_rate",
           "dst_host_srv_diff_host_rate", "dst_host_serror_rate", "dst_host_srv_serror_rate",
           "dst_host_rerror_rate", "dst_host_srv_rerror_rate", "attack", "last_flag"]
df.columns = headers

# Se buscan los posibles datos nulos
print(df.isnull().sum())

# Se listan los valores únicos de la columna 'protocol_type' y se almacena la cantidad de estos valores unicos
print(df['protocol_type'].unique())
print(len(df['protocol_type'].unique()))

# Se cambian los valores de la columna 'protocol_type' por valores numéricos
df['protocol_type'] = df['protocol_type'].replace(df['protocol_type'].unique(), range(len(df['protocol_type'].unique())))

# Se listan los valores únicos de la columna 'service' y se almacena la cantidad de estos valores unicos
print(df['service'].unique())
print(len(df['service'].unique()))

# Se cambian los valores de la columna 'service' por valores numéricos
df['service'] = df['service'].replace(df['service'].unique(), range(len(df['service'].unique())))




df.to_csv('dataset_train.csv', index=False, header=True)