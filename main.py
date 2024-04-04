import limpieza_de_datos
import pandas as pd
import nn_model


# Limpieza de los datasets
# limpieza_de_datos.clean_file('Data_train.csv', 'dataset_train.csv')
# limpieza_de_datos.clean_file('Data_test.csv', 'dataset_test.csv')

# Entrenamiento de la red neuronal
nn_model.train(lr=0.001, epochs=10000, prints=100)
nn_model.print_plots()
nn_model.test()
