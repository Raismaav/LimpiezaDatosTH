import torch
import numpy as np
import pandas as pd
from torch import nn, optim
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset

data_train = pd.read_csv('datasets/dataset_train.csv')
data_test = pd.read_csv('datasets/dataset_test.csv')

# Preprocesar los datos
dataset_train = data_train.drop('attack', axis=1).values
labels_train = data_train['attack'].values

dataset_test = data_test.drop('attack', axis=1).values
labels_test = data_test['attack'].values

# Crear el normalizador
scaler = StandardScaler()

# Ajustar el normalizador a los datos de entrenamiento y transformarlos
dataset_train = scaler.fit_transform(dataset_train)

# Utilizar el mismo normalizador para transformar los datos de entrenamiento
dataset_train = scaler.transform(dataset_train)
dataset_test = scaler.transform(dataset_test)

# Convertir los datos a tensores
dataset_train = torch.FloatTensor(dataset_train).to('cpu')
dataset_test = torch.FloatTensor(dataset_test).to('cpu')
labels_train = torch.FloatTensor(labels_train).to('cpu')
labels_test = torch.FloatTensor(labels_test).to('cpu')

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.fc1 = nn.Linear(dataset_train.shape[1], 64)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(0.3)
        self.fc2 = nn.Linear(64, 32)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(0.3)
        self.fc3 = nn.Linear(32, 32)
        self.relu3 = nn.ReLU()
        self.fc4 = nn.Linear(32, 5)
        self.relu4 = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, input):
        output = self.fc1(input)
        output = self.relu1(output)
        output = self.dropout1(output)
        output = self.fc2(output)
        output = self.relu2(output)
        output = self.dropout2(output)
        output = self.fc3(output)
        output = self.relu3(output)
        output = self.fc4(output)
        output = self.relu4(output)
        output = self.sigmoid(output)
        return output


def train(model_name='model', lr=0.01, epochs=100, prints=10):
    model = MyModel()
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    history = pd.DataFrame()
    higher_accuracy = 0

    for epoch in range(epochs):
        predictions = model(dataset_train)
        loss = loss_fn(predictions, labels_train.long())
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if epoch % prints == 0:
            print(f'Epoch: {epoch}, Loss: {round(loss.item(), 4)}')

        with torch.no_grad():
            predictions = model(dataset_test)
            predictions = predictions.round()
            predictions = torch.argmax(predictions, dim=1)
            accuracy = predictions.eq(labels_test).sum() / float(labels_test.shape[0]) * 100




            if epoch % prints == 0:
                print(f'Accuracy: {round(accuracy.item(), 4)}%\n')

        df_tmp = pd.DataFrame(data={
            'Epoch': epoch,
            'Loss': round(loss.item(), 4),
            'Accuracy': round(accuracy.item(), 4),
        }, index=[0])
        history = pd.concat([history, df_tmp], ignore_index=True)

        #Se almacena el modelo en el punto con mayor precisión
        if accuracy > higher_accuracy:
            higher_accuracy = accuracy
            print(higher_accuracy)
            history.to_csv('mejor_resultado.csv', index=False, header=True)
            torch.save(model.state_dict(), f'best_{model_name}.pth')

    history.to_csv('resultados.csv', index=False, header=True)
    torch.save(model.state_dict(), f'{model_name}.pth')


def print_plots():
    import matplotlib.pyplot as plt
    import seaborn as sns
    df = pd.read_csv('resultados.csv')
    df = df.melt(id_vars=['Epoch'], value_vars=['Loss', 'Accuracy'])

    plt.figure(figsize=(12, 8))
    sns.lineplot(x='Epoch', y='value', hue='variable', data=df)
    plt.savefig('plot.png')
    plt.clf()


def create_confusion_matrix(true_labels, predicted_labels):
    # Crea la matriz de confusión
    from sklearn.metrics import confusion_matrix
    import matplotlib.pyplot as plt
    import seaborn as sns

    cm = confusion_matrix(true_labels, predicted_labels)

    # Visualiza la matriz de confusión como un gráfico de calor sin anotaciones
    plt.figure(figsize=(10, 7))
    sns.heatmap(cm, annot=False, cmap='Blues')
    plt.xlabel('Predicted')
    plt.ylabel('Expected')
    plt.show()

    return cm  # Devuelve la matriz de confusión


def test(model_name='model.pth'):
    import matplotlib.pyplot as plt
    import seaborn as sns
    model = MyModel()
    model.load_state_dict(torch.load(model_name))
    model.eval()

    with torch.no_grad():
        predictions = model(dataset_test)
        predictions = predictions.round()
        predictions = torch.argmax(predictions, dim=1)
        accuracy = predictions.eq(labels_test).sum() / float(labels_test.shape[0]) * 100
        print(f'Accuracy: {round(accuracy.item(), 4)}%\n')

    cm = create_confusion_matrix(labels_test, predictions)
    sns.heatmap(cm, annot=True)
    plt.savefig('matrix.png')
    plt.clf()


def eval(data, model_name='model.pth', rounded=False):
    # Carga el modelo
    model = MyModel()
    model.load_state_dict(torch.load(model_name))
    model.eval()

    # Normaliza los datos
    data = np.array(data).reshape(1, -1)
    data = scaler.transform(data)

    # Convierte los datos a tensores de PyTorch y asegúrate de que estén en el mismo dispositivo que el modelo
    data = torch.tensor(data).float().to('cpu')  # Cambia 'cpu' a 'cuda' si estás usando una GPU

    # Haz la predicción
    with torch.no_grad():
        predictions = model(data)
        if rounded:
            predictions = predictions.round()

    # Convierte las predicciones a una lista y devuélvelas
    return predictions.tolist()
