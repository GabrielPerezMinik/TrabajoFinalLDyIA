import pandas as pd
import numpy as np
from datetime import date
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error,mean_absolute_error
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader, TensorDataset



df = pd.read_csv("data/clean_data.csv",encoding="latin-1",low_memory=False)  #iso-8859-1

df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate']).dt.date
df.set_index('InvoiceDate', inplace=True)

df['profit'] = df['UnitPrice'] * df['Quantity']

# Filtrar datos de entrenamiento y prueba
train_df = df.loc[date(2010, 12, 1):date(2011, 11, 8)].copy()
test_df = df.loc[date(2011, 11, 9):date(2011, 12, 9)].copy()

# Preprocesar datos
train_df.loc[:, 'total_sales'] = train_df['Quantity'] * train_df['UnitPrice']
test_df.loc[:, 'total_sales'] = test_df['Quantity'] * test_df['UnitPrice']
scaler = MinMaxScaler()
train_scaled = scaler.fit_transform(train_df[['total_sales']])
print("Train Data Shape:", train_df.shape)
print("Test Data Shape:", test_df.shape)
test_scaled = scaler.transform(test_df[['total_sales']])


def create_sequences(data, seq_length):
    sequences, targets = [], []
    for i in range(len(data) - seq_length):
        sequences.append(data[i:i + seq_length])
        targets.append(data[i + seq_length])
    return np.array(sequences), np.array(targets)



seq_length = 30  # Usamos 30 días de historial para predecir
X_train, y_train = create_sequences(train_scaled, seq_length)
X_test, y_test = create_sequences(test_scaled, seq_length)


# Convertir a tensores de PyTorch
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32)

# Crear DataLoader para entrenamiento
batch_size = 32
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

# Definir modelo LSTM
class LSTMModel(nn.Module):
    def __init__(self, input_size=1, hidden_size=50, num_layers=2):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        return self.fc(lstm_out[:, -1, :])  # Tomamos la última salida del LSTM


# Instanciar modelo y enviar a GPU si está disponible
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("use: " + str(device) + " to train the model" )
model = LSTMModel().to(device)

# Definir función de pérdida y optimizador
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Entrenamiento del modelo
epochs = 20
for epoch in range(epochs):
    model.train()
    epoch_loss = 0
    for batch_X, batch_y in train_dataloader:
        batch_X, batch_y = batch_X.to(device), batch_y.to(device)
        optimizer.zero_grad()
        predictions = model(batch_X)
        loss = criterion(predictions, batch_y)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
    print(f"Epoch {epoch + 1}/{epochs}, Loss: {epoch_loss / len(train_dataloader):.6f}")

# Evaluación del modelo
model.eval()
y_pred = model(X_test_tensor.to(device)).cpu().detach().numpy()
y_test_actual = y_test_tensor.numpy()

# Invertir escalado para graficar
y_pred_inv = scaler.inverse_transform(y_pred)
y_test_inv = scaler.inverse_transform(y_test_actual)

epsilon=1e-10

def mean_absolute_percentage_error(y_true, y_pred,epsilon=1e-10 ):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    # Evitar división por cero usando np.maximum
    return np.mean(np.abs((y_true - y_pred) / np.maximum(np.abs(y_true), epsilon))) * 100

ape = np.abs((y_test_inv - y_pred_inv) / np.maximum(np.abs(y_test_inv), epsilon)) * 100
squared_error = (y_test_inv - y_pred_inv) ** 2
absolute_error = np.abs(y_test_inv - y_pred_inv)

mape_global = mean_absolute_percentage_error(y_test_inv, y_pred_inv)
rmse_global = np.sqrt(mean_squared_error(y_test_inv, y_pred_inv))
mae_global = mean_absolute_error(y_test_inv, y_pred_inv)

print(f"MAPE Global: {mape_global:.2f}%")
print(f"RMSE Global: {rmse_global:.4f}")
print(f"MAE Global: {mae_global:.4f}")

# Graficar predicciones vs datos reales
plt.figure(figsize=(12, 6))
#plt.plot(test_df.index[seq_length:], y_test_actual, label='Actual Sales', color='blue')
#plt.plot(test_df.index[seq_length:], y_pred , label='Predicted Sales', color='red', linestyle='dashed')
plt.plot(test_df.index[seq_length:], y_test_inv.flatten(), label='Actual Sales', color='blue')
plt.plot(test_df.index[seq_length:], y_pred_inv.flatten(), label='Predicted Sales', color='red', linestyle='dashed')
plt.xlabel('Date')
plt.ylabel('Total sales')
plt.title('Prediction of total sales with LSTM')
plt.legend()
plt.show()



# Guardar el modelo
#torch.save(model.state_dict(), "models/lstm_sales_model.pth")

# Suponiendo que test_df es el DataFrame de test original y que al crear secuencias
# se perdieron las primeras 'seq_length' fechas, usaremos:
dates = test_df.index[seq_length:]

# Verifica que la longitud de 'dates' coincida con la de los arrays de error
if len(dates) != len(ape):
    print("¡Atención! La longitud de las fechas y de los errores no coincide.")
    # Puedes crear un array de índices o fechas artificiales si lo prefieres:
    dates = np.arange(len(ape))

# Graficar los errores por muestra en tres subplots:
fig, axs = plt.subplots(3, 1, figsize=(12, 15))

# Gráfico del APE (MAPE por muestra)
axs[0].plot(dates, ape, marker='o', color='purple')
axs[0].set_title('Error Porcentual Absolute (APE) for sample')
axs[0].set_ylabel('Error (%)')
axs[0].set_xlabel('Date')

# Gráfico del error cuadrático (MSE por muestra)
axs[1].plot(dates, squared_error, marker='o', color='red')
axs[1].set_title('Error Cuadrático for sample')
axs[1].set_ylabel('Error Cuadrático')
axs[1].set_xlabel('Date')

# Gráfico del error absoluto (MAE por muestra)
axs[2].plot(dates, absolute_error, marker='o', color='blue')
axs[2].set_title('Error Absolute for sample')
axs[2].set_ylabel('Error Absolute')
axs[2].set_xlabel('Date')

plt.tight_layout()
plt.show()