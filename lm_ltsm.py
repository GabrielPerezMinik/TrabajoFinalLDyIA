import pandas as pd
import numpy as np
from datetime import date
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader, TensorDataset


class Lltsm:
    # =============================================================================
    # Data Loading and Preprocessing
    # =============================================================================
    def load_and_preprocess_data(csv_path):
        """
        Loads the CSV file, converts 'InvoiceDate' to date format,
        sets it as index, and calculates profit.

        :param csv_path: Path to the CSV file.
        :return: Preprocessed DataFrame.
        """
        df = pd.read_csv(csv_path, encoding="latin-1", low_memory=False)
        # Convert 'InvoiceDate' to date and set as index
        df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate']).dt.date
        df.set_index('InvoiceDate', inplace=True)
        # Calculate profit
        df['profit'] = df['UnitPrice'] * df['Quantity']
        return df


    def split_train_test(df):
        """
        Splits the DataFrame into training and testing sets based on date ranges.

        :param df: Preprocessed DataFrame.
        :return: Tuple (train_df, test_df).
        """
        train_df = df.loc[date(2010, 12, 1):date(2011, 11, 8)].copy()
        test_df = df.loc[date(2011, 11, 9):date(2011, 12, 9)].copy()
        return train_df, test_df


    def add_total_sales(df):
        """
        Adds a 'total_sales' column computed as Quantity * UnitPrice.

        :param df: DataFrame.
        :return: DataFrame with 'total_sales' column.
        """
        df.loc[:, 'total_sales'] = df['Quantity'] * df['UnitPrice']
        return df


    # =============================================================================
    # Sequence Creation
    # =============================================================================
    def create_sequences(data, seq_length):
        """
        Creates sequences from the data where each sequence of length `seq_length`
        is used to predict the following value.

        :param data: Numpy array of scaled data.
        :param seq_length: Length of each sequence.
        :return: Tuple of numpy arrays (sequences, targets).
        """
        sequences, targets = [], []
        for i in range(len(data) - seq_length):
            sequences.append(data[i:i + seq_length])
            targets.append(data[i + seq_length])
        return np.array(sequences), np.array(targets)


    # =============================================================================
    # Define LSTM Model
    # =============================================================================
    class LSTMModel(nn.Module):
        """
        LSTM model for time series prediction.
        The model consists of an LSTM layer followed by a fully connected layer.
        """

        def __init__(self, input_size=1, hidden_size=50, num_layers=2):
            super(Lltsm.LSTMModel, self).__init__()
            self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
            self.fc = nn.Linear(hidden_size, 1)

        def forward(self, x):
            lstm_out, _ = self.lstm(x)
            # Return the output from the last time step
            return self.fc(lstm_out[:, -1, :])


# =============================================================================
# Main Function
# =============================================================================
def run():
    # File path to CSV
    csv_path = "data/clean_data.csv"

    # Load and preprocess the data
    df = Lltsm.load_and_preprocess_data(csv_path)

    # Split data into train and test sets
    train_df, test_df = Lltsm.split_train_test(df)

    # Add 'total_sales' column to both sets
    train_df = Lltsm.add_total_sales(train_df)
    test_df = Lltsm.add_total_sales(test_df)

    # Scale the 'total_sales' column using MinMaxScaler
    scaler = MinMaxScaler()
    train_scaled = scaler.fit_transform(train_df[['total_sales']])
    test_scaled = scaler.transform(test_df[['total_sales']])

    print("Train Data Shape:", train_df.shape)
    print("Test Data Shape:", test_df.shape)

    # Create sequences with 30 days of history for prediction
    seq_length = 30
    X_train, y_train = Lltsm.create_sequences(train_scaled, seq_length)
    X_test, y_test = Lltsm.create_sequences(test_scaled, seq_length)

    # Convert numpy arrays to PyTorch tensors
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test, dtype=torch.float32)

    # Create DataLoader for training
    batch_size = 32
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # Initialize the LSTM model and send to device (GPU if available)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)
    model = Lltsm.LSTMModel().to(device)

    # Define loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Train the model
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

    # Evaluate the model
    model.eval()
    y_pred = model(X_test_tensor.to(device)).cpu().detach().numpy()
    y_test_actual = y_test_tensor.numpy()

    # Inverse transform predictions and actual values to original scale
    y_pred_inv = scaler.inverse_transform(y_pred)
    y_test_inv = scaler.inverse_transform(y_test_actual)

    epsilon = 1e-10

    def mean_absolute_percentage_error(y_true, y_pred, epsilon=1e-10):
        """
        Computes the Mean Absolute Percentage Error (MAPE).

        :param y_true: True values.
        :param y_pred: Predicted values.
        :param epsilon: Small value to avoid division by zero.
        :return: MAPE in percentage.
        """
        y_true, y_pred = np.array(y_true), np.array(y_pred)
        return np.mean(np.abs((y_true - y_pred) / np.maximum(np.abs(y_true), epsilon))) * 100

    # Compute error metrics per sample
    ape = np.abs((y_test_inv - y_pred_inv) / np.maximum(np.abs(y_test_inv), epsilon)) * 100
    squared_error = (y_test_inv - y_pred_inv) ** 2
    absolute_error = np.abs(y_test_inv - y_pred_inv)

    # Calculate global error metrics
    mape_global = mean_absolute_percentage_error(y_test_inv, y_pred_inv)
    rmse_global = np.sqrt(mean_squared_error(y_test_inv, y_pred_inv))
    mae_global = mean_absolute_error(y_test_inv, y_pred_inv)

    print(f"Global MAPE: {mape_global:.2f}%")
    print(f"Global RMSE: {rmse_global:.4f}")
    print(f"Global MAE: {mae_global:.4f}")

    # Plot predictions vs actual sales
    plt.figure(figsize=(12, 6))
    plt.plot(test_df.index[seq_length:], y_test_inv.flatten(), label='Actual Sales', color='blue')
    plt.plot(test_df.index[seq_length:], y_pred_inv.flatten(), label='Predicted Sales', color='red', linestyle='dashed')
    plt.xlabel('Date')
    plt.ylabel('Total Sales')
    plt.title('Prediction of Total Sales with LSTM')
    plt.legend()
    plt.show()

    # Save the trained model
    torch.save(model.state_dict(), "models/lstm_sales_model.pth")

    # Plot error metrics per sample
    dates = test_df.index[seq_length:]
    if len(dates) != len(ape):
        print("Warning: Length of dates and error arrays do not match.")
        dates = np.arange(len(ape))

    fig, axs = plt.subplots(3, 1, figsize=(12, 15))
    axs[0].plot(dates, ape, marker='o', color='purple')
    axs[0].set_title('Absolute Percentage Error (APE) per Sample')
    axs[0].set_ylabel('Error (%)')
    axs[0].set_xlabel('Date')

    axs[1].plot(dates, squared_error, marker='o', color='red')
    axs[1].set_title('Squared Error per Sample')
    axs[1].set_ylabel('Squared Error')
    axs[1].set_xlabel('Date')

    axs[2].plot(dates, absolute_error, marker='o', color='blue')
    axs[2].set_title('Absolute Error per Sample')
    axs[2].set_ylabel('Absolute Error')
    axs[2].set_xlabel('Date')

    plt.tight_layout()
    plt.show()
