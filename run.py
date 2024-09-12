import torch
import torch.nn as nn
import torch.optim as optim
from abc import ABC, abstractmethod
import numpy as np
import pandas as pd
from sktime.performance_metrics.forecasting import mean_absolute_percentage_error
from sktime.split import SlidingWindowSplitter
from torch.utils.data import DataLoader, TensorDataset
from datetime import datetime
import json
import pickle
import numpy as np
import pandas as pd
import os
import argparse
import time

os.environ["CUDA_VISIBLE_DEVICES"] = "0"


def custom_date_parser(date_str) -> pd.DatetimeIndex:
    cleaned_date_str = date_str.replace(" b", "").strip()
    return pd.to_datetime(cleaned_date_str, format='%d.%m.%Y %H:%M:%S')


# Define the Encoder
class EncoderLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers):
        super(EncoderLSTM, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # Define LSTM
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
    
    def forward(self, x):
        # Initialize hidden and cell state
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)
        
        # Pass input through LSTM
        output, (hn, cn) = self.lstm(x, (h0, c0))
        
        # hn and cn will be passed to the decoder
        return output, hn, cn


class DecoderLSTM(nn.Module):
    def __init__(self, output_dim, hidden_dim, num_layers):
        super(DecoderLSTM, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # LSTM layer
        self.lstm = nn.LSTM(output_dim, hidden_dim, num_layers, batch_first=True)
        
        # Fully connected layer to output real values
        self.fc = nn.Linear(hidden_dim, output_dim)
    
    def forward(self, x, hidden, cell):
        # Pass input through the LSTM
        output, (hn, cn) = self.lstm(x, (hidden, cell))
        
        # Pass through the fully connected layer to get real-valued predictions
        output = self.fc(output)
        return output, hn, cn


class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, device):
        super(Seq2Seq, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device
    
    def forward(self, source, target, teacher_forcing_ratio=0.5):
        batch_size = source.size(0)
        target_len = target.size(1)
        target_dim = target.size(2)
        
        # Tensor to store decoder outputs (real-valued)
        outputs = torch.zeros(batch_size, target_len - 1, target_dim).to(self.device)
        
        # Encode the source sequence
        encoder_output, hidden, cell = self.encoder(source)
        
        # First input to the decoder is the first time step of the target
        decoder_input = target[:, 0, :].unsqueeze(1)  # (batch_size, 1, output_dim)
        
        for t in range(0, target_len - 1):
            # Pass through the decoder
            output, hidden, cell = self.decoder(decoder_input, hidden, cell)
            
            # Store the output (real-valued prediction)
            outputs[:, t, :] = output.squeeze(1)
            
            # Decide whether to use teacher forcing
            teacher_force = torch.rand(1).item() < teacher_forcing_ratio
            decoder_input = target[:, t, :].unsqueeze(1) if teacher_force else output
        
        return outputs


def combined_loss(y_true_load, y_pred_load, y_true_outlier, alpha: float = 10.):
    mse_loss = nn.MSELoss()(y_pred_load, y_true_load)
    outlier_mse_loss = nn.MSELoss()(y_pred_load * y_true_outlier, y_true_load * y_true_outlier)
    
    total_loss = mse_loss + alpha * outlier_mse_loss
    return total_loss


class Forecaster(ABC):    
    def __init__(self, year: int, ig: int, lookback:int, forecast_horizon: int, mode: str = "tune"):
        self.year = year
        self.ig = ig
        self.dataset, self.train_idx, self.val_idx = self.load_data(year, ig, mode)
        self.lookback = lookback
        self.forecast_horizon = forecast_horizon
        self.is_trained = False
        self.mode = mode

    @staticmethod
    def load_data(year: int, ig: int, mode: str) -> tuple[pd.DataFrame, pd.Index, pd.Index]:
        if year == 2017:
            ig_str = f"LG {ig:02d}"
        else:
            ig_str = f"LG {ig:01d}"

        train = pd.read_csv(f"{mode}/{year}_train.csv")
        val = pd.read_csv(f"{mode}/{year}_test.csv")
        dataset = pd.concat([train, val], axis=0)

        dataset = dataset[["Time stamp", ig_str]]
        dataset[ig_str] = dataset[ig_str].astype(float)
        dataset.columns = ["Time stamp", "target"]
        dataset["Time stamp"] = pd.to_datetime(dataset.loc[:, "Time stamp"], errors='coerce')
        
        # drop rows with duplicate timestamps
        dataset = dataset.drop_duplicates(subset=["Time stamp"], keep="first")

        train_idx = dataset.iloc[:int(len(dataset) * 0.8), :].index
        val_idx = dataset.iloc[int(len(dataset) * 0.8):, :].index

        return dataset, train_idx, val_idx
    
    @staticmethod
    def get_public_holidays(year: int) -> list[datetime]:
        with open(f"holidays_{year}.json", 'r', encoding='utf-8') as f:
            data = json.load(f)

        # Extract unique dates and convert them to datetime objects
        holiday_dates = set()
        for state in data:
            for holiday_info in data[state].values():
                date_str = holiday_info["datum"]
                # holiday_dates.add(datetime.strptime(date_str, "%Y-%m-%d"))
                holiday_dates.add(datetime.strptime(date_str, "%Y-%m-%d").date())

        return list(holiday_dates)

    @abstractmethod
    def preprocess(self):
        raise

    @abstractmethod
    def train(self):
        pass

    @abstractmethod
    def predict(self):
        pass

    @abstractmethod
    def update(self, train_data):
        pass


class TorchForecaster(Forecaster):
    def __init__(
            self,
            year: int,
            ig: int,
            model: nn.Module,
            loss_fn: nn.Module,
            learning_rate: float = 0.001,
            batch_size: int = 32,
            peak_threshold: float = 0.85,
            lookback: int = 672,
            forecast_horizon: int = 48,
            mode: str = "tune",
            alpha: float = 2.
        ):
        super().__init__(year, ig, lookback, forecast_horizon, mode=mode)
        self.model = model
        self.loss_fn = loss_fn
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.alpha = alpha

        self.peak_threshold = peak_threshold

        self.mean = self.dataset.loc[self.train_idx, "target"].mean()
        self.std = self.dataset.loc[self.train_idx, "target"].std()
        self.peak_value = self.dataset.loc[self.train_idx, "target"].max() * self.peak_threshold
        self.normalized_peak_vaue = (self.peak_value - self.mean) / self.std

        self.train_splitter = SlidingWindowSplitter(
            fh=range(self.forecast_horizon + 1), 
            window_length=self.lookback,
            step_length=1
        )

        self.val_splitter = SlidingWindowSplitter(
            fh=range(self.forecast_horizon), 
            window_length=self.lookback,
            step_length=1
        )

        self.holidays = self.get_public_holidays(self.year)

        self.training_process = []

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.n_updates = 0
        self.total_epochs = 0

    def augment(self, train_data: pd.DataFrame) -> pd.DataFrame:
        """Augment the training data with additional features."""

        train_data.loc[:, "target_normalized"] = (train_data.loc[:, "target"] - self.mean) / self.std
        train_data.loc[:, "is_peak"] = (train_data.loc[:, "target_normalized"] >= self.normalized_peak_vaue).astype(int)

        train_data.loc[:, "dow"] = train_data["Time stamp"].dt.dayofweek
        train_data.loc[:, "hour"] = train_data["Time stamp"].dt.hour

        train_data.loc[:, "is_holiday"] = train_data["Time stamp"].dt.date.isin(self.holidays).astype(int)

        return train_data

    def preprocess(self):
        """Data cleaning and normalization"""
        self.dataset.loc[:, "target"] = self.dataset.loc[:, "target"].interpolate()
        self.dataset = self.dataset.dropna(subset=["target"])

        self.dataset = self.augment(self.dataset)

    def get_dataset(self, split: str = "train") -> TensorDataset:
        print(f"Creating {split} dataset...")
        if split == "train":
            data = self.dataset.iloc[:int(len(self.dataset) * 0.8), :]
            splitter = self.train_splitter
        else:
            data_train = self.dataset.iloc[-self.lookback:, :]
            data_val = self.dataset.iloc[int(len(self.dataset) * 0.8):, :]
            data = pd.concat([data_train, data_val], axis=0, ignore_index=True)
            splitter = self.val_splitter

        X_train_windows = []
        y_train_windows = []

        X_data = data[["target_normalized", "dow", "hour", "is_peak", "is_holiday"]].values
        y_data = data[["target_normalized", "is_peak"]].values

        for X_idx, y_idx in splitter.split(X_data):
            X_train, y_train = X_data[X_idx], y_data[y_idx]

            X_train_windows.append(X_train)
            y_train_windows.append(y_train)

        X_train_tensor = torch.tensor(np.array(X_train_windows), dtype=torch.float32)
        y_train_tensor = torch.tensor(np.array(y_train_windows), dtype=torch.float32)

        dataset = TensorDataset(X_train_tensor, y_train_tensor)

        print("Done.")

        return dataset

    def train(self, epochs: int):
        """Fit the internal model(s)"""
        optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)

        train_dataset = self.get_dataset(split="train")
        val_dataset = self.get_dataset(split="val")
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False)
        
        best_model_state_dict = None
        best_loss = np.inf

        for epoch in range(self.total_epochs, self.total_epochs + epochs):
            self.model.train()
            train_loss = []
            for i, (x_window, y_window) in enumerate(train_loader):
                x_window = x_window.to(self.device)
                y_window = y_window.to(self.device)

                y_window_load = y_window[:, :, 0].unsqueeze(-1)
                y_window_outlier = y_window[:, :, 1].unsqueeze(-1)

                optimizer.zero_grad()
                y_pred_window = self.model(x_window, y_window_load)
                loss = combined_loss(
                    y_true_load=y_window_load[:, 1:, :],
                    y_pred_load=y_pred_window,
                    y_true_outlier=y_window_outlier[:, 1:, :],
                    alpha=self.alpha
                )
                loss.backward()
                optimizer.step()
                train_loss += [loss.item()]
            train_loss = np.mean(train_loss)

            self.model.eval()
            val_loss = []
            with torch.no_grad():
                for x_window, y_window in val_loader:
                    x_window = x_window.to(self.device)
                    y_window = y_window.to(self.device)

                    y_window_load = y_window[:, :, 0].unsqueeze(-1)
                    y_window_outlier = y_window[:, :, 1].unsqueeze(-1)

                    y_pred_window = self.model(x_window, y_window_load, teacher_forcing_ratio=0.)
                    loss = combined_loss(
                        y_true_load=y_window_load[:, 1:, :],
                        y_pred_load=y_pred_window,
                        y_true_outlier=y_window_outlier[:, 1:, :],
                        alpha=self.alpha
                    )
                    val_loss += [loss.item()]
            val_loss = np.mean(val_loss)

            if val_loss < best_loss:
                best_loss = val_loss
                best_model_state_dict = self.model.state_dict()

            print(f"Epoch {epoch + 1} - Train Loss: {train_loss:.2f}, Val Loss: {val_loss:.2f}")
            self.training_process +=[{
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_loss
            }]

            assert best_model_state_dict is not None
            self.model.load_state_dict(best_model_state_dict)

        self.total_epochs += epochs
        pd.DataFrame(self.training_process).to_csv(f"training_process_{self.year}_{self.ig}.csv", index=False)
        self.is_trained = True
            
    def predict(self, time_index: pd.Index) -> np.ndarray:
        """Predict the next value(s)"""
        self.model.eval()

        train_data = self.dataset.iloc[-self.lookback:, :]
        x_window = train_data.loc[:, ["target_normalized", "dow", "hour", "is_peak", "is_holiday"]].values
        
        x_window = torch.tensor(x_window, dtype=torch.float32).to(self.device).unsqueeze(0)

        y_target_load = torch.zeros(1, self.forecast_horizon, 1, dtype=torch.float32).to(self.device)

        extracted_value = x_window[:, -1, 0].unsqueeze(1).unsqueeze(2)  # Shape: [1, 1, 1]

        y_window_load = torch.cat((extracted_value, y_target_load), dim=1)

        y_pred_window = self.model(x_window, y_window_load, teacher_forcing_ratio=0.)

        y_pred_window = y_pred_window.squeeze()

        y_pred_window = y_pred_window.detach().cpu().numpy()

        y_pred_window = y_pred_window * self.std + self.mean
        y_pred_window = np.clip(y_pred_window, 0, None)
        
        return y_pred_window

    def update(self, train_data: pd.DataFrame):
        """Update the model with the new value (if required)"""
        train_data = train_data.reset_index()
        train_data.columns = ["Time stamp", "target"]

        self.dataset = train_data

        self.dataset = self.dataset.drop_duplicates(subset=["Time stamp"], keep="first")
        self.dataset = self.augment(self.dataset)

        self.n_updates += 1

def train_and_evaluate(year: int):
    input_dim = 5
    output_dim = 1  
    hidden_dim = 64 
    num_layers = 2  
    learning_rate = 0.001
    epochs = 2
    finetune_epochs = 2
    alpha = 2.
    finetune_frequency = 115 // finetune_epochs

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if year == 2016:
        load_profiles = pd.read_csv('LoadProfile_20IPs_2016.csv', skiprows=1, delimiter=";", index_col=0, date_parser=custom_date_parser)
    else:
        load_profiles = pd.read_csv('LoadProfile_30IPs_2017.csv', skiprows=1, delimiter=";", index_col=0, date_parser=custom_date_parser)
    
    load_profiles = load_profiles[~load_profiles.index.duplicated(keep='first')]
    load_profiles = load_profiles.interpolate()
    
    actuals = pd.read_csv(f'tobi/{year}_actuals.csv', index_col=0, parse_dates=True)
    peak_actuals = pd.read_csv(f'tobi/{year}_peak_actuals.csv', index_col=0, parse_dates=True)

    forecasts = actuals.copy()
    forecasts.loc[:, forecasts.columns != 'dataset_id'] = 0

    # for dataset_id in actuals['dataset_id'].unique():
    for load in [x for x in actuals.columns if x != 'dataset_id']:
        start = time.time()
        actuals_load = actuals[[load, 'dataset_id']]

        encoder = EncoderLSTM(input_dim, hidden_dim, num_layers).to(device)
        decoder = DecoderLSTM(output_dim, hidden_dim, num_layers).to(device)
        model = Seq2Seq(encoder, decoder, device).to(device)

        forecaster = TorchForecaster(
            year=year,
            ig=int(load.replace("LG", "").strip()),
            model=model,
            learning_rate=learning_rate,
            loss_fn=combined_loss,
            lookback=672,
            forecast_horizon=48,
            mode="test",
            alpha=alpha
        )
        forecaster.preprocess()

        def the_fancy_forecaster(time_index_to_forecast, train_data):
            forecaster.update(train_data)

            if not forecaster.is_trained:
                print(f"Training year {year} for load {load}...")
                forecaster.train(epochs=epochs)
                print("Done.")
            elif forecaster.n_updates % finetune_frequency == 0:
                print(f"Fine-tuning year {year} for load {load}...")
                forecaster.train(epochs=1)
                print("Done.")

            prediction = forecaster.predict(time_index_to_forecast)
            return prediction

        for dataset_id in actuals_load['dataset_id'].unique():
            actuals_i = actuals_load[actuals_load['dataset_id'] == dataset_id]
            actuals_i_j = actuals_i[load]
            start_of_test = actuals_i_j.index[0]
            train_data_i_j = load_profiles[load][:start_of_test - pd.Timedelta(minutes=15)]
            forecast = the_fancy_forecaster(actuals_i_j.index, train_data_i_j)
            forecasts.loc[actuals_i_j.index, load] = forecast

        runtime = time.time() - start
        print(f"Minutes taken: {runtime / 60}")

    def calculate_rmse(actual, predicted):
        return np.sqrt(((actual - predicted) ** 2).mean())

    # Calculate RMSE for each pair of actual and predicted columns
    rmse_results = {}
    rmse_peak_results = {}

    for col in [x for x in forecasts.columns if x != 'dataset_id']:
        forecast = forecasts[col]
        actual = actuals[col]
        peak_actual = peak_actuals[col]
        rmse_results[col] = calculate_rmse(actual, forecast)
        rmse_peak_results[col] = calculate_rmse(peak_actual[peak_actual != 0], forecast[peak_actual != 0])

    # Output RMSE for each column pair
    print(rmse_results)
    print(rmse_peak_results)

    forecasts.to_csv(f"forecasts_{year}.csv")

    # save results as pickle files
    with open(f"results_{year}.pkl", "wb") as f:
        pickle.dump(rmse_results, f)

    with open(f"results_peak_{year}.pkl", "wb") as f:
        pickle.dump(rmse_peak_results, f)





if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--year", type=int, required=True)
    args = parser.parse_args()

    train_and_evaluate(args.year)
    plot(args.year)


