import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
import argparse
import pickle
import json
import numpy as np


sns.set(style="whitegrid")
sns.set_palette("colorblind")


def custom_date_parser(date_str) -> pd.DatetimeIndex:
    cleaned_date_str = date_str.replace(" b", "").strip()
    return pd.to_datetime(cleaned_date_str, format='%d.%m.%Y %H:%M:%S')


def plot(year: int):
    with open(f"results_{year}.pkl", "rb") as f:
        rmse = pickle.load(f)

    with open(f"results_peak_{year}.pkl", "rb") as f:
        peak_rmse = pickle.load(f)

    with open(f"results_{year}_autogluon.json", "rb") as f:
        rmse_autogluon = json.load(f)

    with open(f"results_peak_{year}_autogluon.json", "rb") as f:
        peak_rmse_autogluon = json.load(f)

    # Create a DataFrame from the dictionaries
    df = pd.DataFrame({
        "Load": list(rmse.keys()), 
        "RMSE": list(rmse.values()), 
        "RMSE (Peaks)": list(peak_rmse.values())
    })

    # Create a DataFrame from the dictionaries for both models
    df = pd.DataFrame({
        "Load": list(rmse.keys()), 
        "LSTM: RMSE": list(rmse.values()), 
        "AutoGluon: RMSE": list(rmse_autogluon.values()), 
        "LSTM: RMSE (Peaks)": list(peak_rmse.values()),
        "AutoGluon: RMSE (Peaks)": list(peak_rmse_autogluon.values())
    })

    if year == 2016:
        load_profiles = pd.read_csv('LoadProfile_20IPs_2016.csv', skiprows=1, delimiter=";", index_col=0, date_parser=custom_date_parser)
    else:
        load_profiles = pd.read_csv('LoadProfile_30IPs_2017.csv', skiprows=1, delimiter=";", index_col=0, date_parser=custom_date_parser)

    means = load_profiles.mean()

    df['LSTM: RMSE'] = df['LSTM: RMSE'] / means[df['Load']].values
    df['AutoGluon: RMSE'] = df['AutoGluon: RMSE'] / means[df['Load']].values
    df['LSTM: RMSE (Peaks)'] = df['LSTM: RMSE (Peaks)'] / means[df['Load']].values
    df['AutoGluon: RMSE (Peaks)'] = df['AutoGluon: RMSE (Peaks)'] / means[df['Load']].values

    df["Load"] = df["Load"].str.replace("LG ", "").astype(int)

    # Melt the DataFrame to long format for Seaborn
    df_melted = df.melt(id_vars="Load", 
                        value_vars=["LSTM: RMSE", "AutoGluon: RMSE", "LSTM: RMSE (Peaks)", "AutoGluon: RMSE (Peaks)"],
                        var_name="Metric", value_name="Value")

    # Create the bar plot with an appropriate figure size
    plt.figure(figsize=(8, 5))

    custom_palette = {
        "LSTM: RMSE": "steelblue",
        "AutoGluon: RMSE": "darkorange",
        "LSTM: RMSE (Peaks)": "lightblue",
        "AutoGluon: RMSE (Peaks)": "orange",
    }

    # Use Seaborn to create a grouped barplot
    sns.barplot(
        x="Load",
        y="Value",
        hue="Metric",
        data=df_melted,
        palette=custom_palette,
        dodge=True,
    )

    # Adjust bar width and spacing by controlling dodge and width
    for patch in plt.gca().patches:
        if "AutoGluon" in str(patch.get_label()):
            patch.set_width(patch.get_width() * 0.8)  # Make AutoGluon bars narrower for spacing

    # Add labels and title
    plt.title(f"Comparison of Metrics Across LGs for {year}", fontsize=16)
    plt.xlabel("LG", fontsize=14)
    plt.ylabel("Metric Value (RMSE)", fontsize=14)

    # Adjust the layout and display the legend
    plt.legend(title="Metric", loc="best", fontsize=10)
    plt.tight_layout()

    # Save the plot to a file
    plt.savefig(f"plots/results_{year}.png", dpi=500)

    # Show the plot (optional, you can remove this in production)
    plt.show()

    # print average RSME and peak RSME for both models
    print("LSTM RMSE:", round(df["LSTM: RMSE"].mean(), 3))
    print("AutoGluon RMSE:", round(df["AutoGluon: RMSE"].mean(), 3))
    print("LSTM Peak RMSE:", round(df["LSTM: RMSE (Peaks)"].mean(), 3))
    print("AutoGluon Peak RMSE:", round(df["AutoGluon: RMSE (Peaks)"].mean(), 3))


def plot_learning_curve(
        data: pd.DataFrame,
        train_epochs: int,
        year: int, 
        lg: int,
        show_train_period: bool = True,
        show_val_min: bool = True,
    ):
    data = data.rename(columns={"train_loss": "Training Loss", "val_loss": "Validation Loss"})

    df_melted = data.melt(id_vars="epoch", value_vars=["Training Loss", "Validation Loss"], 
                        var_name="Loss Type", value_name="Loss")

    # Create the lineplot
    plt.figure(figsize=(8, 5))
    g = sns.lineplot(x="epoch", y="Loss", hue="Loss Type", data=df_melted)

    if show_val_min:
        # Add a vertical dotted line at train_epochs
        cumulative_min_val_loss = data["Validation Loss"].cummin()

        # Plot the cumulative minimum as a step function
        val_loss_color = g.get_lines()[1].get_color()  # Assuming "Validation Loss" is the second line
        plt.step(
            data["epoch"],
            cumulative_min_val_loss,
            where="post",
            color=val_loss_color, 
            linestyle="--",
            label="Validation Loss (min)",
            alpha=0.5,
        )

    if show_train_period:
        plt.axvspan(0, train_epochs, color="gray", alpha=0.15, label=f"Training Period")

    # Add labels and title
    plt.title(f"LSTM: Training and Validation Loss Over Epochs for LG {lg} in {year}", fontsize=14)
    plt.xlabel("Epoch", fontsize=12)
    plt.ylabel("Loss", fontsize=12)

    # Display the plot with a legend
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"plots/learning_curve_{year}_{lg}.png", dpi=500)


def plot_all_learning_curves(
        year: int,
        train_epochs: int,
        show_train_period: bool = True,
        show_val_min: bool = True,
    ):
    # get all files in this directory
    for file in os.listdir():
        if file.startswith(f"training_process_{year}"):
            # extract lg from file
            lg = int(file.split("_")[-1].split(".")[0])
            data = pd.read_csv(file)
            plot_learning_curve(
                data=data,
                train_epochs=train_epochs,
                year=year,
                lg=lg,
                show_train_period=show_train_period,
                show_val_min=show_val_min
            )


def calculate_rmse(actual, predicted):
    return np.sqrt(((actual - predicted) ** 2).mean())
    

def plot_forecast(year: int):
    if year == 2016:
        load_profiles = pd.read_csv('LoadProfile_20IPs_2016.csv', skiprows=1, delimiter=";", index_col=0, date_parser=custom_date_parser)
    else:
        load_profiles = pd.read_csv('LoadProfile_30IPs_2017.csv', skiprows=1, delimiter=";", index_col=0, date_parser=custom_date_parser)

    actuals = pd.read_csv(f'tobi/{year}_actuals.csv', index_col=0, parse_dates=True)
    peak_actuals = pd.read_csv(f'tobi/{year}_peak_actuals.csv', index_col=0, parse_dates=True)
    forecasts =  pd.read_csv(f'forecasts_{year}.csv', index_col=0, parse_dates=True)


    best_forecast = None
    best_actuals = None
    best_peak_treshold = None

    best_rmse = np.inf

    for load in [x for x in peak_actuals.columns if x != 'dataset_id']:
        peak_actuals_load = peak_actuals[[load, 'dataset_id']]
        actuals_load = actuals[[load, 'dataset_id']]

        for dataset_id in peak_actuals_load['dataset_id'].unique():
            peak_actuals_i = peak_actuals_load[peak_actuals_load['dataset_id'] == dataset_id]
            actuals_i = actuals_load[actuals_load['dataset_id'] == dataset_id]
            peak_actuals_i_j = peak_actuals_i[load]
            actuals_i_j = actuals_i[load]

            # get 85% of the peak load for the current load in the first 8 months
            load_profiles_load = load_profiles[load]
            max_load = load_profiles_load.max()
            max_load_threshold = max_load * 0.85
            
            forecast = forecasts.loc[peak_actuals_i_j.index, load]
            rmse = calculate_rmse(peak_actuals_i_j[peak_actuals_i_j != 0], forecast[peak_actuals_i_j != 0])
            if len(peak_actuals_i_j[peak_actuals_i_j != 0]) > 4 and rmse < best_rmse:
                best_rmse = rmse
                best_forecast = forecast
                best_actuals = actuals_i_j
                best_peak_treshold = max_load_threshold

    # Convert the series to DataFrame
    best_forecast = best_forecast.reset_index()
    best_forecast.columns = ['Time stamp', 'Predicted Load']

    best_actuals = best_actuals.reset_index()
    best_actuals.columns = ['Time stamp', 'Actual Load']

    # Merge both DataFrames based on the Time stamp
    df = pd.merge(best_forecast, best_actuals, on='Time stamp')

    # Convert Time stamp to datetime
    df['Time stamp'] = pd.to_datetime(df['Time stamp'])

    # Extract the hour for x-axis tick labels
    df['Hour'] = df['Time stamp'].dt.strftime('%H:%M')

    # Plot using Seaborn
    plt.figure(figsize=(8, 5))
    sns.lineplot(x='Hour', y='Predicted Load', data=df, label='Predicted Load')
    sns.lineplot(x='Hour', y='Actual Load', data=df, label='Actual Load')

    # plot horizontal line at 85% of the peak load
    plt.axhline(best_peak_treshold, color='red', linestyle='--', label='85% of Peak Load')

    # Set x-ticks to display only every 4th value
    ticks = df['Hour'][::4]  # Select every 4th value
    plt.xticks(ticks)

    # set y lim 
    plt.ylim(2400, 2600)

    # Display the plot with a legend
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"plots/best_rsme_{year}.png", dpi=500)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--year", type=int, required=True)
    parser.add_argument("--train_epochs", type=int, default=20)
    parser.add_argument("--show_train_period", type=bool, default=True)
    parser.add_argument("--show_val_min", type=bool, default=True)
    args = parser.parse_args()

    plot_all_learning_curves(
        year=args.year, 
        train_epochs=20,
        show_train_period=args.show_train_period,
        show_val_min=args.show_val_min
    )
    plot(args.year)
    plot_forecast(args.year)