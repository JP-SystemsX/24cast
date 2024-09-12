import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
import argparse
import pickle


sns.set(style="whitegrid")
sns.set_palette("colorblind")


def plot(year: int):
    with open(f"results_{year}.pkl", "rb") as f:
        rmse = pickle.load(f)

    with open(f"results_peak_{year}.pkl", "rb") as f:
        peak_rmse = pickle.load(f)

    # Create a DataFrame from the dictionaries
    df = pd.DataFrame({
        'Load': list(rmse.keys()), 
        'RMSE': list(rmse.values()), 
        'RMSE (Peaks)': list(peak_rmse.values())
    })

    # Melt the DataFrame to long format for Seaborn
    df_melted = df.melt(id_vars='Load', value_vars=['RMSE', 'RMSE (Peaks)'], var_name='Metric', value_name='Value')

    # Create the bar plot
    plt.figure(figsize=(8, 5))
    sns.barplot(x='Load', y='Value', hue='Metric', data=df_melted, palette='muted')

    # Add labels and title
    plt.title('Comparison of Metrics Across LGs', fontsize=14)
    plt.xlabel('LG', fontsize=12)
    plt.ylabel('Metric Value', fontsize=12)

    # Display the plot
    plt.tight_layout()
    plt.savefig(f"plots/results_{year}.png", dpi=500)


def plot_learning_curve(data: pd.DataFrame, train_epochs: int, year: int, lg: int):
    data = data.rename(columns={'train_loss': 'Training Loss', 'val_loss': 'Validation Loss'})

    df_melted = data.melt(id_vars='epoch', value_vars=['Training Loss', 'Validation Loss'], 
                        var_name='Loss Type', value_name='Loss')

    # Create the lineplot
    plt.figure(figsize=(8, 5))
    sns.lineplot(x='epoch', y='Loss', hue='Loss Type', data=df_melted, marker='o')

    # Add a vertical dotted line at train_epochs
    plt.axvline(x=train_epochs, color='red', linestyle='--', label=f'Train Epochs = {train_epochs}')

    # Add labels and title
    plt.title('Training and Validation Loss Over Epochs', fontsize=14)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Loss', fontsize=12)

    # Display the plot with a legend
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"plots/learning_curve_{year}_{lg}.png", dpi=500)


def plot_all_learning_curves(year: int, train_epochs: int):
    # get all files in this directory
    for file in os.listdir():
        if file.startswith(f"training_process_{year}"):
            # extract lg from file
            lg = int(file.split("_")[-1].split(".")[0])
            data = pd.read_csv(file)
            plot_learning_curve(data, train_epochs, year, lg)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--year", type=int, required=True)
    args = parser.parse_args()

    plot_all_learning_curves(args.year, 2)
    plot(args.year)