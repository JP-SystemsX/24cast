import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
import argparse
import pickle
import json

sns.set(style="whitegrid")
sns.set_palette("colorblind")


def plot(year: int):
    # with open(f"results_{year}.pkl", "rb") as f:
    #     rmse = pickle.load(f)

    # with open(f"results_peak_{year}.pkl", "rb") as f:
    #     peak_rmse = pickle.load(f)

    with open(f"results_{year}_autogluon.json", "rb") as f:
        rmse_autogluon = json.load(f)

    with open(f"results_peak_{year}_autogluon.json", "rb") as f:
        peak_rmse_autogluon = json.load(f)

    rmse = rmse_autogluon
    peak_rmse = peak_rmse_autogluon

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

    df["Load"] = df["Load"].str.replace("LG ", "")

    # Melt the DataFrame to long format for Seaborn
    df_melted = df.melt(id_vars="Load", 
                        value_vars=["LSTM: RMSE", "AutoGluon: RMSE", "LSTM: RMSE (Peaks)", "AutoGluon: RMSE (Peaks)"],
                        var_name="Metric", value_name="Value")

    # Create the bar plot with an appropriate figure size
    plt.figure(figsize=(8, 5))

    custom_palette = {
        "LSTM: RMSE": "steelblue",
        "AutoGluon: RMSE": "lightblue",
        "LSTM: RMSE (Peaks)": "seagreen",
        "AutoGluon: RMSE (Peaks)": "lightgreen"
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
    plt.title("LSTM: Training and Validation Loss Over Epochs", fontsize=14)
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