import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import numpy as np
import sys
import argparse

def plot_throughput_surface(csv_path, output_path, x_axis, y_axis, fixed_axis_value):
    # Load the CSV data
    data = pd.read_csv(csv_path)

    fixed_axis, fixed_value = fixed_axis_value.split('=')
    fixed_value = int(fixed_value)
    filtered_data = data[data[fixed_axis] == fixed_value]

    # Ensure we have valid x, y, and throughput columns
    if x_axis not in filtered_data.columns or y_axis not in filtered_data.columns or 'throughput' not in filtered_data.columns:
        print(f"Error: Missing columns in the data for axes {x_axis}, {y_axis}, or 'throughput'")
        return

    x_values = filtered_data[x_axis].unique()
    y_values = filtered_data[y_axis].unique()
    
    # Create a grid for the X and Y axes
    x_grid, y_grid = np.meshgrid(x_values, y_values)
    throughput_grid = filtered_data.pivot(index=y_axis, columns=x_axis, values='throughput').values

    # Create the plot
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')
    surf = ax.plot_surface(x_grid, y_grid, throughput_grid, cmap=cm.viridis, edgecolor='k', linewidth=0.5)
    ax.set_xlabel(x_axis)
    ax.set_ylabel(y_axis)
    ax.set_zlabel('Throughput')
    ax.set_title(f'Throughput Surface Plot (fixed {fixed_axis}={fixed_value})')
    fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5)

    plt.savefig(output_path)

# Run the script
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="3D Surface Plot for Throughput")
    parser.add_argument("csv_path", type=str, help="Path to the CSV file with throughput data")
    parser.add_argument("output_path", type=str, help="Path to save the output plot image")
    parser.add_argument("--x_axis", type=str, choices=["M", "K", "N"], required=True, help="Column to use as the X-axis")
    parser.add_argument("--y_axis", type=str, choices=["M", "K", "N"], required=True, help="Column to use as the Y-axis")
    parser.add_argument("--fixed_value", type=str, required=True, help="Fixed axis and its value in the form <Axis>=<Value> (e.g., K=128)")

    args = parser.parse_args()

    # Check if x_axis and y_axis are the same
    if args.x_axis == args.y_axis:
        print("Error: X and Y axes must be different.")
        sys.exit(1)

    plot_throughput_surface(args.csv_path, args.output_path, args.x_axis, args.y_axis, args.fixed_value)
