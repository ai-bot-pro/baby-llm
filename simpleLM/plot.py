# from: https://github.com/antirez/simple-language-model/blob/main/plot.py
import numpy as np
import matplotlib.pyplot as plt
import sys
import os


def show(filenames):
    # Number of files
    num_files = len(filenames)

    # Single file case
    if num_files == 1:
        filename = filenames[0]
        try:
            data = np.loadtxt(filename)
        except IOError:
            print(f"Error: File {filename} not found or cannot be read.")
            sys.exit(1)

        plt.figure(figsize=(10, 6))
        plt.plot(data[:, 0], data[:, 1], label="Training Loss")
        plt.plot(data[:, 0], data[:, 2], label="Validation Loss")
        plt.xlabel("Iteration")
        plt.ylabel("Loss")
        plt.title(f"Training and Validation Loss - {os.path.basename(filename)}")
        plt.legend()
        plt.show()

    # Multiple file case
    else:
        plt.figure(figsize=(10, 6))

        for filename in filenames:
            try:
                data = np.loadtxt(filename)
            except IOError:
                print(f"Error: File {filename} not found or cannot be read.")
                continue

            plt.plot(
                data[:, 0], data[:, 2], label=f"Validation Loss - {os.path.basename(filename)}"
            )

        plt.xlabel("Iteration")
        plt.ylabel("Loss")
        plt.title("Validation Loss Comparison")
        plt.legend()
        plt.show()


if __name__ == "__main__":
    # Check if at least one filename is provided
    if len(sys.argv) < 2:
        print("Error: No filename provided.")
        sys.exit(1)

    # Extract filenames from command line arguments
    filenames = sys.argv[1:]
    show(filenames)
