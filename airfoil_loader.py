def load_airfoil_data(file_path):
    """
    Loads airfoil X and Y coordinates from a text file.

    Args:
        file_path (str): The path to the text file.

    Returns:
        tuple: A tuple containing two lists: x_values and y_values.
               Returns empty lists if there's an error.
    """
    x_vals = []
    y_vals = []
    try:
        with open(file_path, "r") as file:
            for line in file:
                if not line.strip():
                    continue
                parts = line.strip().split()
                try:
                    x = float(parts[0])
                    y = float(parts[1])
                    x_vals.append(x)
                    y_vals.append(y)
                except (ValueError, IndexError):
                    continue
    except FileNotFoundError:
        print(f"Error: Airfoil data file '{file_path}' not found.")
    except Exception as e:
        print(f"Error reading airfoil data: {e}")
    return x_vals, y_vals

if __name__ == '__main__':
    # Example usage if you run this script directly
    import matplotlib.pyplot as plt
    file_path = "NACA_2414.txt"
    x, y = load_airfoil_data(file_path)
    if x and y:
        plt.figure(figsize=(8, 6))
        plt.scatter(x, y, color='black', s=36)
        plt.xlabel("X")
        plt.ylabel("Y")
        plt.title("Plot of Coordinates from NACA_2414.txt")
        plt.grid(True)
        plt.show()