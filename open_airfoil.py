import matplotlib.pyplot as plt

# Replace with your actual file path
file_path = "NACA_2414.txt"

# Lists to hold the X and Y values
x_vals = []
y_vals = []

# Read the file
with open(file_path, "r") as file:
    for line in file:
        # Skip empty lines
        if not line.strip():
            continue
        # Split by whitespace and extract first two columns
        parts = line.strip().split()
        try:
            x = float(parts[0])
            y = float(parts[1])
            x_vals.append(x)
            y_vals.append(y)
        except (ValueError, IndexError):
            # Skip lines that don't have at least two valid numbers
            continue

# Plot the coordinates
plt.figure(figsize=(8, 6))
plt.scatter(x_vals, y_vals, color='black', s=36)  # s is marker size (6^2)
plt.xlabel("X")
plt.ylabel("Y")
plt.title("Plot of Coordinates from NACA_2414.txt")
plt.grid(True)
plt.show()

print("V values")
print(x_vals)
