import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.ndimage import binary_fill_holes
import airfoil_loader  # Import the separate loader script

"""
Simulate flow past an airfoil using Lattice Boltzmann Method
for an isothermal fluid
"""

def main():
    """ Lattice Boltzmann Simulation """

    # Simulation parameters
    Nx                     = 400    # resolution x-dir
    Ny                     = 150    # resolution y-dir (adjust based on airfoil extent)
    rho0                   = 100    # average density
    tau                    = 0.6    # collision timescale
    Nt                     = 4000   # number of timesteps
    plotRealTime = True # switch on for plotting as the simulation goes along

    # Lattice speeds / weights
    NL = 9
    idxs = np.arange(NL)
    cxs = np.array([0, 0, 1, 1, 1, 0,-1,-1,-1])
    cys = np.array([0, 1, 1, 0,-1,-1,-1, 0, 1])
    weights = np.array([4/9,1/9,1/36,1/9,1/36,1/9,1/36,1/9,1/36]) # sums to 1

    # Initial Conditions
    F = np.ones((Ny,Nx,NL)) * rho0 / NL
    np.random.seed(42)
    F += 0.01*np.random.randn(Ny,Nx,NL)
    X, Y = np.meshgrid(range(Nx), range(Ny))
    F[:,:,3] += 2 * (1+0.2*np.cos(2*np.pi*X/Nx*4))
    rho = np.sum(F,2)
    for i in idxs:
        F[:,:,i] *= rho0 / rho

    # Load airfoil coordinates using the separate script
    file_path = "NACA_2414.txt"
    x_vals, y_vals = airfoil_loader.load_airfoil_data(file_path)

    # Normalize and scale airfoil coordinates to the grid
    min_x_airfoil, max_x_airfoil = np.min(x_vals), np.max(x_vals)
    min_y_airfoil, max_y_airfoil = np.min(y_vals), np.max(y_vals)

    # Center the airfoil around x = Nx/4 and y = Ny/2, and scale it
    scaled_airfoil_x = (x_vals - min_x_airfoil) / (max_x_airfoil - min_x_airfoil) * (Nx / 3) + Nx / 6
    scaled_airfoil_y = (y_vals - (min_y_airfoil + max_y_airfoil) / 2) / (max_y_airfoil - min_y_airfoil) * (Ny / 2.5) + Ny / 2

    # Create a boolean mask for the airfoil boundary
    airfoil_boundary = np.zeros((Ny, Nx), dtype=bool)
    for i in range(len(scaled_airfoil_x) - 1):
        x1, y1 = scaled_airfoil_x[i], scaled_airfoil_y[i]
        x2, y2 = scaled_airfoil_x[i+1], scaled_airfoil_y[i+1]
        x_coords, y_coords = np.linspace(min(x1, x2), max(x1, x2), 100), np.linspace(min(y1, y2), max(y1, y2), 100)
        for x_c, y_c in zip(x_coords, y_coords):
            ix, iy = int(round(x_c)), int(round(y_c))
            if 0 <= iy < Ny and 0 <= ix < Nx:
                airfoil_boundary[iy, ix] = True

    # Fill the interior of the airfoil
    airfoil_solid = binary_fill_holes(airfoil_boundary)

    # Prep figure
    fig = plt.figure(figsize=(8, 4), dpi=80)

    # Simulation Main Loop
    for it in range(Nt):
        print(it)

        # Drift
        for i, cx, cy in zip(idxs, cxs, cys):
            F[:,:,i] = np.roll(F[:,:,i], cx, axis=1)
            F[:,:,i] = np.roll(F[:,:,i], cy, axis=0)

        # Set reflective boundaries for the airfoil
        bndryF = F[airfoil_solid, :]
        bndryF = bndryF[:, [0, 5, 6, 7, 8, 1, 2, 3, 4]]
        F[airfoil_solid, :] = bndryF

        # Calculate fluid variables
        rho = np.sum(F,2)
        ux  = np.sum(F*cxs,2) / rho
        uy  = np.sum(F*cys,2) / rho

        # Apply Collision
        Feq = np.zeros(F.shape)
        for i, cx, cy, w in zip(idxs, cxs, cys, weights):
            Feq[:,:,i] = rho * w * ( 1 + 3*(cx*ux+cy*uy)  + 9*(cx*ux+cy*uy)**2/2 - 3*(ux**2+uy**2)/2 )

        F += -(1.0/tau) * (F - Feq)

        # Apply boundary
        F[airfoil_solid, :] = bndryF

        # plot in real time
        if (plotRealTime and (it % 10) == 0) or (it == Nt-1):
            plt.cla()
            ux[airfoil_solid] = 0
            uy[airfoil_solid] = 0
            vorticity = (np.roll(ux, -1, axis=0) - np.roll(ux, 1, axis=0)) - (np.roll(uy, -1, axis=1) - np.roll(uy, 1, axis=1))
            vorticity[airfoil_solid] = np.nan
            vorticity = np.ma.array(vorticity, mask=airfoil_solid)
            plt.imshow(vorticity, cmap='bwr')
            plt.imshow(~airfoil_solid, cmap='gray', alpha=0.3)
            plt.clim(-.1, .1)
            ax = plt.gca()
            ax.invert_yaxis()
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)
            ax.set_aspect('equal')
            plt.pause(0.001)

    # Save figure
    plt.savefig('latticeboltzmann_airfoil.png',dpi=240)
    plt.show()

    return 0

if __name__== "__main__":
  main()