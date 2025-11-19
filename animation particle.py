import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Ellipse
from IPython.display import Image, display
from matplotlib.colors import LinearSegmentedColormap
# Parameters for the grid and simulation
Nx, Ny = 256, 256            # grid points in x and y
Lx, Ly = 4.0, 4.0            # spatial domain: x, y in [-Lx, Lx] and [-Ly, Ly]
x = np.linspace(-Lx, Lx, Nx)
y = np.linspace(-Ly, Ly, Ny)
dx = x[1] - x[0]
dy = y[1] - y[0]
X, Y = np.meshgrid(x, y, indexing='ij')

# Ellipse parameters for the potential well:
a = 3      # semi-major axis length (along x)
b = 1.5      # semi-minor axis length (along y)
inside = (X/a)**2 + (Y/b)**2 <= 1  # boolean mask: True if inside the ellipse

# Define the potential:
#  V = 0 inside the ellipse and a high value (V0) outside.
V0 = 1e6
V = np.zeros((Nx, Ny))
V[~inside] = V0

# Initial wavefunction: a Gaussian wavepacket inside the ellipse with momentum along x.
x0, y0 = 0.5, 0        # initial center
sigma = 0.2              # width of the packet
kx0 = -50.0               # initial momentum along x (in arbitrary units)
ky0 = 0.0                # initial momentum along y
psi = np.exp(-((X - x0)**2 + (Y - y0)**2) / (2 * sigma**2)) * np.exp(1j * (kx0 * X + ky0 * Y))
# Normalize psi
norm = np.sqrt(np.sum(np.abs(psi)**2) * dx * dy)
psi = psi / norm

# Second wavefunction: a Gaussian wavepacket with different initial position and momentum
x0_2, y0_2 = -0.5, 0        # different initial center
sigma_2 = sigma                 # different width
kx0_2 = -kx0                  # different initial momentum along x
ky0_2 = 0.0                    # different initial momentum along y
psi2 = np.exp(-((X - x0_2)**2 + (Y - y0_2)**2) / (2 * sigma_2**2)) * np.exp(1j * (kx0_2 * X + ky0_2 * Y))
# Normalize psi2
norm2 = np.sqrt(np.sum(np.abs(psi2)**2) * dx * dy)
psi2 = psi2 / norm2

# Simulation parameters
dt = 1e-3                 # time step
steps_per_frame = 2      # number of time steps per animation frame (increased from 1)
hbar = 1.0
m = 1.0
total_time = 1         # total simulation time (in simulation time units)

# Calculate the number of frames based on the total time
frames = int(total_time / (dt * steps_per_frame))

# Precompute the kinetic operator factor (in momentum space) for a half time step.
kx = 2 * np.pi * np.fft.fftfreq(Nx, d=dx)
ky = 2 * np.pi * np.fft.fftfreq(Ny, d=dy)
KX, KY = np.meshgrid(kx, ky, indexing='ij')
T_phase = np.exp(-1j * (KX**2 + KY**2) / (2 * m) * (dt / 2) / hbar)

# Precompute the potential operator factor (in position space) for a full time step.
V_phase = np.exp(-1j * V * dt / hbar)

def step(psi):
    """Evolve psi one time step using the split-operator method."""
    # First half-step: kinetic evolution
    psi_k = np.fft.fft2(psi)
    psi_k *= T_phase
    psi = np.fft.ifft2(psi_k)
    
    # Full step: potential evolution in position space
    psi *= V_phase
    
    # Second half-step: kinetic evolution
    psi_k = np.fft.fft2(psi)
    psi_k *= T_phase
    psi = np.fft.ifft2(psi_k)
    
    # Enforce the infinite well condition: psi = 0 outside the ellipse
    psi[~inside] = 0
    
    # Renormalize the wavefunction to maintain total probability = 1
    norm = np.sqrt(np.sum(np.abs(psi)**2) * dx * dy)
    if norm > 0:  # Avoid division by zero
        psi = psi / norm
    
    return psi

# Set up the figure and axis for animation.
fig, ax = plt.subplots(1, 3, figsize=(15, 5))

# Calculate a good maximum value for the colormap (adjust this multiplier as needed)
max_density1 = np.max(np.abs(psi)**2)
max_density2 = np.max(np.abs(psi2)**2)
max_combined = np.max(np.abs(psi)**2 + np.abs(psi2)**2)

# Scale factor that adjusts based on domain size
# Reference area is 4.0 (with default Lx=Ly=2.0)
reference_area = 4.0
current_area = 4 * Lx * Ly
area_factor = reference_area / current_area

vmax_value1 = max_density1 * 0.5 * area_factor
vmax_value2 = max_density2 * 0.5 * area_factor
vmax_combined = max_combined * 0.5 * area_factor

# Create three subplots: psi1, psi2, and combined
im1 = ax[0].imshow(np.abs(psi)**2, extent=[-Lx, Lx, -Ly, Ly], origin='lower', 
               cmap='magma', vmin=0, vmax=vmax_value1)
im2 = ax[1].imshow(np.abs(psi2)**2, extent=[-Lx, Lx, -Ly, Ly], origin='lower', 
               cmap='magma', vmin=0, vmax=vmax_value2)
im3 = ax[2].imshow(np.abs(psi)**2 + np.abs(psi2)**2, extent=[-Lx, Lx, -Ly, Ly], origin='lower', 
               cmap='magma', vmin=0, vmax=vmax_combined)

ax[0].set_title('Particle 1 |ψ₁|²')
ax[1].set_title('Particle 2 |ψ₂|²')
ax[2].set_title('Combined |ψ₁|² + |ψ₂|²')

# Create a list to store the ellipse patches
ell_patches = []

for at in ax:
    at.set_xlabel('x')
    at.set_ylabel('y')
    at.set_xlim([-Lx, Lx])
    at.set_ylim([-Ly, Ly])
    
    # Draw the ellipse representing the well boundary
    ell = Ellipse((0, 0), width=2*b, height=2*a, edgecolor='white', facecolor='none', lw=2)
    at.add_patch(ell)
    ell_patches.append(ell)

def animate(frame):
    global psi, psi2
    for _ in range(steps_per_frame):
        psi = step(psi)
        psi2 = step(psi2)
    
    # # Display progress in terminal
    progress = (frame + 1) / frames * 100
    progress_bar_length = 20
    filled_length = int(progress_bar_length * (frame + 1) // frames)
    bar = '█' * filled_length + '-' * (progress_bar_length - filled_length)
    print(f'\rSimulation progress: [{bar}] {progress:.1f}% (Frame {frame+1}/{frames})', end='')
    if frame == frames - 1:
        print()  # Add newline after completing the simulation
    
    im1.set_data(np.abs(psi)**2)
    im2.set_data(np.abs(psi2)**2)
    im3.set_data(np.abs(psi)**2 + np.abs(psi2)**2)
    
    # Return both the image objects and ellipse patches
    return [im1, im2, im3] + ell_patches

ani = animation.FuncAnimation(fig, animate, frames=frames, interval=20, blit=True)
plt.tight_layout()
plt.show(block=False)  # Make non-blocking so code continues
plt.pause(0.1)  # Pause to allow the animation to start

# Save the animation as a GIF using PillowWriter and display it.
gif_filename = r"c:\Users\Volkan\Desktop\quantum_particles1.gif"  # Save to desktop with full path
writer = animation.PillowWriter(fps=60)
ani.save(gif_filename, writer=writer)
plt.close(fig)
display(Image(filename=gif_filename))
