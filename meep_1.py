#original human-written meep code

import meep as mp
import numpy as np
import matplotlib.pyplot as plt

# PCSEL structure parameters
resolution = 35  # pixels per micron
wavelength = 0.98  # operating wavelength (in microns)
pml_thickness = 0.5  # thickness of perfectly matched layer (PML)
hole_radius = 0.1  # radius of the air holes (in microns)
lattice_spacing = 0.3  # spacing between lattice points (in microns)
active_layer_thickness = 0.5  # thickness of the active layer (in microns)
n_sub_thickness = 0.2  # thickness of n-substrate layer (in microns)
n_clad_thickness = 0.3  # thickness of n-cladding layer (in microns)
p_clad_thickness = 0.3  # thickness of p-cladding layer (in microns)

# Define the materials
n_inp = 3.4  # refractive index of InP
n_air = 1.0  # refractive index of air
n_substrate = 3.0  # refractive index of substrate
n_cladding = 3.2  # refractive index of cladding (n-type)
p_cladding = 3.2  # refractive index of cladding (p-type)

# Define the PCSEL structure
geometry = [
    mp.Block(
        mp.Vector3(mp.inf, mp.inf, pml_thickness),
        center=mp.Vector3(0,0,0),
        material=mp.Medium(index=n_substrate)
    ),
    mp.Block(
        mp.Vector3(mp.inf, mp.inf, n_clad_thickness),
        center=mp.Vector3(0, 0, -n_clad_thickness / 2 - n_sub_thickness/2),
        material=mp.Medium(index=n_cladding)
    ),
    mp.Block(
        mp.Vector3(mp.inf, mp.inf, active_layer_thickness),
        center=mp.Vector3(0, 0, -n_clad_thickness - active_layer_thickness / 2 - n_sub_thickness/2),
        material=mp.Medium(index=n_inp)
    ),
    mp.Block(
        mp.Vector3(mp.inf, mp.inf, active_layer_thickness),
        center=mp.Vector3(0, 0, -n_clad_thickness - active_layer_thickness - active_layer_thickness/2 - n_sub_thickness/2),
        material=mp.Medium(index=n_inp)
    ),
    mp.Block(
        mp.Vector3(mp.inf, mp.inf, p_clad_thickness),
        center=mp.Vector3(0, 0, -n_clad_thickness - 2 * active_layer_thickness - p_clad_thickness / 2 - n_sub_thickness/2),
        material=mp.Medium(index=p_cladding)
    ),
]

# Add photonic crystal layer with air holes
num_holes_x = 50
num_holes_y = 50
pc_thickness = active_layer_thickness

hole_spacing_x = lattice_spacing
hole_spacing_y = lattice_spacing

pc_x = num_holes_x * hole_spacing_x
pc_y = num_holes_y * hole_spacing_y

pc_x_center = 0
pc_y_center = 0

for i in range(num_holes_x):
    for j in range(num_holes_y):
        hole_center = mp.Vector3(
            pc_x_center - pc_x / 2 + hole_spacing_x * i + hole_spacing_x / 2,
            pc_y_center - pc_y / 2 + hole_spacing_y * j + hole_spacing_y / 2,
            -n_clad_thickness - active_layer_thickness - active_layer_thickness/2 - n_sub_thickness/2
        )
        geometry.append(
            mp.Cylinder(radius=hole_radius, height=pc_thickness, center=hole_center, material=mp.air)
        )

# Set up the simulation parameters
cell_size = mp.Vector3(pc_x, pc_y, pml_thickness + pc_thickness + n_sub_thickness + n_clad_thickness + active_layer_thickness + p_clad_thickness)
boundary_layers = [mp.PML(pml_thickness)]  # boundary conditions
k_points = mp.interpolate(10, [mp.Vector3(), mp.Vector3(0.5, 0, 0)])  # k-points for Bloch boundary conditions

# Define the source
source_position = mp.Vector3(0, 0)
source = mp.Source(mp.GaussianSource(wavelength, fwidth=0.2 * wavelength), component=mp.Ez, center=source_position)

# Set up the simulation
sim = mp.Simulation(
    resolution=resolution,
    geometry=geometry,
    cell_size=cell_size,
    boundary_layers=boundary_layers,
    k_point=k_points,
    sources=[source]
)

# Run the simulation
sim.run(until_after_sources=200)

# Retrieve and compute simulation results
Q = sim.mode_field_decay(Q=1, freq=1 / wavelength)
FWHM = Q.fwhm()
resonance_wavelength = 1 / sim.meep_time()
emitted_power = sim.get_emw_poynting().real
far_field = sim.get_farfield(num_points=100, center=source_position)
mode_area = sim.mode_field_area()

# Print the computed values
print("Q factor:", Q.Q)
print("FWHM:", FWHM)
print("Resonance wavelength:", resonance_wavelength)
print("Emitted power:", emitted_power)
print("Mode area:", mode_area)

# Plot the far-field pattern
plt.figure()
plt.plot(np.degrees(far_field[:, 0]), far_field[:, 1])
plt.xlabel("Angle (degrees)")
plt.ylabel("Power (arb. units)")
plt.title("Far-Field Pattern")

plt.show()
