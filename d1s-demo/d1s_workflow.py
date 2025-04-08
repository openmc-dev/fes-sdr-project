from pathlib import Path
import openmc
from openmc.deplete import d1s
import numpy as np
import pandas as pd

openmc.config['chain_file'] = Path('chain_fe.xml')

results_dict = {f'det_{i}': [] for i in range(6)}

n = 1

#===========================================================================
# Set up model

width = 5.0
x0 = openmc.XPlane(-width/2)
x1 = openmc.XPlane(width/2)
y0 = openmc.YPlane(-width/2)
y1 = openmc.YPlane(width/2)
w = 10.0
z_values = np.linspace(-w, w, n + 1)
z_planes = [openmc.ZPlane(z) for z in z_values]

activation_cells = []
for z0, z1 in zip(z_planes[:-1], z_planes[1:]):
    cell = openmc.Cell(region=+x0 & -x1 & +y0 & -y1 & +z0 & -z1)
    mat = openmc.Material()
    mat.set_density('g/cm3', 8.0)
    mat.add_element('Fe', 1.0)
    mat.volume = width * width * 2 * w/n
    cell.fill = mat
    activation_cells.append(cell)

detector_locations = [
    (0., 0., 15.),
    (-9., 0., 15.),
    (-10., 0., 5.),
    (-10., 0., -5.),
    (-9., 0., -15.),
    (0., 0., -15.)
]
detect_cells = []
for (x, y, z) in detector_locations:
    sph = openmc.Sphere(x0=x, z0=z, r=1.0)
    detect_cells.append(openmc.Cell(region=-sph))

sph = openmc.Sphere(r=20.0, boundary_type='vacuum')
outer_cell = openmc.Cell(region=-sph & (-x0 | +x1 | -y0 | +y1 | -z_planes[0] | +z_planes[-1]))
for c in detect_cells:
    outer_cell.region &= ~c.region

model = openmc.Model()
model.geometry = openmc.Geometry(activation_cells + detect_cells + [outer_cell])
model.settings.particles = 1_000_000
model.settings.batches = 10
model.settings.run_mode = 'fixed source'
model.settings.source = openmc.IndependentSource(
    space=openmc.stats.Point((0., 0., -w - 1)),
    energy=openmc.stats.delta_function(14.0e6),
    angle=openmc.stats.Monodirectional((0., 0., 1.))
)
model.settings.photon_transport = True
model.settings.use_decay_photons = True

tally = openmc.Tally()
tally.filters = [
    openmc.ParticleFilter(['photon']),
    openmc.CellFilter(detect_cells),
]
tally.scores = ['flux']

model.tallies = [tally]

radionuclides = d1s.prepare_tallies(model)

sp_path = model.run(path='model_d1s.xml')
sp_path = sp_path.rename('statepoint_d1s.h5')

timesteps = [1e4, 24*3600]
source_rates = [1.0e12, 0.0]

# Determine time correction factors
time_factors = d1s.time_correction_factors(
    radionuclides, timesteps=timesteps, source_rates=source_rates)

with openmc.StatePoint(sp_path) as sp:
    t = sp.tallies[tally.id]
    t = d1s.apply_time_correction(t, time_factors, sum_nuclides=True)
    for i, value in enumerate(t.mean.ravel()):
        results_dict[f'det_{i}'].append(value)

df = pd.DataFrame(results_dict)
print(df.to_string())
df.to_pickle('results_d1s.pkl')
