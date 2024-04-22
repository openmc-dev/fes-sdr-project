import openmc
import openmc.model
import openmc.deplete
import numpy as np
import sys

openmc.config['chain_file'] = 'chain_fe.xml'

x0 = openmc.XPlane(-2.5)
x1 = openmc.XPlane(2.5)
y0 = openmc.YPlane(-2.5)
y1 = openmc.YPlane(2.5)
n = int(sys.argv[1])
w = 10.0
z_values = np.linspace(0.0, w, 2)
z_planes = [openmc.ZPlane(z) for z in z_values]

activation_cells = []
for z0, z1 in zip(z_planes[:-1], z_planes[1:]):
    cell = openmc.Cell(region=+x0 & -x1 & +y0 & -y1 & +z0 & -z1)
    mat = openmc.Material()
    mat.set_density('g/cm3', 8.0)
    mat.add_element('Fe', 1.0)
    mat.depletable = True
    mat.volume = 5. * 5.* w/n
    cell.fill = mat
    activation_cells.append(cell)

sph_detect = openmc.Sphere(z0=15.0, r=1.0)
cell_detect = openmc.Cell(region=-sph_detect)

sph = openmc.Sphere(r=20.0, boundary_type='vacuum')
outer_cell = openmc.Cell(region=-sph & +sph_detect & (-x0 | +x1 | -y0 | +y1 | -z_planes[0] | +z_planes[-1]))

model = openmc.Model()
model.geometry = openmc.Geometry(activation_cells + [cell_detect, outer_cell])
model.settings.particles = 100_000
model.settings.batches = 10
model.settings.run_mode = 'fixed source'
model.settings.source = openmc.IndependentSource(
    space=openmc.stats.Point((0., 0., -5.0)),
    energy=openmc.stats.Discrete([14.0e6], [1.0])
)
model.export_to_model_xml('model_neutron.xml')

# Set up mesh for activation
mesh = openmc.RegularMesh()
mesh.lower_left = (-2.5, -2.5, 0.0)
mesh.upper_right = (2.5, 2.5, w)
mesh.dimension = (1, 1, n)

# Get fluxes and microscopic cross sections on mesh
fluxes, micros = openmc.deplete.get_microxs_and_flux(model, mesh)

# Get homogenized materials
activation_mats = mesh.get_homogenized_materials(model)

op = openmc.deplete.IndependentOperator(openmc.Materials(activation_mats),
                                        fluxes, micros, normalization_mode='source-rate')

timesteps = [1e4, 1e3]
source_rates = [1.0e12, 0.0]

integrator = openmc.deplete.PredictorIntegrator(op, timesteps, source_rates=source_rates)
integrator.integrate(final_step=False)

results = openmc.deplete.Results()

photon_sources = []
for mat in activation_mats:
    # Get activated material
    mat: openmc.Material = results[-1].get_material(str(mat.id))

    # Create source (no domain rejection needed)
    energy = mat.get_decay_photon_energy()
    source = openmc.IndependentSource(
        energy=energy,
        particle='photon',
        strength=energy.integral(),
    )
    photon_sources.append(source)

# Create mesh source
model.settings.source = openmc.MeshSource(mesh, [[photon_sources]])

tally = openmc.Tally()
tally.filters = [openmc.CellFilter([cell_detect])]
tally.scores = ['flux']
model.tallies = [tally]
sp_path = model.run(path='model_photon.xml')
sp_path = sp_path.rename('statepoint_mesh.h5')

with openmc.StatePoint(sp_path) as sp:
    t = sp.tallies[tally.id]
    print(t.get_pandas_dataframe())
