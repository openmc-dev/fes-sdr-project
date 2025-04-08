from pathlib import Path
import openmc
import openmc.deplete
import numpy as np
import pandas as pd

openmc.config['chain_file'] = Path('chain_fe.xml').resolve()

results_dict = {f'det_{i}': [] for i in range(6)}

for n in range(1, 11):
    openmc.reset_auto_ids()

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
        mat.depletable = True
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
    model.export_to_model_xml('model_neutron.xml')

    #===========================================================================
    # Run neutron calculation to get fluxes / cross sections

    fluxes, micros = openmc.deplete.get_microxs_and_flux(model, activation_cells)

    #===========================================================================
    # Deplete

    activation_mats = [cell.fill for cell in activation_cells]

    op = openmc.deplete.IndependentOperator(
        activation_mats, fluxes, micros, normalization_mode='source-rate')

    timesteps = [1e4, 24*3600]
    source_rates = [1.0e12, 0.0]
    integrator = openmc.deplete.PredictorIntegrator(op, timesteps, source_rates=source_rates)
    integrator.integrate(final_step=False)

    #===========================================================================
    # Photon calculation

    results = openmc.deplete.Results()

    photon_sources = []
    for cell in activation_cells:
        # Get activated material
        mat = results[-1].get_material(str(cell.fill.id))

        # Create source (no domain rejection needed)
        energy = mat.get_decay_photon_energy(clip_tolerance=0.0)
        source = openmc.IndependentSource(
            space=openmc.stats.Box(*cell.region.bounding_box),
            energy=energy,
            particle='photon',
            strength=energy.integral(),
        )
        photon_sources.append(source)

    model.settings.source = photon_sources

    tally = openmc.Tally()
    tally.filters = [openmc.CellFilter(detect_cells)]
    tally.scores = ['flux']
    model.tallies = [tally]
    sp_path = model.run(path='model_photon.xml')
    sp_path = sp_path.rename(f'statepoint_cell_{n}.h5')

    with openmc.StatePoint(sp_path) as sp:
        t = sp.tallies[tally.id]
        for i, value in enumerate(t.mean.ravel()):
            results_dict[f'det_{i}'].append(value)

df = pd.DataFrame(results_dict)
print(df.to_string())
df.to_pickle('results_r2s_cell.pkl')
