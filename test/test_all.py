import MDAnalysis as mda
import numpy as np
import pytest
from MDAnalysis import transformations

from src.analysis import calc_traj_occupancy
from src.lib import run, get_proton_pos


def test_proton_pos():
    traj = mda.Universe(pytest.TOPFILE, pytest.TRAJFILE)
    transform = transformations.set_dimensions(pytest.CELL)
    traj.trajectory.add_transformations(transform)
    traj.dimensions = pytest.CELL
    O_idx = traj.select_atoms('name O').indices
    H_idx = traj.select_atoms('name H').indices
    proton_trajs = run(traj, O_idx, H_idx)
    long_trajs = np.array([t for t in proton_trajs if len(t) > 20])
    traj_occupancy = calc_traj_occupancy(traj, long_trajs)
    proton_pos = get_proton_pos(traj, traj_occupancy, long_trajs, pytest.CELL)

    assert np.allclose(proton_pos[0], pytest.ref_pos0)
    assert np.allclose(proton_pos[-1], pytest.ref_pos1)
