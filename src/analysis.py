import numpy as np

def calc_traj_occupancy(traj, proton_trajs):
    traj_occupancy = np.full((len(proton_trajs), len(traj.trajectory)), False)
    for i, pt in enumerate(proton_trajs):
        traj_occupancy[i, pt.iframe] = True
    return traj_occupancy

def calc_displacement(proton_pos, CELL):
    cell = np.array(CELL[:3])
    diff = proton_pos[1:] - proton_pos[:-1]
    diff -= np.round(diff / cell) * cell
    pos = [diff[0]]
    for i in range(1, len(diff)):
        pos.append(pos[-1] + diff[i])
    displacement = np.array([np.linalg.norm(pos[i] - pos[0]) ** 2 for i in range(len(pos))])

    return diff, displacement