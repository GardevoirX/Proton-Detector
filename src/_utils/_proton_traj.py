import numpy as np

def _assign_proton_to_trajs(proton_pos, pos2end_dist, active_traj):

    assign = {i: None for i in range(len(proton_pos))}
    update_traj = {i: False for i in range(len(active_traj))}
    while np.any(~np.isinf(pos2end_dist)):
        iproton, itraj = np.unravel_index(
            pos2end_dist.argmin(), pos2end_dist.shape
        )
        # One trajectory can accept only one proton every frame
        pos2end_dist[iproton] = np.inf
        pos2end_dist[:, itraj] = np.inf
        assign[iproton] = itraj
        update_traj[itraj] = True

    return assign, update_traj