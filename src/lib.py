from itertools import combinations
import MDAnalysis as mda
import networkx as nx
import numpy as np
from MDAnalysis.analysis.distances import distance_array, self_distance_array
from rich.progress import track

from .proton_traj import ProtonTrajCollection


def calc_proton_pos(traj: mda.Universe, proton_idx):
    ag = traj.atoms[proton_idx]
    if ag.n_atoms > 1:
        traj.add_bonds(list(combinations(proton_idx, r=2)))
        proton_pos = ag.center_of_geometry(unwrap=True)
        traj.delete_bonds(list(combinations(proton_idx, r=2)))
    else:
        proton_pos = ag.center_of_geometry(pbc=True) if ag.n_atoms != 0 else [np.nan]

    return proton_pos


def calc_dist_matrix(traj, O_pos):
    d = self_distance_array(O_pos, box=traj.dimensions)
    dist = np.zeros((len(O_pos), len(O_pos)))
    dist[np.triu_indices(len(O_pos), 1)] = d
    dist[np.tril_indices(len(O_pos), -1)] = d

    return dist


def find_O_clusters(dist_mat, hydronium_O_idx):
    g = nx.from_numpy_array(dist_mat)
    return [hydronium_O_idx[list(sub_g)] for sub_g in nx.connected_components(g)]


def partition_O(traj, hydronium_candidates, O_idx, O_pos, cutoff: float = 3):
    if len(O_idx[hydronium_candidates]) > 1:
        clusters = find_O_clusters(
            calc_dist_matrix(traj, O_pos[hydronium_candidates]) < cutoff,
            np.arange(len(O_idx))[hydronium_candidates],
        )
    else:
        clusters = [np.arange(len(O_idx))[hydronium_candidates]]

    return clusters


def run(traj, O_idx, H_idx):
    proton_trajs = ProtonTrajCollection(traj.dimensions)
    for iframe, ts in track(enumerate(traj.trajectory), total=len(traj.trajectory)):
        O_pos = ts._pos[O_idx]
        H_pos = ts._pos[H_idx]
        O_H_dist_mat = distance_array(O_pos, H_pos, traj.dimensions)
        dist_idx = np.argsort(O_H_dist_mat, axis=1)
        dist = np.array([O_H_dist_mat[i][idx] for i, idx in enumerate(dist_idx)])
        # At least three H atoms closer to the O atom than 1.3 angstrom
        hydronium_candidates = dist[:, 2] < 1.3
        # O atoms closer than 3 angstrom are considered as a group and can share a proton
        O_clusters = partition_O(traj, hydronium_candidates, O_idx, O_pos)
        proton_idx = []
        proton_p = []
        proton_ori_p = []
        for cluster in O_clusters:
            proton_idx.append(list(set(H_idx[dist_idx[cluster, 2]])))
            proton_p.append(calc_proton_pos(traj, proton_idx[-1]))
            proton_ori_p.append(ts._pos[proton_idx[-1]])
        proton_trajs.append_new_frame(
            proton_p, proton_idx, O_clusters, iframe, proton_ori_p
        )

    return proton_trajs


def get_proton_pos(traj, traj_occupancy, proton_trajs, CELL):
    proton_pos = []
    for iframe in range(len(traj.trajectory)):
        if (num := np.sum(traj_occupancy[:, iframe])) == 1:
            pt = np.array(proton_trajs)[traj_occupancy[:, iframe]][0]
            proton_pos.append(pt.positions[iframe - pt.iframe[0]])
        elif num > 1:
            temp = mda.Universe.empty(num, trajectory=True)
            temp.add_TopologyAttr("name", ["H"] * num)
            ag = temp.select_atoms("all")
            temp.add_bonds([ag])
            temp.atoms.positions = [
                pt.positions[iframe - pt.iframe[0]]
                for pt in np.array(proton_trajs)[traj_occupancy[:, iframe]]
            ]
            temp.dimensions = CELL
            proton_pos.append(ag.center_of_geometry(unwrap=True))
        else:
            proton_pos.append(None)

    # Fill nan
    temp = []
    for i, a in enumerate(proton_pos):
        if isinstance(a, np.ndarray):
            temp.append(a)
        elif i:
            temp.append(temp[i - 1])
        elif i == 0:
            temp.append(np.array([0.0, 0.0, 0.0]))
    proton_pos = np.array(temp)

    return proton_pos
