import numpy as np
from MDAnalysis.analysis.distances import distance_array


class ProtonTraj:
    def __init__(self, position, H_index, O_index, iframe):
        self.positions = [position]
        self.H_index = [H_index]  # The real index, e.g. 1st atom, 2nd atom
        self.O_index = [O_index]  # Index only for O, e.g. 1st oxygen, 2nd oxygen
        self.iframe = [iframe]

    def __len__(self):
        return len(self.positions)

    def __repr__(self) -> str:
        return f"ProtonTraj with {len(self)} frames"

    @property
    def position(self):
        return self.positions[-1]

    def update(self, position, H_index, O_index, iframe, original_pos):
        if (
            (len(H_index) > 1)
            and (len(self.H_index[-1]) == 1)
            and (self.H_index[-1][0] in H_index)
        ):
            index2seq = {i: seq for seq, i in enumerate(H_index)}
            self.positions.append(original_pos[index2seq[self.H_index[-1][0]]])
            self.O_index.append(np.array([O_index[index2seq[self.H_index[-1][0]]]]))
            self.H_index.append(self.H_index[-1])
        else:
            self.positions.append(position)
            self.O_index.append(O_index)
            self.H_index.append(H_index)
        self.iframe.append(iframe)


class ProtonTrajCollection:
    def __init__(self, dimension, cutoff: float = 2.5):
        self.dimension = dimension
        self.cutoff = cutoff
        self.trajs = []
        self.active_traj_idx = []

    def __len__(self):
        return len(self.trajs)

    def __repr__(self) -> str:
        return f"ProtonTrajCollection with {len(self)} trajectories and {len(self.active_traj_idx)} active trajs"

    def __getitem__(self, key):
        return self.trajs[key]

    @property
    def active_traj(self):
        return [self.trajs[i] for i in self.active_traj_idx]

    def append_new_frame(self, proton_pos, proton_idx, O_idx, iframe, ori_pos):
        if np.isnan(proton_pos[0][0]):
            # Found no proton
            self.active_traj_idx = []
            return

        if not self.active_traj_idx:
            # If no active trajectory, create trajectory for every proton found
            for pos, H, O in zip(proton_pos, proton_idx, O_idx):
                self.create_new_traj(pos, H, O, iframe)
            return
        else:
            # Assign proton found to activate trajectories
            traj_ends = np.array([traj.position for traj in self.active_traj])
            pos2end_dist = distance_array(
                np.array(proton_pos), traj_ends, box=self.dimension
            )
            pos2end_dist[pos2end_dist > self.cutoff] = (
                np.inf
            )  # Only the proton within the cutoff can be assigned to one trajectory
            assign = {i: None for i in range(len(proton_pos))}
            update_traj = {i: False for i in range(len(self.active_traj))}
            while np.any(~np.isinf(pos2end_dist)):
                iproton, itraj = np.unravel_index(
                    pos2end_dist.argmin(), pos2end_dist.shape
                )
                # One trajectory can accept only one proton every frame
                pos2end_dist[iproton] = np.inf
                pos2end_dist[:, itraj] = np.inf
                assign[iproton] = itraj
                update_traj[itraj] = True
            # Update or create
            for i in assign:
                if assign[i] is None:
                    self.create_new_traj(proton_pos[i], proton_idx[i], O_idx[i], iframe)
                else:
                    self.active_traj[assign[i]].update(
                        proton_pos[i], proton_idx[i], O_idx[i], iframe, ori_pos[i]
                    )
            for i in sorted(
                [idx for idx in update_traj if not update_traj[idx]], reverse=True
            ):
                del self.active_traj_idx[i]

    def create_new_traj(self, position, H_index, O_index, iframe):
        self.trajs.append(ProtonTraj(position, H_index, O_index, iframe))
        self.active_traj_idx.append(len(self.trajs) - 1)
