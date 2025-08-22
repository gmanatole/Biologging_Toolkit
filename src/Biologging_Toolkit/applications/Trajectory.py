from Biologging_Toolkit.wrapper import Wrapper

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
import netCDF4
import plotly.graph_objects as go

class Trajectory:
    def __init__(self, depid: str, path: str):
        self.depid = depid
        self.path = path
        nc_path = os.path.join(path, depid, f"{depid}_sens.nc")

        nc = netCDF4.Dataset(nc_path)
        data = {var: nc.variables[var][:] for var in nc.variables}
        self.df = pd.DataFrame(data)

    def get_dive_trajectory(self, dive: int, dt: float = 2.0):
        df_dive = self.df[self.df['dives'] == dive].reset_index(drop=True)

        az = np.radians(df_dive['azimuth'].values)
        el = np.radians(df_dive['elevation_angle'].values)

        step = np.ones(len(df_dive)) * dt

        dx = np.cos(el) * np.cos(az)
        dy = np.cos(el) * np.sin(az)
        dz = np.sin(el)

        x = np.cumsum(dx * step)
        y = np.cumsum(dy * step)
        z = df_dive['depth'].values

        return x, y, z, df_dive

    def plot_dive(self, dive: int, dt: float = 2.0):
        x, y, z, _ = self.get_dive_trajectory(dive, dt)

        fig = plt.figure(figsize=(12, 6))
        plt.suptitle(f"Trajectoire {self.depid} - Dive {dive}")

        ax1 = fig.add_subplot(1, 3, 1)
        ax1.plot(x, -z)
        ax1.set_xlabel("x")
        ax1.set_ylabel("Profondeur (m)")

        ax2 = fig.add_subplot(1, 3, 2)
        ax2.plot(x, y)
        ax2.set_xlabel("x")
        ax2.set_ylabel("y")

        ax3 = fig.add_subplot(1, 3, 3, projection='3d')
        ax3.plot(x, y, -z)
        ax3.view_init(elev=0, azim=100)
        ax3.set_xlabel("x")
        ax3.set_ylabel("y")
        ax3.set_zlabel("Profondeur (m)")

        plt.show()

    def save_gif_frames(self, dive: int, dt: float = 2.0, outdir: str = "./temp_frames"):
        x, y, z, _ = self.get_dive_trajectory(dive, dt)

        save_path = os.path.join(outdir, self.depid, str(dive))
        os.makedirs(save_path, exist_ok=True)

        for i in tqdm(range(360), desc=f"Génération frames Dive {dive}"):
            fig = plt.figure(figsize=(6, 6))
            ax3 = fig.add_subplot(111, projection='3d')
            ax3.plot(x, y, -z)
            ax3.view_init(elev=0, azim=i)
            ax3.set_xlabel("x")
            ax3.set_ylabel("y")
            ax3.set_zlabel("Profondeur (m)")

            plt.savefig(os.path.join(save_path, f"{i:03d}.png"), bbox_inches='tight')
            plt.close(fig)

    def plot3D_go(self, dive: int, dt: float = 2.0):
        x, y, z, _ = self.get_dive_trajectory(dive, dt)

        fig = go.Figure(data=[go.Scatter3d(
            x=x, y=y, z=-z,
            mode='lines',
            line=dict(width=5, color='blue')
        )])

        fig.update_layout(
            title=f"Trajectoire {self.depid} - Dive {dive}",
            scene=dict(
                xaxis_title='x',
                yaxis_title='y',
                zaxis_title='Profondeur (m)',
                aspectmode='cube'
            )
        )

        fig.show()

# class Trajectory(Wrapper) :
#     def __init__(self,
#                  depid,
#                  *,
#                  path,
#                  ponderation = 'acoustic'
#                  ):
#         """
#         This class uses processed dataset to reconstruct the animal's trajectory.
#         The main method is to use Euler angles to get the speed from the pitch and vertical speed.
#         If acoustic data is available in the data structure a model can be fitted using the previous speed estimation.
#         """

#         super().__init__(
#             depid,
#             path
#         )

#         self.ponderation = ponderation