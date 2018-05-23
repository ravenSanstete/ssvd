## script to extract the useful data from raw_grid.dat 
import re
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter


C = 20;
ipath = "raw_grid.dat"
opath = "grid.dat"



## FIRST STAGE PROCESSING

# i_f = open(ipath, 'r');
# o_f = open(opath, 'w+');

# for line in i_f:
# 	if(line.startswith("(U ")):
# 		o_f.write(line);

# o_f.close();
# i_f.close();



## SECOND STAGE PROCESSING, plot out the surface point
pat = r"\(U (\d+), V (\d+)\): \(RMSE (\d.\d+), MAE (\d.\d+)\)";
f = open(opath, 'r');
grid_dat_rmse = np.zeros(shape = (C, C))
grid_dat_mae = np.zeros(shape = (C, C))

for line in f:
	m = re.match(pat, line);
	u = int(m.group(1)) - 1;
	v = int(m.group(2)) - 1;
	grid_dat_rmse[u][v] = float(m.group(3));
	grid_dat_mae[u][v] = float(m.group(4));


# Make data.
V = np.arange(1, C + 1, 1)
U = np.arange(1, C + 1, 1)
V, U = np.meshgrid(V, U);



## DRAW RMSE FIGURE
fig = plt.figure()
ax = fig.gca(projection='3d')


# # Plot the surface.
surf = ax.plot_surface(U, V, grid_dat_rmse, cmap=cm.coolwarm,
                       linewidth=0, antialiased=False)

ax.set_xlim(1, 20)
ax.xaxis.set_major_locator(LinearLocator(20));
ax.xaxis.set_major_formatter(FormatStrFormatter('%d'))
ax.set_ylim(1, 20)
ax.yaxis.set_major_locator(LinearLocator(20));
ax.yaxis.set_major_formatter(FormatStrFormatter('%d'))

ax.set_xlabel(r'$K_u$');
ax.set_ylabel(r'$K_v$');
ax.set_zlabel('RMSE');
# Customize the z axis.
ax.set_zlim(np.min(grid_dat_rmse), np.max(grid_dat_rmse));

# Add a color bar which maps values to colors.
fig.colorbar(surf, shrink=0.5, aspect=5)





## DRAW MAE FIGURE
fig = plt.figure()
ax = fig.gca(projection='3d')


# # Plot the surface.
surf = ax.plot_surface(U, V, grid_dat_mae, cmap=cm.coolwarm,
                       linewidth=0, antialiased=False)

ax.set_xlim(1, 20)
ax.xaxis.set_major_locator(LinearLocator(20));
ax.xaxis.set_major_formatter(FormatStrFormatter('%d'))
ax.set_ylim(1, 20)
ax.yaxis.set_major_locator(LinearLocator(20));
ax.yaxis.set_major_formatter(FormatStrFormatter('%d'))

ax.set_xlabel('$K_u$');
ax.set_ylabel('$K_v$');
ax.set_zlabel('MAE');

# Customize the z axis.
ax.set_zlim(np.min(grid_dat_mae), np.max(grid_dat_mae));

# Add a color bar which maps values to colors.
fig.colorbar(surf, shrink=0.5, aspect=5)


plt.show()








