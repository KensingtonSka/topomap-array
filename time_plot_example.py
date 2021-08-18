# -*- coding: utf-8 -*-
"""
Example using the functions in TopomapArray to plot a set of 10:20 EEG sensors
onto a single subplot.

@author: KensingtonSka (Rhys Hobbs)
"""
import numpy as np
import matplotlib.pyplot as plt
from TopomapArray import project_onto_zplane, gen_grid_size, project_onto_grid

# %% Hypothetical sensor positions:
ch_names = ['Fp1', 'Fp2', 'Fz', 'F3', 'F4', 'F7', 'F8', 
            'Cz', 'C3', 'C4', 'T3', 'T4', 
            'Pz', 'P3', 'P4', 'T5', 'T6', 'O1', 'O2']
xyz = [np.array([-21.5,	70.2,   -0.1]),
       np.array([28.4,	69.1,	-0.4]),
       np.array([0.6,    40.9,	53.9]),
       np.array([-35.5,	49.4,	32.4]),
       np.array([40.2,   47.6,	32.1]),
       np.array([-54.8,	33.9,	-3.5]),
       np.array([56.6,	30.8,	-4.1]), 
       np.array([0.8,	-14.7,	73.9]),
       np.array([-52.2,	-16.4,	57.8]),
       np.array([54.1,	-18.0,	57.5]),
       np.array([-70.2,	-21.3,	-10.7]),
       np.array([71.9,	-25.2,	-8.2]),
       np.array([0.2,	-62.1,	64.5]),
       np.array([-39.5,	-76.3,	47.4]),
       np.array([36.8,	-74.9,	49.2]),
       np.array([-61.5,	-65.3,	1.1]),
       np.array([59.3,	-67.6,	3.8]),
       np.array([-26.8,	-100.2,	12.8]),
       np.array([24.1,	-100.5,	14.1])]
xyz = np.array(xyz)

##Triangle example (uncomment to use):
#ch_names = ['U', 'L', 'R']
#length = 1
#xyz = [np.array([0, length, 0.5]),
#       np.array([-length*np.cos(np.pi/6), -length*np.sin(np.pi/6), 0.5]),
#       np.array([length*np.cos(np.pi/6), -length*np.sin(np.pi/6), 0.5])]
#xyz = np.array(xyz)

##Circle example (uncomment to use):
#radius = 1
#n_points = 10
#angle = np.linspace(0, 2*np.pi*(1-(1/n_points)), n_points)
#xyz = radius*np.array([np.cos(angle), np.sin(angle), angle*0]).T

# Generate data to plot:
X   = np.arange(0, 100, 1)
Y = (np.random.rand(len(xyz), 100)*8) - (4 + np.random.rand()*0.2)

# %% Generate subplot positions:
xy = project_onto_zplane(xyz, projection='z', 
                             scale_seperation_distance=2, ch_names=ch_names)
grid_size = gen_grid_size(xy, sbp=2)
grid_pos = project_onto_grid(xy, grid_size, rotation_matrix=-90)

# %% PLOT:
# %% Using grid_size & grid_pos to plot the data onto the subplots:
# I explicitly don't use gridspec becuase I have observed poor performance
# when working with very large grids (~400 squares).

plt.style.use('dark_background')
fig, ax = plt.subplots(grid_size[0], grid_size[1], figsize=(12, 8))
[axi.set_axis_off() for axi in ax.ravel()] #Turn off axes
[axi.tick_params(axis='both', labelsize=8) for axi in ax.ravel()] #Set axis text size

# Generating equal y-limits:
ylim = round(max([abs(Y.max()), abs(Y.min())]))
plt.setp(ax, xlim=[X[0], X[-1]], 
             ylim=[-ylim, ylim])

for i, pos in enumerate(grid_pos):
    ax[pos[0], pos[1]].plot(X, Y[i,:], '-', linewidth=0.5)
    
    #General settings:
    ax[pos[0], pos[1]].set_title(ch_names[i], size=8)
    
    # Ticks and border settings:
    ax[pos[0], pos[1]].axis('on')
    ax[pos[0], pos[1]].spines['right'].set_visible(False)
    ax[pos[0], pos[1]].spines['top'].set_visible(False)
    ax[pos[0], pos[1]].set_xticklabels([])
    ax[pos[0], pos[1]].set_yticklabels([])

#Make the top-left a reference:
ax[0, 0].axis('on')
ax[0, 0].spines['right'].set_visible(False)
ax[0, 0].spines['top'].set_visible(False)
ax[0, 0].set_xlabel('time (s)')
ax[0, 0].set_ylabel('voltage (V)')



