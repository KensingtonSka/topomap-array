# TopoMap Array
The purpose of this repository is to make an easy way to plot data from a collection of sensors
onto a subplot so that each subplot has the same relative position as the sensors in real life.
The functions in `Topomap_Array.py` are used to take a set of xyz coordinates, map them onto 
the z-plane (using three posible projections, see below), and use resulting values to assign 
each coordinate a subplot number within a grid of subplots.

## Projections:
There are presently three projetions:
1. `'z'`: This projection simply sets the z-coordinates to zero.
2. `'z-scale'`: This projection scales the y-coordinates by the magnitude of the z-coordinate.
3. `'stereographic'`: A standard stereographic projection.


## Example:
The below example illustrates the result of a set of xy coordinates sampled from a circle being 
used to generate a circle of subplots using the 'z' projection.
![Circle Array](example_figures/circle_array.png?raw=true "Title")


We can also use more complicated sets of xyz coordiantes points, such as a set of electroencephalogram sensors ('z' projection):
![Circle Array](example_figures/10-20_array.png?raw=true "Title")

