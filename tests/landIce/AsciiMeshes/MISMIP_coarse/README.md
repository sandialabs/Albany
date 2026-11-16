# MISMIP_coarse: coarse MISMIP 2D test mesh and fields

A coarser version of `../MISMIP` (mesh `mismip_2d.exo` and the nodal fields
`thickness.ascii`, `surface_height.ascii`, `bed_topography.ascii`, `averaged_velocity.ascii`),
for tests that need to run faster. The files have the same names as in `../MISMIP`, so a test
switches to the coarse data by changing `AsciiMeshes/MISMIP/` into `AsciiMeshes/MISMIP_coarse/`.

|                 | `../MISMIP`        | `MISMIP_coarse`     |
|-----------------|--------------------|---------------------|
| nodes           | 4187               | 504                 |
| triangles       | 7992               | 880                 |
| rows of nodes   | 25 (dy = 3.33 km)  | 9 (dy = 10 km)      |
| edge length     | 3.85 km            | 11.55 km            |
| x range [km]    | 0 - 642.78         | 0 - 640.86          |

## How it is made

The MISMIP mesh is a lattice of equilateral triangles: rows of nodes at y = j*dy, with spacing
a = 2*dy/sqrt(3) along each row, and consecutive rows shifted by a/2. The coarse mesh is the same
lattice with 3 times the spacing, built on every third row of the fine mesh and on its nodes only.
So:
- every node of the coarse mesh is a node of the fine mesh, and the coarse fields are the values
  of the fine fields at those nodes (no interpolation: the bed is still the MISMIP+ bed, and the
  surface is still consistent with the thickness and the bed where the ice is grounded);
- the node sets and side sets have the same names, ids and meaning as in the fine mesh:
  - node sets: `node_set` (all nodes), `boundary_node_set_0` (empty), `boundary_node_set_1`
    (first node of each row, x ~ 0), `boundary_node_set_2` (last node of each row, except the
    first and last rows), `boundary_node_set_3` (first and last rows, y = 0 and y = 80, except
    their first node);
  - side sets: `boundary_side_set` (whole boundary), `boundary_side_set_1` (boundary without the
    edges between the last nodes of the rows), `boundary_side_set_2` (edges between the last
    nodes of the rows, i.e., the calving front).

The only change of geometry is at the calving front: the last node of the rows alternates between
x = 640.86 and x = 635.09 (in the fine mesh, 640.86 and 642.78), since the fine mesh has no nodes
beyond x = 642.78.

## Regenerating the files

    ./make_coarse_mesh.py --fine-dir ../MISMIP --out-dir . -m 3

(requires Python with numpy and netCDF4). The script first rebuilds the fine mesh from the
lattice (`-m 1`) and checks that it matches `../MISMIP/mismip_2d.exo` (nodes, triangles, node sets
and side sets), then writes the coarse mesh with the layout of the fine one (netCDF3 64-bit
offset Exodus file, as written by IOSS) and the coarse fields. Other factors that divide the 24
row intervals of the fine mesh (2, 4, 6, ...) also work.
