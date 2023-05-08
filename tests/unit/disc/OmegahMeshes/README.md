## contents
- `gis_unstruct_basal_populated.exo`
  - The populated (with fields) greenland basal exodus mesh was created
    by running `tests/landIce/FO_GIS` on the unpopulated
    `tests/landIce/ExoMeshes/gis_unstruct_2d.exo` mesh specified in
    `./tests/landIce/FO_GIS/input_fo_gis_populate_meshes.yaml`.
    Albany master @ bacbdb7 was the approximate version at time of creation.
- `gis_unstruct_basal_populated.osh`
  - Omega\_h mesh created with `exo2osh` tool from scorec Omega\_h  master @ e1be29b
    from `gis_unstruct_basal_populated.exo`
