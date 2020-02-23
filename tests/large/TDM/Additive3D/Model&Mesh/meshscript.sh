#/bin/bash

# This quick script converts a model, a mesh created from that model, and splits the mesh into a specified number of parts
# Input command example: ./meshscript <name_of_model> <appended_mesh_name> <num_processors_to_compute> <num_mesh_parts_to_create>
# Always name the mesh using the same base name as the source model and append a descriptor for that particular mesh
#  i.e. if the model name is "simple_block.smd" the mesh name should be something like "simple_block_10um_mesh.sms"


path1=.          #relative path to model
path2=.          #relative path to mesh (same folder if this script is in the meshes folder)
mdlConvert $path1/$1.smd  $path1/$1.dmg
convert $path1/$1.smd $path2/$1$2.sms  $path2/$1$2.smb
mpirun -n $3 split $path1/$1.dmg $path2/$1$2.smb $path2/$1$2_$4P.smb $4

