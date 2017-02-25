# -*- coding: utf-8 -*-
"""
Created on Fri Mar 20 15:03:12 2015

@author: knkarls
"""

# import the module
import exomerge
# import the mesh
model = exomerge.import_model('surfaceDiffusion.g')
# create a cohesive zone element block 3 from side set 1
model.convert_side_set_to_cohesive_zone(1, 3)
# save the result
model.export_model('surfaceDiffusion_cse.g')
