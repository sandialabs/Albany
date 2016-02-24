//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#ifndef PATH_SIZE_FIELD_HPP
#define PATH_SIZE_FIELD_HPP

#include <apfMesh.h>

namespace Albany {

struct PathSizeParameters {
  double beam_radius;
};

void addPredictedLaserSizeField(apf::Mesh* m,
    apf::Vector3 const& old_point, apf::Vector3 const& new_point,
    PathSizeParameters const& parameters);

}

#endif
