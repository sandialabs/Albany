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
