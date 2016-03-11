#include "PathSizeField.hpp"

#include <apf.h>
#include <apfMesh.h>
#include <apfGeometry.h>

namespace Albany {

struct SampledSize {
  double constant_value;
};

static SampledSize sampleSizeAtOldPoint(apf::Mesh* m,
    apf::Field* size_field,
    apf::Vector3 const& old_point,
    PathSizeParameters const& parameters)
{
  apf::MeshIterator* vertices = m->begin(0);
  apf::MeshEntity* vertex;
  apf::MeshEntity* closest_vertex = 0;
  double closest_distance;
  while ((vertex = m->iterate(vertices))) {
    apf::Vector3 vertex_point;
    m->getPoint(vertex, 0, vertex_point);
    double distance = (vertex_point - old_point).getLength();
    if (closest_vertex == 0 ||
        (distance < closest_distance)) {
      closest_vertex = vertex;
      closest_distance = distance;
    }
  }
  m->end(vertices);
  assert(closest_distance < parameters.beam_radius);
  double closest_value = apf::getScalar(size_field, closest_vertex, 0);
  SampledSize output;
  output.constant_value = closest_value;
  return output;
}

static void applySizeAlongPath(apf::Mesh* m,
    apf::Field* size_field,
    apf::Vector3 const& old_point, apf::Vector3 const& new_point,
    SampledSize const& sampled_size, PathSizeParameters const& parameters)
{
  apf::MeshIterator* vertices = m->begin(0);
  apf::MeshEntity* vertex;
  apf::LineSegment segment(old_point, new_point);
  while ((vertex = m->iterate(vertices))) {
    apf::Vector3 vertex_point;
    m->getPoint(vertex, 0, vertex_point);
    double distance = apf::getDistance(segment, vertex_point);
    if (distance < parameters.beam_radius)
      apf::setScalar(size_field, vertex, 0, sampled_size.constant_value);
  }
  m->end(vertices);
}

void addPredictedLaserSizeField(apf::Mesh* m,
    apf::Field* size_field,
    apf::Vector3 const& old_point, apf::Vector3 const& new_point,
    PathSizeParameters const& parameters)
{
  SampledSize sampled_size = sampleSizeAtOldPoint(
      m, size_field, old_point, parameters);
  applySizeAlongPath(m, size_field, old_point, new_point,
      sampled_size, parameters);
}

}
