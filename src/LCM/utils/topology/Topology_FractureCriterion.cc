//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#include "Topology.h"
#include "Topology_FractureCriterion.h"

namespace LCM{

FractureCriterionTraction::FractureCriterionTraction(
    Topology & topology,
    std::string const & stress_name,
    double const critical_traction,
    double const beta) :
AbstractFractureCriterion(),
topology_(topology),
stk_discretization_(*(topology.getSTKDiscretization())),
stk_mesh_struct_(*(stk_discretization_.getSTKMeshStruct())),
bulk_data_(*(stk_mesh_struct_.bulkData)),
meta_data_(*(stk_mesh_struct_.metaData)),
dimension_(stk_mesh_struct_.numDim),
stress_field_(*(meta_data_.get_field<TensorFieldType>(stress_name))),
critical_traction_(critical_traction),
beta_(beta)
{
  if (&stress_field_ == 0) {
    std::cerr << "ERROR: " << __PRETTY_FUNCTION__;
    std::cerr << '\n';
    std::cerr << "Cannot find field for traction criterion: ";
    std::cerr << stress_name;
    std::cerr << '\n';
    exit(1);
  }
  computeNormals();
}


bool
FractureCriterionTraction::check(Entity const & entity)
{
  stk_classic::mesh::PairIterRelation const
  relations_up = relations_one_up(entity);

  assert(relations_up.size() == 2);

  EntityVector
  nodes = topology_.getBoundaryEntityNodes(entity);

  EntityVector::size_type const
  number_nodes = nodes.size();

  Intrepid::Tensor<double>
  stress(dimension_, Intrepid::ZEROS);

  Intrepid::Tensor<double>
  nodal_stresses(number_nodes);

  nodal_stresses.set_dimension(dimension_);

  // The traction is evaluated at centroid of face, so a simple
  // average yields the value.
  for (EntityVector::size_type i = 0; i < number_nodes; ++i) {

    Entity &
    node = *(nodes[i]);

    double * const
    pstress = stk_classic::mesh::field_data(stress_field_, node);

    nodal_stresses.fill(pstress);

    stress += nodal_stresses;
  }

  stress /= static_cast<double>(number_nodes);

  Intrepid::Index const
  face_index = entity.identifier() - 1;

  Intrepid::Vector<double> const &
  normal = normals_[face_index];

  Intrepid::Vector<double> const
  traction = stress * normal;

  double
  t_n = Intrepid::dot(traction, normal);

  Intrepid::Vector<double> const
  traction_normal = t_n * normal;

  Intrepid::Vector<double> const
  traction_shear = traction - traction_normal;

  double const
  t_s = Intrepid::norm(traction_shear);

  // Ignore compression
  t_n = std::max(t_n, 0.0);

  double const
  effective_traction = std::sqrt(t_s * t_s / beta_ / beta_ + t_n * t_n);

  return effective_traction >= critical_traction_;
}

void
FractureCriterionTraction::computeNormals()
{
  stk_classic::mesh::Selector
  local_selector = meta_data_.locally_owned_part();

  std::vector<Bucket*> const &
  node_buckets = bulk_data_.buckets(NODE_RANK);

  EntityVector
  nodes;

  stk_classic::mesh::get_selected_entities(local_selector, node_buckets, nodes);

  EntityVector::size_type const
  number_nodes = nodes.size();

  std::vector<Intrepid::Vector<double> >
  coordinates(number_nodes);

  Teuchos::ArrayRCP<double> &
  node_coordinates = stk_discretization_.getCoordinates();

  for (EntityVector::size_type i = 0; i < number_nodes; ++i) {

    double const * const
    pointer_coordinates = &(node_coordinates[dimension_ * i]);

    coordinates[i].set_dimension(dimension_);
    coordinates[i].fill(pointer_coordinates);

  }

  EntityRank const
  cell_rank = meta_data_.element_rank();

  std::vector<Bucket*> const &
  face_buckets = bulk_data_.buckets(cell_rank - 1);

  EntityVector
  faces;

  stk_classic::mesh::get_selected_entities(local_selector, face_buckets, faces);

  EntityVector::size_type const
  number_normals = faces.size();

  normals_.resize(number_normals);

  for (EntityVector::size_type i = 0; i < number_normals; ++i) {

    Entity const &
    face = *(faces[i]);

    EntityVector
    nodes = topology_.getBoundaryEntityNodes(face);

    Intrepid::Vector<double> &
    normal = normals_[i];

    normal.set_dimension(dimension_);

    // Depending on the dimension is how the normal is computed.
    // TODO: generalize this for all topologies.
    switch (dimension_) {

    default:
      std::cerr << "ERROR: " << __PRETTY_FUNCTION__ << '\n';
      std::cerr << "Wrong dimension: " << dimension_ << '\n';
      exit(1);
      break;

    case 2:
      {
        int const
        gid0 = nodes[0]->identifier() - 1;

        Intrepid::Index const
        lid0 = stk_discretization_.getNodeMap()->LID(gid0);

        assert(lid0 < number_nodes);

        int const
        gid1 = nodes[1]->identifier() - 1;

        Intrepid::Index const
        lid1 = stk_discretization_.getNodeMap()->LID(gid1);

        assert(lid1 < number_nodes);

        Intrepid::Vector<double>
        v = coordinates[lid1] - coordinates[lid0];

        normal(0) = -v(1);
        normal(1) = v(0);

        normal = Intrepid::unit(normal);
      }
      break;

    case 3:
      {
        int const
        gid0 = nodes[0]->identifier() - 1;

        Intrepid::Index const
        lid0 = stk_discretization_.getNodeMap()->LID(gid0);

        assert(lid0 < number_nodes);

        int const
        gid1 = nodes[1]->identifier() - 1;

        Intrepid::Index const
        lid1 = stk_discretization_.getNodeMap()->LID(gid1);

        assert(lid1 < number_nodes);

        int const
        gid2 = nodes[2]->identifier() - 1;

        Intrepid::Index const
        lid2 = stk_discretization_.getNodeMap()->LID(gid2);

        assert(lid2 < number_nodes);

        Intrepid::Vector<double>
        v1 = coordinates[lid1] - coordinates[lid0];

        Intrepid::Vector<double>
        v2 = coordinates[lid2] - coordinates[lid0];

        normal = Intrepid::cross(v1, v2);

        normal = Intrepid::unit(normal);
      }
      break;

    }

  }

}

} // namespace LCM

