//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#include "AAdapt_STKUnifSizeField.hpp"
#include "Albany_AbstractSTKFieldContainer.hpp"
#include "Epetra_Import.h"

/*
 * Utility functions (from stk_adapt/regression_tests/RegressionTestLocalRefiner.cpp)
 */

static void normalize(double input_normal[3], double normal[3]) {
  double sum = std::sqrt(input_normal[0] * input_normal[0] +
                         input_normal[1] * input_normal[1] +
                         input_normal[2] * input_normal[2]);
  normal[0] = input_normal[0] / sum;
  normal[1] = input_normal[1] / sum;
  normal[2] = input_normal[2] / sum;
}

static void normalize(double input_output_normal[3]) {
  normalize(input_output_normal, input_output_normal);
}

static double distance(double c0[3], double c1[3]) {
  return std::sqrt((c0[0] - c1[0]) * (c0[0] - c1[0]) + (c0[1] - c1[1]) * (c0[1] - c1[1]) + (c0[2] - c1[2]) * (c0[2] - c1[2]));
}

static void difference(double v01[3], double c0[3], double c1[3]) {
  v01[0] = c0[0] - c1[0];
  v01[1] = c0[1] - c1[1];
  v01[2] = c0[2] - c1[2];
}

static double dot(double c0[3], double c1[3]) {
  return c0[0] * c1[0] + c0[1] * c1[1] + c0[2] * c1[2];
}

static double plane_dot_product(double plane_point[3], double plane_normal[3], double point[3]) {
  double normal[3] = {0, 0, 0};
  normalize(plane_normal, normal);
  double dot = 0.0;

  for(int i = 0; i < 3; i++) {
    dot += (point[i] - plane_point[i]) * normal[i];
  }

  return dot;
}

#if 0
static void project_point_to_plane(double plane_point[3], double plane_normal[3], double point[3], double projected_point[3]) {
  double normal[3] = {0, 0, 0};
  normalize(plane_normal, normal);
  double dot = plane_dot_product(plane_point, plane_normal, point);

  for(int i = 0; i < 3; i++) {
    projected_point[i] = plane_point[i] + (point[i] - dot * normal[i]);
  }
}
#endif


bool
AAdapt::STKUnifRefineField::operator()(const stk_classic::mesh::Entity& element,
                                       stk_classic::mesh::FieldBase* field,  const stk_classic::mesh::BulkData& bulkData) {

  /*
    double plane_point[3] = {2, 0, 0};
    double plane_normal[3] = {1, .5, -.5};
  */
  double plane_point[3] = {0, 0.7, 0};
  double plane_normal[3] = {0, 1, 0};

  const stk_classic::mesh::PairIterRelation elem_nodes = element.relations(stk_classic::mesh::fem::FEMMetaData::NODE_RANK);
  unsigned num_node = elem_nodes.size();
  double* f_data = stk_classic::percept::PerceptMesh::field_data_entity(field, element);
  Albany::AbstractSTKFieldContainer::VectorFieldType* coordField = m_eMesh.get_coordinates_field();

  bool found = false;

  for(unsigned inode = 0; inode < num_node - 1; inode++) {
    stk_classic::mesh::Entity& node_i = * elem_nodes[ inode ].entity();
    double* coord_data_i = stk_classic::percept::PerceptMesh::field_data(coordField, node_i);

    for(unsigned jnode = inode + 1; jnode < num_node; jnode++) {
      stk_classic::mesh::Entity& node_j = * elem_nodes[ jnode ].entity();
      double* coord_data_j = stk_classic::percept::PerceptMesh::field_data(coordField, node_j);

      double dot_0 = plane_dot_product(plane_point, plane_normal, coord_data_i);
      double dot_1 = plane_dot_product(plane_point, plane_normal, coord_data_j);

      // if edge crosses the plane...
      if(dot_0 * dot_1 < 0) {
        found = true;
        break;
      }
    }
  }

  if(found) {
    f_data[0] = 1.0;
    //    std::cout << "Splitting: " << element.identifier() << std::endl;
  }

  else {
    f_data[0] = 0.0;
    //    std::cout << "Not splitting: " << element.identifier() << std::endl;
  }

  return false;  // don't terminate the loop
}

bool
AAdapt::STKUnifUnrefineField::operator()(const stk_classic::mesh::Entity& element,
    stk_classic::mesh::FieldBase* field,  const stk_classic::mesh::BulkData& bulkData) {

  const stk_classic::mesh::PairIterRelation elem_nodes = element.relations(stk_classic::mesh::fem::FEMMetaData::NODE_RANK);
  unsigned num_node = elem_nodes.size();
  double* f_data = stk_classic::percept::PerceptMesh::field_data_entity(field, element);
  Albany::AbstractSTKFieldContainer::VectorFieldType* coordField = m_eMesh.get_coordinates_field();

  bool found = true;

  for(unsigned inode = 0; inode < num_node; inode++) {
    stk_classic::mesh::Entity& node = * elem_nodes[ inode ].entity();
    double* coord_data = stk_classic::percept::PerceptMesh::field_data(coordField, node);

    if(coord_data[0] < 0.0 || coord_data[1] < 0.0) { // || coord_data[2] > 1.1)
      found = false;
      break;
    }
  }

  if(found)
    f_data[0] = -1.0;

  else
    f_data[0] = 0.0;

  return false;  // don't terminate the loop
}




