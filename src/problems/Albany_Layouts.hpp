//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#ifndef ALBANY_LAYOUTS_HPP
#define ALBANY_LAYOUTS_HPP

#include <map>
#include <string>

#include "Phalanx_DataLayout.hpp"
#include "Teuchos_RCP.hpp"

namespace Albany {
  /*!
   * \brief Struct to construct and hold DataLayouts
   */
  struct Layouts {

    Layouts (int worksetSize, int numVertices, int numNodes, int numQPts, int numCellDim, int vecDim=-1, int numFace=0);
    Layouts (int numVertices, int numNodes, int numQPts, int numSideDim, int numSpaceDim, int vecDim, std::string sideSetName);

    //! Data Layout for scalar quantity that lives at nodes
    Teuchos::RCP<PHX::DataLayout> node_scalar;
    //! Data Layout for scalar quantity that lives at quad points
    Teuchos::RCP<PHX::DataLayout> qp_scalar;
    //! Data Layout for scalar quantity that lives on a cell
    Teuchos::RCP<PHX::DataLayout> cell_scalar;
    //! Data Layout for scalar quantity that lives on a cell
    Teuchos::RCP<PHX::DataLayout> cell_scalar2;
    //! Data Layout for scalar quantity that lives on a face
    Teuchos::RCP<PHX::DataLayout> face_scalar;
    //! Data Layout for vector quantity that lives at nodes
    Teuchos::RCP<PHX::DataLayout> node_vector;
    //! Data Layout for vector quantity that lives at quad points
    Teuchos::RCP<PHX::DataLayout> qp_vector;
    //! Data Layout for vector quantity that lives on a cell
    Teuchos::RCP<PHX::DataLayout> cell_vector;
    //! Data Layout for vector quantity that lives on a face
    Teuchos::RCP<PHX::DataLayout> face_vector;
    //! Data Layout for gradient quantity that lives at nodes
    Teuchos::RCP<PHX::DataLayout> node_gradient;
    //! Data Layout for gradient quantity that lives at quad points
    Teuchos::RCP<PHX::DataLayout> qp_gradient;
    //! Data Layout for vector quantity that lives at quad points, with dimension of the ambient space
    Teuchos::RCP<PHX::DataLayout> qp_vector_spacedim;
    //! Data Layout for gradient quantity that lives on a cell
    Teuchos::RCP<PHX::DataLayout> cell_gradient;
    //! Data Layout for gradient quantity that lives on a face
    Teuchos::RCP<PHX::DataLayout> face_gradient;
    //! Data Layout for tensor quantity that lives at nodes
    Teuchos::RCP<PHX::DataLayout> node_tensor;
    //! Data Layout for tensor quantity that lives at quad points
    Teuchos::RCP<PHX::DataLayout> qp_tensor;
    //! Data Layout for tensor quantity (cellDim x sideDim) that lives at quad points.
    Teuchos::RCP<PHX::DataLayout> qp_tensor_cd_sd;
    //! Data Layout for tensor quantity that lives on a cell
    Teuchos::RCP<PHX::DataLayout> cell_tensor;
    //! Data Layout for tensor quantity that lives on a face
    Teuchos::RCP<PHX::DataLayout> face_tensor;
    //! Data Layout for tensor gradient quantity that lives at quad points
    Teuchos::RCP<PHX::DataLayout> qp_tensorgradient;
    //! Data Layout for vector gradient quantity that lives at nodes
    Teuchos::RCP<PHX::DataLayout> node_vecgradient;
    //! Data Layout for vector gradient quantity that lives at quad points
    Teuchos::RCP<PHX::DataLayout> qp_vecgradient;
    //! Data Layout for vector gradient quantity that lives on a cell
    Teuchos::RCP<PHX::DataLayout> cell_vecgradient;
    //! Data Layout for vector gradient quantity that lives on a face
    Teuchos::RCP<PHX::DataLayout> face_vecgradient;
    //! Data Layout for third order tensor quantity that lives at nodes
    Teuchos::RCP<PHX::DataLayout> node_tensor3;
    //! Data Layout for third order tensor quantity that lives at quad points
    Teuchos::RCP<PHX::DataLayout> qp_tensor3;
    //! Data Layout for third order tensor quantity that lives on a cell
    Teuchos::RCP<PHX::DataLayout> cell_tensor3;
    //! Data Layout for third order tensor quantity that lives on a face
    Teuchos::RCP<PHX::DataLayout> face_tensor3;
    //! Data Layout for fourth order tensor quantity that lives at nodes
    Teuchos::RCP<PHX::DataLayout> node_tensor4;
    //! Data Layout for fourth order tensor quantity that lives at quad points
    Teuchos::RCP<PHX::DataLayout> qp_tensor4;
    //! Data Layout for fourth order tensor quantity that lives on a cell
    Teuchos::RCP<PHX::DataLayout> cell_tensor4;
    //! Data Layout for fourth order tensor quantity that lives on a face
    Teuchos::RCP<PHX::DataLayout> face_tensor4;
    //! Data Layout for vector quantity that lives at vertices (coordinates) //FIXME: dont coords live at nodes, not vertices?
    Teuchos::RCP<PHX::DataLayout> vertices_vector;
    Teuchos::RCP<PHX::DataLayout> qp_coords;
    //! Data Layout for length 3 quantity  that lives at nodes (shell coordinates)
    Teuchos::RCP<PHX::DataLayout> node_3vector;

    //! Data Layout for scalar basis functions
    Teuchos::RCP<PHX::DataLayout> node_qp_scalar;
    //! Data Layout for gradient basis functions
    Teuchos::RCP<PHX::DataLayout> node_qp_gradient;
    Teuchos::RCP<PHX::DataLayout> node_qp_vector; // Old, but incorrect name

    //! Data Layout for scalar quantity on workset
    Teuchos::RCP<PHX::DataLayout> workset_scalar;
    //! Data Layout for vector quantity on workset
    Teuchos::RCP<PHX::DataLayout> workset_vector;
    //! Data Layout for gradient quantity on workset
    Teuchos::RCP<PHX::DataLayout> workset_gradient;
    //! Data Layout for tensor quantity on workset
    Teuchos::RCP<PHX::DataLayout> workset_tensor;
    //! Data Layout for vector gradient quantity on workset
    Teuchos::RCP<PHX::DataLayout> workset_vecgradient;

    //! Data Layout for scalar quantity that is hosted by nodes
    Teuchos::RCP<PHX::DataLayout> node_node_scalar;
    //! Data Layout for vector quantity that is hosted by nodes
    Teuchos::RCP<PHX::DataLayout> node_node_vector;
    //! Data Layout for tensor quantity that is hosted by nodes
    Teuchos::RCP<PHX::DataLayout> node_node_tensor;
    /*!
     * \brief Dummy Data Layout where one is needed but not accessed
     * For instance, the action of scattering residual data from a
     * Field into the residual vector in the workset struct needs an
     * evaluator, but the evaluator has no natural Field that it computes.
     * So, it computes the Scatter field with this (empty) Dummy layout.
     * Requesting this Dummy Field then activates this evaluator so
     * the action is performed.
     */
    Teuchos::RCP<PHX::DataLayout> shared_param;
    Teuchos::RCP<PHX::DataLayout> shared_param_vec; // same length as other vectors
    Teuchos::RCP<PHX::DataLayout> dummy;

    // For backward compatibility, and simplicitiy, we want to check if
    // the vector length is the same as the spatial dimension. This
    // assumption is hardwired in mechanics problems and we want to
    // test that it is a valid assumption with this bool.
    bool vectorAndGradientLayoutsAreEquivalent;

    // A flag to check whether this layouts structure belongs to a sideset
    bool isSideLayouts;

    std::map<std::string,Teuchos::RCP<Layouts>> side_layouts;
  };

// -------- Enums and utils to automatize layout specification -------- //

// Define an invalid string in one place (to avoid typos-related bugs).
constexpr const char* INVALID_STR = "__INVALID__";

// Mesh entity where a field is located
enum class FieldLocation : int {
  Cell,
  Node,
  QuadPoint
};

inline std::string e2str (const FieldLocation e) {
  switch (e) {
    case FieldLocation::Node:       return "Node";
    case FieldLocation::QuadPoint:  return "QuadPoint";
    case FieldLocation::Cell:       return "Cell";
  }
  return INVALID_STR;
}

// Type of field (scalar, vector, gradient, tensor)
// Note: gradient is just a vector with length equal to the mesh dimension
enum class FieldRankType : int {
  Scalar,
  Vector,
  Gradient,
  Tensor
};

inline std::string e2str (const FieldRankType rank) {
  switch (rank) {
    case FieldRankType::Scalar:     return "Scalar";
    case FieldRankType::Vector:     return "Vector";
    case FieldRankType::Gradient:   return "Gradient";
    case FieldRankType::Tensor:     return "Tensor";
  }

  return INVALID_STR;
}

// Get the field PHX layout from rank and location
inline Teuchos::RCP<PHX::DataLayout>
get_field_layout (const FieldRankType rank,
                  const FieldLocation loc,
                  const Teuchos::RCP<Albany::Layouts>& layouts)
{
  using FRT = FieldRankType;
  using FL  = FieldLocation;

  Teuchos::RCP<PHX::DataLayout> fl;
  if (rank==FRT::Scalar) {
    if (loc==FL::Cell) {
      fl = layouts->cell_scalar2;
    } else if (loc==FL::Node) {
      fl = layouts->node_scalar;
    } else {
      fl = layouts->qp_scalar;
    }
  } else if (rank==FRT::Vector) {
    if (loc==FL::Cell) {
      fl = layouts->cell_vector;
    } else if (loc==FL::Node) {
      fl = layouts->node_vector;
    } else {
      fl = layouts->qp_vector;
    }
  } else if (rank==FRT::Gradient) {
    if (loc==FL::Cell) {
      fl = layouts->cell_gradient;
    } else if (loc==FL::Node) {
      fl = layouts->node_gradient;
    } else {
      fl = layouts->qp_gradient;
    }
  } else if (rank==FRT::Tensor) {
    if (loc==FL::Cell) {
      fl = layouts->cell_tensor;
    } else if (loc==FL::Node) {
      fl = layouts->node_tensor;
    } else {
      fl = layouts->qp_tensor;
    }
  }

  TEUCHOS_TEST_FOR_EXCEPTION(fl.is_null(), std::runtime_error,
      "Error! Failed to create field layout.\n");

  return fl;
}

// Teuchos requires Teuchos::is_printable<T>::type and Teuchos::is_comparable<T>::type
// to be std::true_type in order for T to be storable in a parameter list.
// However, strong enums are not printable by default, so we need an overload for op<<
template<typename T>
typename
std::enable_if<std::is_same<T,FieldLocation>::value ||
               std::is_same<T,FieldRankType>::value,
               std::ostream&>::type
operator<< (std::ostream& out, const T t) {
  out << e2str(t);
  return out;
}

} // namespace Albany

#endif // ALBANY_LAYOUTS_HPP
