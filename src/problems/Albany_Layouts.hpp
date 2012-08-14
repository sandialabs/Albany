/********************************************************************\
*            Albany, Copyright (2010) Sandia Corporation             *
*                                                                    *
* Notice: This computer software was prepared by Sandia Corporation, *
* hereinafter the Contractor, under Contract DE-AC04-94AL85000 with  *
* the Department of Energy (DOE). All rights in the computer software*
* are reserved by DOE on behalf of the United States Government and  *
* the Contractor as provided in the Contract. You are authorized to  *
* use this computer software for Governmental purposes but it is not *
* to be released or distributed to the public. NEITHER THE GOVERNMENT*
* NOR THE CONTRACTOR MAKES ANY WARRANTY, EXPRESS OR IMPLIED, OR      *
* ASSUMES ANY LIABILITY FOR THE USE OF THIS SOFTWARE. This notice    *
* including this sentence must appear on any copies of this software.*
*    Questions to Andy Salinger, agsalin@sandia.gov                  *
\********************************************************************/


#ifndef ALBANY_LAYOUTS_HPP
#define ALBANY_LAYOUTS_HPP

#include <vector>
#include <string>

#include "Teuchos_RCP.hpp"

#include "Phalanx.hpp"
#include "Albany_DataTypes.hpp"

namespace Albany {
  /*!
   * \brief Struct to construct and hold DataLayouts
   */
  struct Layouts {
    Layouts(int worksetSize, int  numVertices, int numNodes, int numQPts, int numDim, int vecDim=-1, int numFace=0);
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
    //! Data Layout for gradient quantity that lives on a cell
    Teuchos::RCP<PHX::DataLayout> cell_gradient;
    //! Data Layout for gradient quantity that lives on a face
    Teuchos::RCP<PHX::DataLayout> face_gradient;
    //! Data Layout for tensor quantity that lives at nodes
    Teuchos::RCP<PHX::DataLayout> node_tensor;
    //! Data Layout for tensor quantity that lives at quad points
    Teuchos::RCP<PHX::DataLayout> qp_tensor;
    //! Data Layout for tensor quantity that lives on a cell
    Teuchos::RCP<PHX::DataLayout> cell_tensor;
    //! Data Layout for tensor quantity that lives on a face
    Teuchos::RCP<PHX::DataLayout> face_tensor;
    //! Data Layout for vector gradient quantity that lives at nodes
    Teuchos::RCP<PHX::DataLayout> node_vecgradient;
    //! Data Layout for vector gradient quantity that lives at quad points
    Teuchos::RCP<PHX::DataLayout> qp_vecgradient;
    //! Data Layout for vector gradient quantity that lives on a cell
    Teuchos::RCP<PHX::DataLayout> cell_vecgradient;
    //! Data Layout for vector gradient quantity that lives on a face
    Teuchos::RCP<PHX::DataLayout> face_vecgradient;
    //! Data Layout for vector quantity that lives at vertices (coordinates)
    Teuchos::RCP<PHX::DataLayout> vertices_vector;
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
    /*!
     * \brief Dummy Data Layout where one is needed but not accessed
     * For instance, the action of scattering residual data from a
     * Field into the residual vector in the workset struct needs an
     * evaluator, but the evaluator has no natural Field that it computes.
     * So, it computes the Scatter field with this (empty) Dummy layout.
     * Requesting this Dummy Field then activates this evaluator so
     * the action is performed.
     */
    Teuchos::RCP<PHX::DataLayout> dummy;

    // For backward compatibility, and simplicitiy, we want to check if
    // the vector length is the same as the spatial dimension. This
    // assumption is hardwired in mechanics problems and we want to 
    // test that it is a valide assumption with this bool.
    bool vectorAndGradientLayoutsAreEquivalent;
  };
}

#endif 
