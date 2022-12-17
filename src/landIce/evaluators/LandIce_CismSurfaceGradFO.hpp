//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#ifndef LANDICE_CISM_SURFACE_GRAD_FO_HPP
#define LANDICE_CISM_SURFACE_GRAD_FO_HPP

#include "Phalanx_config.hpp"
#include "Phalanx_Evaluator_WithBaseImpl.hpp"
#include "Phalanx_Evaluator_Derived.hpp"
#include "Phalanx_MDField.hpp"

#include "Albany_Layouts.hpp"
#include "Albany_ScalarOrdinalTypes.hpp"
#include "PHAL_Dimension.hpp"

namespace LandIce {

/** \brief Construct surface gradient from CISM components

    Given nodal values for dsdx and dsdy, this evaluator computes
    the vector quantity grad(ds) at the quadrature points
*/

template<typename EvalT, typename Traits>
class CismSurfaceGradFO : public PHX::EvaluatorWithBaseImpl<Traits>,
		    public PHX::EvaluatorDerived<EvalT, Traits> {
public:

  typedef typename EvalT::ScalarT ScalarT;

  CismSurfaceGradFO(const Teuchos::ParameterList& p,
                    const Teuchos::RCP<Albany::Layouts>& dl);

  void postRegistrationSetup(typename Traits::SetupData d,
                             PHX::FieldManager<Traits>& vm);

  void evaluateFields(typename Traits::EvalData d);

private:
 
  typedef typename EvalT::MeshScalarT MeshScalarT;

  // Input:
  //! Values at nodes
  PHX::MDField<const MeshScalarT,Cell,Node> dsdx_node;
  PHX::MDField<const MeshScalarT,Cell,Node> dsdy_node;
  PHX::MDField<const RealType,Cell,Node,QuadPoint> BF;

  // Output:
  PHX::MDField<MeshScalarT,Cell,QuadPoint,Dim> gradS_qp;

  unsigned int numQPs, numDims, numNodes;
};

} // namespace LandIce

#endif // LANDICE_CISM_SURFACE_GRAD_FO_HPP
