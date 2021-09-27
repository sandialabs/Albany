//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#ifndef PHAL_ADVECTIONRESID_HPP
#define PHAL_ADVECTIONRESID_HPP

#include "Phalanx_config.hpp"
#include "Phalanx_Evaluator_WithBaseImpl.hpp"
#include "Phalanx_Evaluator_Derived.hpp"
#include "Phalanx_MDField.hpp"
#include "Sacado_ParameterAccessor.hpp"
#include "Albany_Layouts.hpp"

namespace PHAL {

/** \brief Finite Element Interpolation Evaluator

    This evaluator interpolates nodal DOF values to quad points.

*/

template<typename EvalT, typename Traits>
class AdvectionResid : public PHX::EvaluatorWithBaseImpl<Traits>,
		    public PHX::EvaluatorDerived<EvalT, Traits> {

public:

  AdvectionResid(Teuchos::ParameterList const& p,
                                const Teuchos::RCP<Albany::Layouts>& dl); 

  void
  postRegistrationSetup(
      typename Traits::SetupData d,
      PHX::FieldManager<Traits>& vm);

  void
  evaluateFields(typename Traits::EvalData d);

private:

  typedef typename EvalT::ScalarT     ScalarT;
  typedef typename EvalT::MeshScalarT MeshScalarT;

  ScalarT value;  
  
  // Input:
  PHX::MDField<const MeshScalarT, Cell, Node, QuadPoint>      wBF;
  PHX::MDField<ScalarT const, Cell, QuadPoint>                udot;
  PHX::MDField<ScalarT const, Cell, QuadPoint, Dim>           uGrad;
  PHX::MDField<const MeshScalarT, Cell, QuadPoint, Dim>       coordVec;
  bool advectionIsDistParam; //Flag telling code conductivity is distr. param.
  // Advection coefficient components
  PHX::MDField<const ScalarT> a_x;
  PHX::MDField<const ScalarT> a_y;
  PHX::MDField<const ScalarT> a_z;
  // Advection coefficient and its gradient (for distributed advection field)
  PHX::MDField<const ScalarT,Cell,QuadPoint> AdvCoeff;
  PHX::MDField<const ScalarT,Cell,QuadPoint,Dim> AdvCoeffGrad;

  // Output:
  PHX::MDField<ScalarT, Cell, QuadPoint> source;
  PHX::MDField<ScalarT, Cell, Node> residual;

  unsigned int numQPs, numDims, numNodes, worksetSize;
  enum FTYPE {NONE, XSIN};
  FTYPE force_type;
};
}

#endif
