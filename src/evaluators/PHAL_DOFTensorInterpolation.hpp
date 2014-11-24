//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#ifndef PHAL_DOFTENSOR_INTERPOLATION_HPP
#define PHAL_DOFTENSOR_INTERPOLATION_HPP

#include "Phalanx_ConfigDefs.hpp"
#include "Phalanx_Evaluator_WithBaseImpl.hpp"
#include "Phalanx_Evaluator_Derived.hpp"
#include "Phalanx_MDField.hpp"

#include "Albany_Layouts.hpp"

namespace PHAL {
/** \brief Finite Element Interpolation Evaluator

    This evaluator interpolates nodal DOFTensor values to quad points.

*/

template<typename EvalT, typename Traits>
class DOFTensorInterpolation
       : public PHX::EvaluatorWithBaseImpl<Traits>,
 	 public PHX::EvaluatorDerived<EvalT, Traits>  {

public:

  DOFTensorInterpolation(const Teuchos::ParameterList& p,
                      const Teuchos::RCP<Albany::Layouts>& dl);

  void postRegistrationSetup(typename Traits::SetupData d,
                      PHX::FieldManager<Traits>& vm);

  void evaluateFields(typename Traits::EvalData d);

private:

  typedef typename EvalT::ScalarT ScalarT;

  // Input:
  //! Values at nodes
  PHX::MDField<ScalarT,Cell,Node,VecDim,VecDim> val_node;
  //! Basis Functions
  PHX::MDField<RealType,Cell,Node,QuadPoint> BF;

  // Output:
  //! Values at quadrature points
  PHX::MDField<ScalarT,Cell,QuadPoint,VecDim,VecDim> val_qp;

  std::size_t numNodes;
  std::size_t numQPs;
  std::size_t vecDim;
};

//! Specialization for Jacobian evaluation taking advantage of known sparsity
template<typename Traits>
class DOFTensorInterpolation<PHAL::AlbanyTraits::Jacobian, Traits>
      : public PHX::EvaluatorWithBaseImpl<Traits>,
        public PHX::EvaluatorDerived<PHAL::AlbanyTraits::Jacobian, Traits>  {

public:

  DOFTensorInterpolation(const Teuchos::ParameterList& p,
                      const Teuchos::RCP<Albany::Layouts>& dl);

  void postRegistrationSetup(typename Traits::SetupData d,
                      PHX::FieldManager<Traits>& vm);

  void evaluateFields(typename Traits::EvalData d);

private:

  typedef typename PHAL::AlbanyTraits::Jacobian::ScalarT ScalarT;

  // Input:
  //! Values at nodes
  PHX::MDField<ScalarT,Cell,Node,VecDim,VecDim> val_node;
  //! Basis Functions
  PHX::MDField<RealType,Cell,Node,QuadPoint> BF;

  // Output:
  //! Values at quadrature points
  PHX::MDField<ScalarT,Cell,QuadPoint,VecDim,VecDim> val_qp;

  std::size_t numNodes;
  std::size_t numQPs;
  std::size_t vecDim;
  std::size_t offset;
};

}

#endif
