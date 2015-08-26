//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#ifndef PHASESOURCE_HPP
#define PHASESOURCE_HPP

#include "Phalanx_config.hpp"
#include "Phalanx_Evaluator_WithBaseImpl.hpp"
#include "Phalanx_Evaluator_Derived.hpp"
#include "Phalanx_MDField.hpp"
#include "Teuchos_ParameterList.hpp"
#include "Epetra_Vector.h"
#include "Sacado_ParameterAccessor.hpp"
#include "Stokhos_KL_ExponentialRandomField.hpp"
#include "Teuchos_Array.hpp"
#include "Albany_Layouts.hpp"

namespace AMP {
///
/// \brief Phase Source
///
/// This evaluator computes the source function to a 
/// phase-change/heat equation problem
///
template<typename EvalT, typename Traits>
class PhaseSource : 
  public PHX::EvaluatorWithBaseImpl<Traits>,
  public PHX::EvaluatorDerived<EvalT, Traits>
{

public:

  PhaseSource(Teuchos::ParameterList& p,
      const Teuchos::RCP<Albany::Layouts>& dl);

  void 
  postRegistrationSetup(typename Traits::SetupData d,
      PHX::FieldManager<Traits>& vm);

  void 
  evaluateFields(typename Traits::EvalData d);

private:

  typedef typename EvalT::ScalarT ScalarT;
  typedef typename EvalT::MeshScalarT MeshScalarT;

  ScalarT laser_power;
  void init_constant(ScalarT value, Teuchos::ParameterList& p);

  PHX::MDField<MeshScalarT,Cell,QuadPoint,Dim> coord_;

  PHX::MDField<ScalarT,Cell,Node> source_;

  unsigned int num_qps_;
  unsigned int num_dims_;
  unsigned int num_nodes_;
  unsigned int workset_size_;

  Teuchos::RCP<const Teuchos::ParameterList>
     getValidPhaseSourceParameters() const;
};
}

#endif
