//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#ifndef LOCAL_POROSITY_HPP
#define LOCAL_POROSITY_HPP

#include "Phalanx_config.hpp"
#include "Phalanx_Evaluator_WithBaseImpl.hpp"
#include "Phalanx_Evaluator_Derived.hpp"
#include "Phalanx_MDField.hpp"
#include "Teuchos_ParameterList.hpp"
#include "Epetra_Vector.h"
#include "Sacado_ParameterAccessor.hpp"
#include "Teuchos_Array.hpp"
#include "Albany_Layouts.hpp"

namespace AMP {
///
/// \brief Local Porosity
///
/// This evaluator computes the local porosity
/// for a phase-change/heat equation problem
///
template<typename EvalT, typename Traits>
class Local_Porosity : 
  public PHX::EvaluatorWithBaseImpl<Traits>,
  public PHX::EvaluatorDerived<EvalT, Traits>
{

public:

  Local_Porosity(Teuchos::ParameterList& p,
      const Teuchos::RCP<Albany::Layouts>& dl);

  void 
  postRegistrationSetup(typename Traits::SetupData d,
      PHX::FieldManager<Traits>& vm);

  void 
  evaluateFields(typename Traits::EvalData d);

private:

  typedef typename EvalT::ScalarT ScalarT;
  typedef typename EvalT::MeshScalarT MeshScalarT;

  // we need to store the initial value of porosity which is 0 in the substrate and some value in the powder
  ScalarT Initial_porosity;


  PHX::MDField<MeshScalarT,Cell,QuadPoint,Dim> coord_;
  PHX::MDField<ScalarT,Cell,QuadPoint> porosity_;
  PHX::MDField<ScalarT,Cell,QuadPoint> psi_;

  unsigned int num_qps_;
  unsigned int num_dims_;
  unsigned int num_nodes_;
  unsigned int workset_size_;

  Teuchos::RCP<const Teuchos::ParameterList>
    getValidLocal_PorosityParameters() const;

};
}

#endif
