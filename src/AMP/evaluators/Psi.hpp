//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#ifndef PSI_HPP
#define PSI_HPP

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
/// \brief  Psi
///
/// This evaluator computes the Psi State Variables to a 
/// phase-change/heat equation problem
///
template<typename EvalT, typename Traits>
class Psi : 
  public PHX::EvaluatorWithBaseImpl<Traits>,
  public PHX::EvaluatorDerived<EvalT, Traits>
{

public:

  Psi(Teuchos::ParameterList& p,
      const Teuchos::RCP<Albany::Layouts>& dl);

  void 
  postRegistrationSetup(typename Traits::SetupData d,
      PHX::FieldManager<Traits>& vm);

  void 
  evaluateFields(typename Traits::EvalData d);

private:

  typedef typename EvalT::ScalarT ScalarT;
  typedef typename EvalT::MeshScalarT MeshScalarT;

  // This store the value of Psi per block
  ScalarT constant_value_;

  PHX::MDField<ScalarT,Cell,QuadPoint> T_;
  PHX::MDField<ScalarT,Cell,QuadPoint> phi_;

  //State Variables:    
  PHX::MDField<ScalarT,Cell,QuadPoint> psi_;


  unsigned int num_qps_;
  unsigned int num_dims_;
  unsigned int num_nodes_;
  unsigned int workset_size_;

  bool enable_transient_;
  std::string psi_Name_;
  std::string phi_Name_;
  
  Teuchos::RCP<const Teuchos::ParameterList>
    getValidPsiParameters() const; 

};
}

#endif
