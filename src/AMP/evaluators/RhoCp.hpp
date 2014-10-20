//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#ifndef RHOCP_HPP
#define RHOCP_HPP

#include "Phalanx_ConfigDefs.hpp"
#include "Phalanx_Evaluator_WithBaseImpl.hpp"
#include "Phalanx_Evaluator_Derived.hpp"
#include "Phalanx_MDField.hpp"

//------------------------------------------------------------------------------------------
#include "Teuchos_ParameterList.hpp"
#include "Epetra_Vector.h"
#include "Sacado_ParameterAccessor.hpp"
#include "Stokhos_KL_ExponentialRandomField.hpp"
#include "Teuchos_Array.hpp"
//------------------------------------------------------------------------------------------

namespace AMP {
///
/// \brief Rho Cp 
///
/// This evaluator computes the specific heat times 
/// density function for a phase-change/heat 
/// equation problem
///
template<typename EvalT, typename Traits>
class RhoCp : 
  public PHX::EvaluatorWithBaseImpl<Traits>,
  public PHX::EvaluatorDerived<EvalT, Traits>
{

public:

  RhoCp(Teuchos::ParameterList& p,
      const Teuchos::RCP<Albany::Layouts>& dl);

  void 
  postRegistrationSetup(typename Traits::SetupData d,
      PHX::FieldManager<Traits>& vm);

  void 
  evaluateFields(typename Traits::EvalData d);

private:

  typedef typename EvalT::ScalarT ScalarT;
  typedef typename EvalT::MeshScalarT MeshScalarT;

  PHX::MDField<MeshScalarT,Cell,Node,QuadPoint,Dim> coord_;
  PHX::MDField<ScalarT,Cell,Node> rho_cp_;

  unsigned int num_qps_;
  unsigned int num_dims_;
  unsigned int num_nodes_;
  unsigned int workset_size_;

  //! Constant value
   ScalarT constant_value;

    Teuchos::RCP<const Teuchos::ParameterList>
       getValidRhoCpParameters() const;

  //! Convenience function to initialize constant Rho Cp
  void init_constant(ScalarT value, Teuchos::ParameterList& p);


};
}

#endif
