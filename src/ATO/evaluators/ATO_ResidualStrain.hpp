//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#ifndef RESIDUAL_STRESS_HPP
#define RESIDUAL_STRESS_HPP

#include "Phalanx_config.hpp"
#include "Phalanx_Evaluator_WithBaseImpl.hpp"
#include "Phalanx_Evaluator_Derived.hpp"
#include "Phalanx_MDField.hpp"

#include "Teuchos_ParameterList.hpp"
#include "Stokhos_KL_ExponentialRandomField.hpp"
#include "Teuchos_Array.hpp"

namespace ATO {
/** 
 * \brief Evaluates strain tensor KL expansion.
 */

template<typename EvalT, typename Traits>
class ResidualStrain : 
  public PHX::EvaluatorWithBaseImpl<Traits>,
  public PHX::EvaluatorDerived<EvalT, Traits> {
  
public:
  typedef typename EvalT::ScalarT ScalarT;
  typedef typename EvalT::MeshScalarT MeshScalarT;

  ResidualStrain(Teuchos::ParameterList& p);
  
  void postRegistrationSetup(typename Traits::SetupData d,
			     PHX::FieldManager<Traits>& vm);
  
  void evaluateFields(typename Traits::EvalData d);
  
private:

  int numQPs;
  int numDims;
  PHX::MDField<const MeshScalarT,Cell,QuadPoint,Dim> coordVec;
  PHX::MDField<ScalarT,Cell,QuadPoint,Dim,Dim> strain;

  //! Exponential random field
  Teuchos::RCP< Stokhos::KL::ExponentialRandomField<RealType> > shear_xy_kl;
  Teuchos::RCP< Stokhos::KL::ExponentialRandomField<RealType> > shear_xz_kl;
  Teuchos::RCP< Stokhos::KL::ExponentialRandomField<RealType> > shear_yz_kl;
  Teuchos::RCP< Stokhos::KL::ExponentialRandomField<RealType> > vol_strn_kl;

  //! Values of the random variables
  Teuchos::Array<ScalarT> shear_xy_rv;
  Teuchos::Array<ScalarT> shear_xz_rv;
  Teuchos::Array<ScalarT> shear_yz_rv;
  Teuchos::Array<ScalarT> vol_strn_rv;
};
}

#endif
