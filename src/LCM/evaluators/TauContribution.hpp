//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#ifndef TAU_CONTRIBUTION_HPP
#define TAU_CONTRIBUTION_HPP

#include "Phalanx_ConfigDefs.hpp"
#include "Phalanx_Evaluator_WithBaseImpl.hpp"
#include "Phalanx_Evaluator_Derived.hpp"
#include "Phalanx_MDField.hpp"

#include "Teuchos_ParameterList.hpp"
#include "Epetra_Vector.h"
#include "Sacado_ParameterAccessor.hpp"
#include "Stokhos_KL_ExponentialRandomField.hpp"
#include "Teuchos_Array.hpp"

namespace LCM {
/** 
 * \brief Evaluates scalar function for hydro-static
 *  stress contribution on hydrogen transport
 * KL expansion.
 */

template<typename EvalT, typename Traits>
class TauContribution :
  public PHX::EvaluatorWithBaseImpl<Traits>,
  public PHX::EvaluatorDerived<EvalT, Traits>,
  public Sacado::ParameterAccessor<EvalT, SPL_Traits> {
  
public:
  typedef typename EvalT::ScalarT ScalarT;
  typedef typename EvalT::MeshScalarT MeshScalarT;

  TauContribution(Teuchos::ParameterList& p);
  
  void postRegistrationSetup(typename Traits::SetupData d,
			     PHX::FieldManager<Traits>& vm);
  
  void evaluateFields(typename Traits::EvalData d);
  
  ScalarT& getValue(const std::string &n);

private:

  std::size_t numQPs;
  std::size_t numDims;

  RealType Rideal;

  PHX::MDField<MeshScalarT,Cell,QuadPoint,Dim> coordVec;

  PHX::MDField<ScalarT,Cell,QuadPoint> Clattice;
  //PHX::MDField<ScalarT,Cell,QuadPoint> Rideal;
  PHX::MDField<ScalarT,Cell,QuadPoint> temperature;
  PHX::MDField<ScalarT,Cell,QuadPoint> DL;

  //! Is conductivity constant, or random field
  bool is_constant;

  //! Constant value
  ScalarT constant_value;

  //! Partial Molar Volume
  ScalarT VmPartial;

  //! OutPut
  PHX::MDField<ScalarT,Cell,QuadPoint> tauFactor;


  //! Exponential random field
  Teuchos::RCP< Stokhos::KL::ExponentialRandomField<MeshScalarT> > exp_rf_kl;

  //! Values of the random variables
  Teuchos::Array<ScalarT> rv;
};
}

#endif
