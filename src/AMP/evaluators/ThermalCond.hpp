//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#ifndef THERMALCOND_HPP
#define THERMALCOND_HPP

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
/// \brief Thermal Conductivity
///
/// This evaluator computes the thermal conductivity
/// for a phase-change/heat equation problem
///
template<typename EvalT, typename Traits>
class ThermalCond : 
  public PHX::EvaluatorWithBaseImpl<Traits>,
  public PHX::EvaluatorDerived<EvalT, Traits>
{

public:

  ThermalCond(Teuchos::ParameterList& p,
      const Teuchos::RCP<Albany::Layouts>& dl);

  void 
  postRegistrationSetup(typename Traits::SetupData d,
      PHX::FieldManager<Traits>& vm);

  void 
  evaluateFields(typename Traits::EvalData d);

private:

  typedef typename EvalT::ScalarT ScalarT;
  typedef typename EvalT::MeshScalarT MeshScalarT;

  // we need to store the value of powder and substrate
  // powder thermal conductivity
  ScalarT powder_value_;
  // Substrate (solid) thermal conductivity
  ScalarT solid_value_;

  PHX::MDField<MeshScalarT,Cell,QuadPoint,Dim> coord_;
  PHX::MDField<ScalarT,Cell,QuadPoint> k_;
  PHX::MDField<ScalarT,Cell,QuadPoint> psi_;

  unsigned int num_qps_;
  unsigned int num_dims_;
  unsigned int num_nodes_;
  unsigned int workset_size_;

//  // Return initial powder thermal conductivity
//  ScalarT getPowderThermalCondutivity() const;
//  
//  // Return initial substrate thermal conductivity
//  ScalarT getSubstrateThermalCondutivity() const;
  
  
  Teuchos::RCP<const Teuchos::ParameterList>
    getValidThermalCondParameters() const;

};
}

#endif
