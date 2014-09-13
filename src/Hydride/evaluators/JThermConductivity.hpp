//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#ifndef J_THERMAL_CONDUCTIVITY_HPP
#define J_THERMAL_CONDUCTIVITY_HPP

#include "Phalanx_ConfigDefs.hpp"
#include "Phalanx_Evaluator_WithBaseImpl.hpp"
#include "Phalanx_Evaluator_Derived.hpp"
#include "Phalanx_MDField.hpp"

#include "Teuchos_ParameterList.hpp"
#include "Teuchos_Array.hpp"

#include "Albany_Layouts.hpp"

#include "QCAD_MaterialDatabase.hpp"

namespace PHAL {
/** 
 * \brief Evaluates thermal conductivity, either as a constant or a truncated
 * KL expansion.

 */

template<typename EvalT, typename Traits>
class JThermConductivity : 
  public PHX::EvaluatorWithBaseImpl<Traits>,
  public PHX::EvaluatorDerived<EvalT, Traits> {
  
public:
  typedef typename EvalT::ScalarT ScalarT;

  JThermConductivity(Teuchos::ParameterList& p,
               const Teuchos::RCP<Albany::Layouts>& dl);
  
  void postRegistrationSetup(typename Traits::SetupData d,
			     PHX::FieldManager<Traits>& vm);
  
  void evaluateFields(typename Traits::EvalData d);
  
private:       

//! Validate the name strings under "JTherm Conductivity" section in xml input file, 
  Teuchos::RCP<const Teuchos::ParameterList>
               getValidJThermCondParameters() const;

  std::size_t numQPs;
  std::size_t numDims;
  PHX::MDField<ScalarT,Cell,QuadPoint> thermalCond;
  PHX::MDField<ScalarT,Cell,QuadPoint> Temperature;

  //! Conductivity type
  std::string type; 

  //! Constant value
  ScalarT Qh;
  ScalarT R;
  ScalarT Cht;
  ScalarT Vbar;

  //! Material database - holds thermal conductivity among other quantities
  Teuchos::RCP<QCAD::MaterialDatabase> materialDB;

  //! storing the DataLayouts
  const Teuchos::RCP<Albany::Layouts>& dl_;

};
}

#endif
