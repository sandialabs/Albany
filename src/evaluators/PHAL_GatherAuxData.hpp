//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#ifndef PHAL_GATHER_AUXDATA_HPP
#define PHAL_GATHER_AUXDATA_HPP

#include "Phalanx_config.hpp"
#include "Phalanx_Evaluator_WithBaseImpl.hpp"
#include "Phalanx_Evaluator_Derived.hpp"
#include "Phalanx_MDField.hpp"

#include "Albany_Layouts.hpp"

#include "Teuchos_ParameterList.hpp"
#ifdef ALBANY_EPETRA
#include "Epetra_Vector.h"
#endif

namespace PHAL {
/** \brief Gathers auxilliary values from a workset \e auxData vector into 
    the nodal fields of the field manager

    Currently makes an assumption that the stride is constant for dofs
    and that the nmber of dofs is equal to the size of the solution
    names vector.

*/
// **************************************************************
// Base Class with Generic Implementations: Specializations for
// Automatic Differentiation Below 
// **************************************************************

template<typename EvalT, typename Traits>
class GatherAuxData
  : public PHX::EvaluatorWithBaseImpl<Traits>,
    public PHX::EvaluatorDerived<EvalT, Traits>  {
  
public:
  
  GatherAuxData(const Teuchos::ParameterList& p,
		const Teuchos::RCP<Albany::Layouts>& dl);
  
  void postRegistrationSetup(typename Traits::SetupData d,
                      PHX::FieldManager<Traits>& vm);
  
  void evaluateFields(typename Traits::EvalData d);  
protected:

  typedef typename EvalT::ScalarT ScalarT;
  PHX::MDField<ScalarT,Cell,Node> vector_data;
  std::size_t auxDataIndex;
  std::size_t numNodes;
};

// **************************************************************
}

#endif
