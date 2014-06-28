//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#ifndef LCM_GATHER_SPHEREVOLUME_HPP
#define LCM_GATHER_SPHEREVOLUME_HPP

#include "Phalanx_ConfigDefs.hpp"
#include "Phalanx_Evaluator_WithBaseImpl.hpp"
#include "Phalanx_Evaluator_Derived.hpp"
#include "Phalanx_MDField.hpp"

#include "Albany_Layouts.hpp"

#include "Teuchos_ParameterList.hpp"
#include "Epetra_Vector.h"

namespace LCM {
/** \brief Gathers sphere volume into the nodal fields of the field manager.
*/

template<typename EvalT, typename Traits> 
class GatherSphereVolume : public PHX::EvaluatorWithBaseImpl<Traits>,
			   public PHX::EvaluatorDerived<EvalT, Traits>  {
  
public:
  
  GatherSphereVolume(const Teuchos::ParameterList& p,
		     const Teuchos::RCP<Albany::Layouts>& dl);
  
  GatherSphereVolume(const Teuchos::ParameterList& p);

  void postRegistrationSetup(typename Traits::SetupData d,
			     PHX::FieldManager<Traits>& vm);
  
  void evaluateFields(typename Traits::EvalData d);
  
private:

  typedef typename EvalT::ScalarT ScalarT;
  typedef typename EvalT::MeshScalarT MeshScalarT;

  PHX::MDField<RealType,Cell,Vertex> sphereVolume;
 
  std::size_t worksetSize;
  std::size_t numVertices;
};
}

#endif
