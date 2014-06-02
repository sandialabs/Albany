//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#ifndef AERAS_ATMOSPHERE_HPP
#define AERAS_ATMOSPHERE_HPP

#include "Phalanx_ConfigDefs.hpp"
#include "Phalanx_Evaluator_WithBaseImpl.hpp"
#include "Phalanx_Evaluator_Derived.hpp"
#include "Phalanx_MDField.hpp"

#include "Albany_Layouts.hpp"

#include "Teuchos_ParameterList.hpp"
#include "Epetra_Vector.h"

namespace Aeras {

template<typename EvalT, typename Traits> 
class Atmosphere : public PHX::EvaluatorWithBaseImpl<Traits>,
                   public PHX::EvaluatorDerived<EvalT, Traits>  {
  
public:
  
  Atmosphere(Teuchos::ParameterList& p,
             const Teuchos::RCP<Albany::Layouts>& dl);
  // Old constructor, still needed by BCs that use PHX Factory
  Atmosphere(const Teuchos::ParameterList& p);
  
  void postRegistrationSetup(typename Traits::SetupData d,
                      PHX::FieldManager<Traits>& vm);
  
  void evaluateFields(typename Traits::EvalData d);
  
private:

  typedef typename EvalT::ScalarT ScalarT;
  typedef typename EvalT::MeshScalarT MeshScalarT;

  PHX::MDField<MeshScalarT,Cell,Vertex,Dim> coordVec;
  PHX::MDField<MeshScalarT,Cell,Node,Dim,Dim> tracersOld;
  PHX::MDField<MeshScalarT,Cell,Node,Dim,Dim> tracersNew;
  PHX::MDField<ScalarT,Cell,QuadPoint,VecDim> U;  //vecDim works but its really Dim+1

  PHX::MDField<ScalarT,Cell,Node,VecDim> ResidualIn;
  PHX::MDField<ScalarT,Cell,Node,VecDim> ResidualOut;

 
  bool  periodic;
  std::size_t worksetSize;
  std::size_t numNodes;
  std::size_t numCoords;
  std::size_t numTracers;
  std::size_t numLevels;
};
}

#endif
