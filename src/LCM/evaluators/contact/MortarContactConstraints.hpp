//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#ifndef MORTAR_CONTACT_CONSTRAINTS_HPP
#define MORTAR_CONTACT_CONSTRAINTS_HPP

#include "Phalanx_config.hpp"
#include "Phalanx_Evaluator_WithBaseImpl.hpp"
#include "Phalanx_Evaluator_Derived.hpp"
#include "Phalanx_MDField.hpp"

#include "Albany_Layouts.hpp"

#include "Teuchos_ParameterList.hpp"

namespace LCM {

// **************************************************************
// Base Class for code that is independent of evaluation type
// **************************************************************

template<typename EvalT, typename Traits>
class MortarContactBase
  : public PHX::EvaluatorWithBaseImpl<Traits>,
    public PHX::EvaluatorDerived<EvalT, Traits>  {

public:

  MortarContactBase(Teuchos::ParameterList& p,
                    const Teuchos::RCP<Albany::Layouts>& dl);

  void postRegistrationSetup(typename Traits::SetupData d,
                             PHX::FieldManager<Traits>& vm);

// These functions are defined in the specializations
  void evaluateFields(typename Traits::EvalData d) = 0;

protected:

  typedef typename EvalT::ScalarT ScalarT;
  typedef typename EvalT::MeshScalarT MeshScalarT;

  PHX::MDField<ScalarT,Cell,QuadPoint> M_operator; // This evaluator creates M and D, not sure what they look like yet
                                                   // so put in a placeholder

  Teuchos::Array<int> offset;

//! Coordinate vector at vertices
  PHX::MDField<MeshScalarT,Cell,Vertex,Dim> coordVec;

};


template<typename EvalT, typename Traits>
class MortarContact
  : public MortarContactBase<EvalT, Traits>  {

public:

  MortarContact(Teuchos::ParameterList& p,
                const Teuchos::RCP<Albany::Layouts>& dl) :
                      MortarContactBase<EvalT, Traits>(p, dl){}

  void evaluateFields(typename Traits::EvalData d){
	std::cerr << "Calling non-residual MortarContact evaluateFields() called: " << __FILE__ << " line " << __LINE__ << std::endl;
  }

};

// **************************************************************
// **************************************************************
// * Specializations
// **************************************************************
// **************************************************************

// **************************************************************
// Residual
// **************************************************************

template<typename Traits>
class MortarContact<PHAL::AlbanyTraits::Residual, Traits>
  : public MortarContactBase<PHAL::AlbanyTraits::Residual, Traits>  {

public:

  MortarContact(Teuchos::ParameterList& p,
                const Teuchos::RCP<Albany::Layouts>& dl);

  void evaluateFields(typename Traits::EvalData d);

};

}

#endif
