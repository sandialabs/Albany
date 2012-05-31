/********************************************************************\
*            Albany, Copyright (2010) Sandia Corporation             *
*                                                                    *
* Notice: This computer software was prepared by Sandia Corporation, *
* hereinafter the Contractor, under Contract DE-AC04-94AL85000 with  *
* the Department of Energy (DOE). All rights in the computer software*
* are reserved by DOE on behalf of the United States Government and  *
* the Contractor as provided in the Contract. You are authorized to  *
* use this computer software for Governmental purposes but it is not *
* to be released or distributed to the public. NEITHER THE GOVERNMENT*
* NOR THE CONTRACTOR MAKES ANY WARRANTY, EXPRESS OR IMPLIED, OR      *
* ASSUMES ANY LIABILITY FOR THE USE OF THIS SOFTWARE. This notice    *
* including this sentence must appear on any copies of this software.*
*    Questions to Andy Salinger, agsalin@sandia.gov                  *
\********************************************************************/


#ifndef LAMENTSTRESS_HPP
#define LAMENTSTRESS_HPP

#include "Phalanx_ConfigDefs.hpp"
#include "Phalanx_Evaluator_WithBaseImpl.hpp"
#include "Phalanx_Evaluator_Derived.hpp"
#include "Phalanx_MDField.hpp"
#include "PHAL_Dimension.hpp"
#include "LameUtils.hpp"

namespace LCM {
/** \brief Evaluates stress using the Library for Advanced Materials for Engineering with Never-ending Templates (LAMENT).
*/

template<typename EvalT, typename Traits>
class LamentStress : public PHX::EvaluatorWithBaseImpl<Traits>,
		     public PHX::EvaluatorDerived<EvalT, Traits>  {

public:

  LamentStress(Teuchos::ParameterList& p);

  void postRegistrationSetup(typename Traits::SetupData d,
			     PHX::FieldManager<Traits>& vm);

  virtual void evaluateFields(typename Traits::EvalData d);

protected:

  typedef typename EvalT::ScalarT ScalarT;
  typedef typename EvalT::MeshScalarT MeshScalarT;

  // Input:
  PHX::MDField<ScalarT,Cell,QuadPoint,Dim,Dim> defGradField;

  std::string defGradName, stressName;
  unsigned int numQPs;
  unsigned int numDims;
  Teuchos::RCP<PHX::DataLayout> tensor_dl;

  // Output:
  PHX::MDField<ScalarT,Cell,QuadPoint,Dim,Dim> stressField;

  // The LAMENT material model
  Teuchos::RCP<lament::Material<ScalarT> > lamentMaterialModel;

  // The LAMENT material model name
  std::string lamentMaterialModelName;

  // Vector of the state variable names for the LAMENT material model
  std::vector<std::string> lamentMaterialModelStateVariableNames;

  // Vector of the fields corresponding to the LAMENT material model state variables
  std::vector< PHX::MDField<ScalarT,Cell,QuadPoint,Dim,Dim> > lamentMaterialModelStateVariableFields;
};

}

#endif
