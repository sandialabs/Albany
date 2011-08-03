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


#ifndef LAMESTRESS_HPP
#define LAMESTRESS_HPP

#include "Phalanx_ConfigDefs.hpp"
#include "Phalanx_Evaluator_WithBaseImpl.hpp"
#include "Phalanx_Evaluator_Derived.hpp"
#include "Phalanx_MDField.hpp"
#include "LameUtils.hpp"

namespace LCM {
/** \brief Evaluates stress using the Library for Advanced Materials for Engineering (LAME).
*/

template<typename EvalT, typename Traits>
class LameStress : public PHX::EvaluatorWithBaseImpl<Traits>,
		    public PHX::EvaluatorDerived<EvalT, Traits>  {

public:

  LameStress(const Teuchos::ParameterList& p);

  void postRegistrationSetup(typename Traits::SetupData d,
                      PHX::FieldManager<Traits>& vm);

  void evaluateFields(typename Traits::EvalData d);

private:

  typedef typename EvalT::ScalarT ScalarT;
  typedef typename EvalT::MeshScalarT MeshScalarT;

  // Input:
  PHX::MDField<ScalarT,Cell,QuadPoint,Dim,Dim> defGradField;

  std::string defGradName, stressName;
  unsigned int numQPs;
  unsigned int numDims;

  // Output:
  PHX::MDField<ScalarT,Cell,QuadPoint,Dim,Dim> stressField;

  // The LAME material model
  Teuchos::RCP<lame::Material> lameMaterialModel;

  // The LAME material model name
  std::string lameMaterialModelName;

  // Vector of the state variable names for the LAME material model
  std::vector<std::string> lameMaterialModelStateVariableNames;

  // Vector of the fields corresponding to the LAME material model state variables
  std::vector< PHX::MDField<ScalarT,Cell,QuadPoint,Dim,Dim> > lameMaterialModelStateVariableFields;
};
}

#endif
