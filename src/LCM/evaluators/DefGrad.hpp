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


#ifndef DEFGRAD_HPP
#define DEFGRAD_HPP

#include "Phalanx_ConfigDefs.hpp"
#include "Phalanx_Evaluator_WithBaseImpl.hpp"
#include "Phalanx_Evaluator_Derived.hpp"
#include "Phalanx_MDField.hpp"

#include "Intrepid_CellTools.hpp"
#include "Intrepid_Cubature.hpp"

namespace LCM {
/** \brief Deformation Gradient

    This evaluator computes the deformation gradient

*/

template<typename EvalT, typename Traits>
class DefGrad : public PHX::EvaluatorWithBaseImpl<Traits>,
		public PHX::EvaluatorDerived<EvalT, Traits>  {

public:

  DefGrad(const Teuchos::ParameterList& p);

  void postRegistrationSetup(typename Traits::SetupData d,
			     PHX::FieldManager<Traits>& vm);

  void evaluateFields(typename Traits::EvalData d);

private:

  typedef typename EvalT::ScalarT ScalarT;
  typedef typename EvalT::MeshScalarT MeshScalarT;

  // Input:
  PHX::MDField<ScalarT,Cell,QuadPoint,Dim,Dim> GradU;
  PHX::MDField<MeshScalarT,Cell,QuadPoint> weights;
  
  // Output:
  PHX::MDField<ScalarT,Cell,QuadPoint,Dim,Dim> defgrad;
  PHX::MDField<ScalarT,Cell,QuadPoint> J;

  unsigned int numQPs;
  unsigned int numDims;
  unsigned int worksetSize;

  //! flag to compute the weighted average of J
  bool weightedAverage;

  //! stabilization parameter for the weighted average
  ScalarT alpha;
  

};

}
#endif
