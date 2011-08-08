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


#ifndef PHAL_NSNEUTRONEQRESID_HPP
#define PHAL_NSNEUTRONEQRESID_HPP

#include "Phalanx_ConfigDefs.hpp"
#include "Phalanx_Evaluator_WithBaseImpl.hpp"
#include "Phalanx_Evaluator_Derived.hpp"
#include "Phalanx_MDField.hpp"

/** \brief Finite Element Interpolation Evaluator

    This evaluator interpolates nodal DOF values to quad points.

*/
namespace PHAL {

template<typename EvalT, typename Traits>
class NSNeutronEqResid : public PHX::EvaluatorWithBaseImpl<Traits>,
		    public PHX::EvaluatorDerived<EvalT, Traits>  {

public:

  NSNeutronEqResid(const Teuchos::ParameterList& p);

  void postRegistrationSetup(typename Traits::SetupData d,
                      PHX::FieldManager<Traits>& vm);

  void evaluateFields(typename Traits::EvalData d);

private:

  typedef typename EvalT::ScalarT ScalarT;
  typedef typename EvalT::MeshScalarT MeshScalarT;

  // Input:
  PHX::MDField<MeshScalarT,Cell,Node,QuadPoint> wBF;
  PHX::MDField<ScalarT,Cell,QuadPoint> Neutron;
  PHX::MDField<ScalarT,Cell,QuadPoint> T;
  PHX::MDField<ScalarT,Cell,QuadPoint> NeutronDiff;
  PHX::MDField<MeshScalarT,Cell,Node,QuadPoint,Dim> wGradBF;
  PHX::MDField<ScalarT,Cell,QuadPoint,Dim> NGrad;
  PHX::MDField<ScalarT,Cell,QuadPoint> Source;
  PHX::MDField<ScalarT,Cell,QuadPoint> Absorp;
  PHX::MDField<ScalarT,Cell,QuadPoint> Fission;  
  PHX::MDField<ScalarT,Cell,QuadPoint> Tref;  

  // Output:
  PHX::MDField<ScalarT,Cell,Node> NResidual;

  bool haveNeutSource, haveHeat, haveFlow;
  unsigned int numQPs, numDims, numNodes;
  Intrepid::FieldContainer<ScalarT> flux;
  Intrepid::FieldContainer<ScalarT> abscoeff;

 };
}

#endif
