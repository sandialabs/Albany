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


#ifndef J2STRESS_HPP
#define J2STRESS_HPP

#include "Phalanx_ConfigDefs.hpp"
#include "Phalanx_Evaluator_WithBaseImpl.hpp"
#include "Phalanx_Evaluator_Derived.hpp"
#include "Phalanx_MDField.hpp"

/** \brief J2Stress stress response

    This evaluator computes stress based on a uncoupled J2Stress
    potential

*/
namespace LCM {

template<typename EvalT, typename Traits>
class J2Stress : public PHX::EvaluatorWithBaseImpl<Traits>,
		 public PHX::EvaluatorDerived<EvalT, Traits>  {

public:

  J2Stress(const Teuchos::ParameterList& p);

  void postRegistrationSetup(typename Traits::SetupData d,
                      PHX::FieldManager<Traits>& vm);

  void evaluateFields(typename Traits::EvalData d);

private:

  typedef typename EvalT::ScalarT ScalarT;
  typedef typename EvalT::MeshScalarT MeshScalarT;

  typename EvalT::ScalarT norm(Intrepid::FieldContainer<ScalarT>);
  void exponential_map(Intrepid::FieldContainer<ScalarT>, Intrepid::FieldContainer<ScalarT>);

  // Input:
  PHX::MDField<ScalarT,Cell,QuadPoint,Dim,Dim> defgrad;
  PHX::MDField<ScalarT,Cell,QuadPoint> J;
  PHX::MDField<ScalarT,Cell,QuadPoint> elasticModulus;
  PHX::MDField<ScalarT,Cell,QuadPoint> poissonsRatio;
  PHX::MDField<ScalarT,Cell,QuadPoint> yieldStrength;
  PHX::MDField<ScalarT,Cell,QuadPoint> hardeningModulus;

  // Output:
  PHX::MDField<ScalarT,Cell,QuadPoint,Dim,Dim> stress;

  //int numQPs;
  //int numDims;
  std::size_t numQPs;
  std::size_t numDims;

  // scratch space FCs
  Intrepid::FieldContainer<ScalarT> be;
  Intrepid::FieldContainer<ScalarT> s;
  Intrepid::FieldContainer<ScalarT> N;
  Intrepid::FieldContainer<ScalarT> A;
  Intrepid::FieldContainer<ScalarT> expA;
  Intrepid::FieldContainer<ScalarT> Fp;
  Intrepid::FieldContainer<ScalarT> Fpinv;
  Intrepid::FieldContainer<ScalarT> FpinvT;
  Intrepid::FieldContainer<ScalarT> Cpinv;
  Intrepid::FieldContainer<ScalarT> eqps;

  Intrepid::FieldContainer<ScalarT> tmp;
  Intrepid::FieldContainer<ScalarT> tmp2;


};
}

#ifndef PHAL_ETI
#include "J2Stress_Def.hpp"
#endif

#endif
