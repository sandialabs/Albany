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


#ifndef J2DAMAGE_HPP
#define J2SDAMAGE_HPP

#include "Phalanx_ConfigDefs.hpp"
#include "Phalanx_Evaluator_WithBaseImpl.hpp"
#include "Phalanx_Evaluator_Derived.hpp"
#include "Phalanx_MDField.hpp"
#include "Tensor.h"

/** \brief J2Stress with damage dependent response

    This evaluator computes stress based on a uncoupled J2Stress
    potential

*/
namespace LCM {

template<typename EvalT, typename Traits>
class J2Damage : public PHX::EvaluatorWithBaseImpl<Traits>,
		 public PHX::EvaluatorDerived<EvalT, Traits>  {

public:

  J2Damage(const Teuchos::ParameterList& p);

  void postRegistrationSetup(typename Traits::SetupData d,
                      PHX::FieldManager<Traits>& vm);

  void evaluateFields(typename Traits::EvalData d);

private:

  typedef typename EvalT::ScalarT ScalarT;
  typedef typename EvalT::MeshScalarT MeshScalarT;

  // Input:
  PHX::MDField<ScalarT,Cell,QuadPoint,Dim,Dim> defgrad;
  PHX::MDField<ScalarT,Cell,QuadPoint> J;
  PHX::MDField<ScalarT,Cell,QuadPoint> bulkModulus;
  PHX::MDField<ScalarT,Cell,QuadPoint> shearModulus;
  PHX::MDField<ScalarT,Cell,QuadPoint> yieldStrength;
  PHX::MDField<ScalarT,Cell,QuadPoint> hardeningModulus;
  PHX::MDField<ScalarT,Cell,QuadPoint> satMod;
  PHX::MDField<ScalarT,Cell,QuadPoint> satExp;
  PHX::MDField<ScalarT,Cell,QuadPoint> damage;

  // Output:
  PHX::MDField<ScalarT,Cell,QuadPoint,Dim,Dim> stress;
  PHX::MDField<ScalarT,Cell,QuadPoint> dp;
  PHX::MDField<ScalarT,Cell,QuadPoint> seff;
  PHX::MDField<ScalarT,Cell,QuadPoint> energy;
  PHX::MDField<ScalarT,Cell,QuadPoint,Dim,Dim> Fp;
  PHX::MDField<ScalarT,Cell,QuadPoint> eqps;

  std::string fpName, eqpsName;
  unsigned int numQPs;
  unsigned int numDims;

  // scratch space FCs
  Tensor<ScalarT> be;
  Tensor<ScalarT> s;
  Tensor<ScalarT> N;
  Tensor<ScalarT> A;
  Tensor<ScalarT> expA;

  Intrepid::FieldContainer<ScalarT> Fpinv;
  Intrepid::FieldContainer<ScalarT> FpinvT;
  Intrepid::FieldContainer<ScalarT> Cpinv;

};
}

#endif
