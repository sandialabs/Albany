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


#ifndef J2FIBER_HPP
#define J2FIBER_HPP

#include "Phalanx_ConfigDefs.hpp"
#include "Phalanx_Evaluator_WithBaseImpl.hpp"
#include "Phalanx_Evaluator_Derived.hpp"
#include "Phalanx_MDField.hpp"

#include "Tensor.h"

namespace LCM {
/** \brief J2Fiber stress response

    This evaluator computes stress based on a uncoupled J2Fiber
    potential

*/

template<typename EvalT, typename Traits>
class J2Fiber : public PHX::EvaluatorWithBaseImpl<Traits>,
		 public PHX::EvaluatorDerived<EvalT, Traits>  {

public:

  J2Fiber(const Teuchos::ParameterList& p);

  void postRegistrationSetup(typename Traits::SetupData d,
                      PHX::FieldManager<Traits>& vm);

  void evaluateFields(typename Traits::EvalData d);

private:

  typedef typename EvalT::ScalarT ScalarT;
  typedef typename EvalT::MeshScalarT MeshScalarT;

  // Input:
  PHX::MDField<ScalarT,Cell,QuadPoint,Dim,Dim> defgrad;
  PHX::MDField<ScalarT,Cell,QuadPoint> J;
  PHX::MDField<ScalarT,Cell,QuadPoint,Dim> gptLocation;
  PHX::MDField<ScalarT,Cell,QuadPoint> elasticModulus;
  PHX::MDField<ScalarT,Cell,QuadPoint> poissonsRatio;
  PHX::MDField<ScalarT,Cell,QuadPoint> yieldStrength;
  PHX::MDField<ScalarT,Cell,QuadPoint> hardeningModulus;
  PHX::MDField<ScalarT,Cell,QuadPoint> satMod;
  PHX::MDField<ScalarT,Cell,QuadPoint> satExp;

  // Output:
  PHX::MDField<ScalarT,Cell,QuadPoint,Dim,Dim> stress;
  PHX::MDField<ScalarT,Cell,QuadPoint,Dim,Dim> Fp;
  PHX::MDField<ScalarT,Cell,QuadPoint> eqps;
  PHX::MDField<ScalarT,Cell,QuadPoint> energy_J2;
  PHX::MDField<ScalarT,Cell,QuadPoint> energy_f1;
  PHX::MDField<ScalarT,Cell,QuadPoint> energy_f2;
  PHX::MDField<ScalarT,Cell,QuadPoint> damage_J2;
  PHX::MDField<ScalarT,Cell,QuadPoint> damage_f1;
  PHX::MDField<ScalarT,Cell,QuadPoint> damage_f2;

  std::string fpName, eqpsName;
  std::string energy_J2Name, energy_f1Name, energy_f2Name;
  unsigned int numQPs;
  unsigned int numDims;
  unsigned int worksetSize;

  LCM::Tensor<ScalarT> F;
  LCM::Tensor<ScalarT> Fpn;
  LCM::Tensor<ScalarT> Cpinv;
  LCM::Tensor<ScalarT> be;
  LCM::Tensor<ScalarT> s;
  LCM::Tensor<ScalarT> N;
  LCM::Tensor<ScalarT> A;
  LCM::Tensor<ScalarT> expA;

  RealType xiinf_J2;
  RealType tau_J2;

  RealType k_f1;
  RealType q_f1;
  RealType vol_f1;
  RealType xiinf_f1;
  RealType tau_f1;

  RealType k_f2;
  RealType q_f2;
  RealType vol_f2;
  RealType xiinf_f2;
  RealType tau_f2;

  bool isLocalCoord;

  std::vector< RealType > direction_f1;
  std::vector< RealType > direction_f2;
  std::vector< RealType > ringCenter;

};
}

#endif
