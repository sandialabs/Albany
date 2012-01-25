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


#ifndef THERMO_MECHANICAL_STRESS_HPP
#define THERMO_MECHANICAL_STRESS_HPP

#include "Phalanx_ConfigDefs.hpp"
#include "Phalanx_Evaluator_WithBaseImpl.hpp"
#include "Phalanx_Evaluator_Derived.hpp"
#include "Phalanx_MDField.hpp"
#include "Tensor.h"

namespace LCM {
/** \brief ThermoMechanical stress response

    This evaluator computes stress based on a uncoupled Neohookean
    Helmholtz potential with temperature dependence

*/

template<typename EvalT, typename Traits>
class ThermoMechanicalStress : public PHX::EvaluatorWithBaseImpl<Traits>,
			       public PHX::EvaluatorDerived<EvalT, Traits>  {

public:

  ThermoMechanicalStress(const Teuchos::ParameterList& p);

  void postRegistrationSetup(typename Traits::SetupData d,
			     PHX::FieldManager<Traits>& vm);

  void evaluateFields(typename Traits::EvalData d);

private:

  typedef typename EvalT::ScalarT ScalarT;

  // Input:
  PHX::MDField<ScalarT,Cell,QuadPoint,Dim,Dim> F_array;
  PHX::MDField<ScalarT,Cell,QuadPoint> J_array;
  PHX::MDField<ScalarT,Cell,QuadPoint> shearModulus;
  PHX::MDField<ScalarT,Cell,QuadPoint> bulkModulus;
  PHX::MDField<ScalarT,Cell,QuadPoint> temperature;
  PHX::MDField<ScalarT,Cell,QuadPoint> yieldStrength;
  PHX::MDField<ScalarT,Cell,QuadPoint> hardeningModulus;
  PHX::MDField<ScalarT,Cell,QuadPoint> satMod;
  PHX::MDField<ScalarT,Cell,QuadPoint> satExp;
  PHX::MDField<ScalarT,Dummy> deltaTime;

  // Output:
  PHX::MDField<ScalarT,Cell,QuadPoint,Dim,Dim> stress;
  PHX::MDField<ScalarT,Cell,QuadPoint,Dim,Dim> Fp;
  PHX::MDField<ScalarT,Cell,QuadPoint> eqps;
  PHX::MDField<ScalarT,Cell,QuadPoint> mechSource;

  std::string fpName, eqpsName;
  unsigned int numQPs;
  unsigned int numDims;
  RealType thermalExpansionCoeff;
  RealType refTemperature;

  // local Tensors
  Tensor<ScalarT> F, Fpold, Fpinv, Cpinv, be, s, N, A, expA;
};
}

#endif
