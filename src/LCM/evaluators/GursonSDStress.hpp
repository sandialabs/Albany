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


#ifndef GURSONSDSTRESS_HPP
#define GURSONSDSTRESS_HPP

#include "Phalanx_ConfigDefs.hpp"
#include "Phalanx_Evaluator_WithBaseImpl.hpp"
#include "Phalanx_Evaluator_Derived.hpp"
#include "Phalanx_MDField.hpp"

#include "Tensor.h"
#include "Sacado.hpp"

namespace LCM {
/** \brief Gurson small deformation stress response

    This evaluator computes stress based on modified Gurson model.

*/

template<typename EvalT, typename Traits>
class GursonSDStress : public PHX::EvaluatorWithBaseImpl<Traits>,
		 public PHX::EvaluatorDerived<EvalT, Traits>  {

public:

  GursonSDStress(const Teuchos::ParameterList& p);

  void postRegistrationSetup(typename Traits::SetupData d,
					  PHX::FieldManager<Traits>& vm);

  void evaluateFields(typename Traits::EvalData d);

private:

  typedef typename EvalT::ScalarT ScalarT;
  typedef typename EvalT::MeshScalarT MeshScalarT;

  // all local functions used in computing GursonSD model stress:
  ScalarT compute_Y(ScalarT & epVal, ScalarT & Eor3mu);
  ScalarT compute_Phi(LCM::Tensor<ScalarT> & devVal, ScalarT & pVal, ScalarT & fvoidVal, ScalarT & epVal, ScalarT & Eor3mu);
  ScalarT compute_dPhidep(ScalarT & tmp, ScalarT & pN, ScalarT & fvoidN, ScalarT & epN, ScalarT & YN, ScalarT & Eor3mu);
  ScalarT compute_dfdgam(ScalarT & depdgam, ScalarT & tmp, ScalarT & pN, ScalarT & sigeN, ScalarT & J3N,
		ScalarT & fvoidN, ScalarT & epN, ScalarT & YN);

  //Input
  PHX::MDField<ScalarT,Cell,QuadPoint,Dim,Dim> strain;
  PHX::MDField<ScalarT,Cell,QuadPoint> elasticModulus;
  PHX::MDField<ScalarT,Cell,QuadPoint> poissonsRatio;

  unsigned int numQPs;
  unsigned int numDims;

  double f0;
  double Y0;
  double kw;
  double N;
  double q1;
  double q2;
  double q3;
  double eN;
  double sN;
  double fN;
  double fc;
  double ff;
  double flag;

  std::string strainName, stressName;
  std::string voidVolumeName, epName, yieldStrengthName;

  //output
  PHX::MDField<ScalarT,Cell,QuadPoint,Dim,Dim> stress;
  PHX::MDField<ScalarT,Cell,QuadPoint> voidVolume;
  PHX::MDField<ScalarT,Cell,QuadPoint> ep;
  PHX::MDField<ScalarT,Cell,QuadPoint> yieldStrength;

};
}

#endif

