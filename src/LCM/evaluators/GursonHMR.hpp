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


#ifndef GURSONHMR_HPP
#define GURSONHMR_HPP

#include "Phalanx_ConfigDefs.hpp"
#include "Phalanx_Evaluator_WithBaseImpl.hpp"
#include "Phalanx_Evaluator_Derived.hpp"
#include "Phalanx_MDField.hpp"

#include "Tensor.h"
#include "Sacado.hpp"

namespace LCM {
	/** Large Deformation Gurson Model with Hardening Minus Recovery (HMR) law
	 *
	 * This evaluator compute stress based on Gurson damage model
	 *
	 */

	template<typename EvalT, typename Traits>
	class GursonHMR: public PHX::EvaluatorWithBaseImpl<Traits>,
    	public PHX::EvaluatorDerived<EvalT, Traits> {

 public:

   GursonHMR(const Teuchos::ParameterList& p);

   void postRegistrationSetup(typename Traits::SetupData d,
       PHX::FieldManager<Traits>& vm);

   void evaluateFields(typename Traits::EvalData d);

 private:

   typedef typename EvalT::ScalarT ScalarT;
   typedef typename EvalT::MeshScalarT MeshScalarT;
   typedef typename Sacado::Fad::DFad<ScalarT> DFadType;

   // all local functions used in computing GursonFD model stress:
   ScalarT
   compute_Phi(LCM::Tensor<ScalarT, 3> & s, ScalarT & p, ScalarT & fvoid, ScalarT & Y,
   	  ScalarT & isoH, ScalarT & Jacobian);

   void
   compute_ResidJacobian(std::vector<ScalarT> & X,
   	  std::vector<ScalarT> & R, std::vector<ScalarT> & dRdX, const ScalarT & p,
   	  const ScalarT & fvoid, const ScalarT & es, LCM::Tensor<ScalarT, 3> & s,
   	  ScalarT & mu, ScalarT & kappa, ScalarT & H, ScalarT & Y, ScalarT & Rd,
   	  ScalarT & Jacobian);

   //input
   PHX::MDField<ScalarT, Cell, QuadPoint, Dim, Dim> defgrad;
   PHX::MDField<ScalarT, Cell, QuadPoint> J;
   PHX::MDField<ScalarT, Cell, QuadPoint> elasticModulus;
   PHX::MDField<ScalarT, Cell, QuadPoint> poissonsRatio;
   PHX::MDField<ScalarT, Cell, QuadPoint> yieldStrength;
   PHX::MDField<ScalarT, Cell, QuadPoint> hardeningModulus;
   PHX::MDField<ScalarT, Cell, QuadPoint> recoveryModulus;

   std::string fpName, essName, eqpsName, voidVolumeName, isoHardeningName;
   unsigned int numQPs;
   unsigned int numDims;
   unsigned int worksetSize;

   // constant parameters
   RealType f0;
   RealType kw;
   RealType eN;
   RealType sN;
   RealType fN;
   RealType fc;
   RealType ff;
   RealType q1;
   RealType q2;
   RealType q3;

   //output
   PHX::MDField<ScalarT, Cell, QuadPoint, Dim, Dim> stress;
   PHX::MDField<ScalarT, Cell, QuadPoint, Dim, Dim> Fp;
   PHX::MDField<ScalarT, Cell, QuadPoint> ess;
   PHX::MDField<ScalarT, Cell, QuadPoint> eqps;
   PHX::MDField<ScalarT, Cell, QuadPoint> voidVolume;
   PHX::MDField<ScalarT, Cell, QuadPoint> isoHardening;

   // scratch space FCs
   Tensor<ScalarT, 3> be;
   Tensor<ScalarT, 3> logbe;
   Tensor<ScalarT, 3> s;
   Tensor<ScalarT, 3> A;
   Tensor<ScalarT, 3> expA;

   Intrepid::FieldContainer<ScalarT> Fpinv;
   Intrepid::FieldContainer<ScalarT> FpinvT;
   Intrepid::FieldContainer<ScalarT> Cpinv;

  };
}


#endif /* GURSONHMR_HPP_ */
