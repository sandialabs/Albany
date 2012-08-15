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

#ifndef GURSONFD_HPP
#define GURSONFD_HPP

#include "Phalanx_ConfigDefs.hpp"
#include "Phalanx_Evaluator_WithBaseImpl.hpp"
#include "Phalanx_Evaluator_Derived.hpp"
#include "Phalanx_MDField.hpp"

#include "Tensor.h"
#include "Sacado.hpp"

namespace LCM {
  /** \brief Gurson large deformation hyperelastic stress response

   This evaluator computes stress based on original Gurson model.

   */

  template<typename EvalT, typename Traits>
  class GursonFD: public PHX::EvaluatorWithBaseImpl<Traits>,
      public PHX::EvaluatorDerived<EvalT, Traits> {

  public:

    GursonFD(const Teuchos::ParameterList& p);

    void postRegistrationSetup(typename Traits::SetupData d,
        PHX::FieldManager<Traits>& vm);

    void evaluateFields(typename Traits::EvalData d);

  private:

    typedef typename EvalT::ScalarT ScalarT;
    typedef typename EvalT::MeshScalarT MeshScalarT;
    typedef typename Sacado::Fad::DFad<ScalarT> DFadType;

    // all local functions used in computing GursonFD model stress:

    ScalarT
    compute_Phi(LCM::Tensor<ScalarT, 3> & s, ScalarT & p, ScalarT & fvoid,
        ScalarT & eq, ScalarT & K, ScalarT & Y, ScalarT & siginf,
        ScalarT & delta, ScalarT & Jacobian, ScalarT & E);

    void
    compute_ResidJacobian(std::vector<ScalarT> & X, std::vector<ScalarT> & R,
        std::vector<ScalarT> & dRdX, const ScalarT & p, const ScalarT & fvoid,
        const ScalarT & eq, LCM::Tensor<ScalarT, 3> & s, ScalarT & mu,
        ScalarT & kappa, ScalarT & K, ScalarT & Y, ScalarT & siginf,
        ScalarT & delta, ScalarT & Jacobian);

    //Input
    PHX::MDField<ScalarT, Cell, QuadPoint, Dim, Dim> defgrad;
    PHX::MDField<ScalarT, Cell, QuadPoint> J;
    PHX::MDField<ScalarT, Cell, QuadPoint> elasticModulus;
    PHX::MDField<ScalarT, Cell, QuadPoint> poissonsRatio;
    PHX::MDField<ScalarT, Cell, QuadPoint> yieldStrength;
    PHX::MDField<ScalarT, Cell, QuadPoint> hardeningModulus;
    PHX::MDField<ScalarT, Cell, QuadPoint> satMod;
    PHX::MDField<ScalarT, Cell, QuadPoint> satExp;
    PHX::MDField<ScalarT,Dummy> deltaTime;

    std::string fpName, eqpsName, voidVolumeName, defGradName, stressName;
    unsigned int numQPs;
    unsigned int numDims;
    unsigned int worksetSize;

    RealType eq0;
    RealType N;
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

    bool isSaturationH;
    bool isHyper;

    //output
    PHX::MDField<ScalarT, Cell, QuadPoint, Dim, Dim> stress;
    PHX::MDField<ScalarT, Cell, QuadPoint, Dim, Dim> Fp;
    PHX::MDField<ScalarT, Cell, QuadPoint> eqps;
    PHX::MDField<ScalarT, Cell, QuadPoint> voidVolume;

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

#endif

