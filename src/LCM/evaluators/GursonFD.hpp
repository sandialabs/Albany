//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//
#ifndef GURSONFD_HPP
#define GURSONFD_HPP

#include "Phalanx_ConfigDefs.hpp"
#include "Phalanx_Evaluator_WithBaseImpl.hpp"
#include "Phalanx_Evaluator_Derived.hpp"
#include "Phalanx_MDField.hpp"

#include "VectorTensorBase.h"
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
    typedef LCM::Tensor<ScalarT> Tensor;

    // all local functions used in computing GursonFD model stress:

    ScalarT
    YeldFunction(Tensor const & s, ScalarT const & p, ScalarT const & fvoid,
        ScalarT const & eq, ScalarT const & K, ScalarT const & Y,
        ScalarT const & siginf,ScalarT const & delta,
        ScalarT const & Jacobian, ScalarT const & E);

    void
    ResidualJacobian(std::vector<ScalarT> & X,
      std::vector<ScalarT> & R, std::vector<ScalarT> & dRdX,  const ScalarT & p,
      const ScalarT & fvoid, const ScalarT & eq, Tensor & s,
      const ScalarT & shearModulus, const ScalarT & bulkModulus,
      const ScalarT & K, const ScalarT & Y, const ScalarT & siginf,
      const ScalarT & delta, const ScalarT & Jacobian);

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

    Intrepid::FieldContainer<ScalarT> Fpinv;
    Intrepid::FieldContainer<ScalarT> FpinvT;
    Intrepid::FieldContainer<ScalarT> Cpinv;

  };
}

#endif

