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

#ifndef CAPMODELSTRESS_HPP
#define CAPMODELSTRESS_HPP

#include "Phalanx_ConfigDefs.hpp"
#include "Phalanx_Evaluator_WithBaseImpl.hpp"
#include "Phalanx_Evaluator_Derived.hpp"
#include "Phalanx_MDField.hpp"

#include "Tensor.h"

namespace LCM {
  /** \brief CapModelStress stress response

   This evaluator computes stress based on a cap plasticity model.

   */

  template<typename EvalT, typename Traits>
  class CapModelStress: public PHX::EvaluatorWithBaseImpl<Traits>,
      public PHX::EvaluatorDerived<EvalT, Traits> {

  public:

    CapModelStress(const Teuchos::ParameterList& p);

    void postRegistrationSetup(typename Traits::SetupData d,
        PHX::FieldManager<Traits>& vm);

    void evaluateFields(typename Traits::EvalData d);

  private:

    typedef typename EvalT::ScalarT ScalarT;
    typedef typename EvalT::MeshScalarT MeshScalarT;

    // all local functions used in computing cap model stress:
    ScalarT compute_f(LCM::Tensor<ScalarT, 3> & sigma,
        LCM::Tensor<ScalarT, 3> & alpha, ScalarT & kappa);

    LCM::Tensor<ScalarT, 3> compute_dfdsigma(LCM::Tensor<ScalarT, 3> & sigma,
        LCM::Tensor<ScalarT, 3> & alpha, ScalarT & kappa);

    LCM::Tensor<ScalarT, 3> compute_dgdsigma(LCM::Tensor<ScalarT, 3> & sigma,
        LCM::Tensor<ScalarT, 3> & alpha, ScalarT & kappa);

    ScalarT compute_dfdkappa(LCM::Tensor<ScalarT, 3> & sigma,
        LCM::Tensor<ScalarT, 3> & alpha, ScalarT & kappa);

    ScalarT compute_Galpha(ScalarT & J2_alpha);

    LCM::Tensor<ScalarT, 3> compute_halpha(LCM::Tensor<ScalarT, 3> & dgdsigma,
        ScalarT & J2_alpha);

    ScalarT compute_dedkappa(ScalarT & kappa);

    //Input
    PHX::MDField<ScalarT, Cell, QuadPoint, Dim, Dim> strain;
    PHX::MDField<ScalarT, Cell, QuadPoint> elasticModulus;
    PHX::MDField<ScalarT, Cell, QuadPoint> poissonsRatio;

    unsigned int numQPs;
    unsigned int numDims;

    double A;
    double B;
    double C;
    double theta;
    double R;
    double kappa0;
    double W;
    double D1;
    double D2;
    double calpha;
    double psi;
    double N;
    double L;
    double phi;
    double Q;

    std::string strainName, stressName;
    std::string backStressName, capParameterName;

    //output
    PHX::MDField<ScalarT, Cell, QuadPoint, Dim, Dim> stress;
    PHX::MDField<ScalarT, Cell, QuadPoint, Dim, Dim> backStress;
    PHX::MDField<ScalarT, Cell, QuadPoint> capParameter;

  };
}

#endif

