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


#ifndef THERMO_MECHANICAL_ENERGY_RESIDUAL_HPP
#define THERMO_MECHANICAL_ENERGY_RESIDUAL_HPP

#include "Phalanx_ConfigDefs.hpp"
#include "Phalanx_Evaluator_WithBaseImpl.hpp"
#include "Phalanx_Evaluator_Derived.hpp"
#include "Phalanx_MDField.hpp"

namespace LCM {
  /** \brief ThermMechanical Energy Residual

      This evaluator computes the residual for the energy equation
      in the coupled therm-mechanical problem

  */

  template<typename EvalT, typename Traits>
  class ThermoMechanicalEnergyResidual : public PHX::EvaluatorWithBaseImpl<Traits>,
					 public PHX::EvaluatorDerived<EvalT, Traits>  {

  public:

    ThermoMechanicalEnergyResidual(const Teuchos::ParameterList& p);

    void postRegistrationSetup(typename Traits::SetupData d,
			       PHX::FieldManager<Traits>& vm);

    void evaluateFields(typename Traits::EvalData d);

  private:

    typedef typename EvalT::ScalarT ScalarT;
    typedef typename EvalT::MeshScalarT MeshScalarT;

    // Input:
    PHX::MDField<MeshScalarT,Cell,Node,QuadPoint> wBF;
    PHX::MDField<ScalarT,Cell,QuadPoint> Temperature;
    PHX::MDField<ScalarT,Cell,QuadPoint> Tdot;
    PHX::MDField<ScalarT,Cell,QuadPoint> ThermalCond;
    PHX::MDField<MeshScalarT,Cell,Node,QuadPoint,Dim> wGradBF;
    PHX::MDField<ScalarT,Cell,QuadPoint,Dim> TGrad;
    PHX::MDField<ScalarT,Cell,QuadPoint> Source;
    Teuchos::Array<double> convectionVels;
    PHX::MDField<ScalarT,Cell,QuadPoint> rhoCp;
    PHX::MDField<ScalarT,Cell,QuadPoint> Absorption;
    PHX::MDField<ScalarT,Cell,QuadPoint,Dim,Dim> F; // deformation gradient
    PHX::MDField<ScalarT,Cell,QuadPoint,Dim,Dim> mechSource; // mechanical heat source

    // Output:
    PHX::MDField<ScalarT,Cell,Node> TResidual;

    bool haveSource;
    bool haveConvection;
    bool haveAbsorption;
    bool enableTransient;
    bool haverhoCp;
    unsigned int numQPs, numDims, worksetSize;
    Intrepid::FieldContainer<ScalarT> flux;
    Intrepid::FieldContainer<ScalarT> aterm;
    Intrepid::FieldContainer<ScalarT> C;
    Intrepid::FieldContainer<ScalarT> Cinv;
    Intrepid::FieldContainer<ScalarT> CinvTgrad;
  };
}

#endif
