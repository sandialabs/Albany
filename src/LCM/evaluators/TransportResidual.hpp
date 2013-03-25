//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#if !defined(LCM_TransportResidual_hpp)
#define LCM_TransportResidual_hpp

#include <Phalanx_ConfigDefs.hpp>
#include <Phalanx_Evaluator_WithBaseImpl.hpp>
#include <Phalanx_Evaluator_Derived.hpp>
#include <Phalanx_MDField.hpp>

#include "Albany_Layouts.hpp"

namespace LCM {
  /// \brief
  ///
  ///  This evaluator computes the residual for the transport equation
  ///
  template<typename EvalT, typename Traits>
  class TransportResidual : public PHX::EvaluatorWithBaseImpl<Traits>,
                            public PHX::EvaluatorDerived<EvalT, Traits>  {

  public:

    ///
    /// Constructor
    ///
    TransportResidual(const Teuchos::ParameterList& p,
                      const Teuchos::RCP<Albany::Layouts>& dl);

    ///
    /// Phalanx method to allocate space
    ///
    void postRegistrationSetup(typename Traits::SetupData d,
                               PHX::FieldManager<Traits>& vm);

    ///
    /// Implementation of physics
    ///
    void evaluateFields(typename Traits::EvalData d);

  private:

    typedef typename EvalT::ScalarT ScalarT;
    typedef typename EvalT::MeshScalarT MeshScalarT;

    // Input:
    PHX::MDField<MeshScalarT,Cell,QuadPoint> weights_;
    PHX::MDField<MeshScalarT,Cell,Node,QuadPoint> w_bf_;
    PHX::MDField<MeshScalarT,Cell,Node,QuadPoint,Dim> w_grad_bf_;
    PHX::MDField<ScalarT,Cell,QuadPoint> Source;
    PHX::MDField<ScalarT,Cell,QuadPoint,Dim, Dim> DefGrad;
    PHX::MDField<ScalarT,Cell,QuadPoint> elementLength;
    PHX::MDField<ScalarT,Cell,QuadPoint> Dstar;
    PHX::MDField<ScalarT,Cell,QuadPoint> DL;
    PHX::MDField<ScalarT,Cell,QuadPoint> Clattice;
    PHX::MDField<ScalarT,Cell,QuadPoint,Dim> CLGrad;
    PHX::MDField<ScalarT,Cell,QuadPoint,Dim> stressGrad;
    PHX::MDField<ScalarT,Cell,QuadPoint> stabParameter;

    // Input for the strain rate effect
    PHX::MDField<ScalarT,Cell,QuadPoint> Ctrapped;
    PHX::MDField<ScalarT,Cell,QuadPoint> Ntrapped;
    PHX::MDField<ScalarT,Cell,QuadPoint> eqps;
    PHX::MDField<ScalarT,Cell,QuadPoint> eqpsFactor;
    std::string eqpsName;


    // Input for hydro-static stress effect
    PHX::MDField<ScalarT,Cell,QuadPoint,Dim,Dim> Pstress;
    PHX::MDField<ScalarT,Cell,QuadPoint> tauFactor;

    // Time
    PHX::MDField<ScalarT,Dummy> deltaTime;

    //Data from previous time step
    std::string ClatticeName;
    std::string CLGradName;

    bool haveSource;
    bool haveMechSource;
    bool enableTransient;

    unsigned int numNodes;
    unsigned int numQPs;
    unsigned int numDims;
    unsigned int worksetSize;

    // Temporary FieldContainers
    Intrepid::FieldContainer<ScalarT> Hflux;
    Intrepid::FieldContainer<ScalarT> Hfluxdt;
    Intrepid::FieldContainer<ScalarT> C;
    Intrepid::FieldContainer<ScalarT> Cinv;
    Intrepid::FieldContainer<ScalarT> CinvTgrad;
    Intrepid::FieldContainer<ScalarT> CinvTgrad_old;

    Intrepid::FieldContainer<ScalarT> artificalDL;
    Intrepid::FieldContainer<ScalarT> stabilizedDL;

    Intrepid::FieldContainer<ScalarT> tauStress;


    Intrepid::FieldContainer<ScalarT> pterm;
    Intrepid::FieldContainer<ScalarT> tpterm;
    Intrepid::FieldContainer<ScalarT> aterm;

    Intrepid::FieldContainer<ScalarT> tauH;
    Intrepid::FieldContainer<ScalarT> CinvTaugrad;

    ScalarT CLbar, vol ;

    // Temporary Field Containers for stabilization

    Intrepid::FieldContainer<ScalarT> pTTterm;
    Intrepid::FieldContainer<ScalarT> pBterm;
    Intrepid::FieldContainer<ScalarT> pTranTerm;



    ScalarT trialPbar;
    // ScalarT pStrainRateTerm;



    // Output:
    PHX::MDField<ScalarT,Cell,Node> TResidual;
  };
}

#endif
