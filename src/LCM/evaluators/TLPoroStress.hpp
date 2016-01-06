//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#ifndef TLPOROSTRESS_HPP
#define TLPOROSTRESS_HPP

#include "Phalanx_config.hpp"
#include "Phalanx_Evaluator_WithBaseImpl.hpp"
#include "Phalanx_Evaluator_Derived.hpp"
#include "Phalanx_MDField.hpp"

namespace LCM {
/** \brief

    This evaluator obtains effective stress and return
    total stress (i.e. with pore-fluid contribution)
    For now, it does not work for Neohookean AD

*/

template<typename EvalT, typename Traits>
class TLPoroStress : public PHX::EvaluatorWithBaseImpl<Traits>,
		    public PHX::EvaluatorDerived<EvalT, Traits>  {

public:

  TLPoroStress(const Teuchos::ParameterList& p);

  void postRegistrationSetup(typename Traits::SetupData d,
                      PHX::FieldManager<Traits>& vm);

  void evaluateFields(typename Traits::EvalData d);

private:

  typedef typename EvalT::ScalarT ScalarT;
  typedef typename EvalT::MeshScalarT MeshScalarT;

  // Input:
  PHX::MDField<ScalarT,Cell,QuadPoint,Dim,Dim> defGrad;
  PHX::MDField<ScalarT,Cell,QuadPoint> J;
  PHX::MDField<ScalarT,Cell,QuadPoint,Dim,Dim> stress;
  PHX::MDField<ScalarT,Cell,QuadPoint> biotCoefficient;
  PHX::MDField<ScalarT,Cell,QuadPoint> porePressure;

  unsigned int numQPs;
  unsigned int numDims;

  // Work space FCs
  Intrepid2::FieldContainer<ScalarT> F_inv;
  Intrepid2::FieldContainer<ScalarT> F_invT;
  Intrepid2::FieldContainer<ScalarT> JF_invT;
  Intrepid2::FieldContainer<ScalarT> JpF_invT;
  Intrepid2::FieldContainer<ScalarT> JBpF_invT;

  // Material Name
    std::string matModel;


  // Output:
  PHX::MDField<ScalarT,Cell,QuadPoint,Dim,Dim> totstress;
};
}

#endif
