//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#ifndef AERAS_HYDROSTATIC_RESPONSE_L2ERROR_HPP
#define AERAS_HYDROSTATIC_RESPONSE_L2ERROR_HPP

#include "PHAL_SeparableScatterScalarResponse.hpp"
#include "Aeras_Layouts.hpp"
#include "Aeras_Dimension.hpp"

struct DoubleType { 
  typedef double  ScalarT; 
  typedef double MeshScalarT; 
};

namespace Aeras{

template<typename EvalT, typename Traits>
class HydrostaticResponseL2Error :
  public PHAL::SeparableScatterScalarResponse<EvalT,Traits>
{
public:
  typedef typename EvalT::ScalarT ScalarT;
  typedef typename EvalT::MeshScalarT MeshScalarT;

  HydrostaticResponseL2Error(Teuchos::ParameterList& p,
                       const Teuchos::RCP<Aeras::Layouts>& dl);

  void postRegistrationSetup(typename Traits::SetupData d,
                                   PHX::FieldManager<Traits>& vm);

  void preEvaluate(typename Traits::PreEvalData d);

  void evaluateFields(typename Traits::EvalData d);

  void postEvaluate(typename Traits::PostEvalData d);

private:
  Teuchos::RCP<const Teuchos::ParameterList> getValidResponseParameters() const;
  Teuchos::RCP<Teuchos::FancyOStream> out;

  std::string refSolName; //name of reference solution
  enum REF_SOL_NAME {ZERO, BAROCLINIC_UNPERTURBED};
  REF_SOL_NAME ref_sol_name;
  double inputData; // constant read in from parameter list that may be used in specifying reference solution

  std::size_t numQPs;
  std::size_t numDims;

  PHX::MDField<ScalarT> field;
  PHX::MDField<MeshScalarT,Cell,QuadPoint> weighted_measure;
  PHX::MDField<MeshScalarT> sphere_coord;
  PHX::MDField<ScalarT,Cell,QuadPoint,Level,Dim> velocity;
  PHX::MDField<ScalarT,Cell,QuadPoint,Level> temperature;
  PHX::MDField<ScalarT,Cell,QuadPoint> spressure;
  const int numLevels;
  int responseSize; 
};

}

#endif




