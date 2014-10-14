//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#ifndef AERAS_ATMOSPHERE_MOISTURE_HPP
#define AERAS_ATMOSPHERE_MOISTURE_HPP

#include "Phalanx_ConfigDefs.hpp"
#include "Phalanx_Evaluator_WithBaseImpl.hpp"
#include "Phalanx_Evaluator_Derived.hpp"
#include "Phalanx_MDField.hpp"

#include "Aeras_Layouts.hpp"
#include "Aeras_Dimension.hpp"

#include "Teuchos_ParameterList.hpp"

namespace Aeras {

//static
//void kessler(int Km, double dt_in,
//             std::vector<double> & rho, 
//             std::vector<double> & p, 
//             std::vector<double> & exner, 
//             std::vector<double> & dz8w,
//             std::vector<double> & t,  
//             std::vector<double> & qv, 
//             std::vector<double> & qc, 
//             std::vector<double> & qr,
//             double &rainnc,  double &rainncv,
//             std::vector<double> &z);

template<typename EvalT, typename Traits> 
class Atmosphere_Moisture : public PHX::EvaluatorWithBaseImpl<Traits>,
                            public PHX::EvaluatorDerived<EvalT, Traits>  {
  
public:
  typedef typename EvalT::ScalarT ScalarT;
  typedef typename EvalT::MeshScalarT MeshScalarT;
  
  Atmosphere_Moisture(Teuchos::ParameterList& p,
                      const Teuchos::RCP<Aeras::Layouts>& dl);
  
  void postRegistrationSetup(typename Traits::SetupData d,
                      PHX::FieldManager<Traits>& vm);
  
  void evaluateFields(typename Traits::EvalData d);
  
private:
void kessler(const int Km, const double dt_in,
             const std::vector<double> & rho, 
             const std::vector<double> & p, 
             const std::vector<double> & exner, 
             const std::vector<double> & dz8w,
             std::vector<double> & t,  
             std::vector<double> & qv, 
             std::vector<double> & qc, 
             std::vector<double> & qr,
             double &rainnc,  double &rainncv,
             const std::vector<double> &z);

  PHX::MDField<ScalarT,Cell,QuadPoint,Level,VecDim> Velx;
  PHX::MDField<ScalarT,Cell,QuadPoint,VecDim> Temp;
  PHX::MDField<ScalarT,Cell,QuadPoint,VecDim> Density;
  PHX::MDField<ScalarT,Cell,QuadPoint,VecDim> GeoPotential;
  PHX::MDField<ScalarT,Cell,QuadPoint,VecDim> Pressure;
  PHX::MDField<ScalarT,Cell,QuadPoint,VecDim> Pi;
  PHX::MDField<ScalarT,Cell,QuadPoint,VecDim> PiDot;
  PHX::MDField<ScalarT,Cell,QuadPoint,VecDim> TempSrc;

  std::map<std::string, PHX::MDField<ScalarT,Cell,QuadPoint> > TracerIn;
  std::map<std::string, PHX::MDField<ScalarT,Cell,QuadPoint> > TracerSrc;


  const Teuchos::ArrayRCP<std::string> tracerNames;
  const Teuchos::ArrayRCP<std::string> tracerSrcNames;
  std::map<std::string, std::string>   namesToSrc;
 
  bool compute_cloud_physics;

  const int numQPs;
  const int numDims;
  const int numLevels;
};
}

#endif
