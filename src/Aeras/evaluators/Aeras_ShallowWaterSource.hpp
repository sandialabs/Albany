//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#ifndef AERAS_SHALLOWWATERSOURCE_HPP
#define AERAS_SHALLOWWATERSOURCE_HPP

#include "Phalanx_ConfigDefs.hpp"
#include "Phalanx_Evaluator_WithBaseImpl.hpp"
#include "Phalanx_Evaluator_Derived.hpp"
#include "Phalanx_MDField.hpp"
#include "Sacado_ParameterAccessor.hpp" 
#include "Albany_Layouts.hpp"

namespace Aeras {
/** \brief Finite Element Interpolation Evaluator

    This evaluator interpolates nodal DOF values to quad points.

*/

template<typename EvalT, typename Traits>
class ShallowWaterSource : public PHX::EvaluatorWithBaseImpl<Traits>,
		    public PHX::EvaluatorDerived<EvalT, Traits>,
		    public Sacado::ParameterAccessor<EvalT, SPL_Traits> {

public:

  typedef typename EvalT::ScalarT ScalarT;

  ShallowWaterSource(const Teuchos::ParameterList& p,
              const Teuchos::RCP<Albany::Layouts>& dl);

  void postRegistrationSetup(typename Traits::SetupData d,
                      PHX::FieldManager<Traits>& vm);

  void evaluateFields(typename Traits::EvalData d);
  
  ScalarT& getValue(const std::string &n); 
 
  typedef typename EvalT::MeshScalarT MeshScalarT;

private:

          
  void get_coriolis(std::size_t cell, Intrepid::FieldContainer<ScalarT>  & coriolis);

  // Input:
  PHX::MDField<MeshScalarT,Cell,QuadPoint, Dim> sphere_coord;

  enum SOURCETYPE {NONE, TC4};
  SOURCETYPE sourceType;

  // Output:
  PHX::MDField<ScalarT,Cell,QuadPoint,VecDim> source;

  std::size_t numQPs, numDims, numNodes, vecDim, spatialDim;
          
  ScalarT gravity; // gravity parameter -- Sacado-ized for sensitivities
  ScalarT Omega;   //rotation of earth  -- Sacado-ized for sensitivities

          
  ScalarT earthRadius; //Earth radius

  ScalarT myPi; // a local copy of pi
       
  ///// SW TC4 parameters and routines
  ScalarT SU0;
  ScalarT PHI0;
  ScalarT RLON0;
  ScalarT RLAT0;
          
  ScalarT ALFA; //spelling is correct
  ScalarT SIGMA;
  ScalarT NPWR;
          
  ScalarT dbubf(const ScalarT lat);
  ScalarT bubfnc(const ScalarT lat);
  ScalarT d2bubf(const ScalarT lat);
          
          
};
  
  

  

}

#endif
