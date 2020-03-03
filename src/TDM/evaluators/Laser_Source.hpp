//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#ifndef TDM_LASER_SOURCE_HPP
#define TDM_LASER_SOURCE_HPP

#include "Phalanx_config.hpp"
#include "Phalanx_Evaluator_WithBaseImpl.hpp"
#include "Phalanx_Evaluator_Derived.hpp"
#include "Phalanx_MDField.hpp"
#include "Teuchos_ParameterList.hpp"
#include "Epetra_Vector.h"
#include "Sacado_ParameterAccessor.hpp"
#include "Teuchos_Array.hpp"
#include "Albany_Layouts.hpp"

#include "Laser.hpp"

namespace TDM {
  template<typename EvalT, typename Traits>
  class Laser_Source : 
    public PHX::EvaluatorWithBaseImpl<Traits>,
    public PHX::EvaluatorDerived<EvalT, Traits>
  {

  public:

    Laser_Source(Teuchos::ParameterList& p,
		 const Teuchos::RCP<Albany::Layouts>& dl);

    void 
    postRegistrationSetup(typename Traits::SetupData d,
			  PHX::FieldManager<Traits>& vm);

    void 
    evaluateFields(typename Traits::EvalData d);

  private:

    typedef typename EvalT::ScalarT ScalarT;
    typedef typename EvalT::MeshScalarT MeshScalarT;

    //From materials file
    ScalarT absortivity;
    ScalarT reflectivity;
    ScalarT laser_beam_radius;
    ScalarT laser_pulse_frequency;
    ScalarT average_laser_power;
    ScalarT initial_porosity;
    ScalarT powder_diameter;
    ScalarT powder_layer_thickness;
    std::string sim_type;
    std::string laser_path_filename;
  
    PHX::MDField<MeshScalarT,Cell,QuadPoint,Dim> coord_;
    PHX::MDField<ScalarT,Cell,QuadPoint> laser_source_;
    PHX::MDField<ScalarT,Dummy> time;
    PHX::MDField<ScalarT,Dummy> deltaTime;
    PHX::MDField<ScalarT,Cell,QuadPoint> psi1_;
    PHX::MDField<ScalarT,Cell,QuadPoint> psi2_;
    PHX::MDField<ScalarT,Cell,QuadPoint> depth_;    

    unsigned int num_qps_;
    unsigned int num_dims_;
    unsigned int num_nodes_;
    unsigned int workset_size_;

    //Create an object that will import and contain an array of the laser path data
    Laser Laser_object;

    // variable use to decide if subtractive is true or false
    bool Subtractive_;

    Teuchos::RCP<const Teuchos::ParameterList>
    getValidLaser_SourceParameters() const;
  };
}

#endif
