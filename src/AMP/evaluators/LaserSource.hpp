//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#ifndef LASERSOURCE_HPP
#define LASERSOURCE_HPP

#include "Phalanx_config.hpp"
#include "Phalanx_Evaluator_WithBaseImpl.hpp"
#include "Phalanx_Evaluator_Derived.hpp"
#include "Phalanx_MDField.hpp"
#include "Teuchos_ParameterList.hpp"
#include "Epetra_Vector.h"
#include "Sacado_ParameterAccessor.hpp"
#include "Stokhos_KL_ExponentialRandomField.hpp"
#include "Teuchos_Array.hpp"
#include "Albany_Layouts.hpp"

#include "Laser.hpp"

namespace AMP {
///
/// \brief Laser Source
///
/// This evaluator computes the moving laser source as a function of space and time to a 
/// Phase-change/heat equation problem
///
template<typename EvalT, typename Traits>
class LaserSource : 
  public PHX::EvaluatorWithBaseImpl<Traits>,
  public PHX::EvaluatorDerived<EvalT, Traits>
{

public:

  LaserSource(Teuchos::ParameterList& p,
      const Teuchos::RCP<Albany::Layouts>& dl);

  void 
  postRegistrationSetup(typename Traits::SetupData d,
      PHX::FieldManager<Traits>& vm);

  void 
  evaluateFields(typename Traits::EvalData d);

private:

  typedef typename EvalT::ScalarT ScalarT;
  typedef typename EvalT::MeshScalarT MeshScalarT;

  ScalarT porosity;
  ScalarT particle_dia;
  ScalarT laser_beam_radius;
  ScalarT laser_power;
  ScalarT powder_hemispherical_reflectivity;
  
  void init_constant_porosity(ScalarT value_porosity, Teuchos::ParameterList& p);
  void init_constant_particle_dia(ScalarT value_particle_dia, Teuchos::ParameterList& p);
  void init_constant_laser_beam_radius(ScalarT value_laser_beam_radius, Teuchos::ParameterList& p);
  void init_constant_laser_power(ScalarT value_laser_power, Teuchos::ParameterList& p);
  void init_constant_powder_hemispherical_reflectivity(ScalarT value_powder_hemispherical_reflectivity, Teuchos::ParameterList& p);

  PHX::MDField<MeshScalarT,Cell,QuadPoint,Dim> coord_;
  PHX::MDField<ScalarT,Cell,QuadPoint> laser_source_;
  PHX::MDField<ScalarT,Dummy> time;
  PHX::MDField<ScalarT,Dummy> deltaTime;

  unsigned int num_qps_;
  unsigned int num_dims_;
  unsigned int num_nodes_;
  unsigned int workset_size_;

  Laser LaserData_;

  Teuchos::RCP<const Teuchos::ParameterList>
     getValidLaserSourceParameters() const;
};
}

#endif
