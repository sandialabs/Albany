//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#ifndef MULTISCALE_THERMAL_CONDUCTIVITY_HPP
#define MULTISCALE_THERMAL_CONDUCTIVITY_HPP

#include "Phalanx_config.hpp"
#include "Phalanx_Evaluator_WithBaseImpl.hpp"
#include "Phalanx_Evaluator_Derived.hpp"
#include "Phalanx_MDField.hpp"

#include "Teuchos_ParameterList.hpp"
#include "Sacado_ParameterAccessor.hpp"
#ifdef ALBANY_STOKHOS
#include "Stokhos_KL_ExponentialRandomField.hpp"
#endif
#include "Teuchos_Array.hpp"

#include "QCAD_MaterialDatabase.hpp"

#include <zmq.h>

namespace AFRL {
/**
 * \brief Evaluates thermal conductivity, either as a constant or a truncated
 * KL expansion.

This class may be used in two ways.

1. The simplest is to use a constant thermal conductivity across the entire domain (one element block,
one material), say with a value of 5.0. In this case, one would declare at the "Problem" level, that a
constant thermal conductivity was being used, and its value was 5.0:

<ParameterList name="Problem">
   ...
    <ParameterList name="Thermal Conductivity">
       <Parameter name="Thermal Conductivity Type" type="string" value="Constant"/>
       <Parameter name="Value" type="double" value="5.0"/>
    </ParameterList>
</ParameterList>

An example of this is test problem is SteadyHeat2DInternalNeumann

2. The other extreme is to have a multiple element block problem, say 3, with each element block corresponding
to a material. Each element block has its own field manager, and different evaluators are used in each element
block. See the test problem Heat2DMMCylWithSource for an example of this use case.

 */

template<typename EvalT, typename Traits>
class MultiScaleThermalConductivity :
  public PHX::EvaluatorWithBaseImpl<Traits>,
  public PHX::EvaluatorDerived<EvalT, Traits>,
  public Sacado::ParameterAccessor<EvalT, SPL_Traits> {

public:
  typedef typename EvalT::ScalarT ScalarT;
  typedef typename EvalT::MeshScalarT MeshScalarT;

  enum SG_RF {CONSTANT, UNIFORM, LOGNORMAL};

  MultiScaleThermalConductivity(Teuchos::ParameterList& p);
  // virtual ~MultiScaleThermalConductivity();

  void postRegistrationSetup(typename Traits::SetupData d,
			     PHX::FieldManager<Traits>& vm);

  void evaluateFields(typename Traits::EvalData d);

  ScalarT& getValue(const std::string &n);

private:

//! Validate the name strings under "Thermal Conductivity" section in xml input file,
  Teuchos::RCP<const Teuchos::ParameterList>
               getValidThermalCondParameters() const;

  enum ComputeMode {Constant, Series, Remote};
  ComputeMode computeMode;

  std::size_t numQPs;
  std::size_t numDims;
  PHX::MDField<MeshScalarT,Cell,QuadPoint,Dim> coordVec;
  PHX::MDField<ScalarT,Cell,QuadPoint> thermalCond;
  PHX::MDField<ScalarT,Cell,QuadPoint> temperature;
  PHX::MDField<ScalarT,Cell,QuadPoint,Dim> gradTemperature;

  //! Conductivity type
  std::string type;

  //! Constant value
  ScalarT constant_value;

#ifdef ALBANY_STOKHOS
  //! Exponential random field
  Teuchos::RCP< Stokhos::KL::ExponentialRandomField<MeshScalarT> > exp_rf_kl;
#endif

  //! Values of the random variables
  Teuchos::Array<ScalarT> rv;

  //! Material database - holds thermal conductivity among other quantities
  Teuchos::RCP<QCAD::MaterialDatabase> materialDB;

  //! material, Description file and index for representative volume element
  //  (RVE)
  struct RepresentativeVolumeElement
  {
    RepresentativeVolumeElement() : context(0), socket(0) {}
    ~RepresentativeVolumeElement()
    {
      if (socket) zmq_close(socket);
      if (context) zmq_ctx_destroy(context);
    }
    std::string material;
    std::string descriptionfile;
    int id;
    void* context;
    void* socket;
  };
  RepresentativeVolumeElement RVE;

  //! Convenience function to initialize constant thermal conductivity
  void init_constant(ScalarT value, Teuchos::ParameterList& p);

  //! Convenience function to initialize thermal conductivity based on
  //  Truncated KL Expansion || Log Normal RF
  void init_KL_RF(std::string &type, Teuchos::ParameterList& subList, Teuchos::ParameterList& p);

  //! Convenience function to initialize thermal conductivity based on
  //  external computation
  void init_remote(std::string &type, std::string& microscaleExe,
                   std::string& descriptionFile, int id,
                   Teuchos::ParameterList& p);
  double get_remote(double time, double previousTime, const ScalarT& temperature,
                    const Teuchos::Array<ScalarT>& gradT) const;


  SG_RF randField;
};
}

#endif
