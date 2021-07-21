//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#ifndef PHAL_THERMAL_CONDUCTIVITY_HPP
#define PHAL_THERMAL_CONDUCTIVITY_HPP

#include "Albany_config.h"

#include "Phalanx_config.hpp"
#include "Phalanx_Evaluator_WithBaseImpl.hpp"
#include "Phalanx_Evaluator_Derived.hpp"
#include "Phalanx_MDField.hpp"

#include "Teuchos_ParameterList.hpp"
#include "Sacado_ParameterAccessor.hpp"
#include "Teuchos_Array.hpp"

#include "Albany_MaterialDatabase.hpp"

namespace PHAL {
/**
 * \brief Evaluates thermal conductivity, either as a constant or a truncated
 * KL expansion.

This class may be used in two ways.

1. The simplest is to use a constant thermal conductivity across the entire domain (one element block,
one material), say with a value of 5.0. In this case, one would declare at the "Problem" level, that a
constant thermal conductivity was being used, and its value was 5.0:

<ParameterList name="Problem">
   ...
    <ParameterList name="ThermalConductivity">
       <Parameter name="ThermalConductivity Type" type="string" value="Constant"/>
       <Parameter name="Value" type="double" value="5.0"/>
    </ParameterList>
</ParameterList>

An example of this is test problem is SteadyHeat2DInternalNeumann

2. The other extreme is to have a multiple element block problem, say 3, with each element block corresponding
to a material. Each element block has its own field manager, and different evaluators are used in each element
block. See the test problem Heat2DMMCylWithSource for an example of this use case.

 */

template<typename EvalT, typename Traits>
class ThermalConductivity :
  public PHX::EvaluatorWithBaseImpl<Traits>,
  public PHX::EvaluatorDerived<EvalT, Traits>,
  public Sacado::ParameterAccessor<EvalT, SPL_Traits> {

public:
  typedef typename EvalT::ScalarT ScalarT;
  typedef typename EvalT::MeshScalarT MeshScalarT;

  enum SG_RF {CONSTANT, UNIFORM, LOGNORMAL};

  ThermalConductivity(Teuchos::ParameterList& p);

  void postRegistrationSetup(typename Traits::SetupData d,
           PHX::FieldManager<Traits>& vm);

  void evaluateFields(typename Traits::EvalData d);

  ScalarT& getValue(const std::string &n);

private:

//! Validate the name strings under "ThermalConductivity" section in xml input file,
  Teuchos::RCP<const Teuchos::ParameterList>
               getValidThermalCondParameters() const;

  bool is_constant;

  std::size_t numQPs;
  std::size_t numDims;
  PHX::MDField<const MeshScalarT,Cell,QuadPoint,Dim> coordVec;
  PHX::MDField<ScalarT,Cell,QuadPoint> thermalCond;

  //! Conductivity type
  std::string type;

  //! Constant value
  ScalarT constant_value;

  //! Values of the random variables
  Teuchos::Array<ScalarT> rv;

  //! Material database - holds thermal conductivity among other quantities
  Teuchos::RCP<Albany::MaterialDatabase> materialDB;

  //! Convenience function to initialize constant thermal conductivity
  void init_constant(ScalarT value, Teuchos::ParameterList& p);

  //! Convenience function to initialize thermal conductivity based on
  //  Truncated KL Expansion || Log Normal RF
  void init_KL_RF(std::string &type, Teuchos::ParameterList& subList, Teuchos::ParameterList& p);

  SG_RF randField;

};
}

#endif
