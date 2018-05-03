//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#include "FELIX_ParamEnum.hpp"

#include "Phalanx_DataLayout.hpp"
#include "Phalanx_TypeStrings.hpp"

namespace FELIX {

//**********************************************************************
template<typename EvalT, typename Traits, bool ThermoCoupled>
FlowFactorA<EvalT, Traits, ThermoCoupled>::FlowFactorA (const Teuchos::ParameterList& p,
                                         const Teuchos::RCP<Albany::Layouts>& dl) :
  flowFactor (p.get<std::string> ("Flow Factor A Variable Name"), dl->cell_scalar2)
{
  std::string flowFactorType;
  if(p.isParameter("Flow Rate Type"))
    flowFactorType = p.get<std::string>("Flow Rate Type");
  else
    flowFactorType = "Uniform";

  if (flowFactorType == "Uniform")
  {
    flowFactor_type = UNIFORM;
    flowFactorParam = PHX::MDField<const ScalarT, Dim>(ParamEnumName::FlowFactorA,dl->shared_param);

    this->addDependentField(flowFactorParam);
    this->addEvaluatedField(flowFactor);

#ifdef OUTPUT_TO_SCREEN
    *out << "Uniform Flow Factor A" << std::endl;
#endif
  }
  else if (flowFactorType == "Given Field")
  {
    flowFactor_type = GIVEN_FIELD;
    given_flow_factor = decltype(given_flow_factor)(p.get<std::string> ("Given Flow Factor Field Name"), dl->cell_scalar2);
    this->addDependentField(given_flow_factor);
#ifdef OUTPUT_TO_SCREEN
    *out << "Flow factor read in from file (exodus or ascii) or passed in from CISM." << std::endl;
#endif
  }
  else if (flowFactorType == "Temperature Based")
  {
    flowFactor_type = TEMPERATURE_BASED;
    temperature = decltype(temperature)(p.get<std::string> ("Temperature Variable Name"), dl->cell_scalar2);
    this->addDependentField(temperature);
    this->addEvaluatedField(flowFactor);
#ifdef OUTPUT_TO_SCREEN
    *out << "Flow factor computed using temperature field." << std::endl;
#endif
  }
  else {
    TEUCHOS_TEST_FOR_EXCEPTION(true, Teuchos::Exceptions::InvalidParameter,
      std::endl << "Error in FELIX::FlowFactorA:  \"" << flowFactorType << "\" is not a valid parameter for Flow Rate Type" << std::endl);
  }

  this->setName("FlowFactorA"+PHX::typeAsString<EvalT>());
}

//**********************************************************************
template<typename EvalT, typename Traits, bool ThermoCoupled>
void FlowFactorA<EvalT, Traits, ThermoCoupled>::
postRegistrationSetup(typename Traits::SetupData d,
                      PHX::FieldManager<Traits>& fm)
{
  switch (flowFactor_type)
  {
    case UNIFORM:
      this->utils.setFieldData(flowFactorParam,fm);
      break;
    case TEMPERATURE_BASED:
      this->utils.setFieldData(temperature,fm);
      break;
    case GIVEN_FIELD:
      this->utils.setFieldData(given_flow_factor,fm);
      break;
    default:
      ; //nothing to do
  }

  this->utils.setFieldData(flowFactor,fm);
}

//**********************************************************************
template<typename EvalT, typename Traits, bool ThermoCoupled>
void FlowFactorA<EvalT, Traits, ThermoCoupled>::
evaluateFields (typename Traits::EvalData workset)
{
  TempScalarT A;
  switch (flowFactor_type)
  {
    case UNIFORM:
      A = Albany::convertScalar<ScalarT,TempScalarT>(flowFactorParam(0));
      for (int cell=0; cell<workset.numCells; ++cell)
        flowFactor(cell) = A;
      break;

    case GIVEN_FIELD:
      for (int cell=0; cell<workset.numCells; ++cell)
        flowFactor(cell) = given_flow_factor(cell);
      break;

    case TEMPERATURE_BASED:
      for (int cell=0; cell<workset.numCells; ++cell)
        flowFactor(cell) = (temperature(cell) < 263) ? 1.3e7 / std::exp (6.0e4 / 8.314 / temperature(cell))
                                                   : 6.26e22 / std::exp (1.39e5 / 8.314 / temperature(cell));
      break;

    default:
      TEUCHOS_TEST_FOR_EXCEPTION (true, std::logic_error, "Error! Invalid flow rate type. However, you should have got an error before...\n");
  }
}

} // Namespace FELIX
