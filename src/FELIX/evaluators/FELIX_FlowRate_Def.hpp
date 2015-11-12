//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#include "Phalanx_DataLayout.hpp"
#include "Phalanx_TypeStrings.hpp"

namespace FELIX {

//**********************************************************************
template<typename EvalT, typename Traits>
FlowRate<EvalT, Traits>::FlowRate (const Teuchos::ParameterList& p,
                                   const Teuchos::RCP<Albany::Layouts>& dl) :
  flowRate (p.get<std::string> ("Flow Rate Variable Name"), dl->cell_scalar2)
{
  Teuchos::ParameterList* visc_list = p.get<Teuchos::ParameterList*>("Parameter List");

  std::string flowRateType;
  if(visc_list->isParameter("Flow Rate Type"))
    flowRateType = visc_list->get<std::string>("Flow Rate Type");
  else
    flowRateType = "Uniform";

  if (flowRateType == "Uniform")
  {
    flowRate_type = UNIFORM;
    A = visc_list->get<double>("Glen's Law A");
#ifdef OUTPUT_TO_SCREEN
    *out << "Uniform Flow Rate A: " << A << std::endl;
#endif
  }
  else if (flowRateType == "Given Field")
  {
    flowRate_type = GIVEN_FIELD;
    given_flow_rate = PHX::MDField<ScalarT,Cell>(p.get<std::string> ("Given Flow Rate Field Name"), dl->cell_scalar2);
    this->addDependentField(given_flow_rate);
#ifdef OUTPUT_TO_SCREEN
    *out << "Flow Rate read in from file (exodus or ascii) or passed in from CISM." << std::endl;
#endif
  }
  else if (flowRateType == "Temperature Based")
  {
    flowRate_type = TEMPERATURE_BASED;
    temperature = PHX::MDField<ScalarT,Cell>(p.get<std::string> ("Temperature Variable Name"), dl->cell_scalar2);
    this->addDependentField(temperature);
#ifdef OUTPUT_TO_SCREEN
    *out << "Flow Rate computed using temperature field." << std::endl;
#endif
  }
  else {
    TEUCHOS_TEST_FOR_EXCEPTION(true, Teuchos::Exceptions::InvalidParameter,
      std::endl << "Error in FELIX::ViscosityFO:  \"" << flowRateType << "\" is not a valid parameter for Flow Rate Type" << std::endl);
  }

  this->addEvaluatedField(flowRate);

  this->setName("FlowRate"+PHX::typeAsString<EvalT>());
}

//**********************************************************************
template<typename EvalT, typename Traits>
void FlowRate<EvalT, Traits>::
postRegistrationSetup(typename Traits::SetupData d,
                      PHX::FieldManager<Traits>& fm)
{
  switch (flowRate_type)
  {
    case TEMPERATURE_BASED:
      this->utils.setFieldData(temperature,fm);
      break;
    case GIVEN_FIELD:
      this->utils.setFieldData(given_flow_rate,fm);
  }

  this->utils.setFieldData(flowRate,fm);
}

//**********************************************************************
template<typename EvalT, typename Traits>
void FlowRate<EvalT, Traits>::evaluateFields (typename Traits::EvalData workset)
{
  switch (flowRate_type)
  {
    case UNIFORM:
      for (int cell=0; cell<workset.numCells; ++cell)
        flowRate(cell) = A;
      break;

    case GIVEN_FIELD:
      for (int cell=0; cell<workset.numCells; ++cell)
        flowRate(cell) = given_flow_rate(cell);
      break;

    case TEMPERATURE_BASED:
      for (int cell=0; cell<workset.numCells; ++cell)
        flowRate(cell) = (temperature(cell) < 263) ? 1.3e7 / std::exp (6.0e4 / 8.314 / temperature(cell))
                                                   : 6.26e22 / std::exp (1.39e5 / 8.314 / temperature(cell));
      break;

    default:
      TEUCHOS_TEST_FOR_EXCEPTION (true, std::logic_error, "Error! Invalid flow rate type. However, you should have got an error before...\n");
  }
}

} // Namespace FELIX
