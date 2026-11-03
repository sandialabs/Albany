//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#include "Phalanx_DataLayout.hpp"
#include "Phalanx_Print.hpp"

#include "LandIce_FlowRate.hpp"

namespace LandIce {

//**********************************************************************
template<typename EvalT, typename Traits, typename TempST>
FlowRate<EvalT, Traits, TempST>::
FlowRate (const Teuchos::ParameterList& p,
          const Teuchos::RCP<Albany::Layouts>& dl) :
  flowRate (p.get<std::string> ("Flow Rate Variable Name"), dl->cell_scalar2)
{
  Teuchos::ParameterList* visc_list = p.get<Teuchos::ParameterList*>("Parameter List");

  std::string flowRateType;
  if(visc_list->isParameter("Flow Rate Type")) {
    flowRateType = visc_list->get<std::string>("Flow Rate Type");
  } else {
    flowRateType = "Uniform";
  }

  if (flowRateType == "Uniform") {
    flowRate_type = UNIFORM;
    A = visc_list->get<double>("Glen's Law A");
#ifdef OUTPUT_TO_SCREEN
    *out << "Uniform Flow Rate A: " << A << std::endl;
#endif
  } else if (flowRateType == "Given Field") {
    flowRate_type = GIVEN_FIELD;
    given_flow_rate = decltype(given_flow_rate)(p.get<std::string> ("Given Flow Rate Field Name"), dl->cell_scalar2);
    this->addDependentField(given_flow_rate);
#ifdef OUTPUT_TO_SCREEN
    *out << "Flow Rate read in from file (exodus or ascii) or passed in from CISM." << std::endl;
#endif
  } else if (flowRateType == "Temperature Based") {
    flowRate_type = TEMPERATURE_BASED;
    temperature = decltype(temperature)(p.get<std::string> ("Temperature Variable Name"), dl->cell_scalar2);
    this->addDependentField(temperature);

    // Same parameters and defaults as LandIce::ViscosityFO
    auto& arrhenius_list = visc_list->sublist("Arrhenius Relationship");
    actenh     = arrhenius_list.get("Q+",1.39e5);
    actenl     = arrhenius_list.get("Q-",6.0e4);
    gascon     = arrhenius_list.get("Gas Constant",8.314);
    switchingT = arrhenius_list.get("T*",263.15);
    arrmlh     = arrhenius_list.get("A+",1.733e3);
    arrmll     = arrhenius_list.get("A-",3.613e-13);
#ifdef OUTPUT_TO_SCREEN
    *out << "Flow Rate computed using temperature field." << std::endl;
#endif
  } else {
    TEUCHOS_TEST_FOR_EXCEPTION(true, Teuchos::Exceptions::InvalidParameter,
      std::endl << "Error in LandIce::ViscosityFO:  \"" << flowRateType << "\" is not a valid parameter for Flow Rate Type" << std::endl);
  }

  this->addEvaluatedField(flowRate);

  this->setName("FlowRate"+PHX::print<EvalT>());
}

template<typename EvalT, typename Traits, typename TempST>
void FlowRate<EvalT,Traits,TempST>::
postRegistrationSetup(typename Traits::SetupData d,
                      PHX::FieldManager<Traits>& /* fm */)
{
  d.fill_field_dependencies(this->dependentFields(),this->evaluatedFields());
}

//**********************************************************************
template<typename EvalT, typename Traits, typename TempST>
void FlowRate<EvalT, Traits, TempST>::evaluateFields (typename Traits::EvalData workset)
{
  switch (flowRate_type) {
    case UNIFORM:
      for (unsigned int cell=0; cell<workset.numCells; ++cell)
        flowRate(cell) = A;
      break;

    case GIVEN_FIELD:
      for (unsigned int cell=0; cell<workset.numCells; ++cell)
        flowRate(cell) = given_flow_rate(cell);
      break;

    case TEMPERATURE_BASED: {
      for (unsigned int cell=0; cell<workset.numCells; ++cell)
        flowRate(cell) = (temperature(cell) < switchingT) ? arrmll / std::exp (actenl / gascon / temperature(cell))
                                                   : arrmlh / std::exp (actenh / gascon / temperature(cell));
      break;
    }

    default:
      TEUCHOS_TEST_FOR_EXCEPTION (true, std::logic_error, "Error! Invalid flow rate type. However, you should have got an error before...\n");
  }
}

} // Namespace LandIce
