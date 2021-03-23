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
      constexpr double actenh (1.39e5);       // [J mol-1]
      constexpr double actenl (6.0e4);        // [J mol-1]
      constexpr double gascon (8.314);        // [J mol-1 K-1]
      constexpr double switchingT (263.15);   // [K]
      constexpr double arrmlh (1.733e3);      // [Pa-3 s-1]
      constexpr double arrmll (3.613e-13);    // [Pa-3 s-1]
      constexpr double k4scyr (3.1536e19);    // [s y-1]
      constexpr double arrmh (k4scyr*arrmlh); // [Pa-3 yr-1]
      constexpr double arrml (k4scyr*arrmll); // [Pa-3 yr-1]

      for (unsigned int cell=0; cell<workset.numCells; ++cell)
        flowRate(cell) = (temperature(cell) < switchingT) ? arrml / std::exp (actenl / gascon / temperature(cell))
                                                   : arrmh / std::exp (actenh / gascon / temperature(cell));
      break;
    }

    default:
      TEUCHOS_TEST_FOR_EXCEPTION (true, std::logic_error, "Error! Invalid flow rate type. However, you should have got an error before...\n");
  }
}

} // Namespace LandIce
