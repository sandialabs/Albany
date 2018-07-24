//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#include "LandIce_ParamEnum.hpp"

#include "Phalanx_DataLayout.hpp"
#include "Phalanx_TypeStrings.hpp"

namespace LandIce {

//**********************************************************************
template<typename EvalT, typename Traits, bool ThermoCoupled>
IceSoftness<EvalT, Traits, ThermoCoupled>::IceSoftness (const Teuchos::ParameterList& p,
                                         const Teuchos::RCP<Albany::Layouts>& dl) :
  ice_softness (p.get<std::string> ("Ice Softness Variable Name"), dl->cell_scalar2)
{
  std::string ice_softnessType;
  if(p.isParameter("Ice Softness Type")) {
    ice_softnessType = p.get<std::string>("Ice Softness Type");
  } else {
    ice_softnessType = "Uniform";
  }

  if (ice_softnessType == "Uniform")
  {
    ice_softness_type = UNIFORM;
    A = p.get<Teuchos::ParameterList*>("LandIce Physical Parameters")->get<double>("Ice Softness");

#ifdef OUTPUT_TO_SCREEN
    *out << "Uniform Ice Softness" << std::endl;
#endif
  }
  else if (ice_softnessType == "Given Field")
  {
    ice_softness_type = GIVEN_FIELD;
    given_ice_softness = decltype(given_ice_softness)(p.get<std::string> ("Given Ice Softness Field Name"), dl->cell_scalar2);
    this->addDependentField(given_ice_softness);
#ifdef OUTPUT_TO_SCREEN
    *out << "Ice softness read in from file (exodus or ascii) or passed in from CISM." << std::endl;
#endif
  }
  else if (ice_softnessType == "Temperature Based")
  {
    ice_softness_type = TEMPERATURE_BASED;
    temperature = decltype(temperature)(p.get<std::string> ("Temperature Variable Name"), dl->cell_scalar2);
    this->addDependentField(temperature);
#ifdef OUTPUT_TO_SCREEN
    *out << "Ice softness computed using temperature field." << std::endl;
#endif
  }
  else {
    TEUCHOS_TEST_FOR_EXCEPTION(true, Teuchos::Exceptions::InvalidParameter,
      std::endl << "Error in LandIce::IceSoftness:  \"" << ice_softnessType << "\" is not a valid parameter for Ice Softness Type" << std::endl);
  }

  this->addEvaluatedField(ice_softness);

  this->setName("IceSoftness"+PHX::typeAsString<EvalT>());
}

//**********************************************************************
template<typename EvalT, typename Traits, bool ThermoCoupled>
void IceSoftness<EvalT, Traits, ThermoCoupled>::
postRegistrationSetup(typename Traits::SetupData d,
                      PHX::FieldManager<Traits>& fm)
{
  switch (ice_softness_type)
  {
    case UNIFORM:
      break;
    case TEMPERATURE_BASED:
      this->utils.setFieldData(temperature,fm);
      break;
    case GIVEN_FIELD:
      this->utils.setFieldData(given_ice_softness,fm);
      break;
    default:
      ; //nothing to do
  }

  this->utils.setFieldData(ice_softness,fm);
}

//**********************************************************************
template<typename EvalT, typename Traits, bool ThermoCoupled>
void IceSoftness<EvalT, Traits, ThermoCoupled>::
evaluateFields (typename Traits::EvalData workset)
{
  switch (ice_softness_type)
  {
    case UNIFORM:
      for (int cell=0; cell<workset.numCells; ++cell)
        ice_softness(cell) = A;
      break;

    case GIVEN_FIELD:
      for (int cell=0; cell<workset.numCells; ++cell)
        ice_softness(cell) = given_ice_softness(cell);
      break;

    case TEMPERATURE_BASED:
      for (int cell=0; cell<workset.numCells; ++cell)
        ice_softness(cell) = (temperature(cell) < 263) ? 1.3e7 / std::exp (6.0e4 / 8.314 / temperature(cell))
                                                   : 6.26e22 / std::exp (1.39e5 / 8.314 / temperature(cell));
      break;

    default:
      TEUCHOS_TEST_FOR_EXCEPTION (true, std::logic_error, "Error! Invalid ice softness type. However, you should have got an error before...\n");
  }
}

} // Namespace LandIce
