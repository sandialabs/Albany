//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#include <fstream>
#include "Teuchos_TestForException.hpp"
#include "Phalanx_DataLayout.hpp"
#include "Sacado_ParameterRegistration.hpp"
#include "Albany_Utils.hpp"

namespace PHAL {

template<typename EvalT, typename Traits>
JThermConductivity<EvalT, Traits>::
JThermConductivity(Teuchos::ParameterList& p,
      const Teuchos::RCP<Albany::Layouts>& dl) :
  dl_(dl),

  Temperature (p.get<std::string>                   ("Temperature Name"), dl->qp_scalar),
  thermalCond(p.get<std::string>("QP Variable Name"), dl->qp_scalar)

{

  Teuchos::ParameterList* cond_list = 
    p.get<Teuchos::ParameterList*>("Parameter List");

  Teuchos::RCP<const Teuchos::ParameterList> reflist = 
               this->getValidJThermCondParameters();

  // Check the parameters contained in the input file. Do not check the defaults
  // set programmatically
  cond_list->validateParameters(*reflist, 0, 
    Teuchos::VALIDATE_USED_ENABLED, Teuchos::VALIDATE_DEFAULTS_DISABLED);

  std::vector<PHX::DataLayout::size_type> dims;
  dl_->qp_vector->dimensions(dims);
  numQPs  = dims[1];
  numDims = dims[2];

  std::string ebName = 
    p.get<std::string>("Element Block Name", "Missing");

  type = cond_list->get("Material Parameters Type", "Block Dependent");

  if (type == "Block Dependent") 
  {
    // We have a multiple material problem and need to map element blocks to material data

    if(p.isType<Teuchos::RCP<Albany::MaterialDatabase> >("MaterialDB")){
       materialDB = p.get< Teuchos::RCP<Albany::MaterialDatabase> >("MaterialDB");
    }
    else {
       TEUCHOS_TEST_FOR_EXCEPTION(true, Teuchos::Exceptions::InvalidParameter,
		     std::endl <<
		     "Error! Must specify a material database if using block dependent " << 
		     "material properties" << std::endl);
    }

    // Get the sublist for thermal conductivity for the element block in the mat DB (the material in the
    // elem block ebName.
    {

    Teuchos::ParameterList& subList = materialDB->getElementBlockSublist(ebName, "Q_h");

    std::string typ = subList.get("Q_h Type", "Constant");

    if (typ == "Constant") {

       Qh = subList.get("Value", 0.0);

    }
    }
    {

    Teuchos::ParameterList& subList = materialDB->getElementBlockSublist(ebName, "R");

    std::string typ = subList.get("R Type", "Constant");

    if (typ == "Constant") {

       R = subList.get("Value", 0.0);

    }
    }
    {

    Teuchos::ParameterList& subList = materialDB->getElementBlockSublist(ebName, "C_{H,Tot}");

    std::string typ = subList.get("C_{H,Tot} Type", "Constant");

    if (typ == "Constant") {

       Cht = subList.get("Value", 0.0);

    }
    }
    {

    Teuchos::ParameterList& subList = materialDB->getElementBlockSublist(ebName, "Vbar");

    std::string typ = subList.get("Vbar Type", "Constant");

    if (typ == "Constant") {

       Vbar = subList.get("Value", 0.0);

    }
    }
  } // Block dependent

  else {
    TEUCHOS_TEST_FOR_EXCEPTION(true, Teuchos::Exceptions::InvalidParameter,
		       "Must specify material parameters in the material database" << type);
  } 

  this->addEvaluatedField(thermalCond);
  this->addDependentField(Temperature);
  this->setName("JTherm Conductivity"+PHX::typeAsString<EvalT>());
}

// **********************************************************************
template<typename EvalT, typename Traits>
void JThermConductivity<EvalT, Traits>::
postRegistrationSetup(typename Traits::SetupData d,
                      PHX::FieldManager<Traits>& fm)
{
  this->utils.setFieldData(thermalCond,fm);
  this->utils.setFieldData(Temperature,fm);
}

// **********************************************************************
template<typename EvalT, typename Traits>
void JThermConductivity<EvalT, Traits>::
evaluateFields(typename Traits::EvalData workset)
{

/*
  Here we calculate the qp values for the multiplier to the Grad T term, i.e.

  [ R \log{(C_{H, Tot} \bar{V})} - Q_H / T ]

  The result of this eval at each qp goes into thermalCond()
*/

    for (std::size_t cell=0; cell < workset.numCells; ++cell) {
      for (std::size_t qp=0; qp < numQPs; ++qp) {
         thermalCond(cell, qp) = R * std::log(Cht * Vbar) - Qh / Temperature(cell, qp);
      }
    }

}

// **********************************************************************
template<typename EvalT,typename Traits>
Teuchos::RCP<const Teuchos::ParameterList>
JThermConductivity<EvalT,Traits>::getValidJThermCondParameters() const
{
  Teuchos::RCP<Teuchos::ParameterList> validPL =
       rcp(new Teuchos::ParameterList("Valid JTherm Conductivity Params"));;

  validPL->set<std::string>("Qh Type", "Constant", 
               "Constant Qh across the entire domain");
  validPL->set<std::string>("R Type", "Constant", 
               "Constant R across the entire domain");
  validPL->set<std::string>("C_{H,Tot} Type", "Constant", 
               "Constant C_{H,Tot} across the entire domain");
  validPL->set<std::string>("Vbar Type", "Constant", 
               "Constant Vbar across the entire domain");
  validPL->set<double>("Value", 1.0, "Constant material parameter value");

  return validPL;
}

// **********************************************************************
// **********************************************************************
}

