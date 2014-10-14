//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#include <fstream>
#include "Teuchos_TestForException.hpp"
#include "Phalanx_DataLayout.hpp"
#include "Sacado_ParameterRegistration.hpp"
#include "Albany_Utils.hpp"

//Radom field types
/*
enum SG_RF {CONSTANT, UNIFORM, LOGNORMAL};
const int num_sg_rf = 3;
const SG_RF sg_rf_values[] = {CONSTANT, UNIFORM, LOGNORMAL};
const char *sg_rf_names[] = {"Constant", "Uniform", "Log-Normal"};

SG_RF randField = CONSTANT;
*/

namespace PHAL {

template<typename EvalT, typename Traits>
Permittivity<EvalT, Traits>::
Permittivity(Teuchos::ParameterList& p) :
  permittivity(p.get<std::string>("QP Variable Name"),
              p.get<Teuchos::RCP<PHX::DataLayout> >("QP Scalar Data Layout"))
{

  randField = CONSTANT;

  Teuchos::ParameterList* cond_list =
    p.get<Teuchos::ParameterList*>("Parameter List");

  Teuchos::RCP<const Teuchos::ParameterList> reflist =
    this->getValidPermittivityParameters();

  // Check the parameters contained in the input file. Do not check the defaults
  // set programmatically
  cond_list->validateParameters(*reflist, 0,
    Teuchos::VALIDATE_USED_ENABLED, Teuchos::VALIDATE_DEFAULTS_DISABLED);

  Teuchos::RCP<PHX::DataLayout> vector_dl =
    p.get< Teuchos::RCP<PHX::DataLayout> >("QP Vector Data Layout");
  std::vector<PHX::DataLayout::size_type> dims;
  vector_dl->dimensions(dims);
  numQPs  = dims[1];
  numDims = dims[2];

  std::string ebName =
    p.get<std::string>("Element Block Name", "Missing");

  type = cond_list->get("Permittivity Type", "Constant");

  if (type == "Constant") {

    ScalarT value = cond_list->get("Value", 1.0);
    init_constant(value, p);

  }

  else if (type == "Truncated KL Expansion" || type == "Log Normal RF") {

    init_KL_RF(type, *cond_list, p);

  }

  else if (type == "Block Dependent")
  {
    // We have a multiple material problem and need to map element blocks to material data

    if(p.isType<Teuchos::RCP<QCAD::MaterialDatabase> >("MaterialDB")){
       materialDB = p.get< Teuchos::RCP<QCAD::MaterialDatabase> >("MaterialDB");
    }
    else {
       TEUCHOS_TEST_FOR_EXCEPTION(
         true, Teuchos::Exceptions::InvalidParameter,
         std::endl <<
         "Error! Must specify a material database if using block dependent " <<
         "permittivity" << std::endl);
    }

    // Get the sublist for permittivity for the element block in the mat DB (the material in the
    // elem block ebName.

    Teuchos::ParameterList& subList = materialDB->getElementBlockSublist(ebName, "Permittivity");

    std::string typ = subList.get("Permittivity Type", "Constant");

    if (typ == "Constant") {

       ScalarT value = subList.get("Value", 1.0);
       init_constant(value, p);

    }
    else if (typ == "Truncated KL Expansion" || typ == "Log Normal RF") {

       init_KL_RF(typ, subList, p);

    }
  } // Block dependent

  else {
    TEUCHOS_TEST_FOR_EXCEPTION(true, Teuchos::Exceptions::InvalidParameter,
                       "Invalid permittivity type " << type);
  }

  this->addEvaluatedField(permittivity);
  this->setName("Permittivity"+PHX::TypeString<EvalT>::value);
}

template<typename EvalT, typename Traits>
void
Permittivity<EvalT, Traits>::
init_constant(ScalarT value, Teuchos::ParameterList& p){

    is_constant = true;
    randField = CONSTANT;

    constant_value = value;

    // Add permittivity as a Sacado-ized parameter
    Teuchos::RCP<ParamLib> paramLib =
      p.get< Teuchos::RCP<ParamLib> >("Parameter Library", Teuchos::null);

    new Sacado::ParameterRegistration<EvalT, SPL_Traits>(
      "Permittivity", this, paramLib);

} // init_constant

template<typename EvalT, typename Traits>
void
Permittivity<EvalT, Traits>::
init_KL_RF(std::string &type, Teuchos::ParameterList& sublist, Teuchos::ParameterList& p){

    is_constant = false;

    if (type == "Truncated KL Expansion")
      randField = UNIFORM;
    else if (type == "Log Normal RF")
      randField = LOGNORMAL;

    Teuchos::RCP<PHX::DataLayout> scalar_dl =
      p.get< Teuchos::RCP<PHX::DataLayout> >("QP Scalar Data Layout");
    Teuchos::RCP<PHX::DataLayout> vector_dl =
      p.get< Teuchos::RCP<PHX::DataLayout> >("QP Vector Data Layout");
    PHX::MDField<MeshScalarT,Cell,QuadPoint,Dim>
      fx(p.get<std::string>("QP Coordinate Vector Name"), vector_dl);
    coordVec = fx;
    this->addDependentField(coordVec);

    exp_rf_kl =
      Teuchos::rcp(new Stokhos::KL::ExponentialRandomField<MeshScalarT>(sublist));
    int num_KL = exp_rf_kl->stochasticDimension();

    // Add KL random variables as Sacado-ized parameters
    rv.resize(num_KL);
    Teuchos::RCP<ParamLib> paramLib =
      p.get< Teuchos::RCP<ParamLib> >("Parameter Library", Teuchos::null);
    for (int i=0; i<num_KL; i++) {
      std::string ss = Albany::strint("Permittivity KL Random Variable",i);
      new Sacado::ParameterRegistration<EvalT, SPL_Traits>(ss, this, paramLib);
      rv[i] = sublist.get(ss, 0.0);
    }

} // (type == "Truncated KL Expansion" || type == "Log Normal RF")

// **********************************************************************
template<typename EvalT, typename Traits>
void Permittivity<EvalT, Traits>::
postRegistrationSetup(typename Traits::SetupData d,
                      PHX::FieldManager<Traits>& fm)
{
  this->utils.setFieldData(permittivity,fm);
  if (!is_constant)
      this->utils.setFieldData(coordVec,fm);
}

// **********************************************************************
template<typename EvalT, typename Traits>
void Permittivity<EvalT, Traits>::
evaluateFields(typename Traits::EvalData workset)
{
  if (is_constant) {
    for (std::size_t cell=0; cell < workset.numCells; ++cell) {
      for (std::size_t qp=0; qp < numQPs; ++qp) {
         permittivity(cell,qp) = constant_value;
      }
    }
  }

  else {
    for (std::size_t cell=0; cell < workset.numCells; ++cell) {
      for (std::size_t qp=0; qp < numQPs; ++qp) {
          Teuchos::Array<MeshScalarT> point(numDims);
          for (std::size_t i=0; i<numDims; i++)
              point[i] = Sacado::ScalarValue<MeshScalarT>::eval(coordVec(cell,qp,i));
          if (randField == UNIFORM)
              permittivity(cell,qp) = exp_rf_kl->evaluate(point, rv);
          else if (randField == LOGNORMAL)
              permittivity(cell,qp) = std::exp(exp_rf_kl->evaluate(point, rv));
      }
    }
  }
}

// **********************************************************************
template<typename EvalT,typename Traits>
typename Permittivity<EvalT,Traits>::ScalarT&
Permittivity<EvalT,Traits>::getValue(const std::string &n)
{
  if (is_constant) {
    return constant_value;
  }

  for (int i=0; i<rv.size(); i++) {
    if (n == Albany::strint("Permittivity KL Random Variable",i))
      return rv[i];
  }
  TEUCHOS_TEST_FOR_EXCEPTION(true, Teuchos::Exceptions::InvalidParameter,
                     std::endl <<
                     "Error! Logic error in getting paramter " << n
                     << " in Permittivity::getValue()" << std::endl);
  return constant_value;
}

// **********************************************************************
template<typename EvalT,typename Traits>
Teuchos::RCP<const Teuchos::ParameterList>
Permittivity<EvalT,Traits>::getValidPermittivityParameters() const
{
  Teuchos::RCP<Teuchos::ParameterList> validPL =
       rcp(new Teuchos::ParameterList("Valid Permittivity Params"));;

  validPL->set<std::string>("Permittivity Type", "Constant",
               "Constant permittivity across the entire domain");
  validPL->set<double>("Value", 1.0, "Constant permittivity value");

// Truncated KL parameters

  validPL->set<int>("Number of KL Terms", 2, "");
  validPL->set<double>("Mean", 0.2, "");
  validPL->set<double>("Standard Deviation", 0.1, "");
  validPL->set<std::string>("Domain Lower Bounds", "{0.0 0.0}", "");
  validPL->set<std::string>("Domain Upper Bounds", "{1.0 1.0}", "");
  validPL->set<std::string>("Correlation Lengths", "{1.0 1.0}", "");
  return validPL;
}

// **********************************************************************
// **********************************************************************
}
