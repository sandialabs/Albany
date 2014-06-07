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

namespace LCM {

template<typename EvalT, typename Traits>
PoissonsRatio<EvalT, Traits>::
PoissonsRatio(Teuchos::ParameterList& p) :
  poissonsRatio(p.get<std::string>("QP Variable Name"),
      p.get<Teuchos::RCP<PHX::DataLayout> >("QP Scalar Data Layout"))
{
  Teuchos::ParameterList* pr_list = 
    p.get<Teuchos::ParameterList*>("Parameter List");
  
  Teuchos::RCP<ParamLib> paramLib = 
    p.get< Teuchos::RCP<ParamLib> >("Parameter Library", Teuchos::null);

  Teuchos::RCP<PHX::DataLayout> vector_dl =
    p.get< Teuchos::RCP<PHX::DataLayout> >("QP Vector Data Layout");
  std::vector<PHX::DataLayout::size_type> dims;
  vector_dl->dimensions(dims);
  numQPs  = dims[1];
  numDims = dims[2];

  std::string type = pr_list->get("Poissons Ratio Type", "Constant");
  if (type == "Constant") {
    is_constant = true;
    constant_value = pr_list->get<double>("Value");

    // Add Poissons Ratio as a Sacado-ized parameter
    new Sacado::ParameterRegistration<EvalT, SPL_Traits>(
      "Poissons Ratio", this, paramLib);
  }
  else if (type == "Truncated KL Expansion") {
    is_constant = false;
    PHX::MDField<MeshScalarT,Cell,QuadPoint,Dim>
      fx(p.get<std::string>("QP Coordinate Vector Name"), vector_dl);
    coordVec = fx;
    this->addDependentField(coordVec);

    exp_rf_kl = 
      Teuchos::rcp(new Stokhos::KL::ExponentialRandomField<MeshScalarT>(*pr_list));
    int num_KL = exp_rf_kl->stochasticDimension();

    // Add KL random variables as Sacado-ized parameters
    rv.resize(num_KL);
    for (int i=0; i<num_KL; i++) {
      std::string ss = Albany::strint("Poissons Ratio KL Random Variable",i);
      new Sacado::ParameterRegistration<EvalT, SPL_Traits>(ss, this, paramLib);
      rv[i] = pr_list->get(ss, 0.0);
    }
  }
  else {
    TEUCHOS_TEST_FOR_EXCEPTION(true, Teuchos::Exceptions::InvalidParameter,
			       "Invalid Poissons ratio type " << type);
  } 

  // Optional dependence on Temperature (nu = nu_ + dnudT * T)
  // Switched ON by sending Temperature field in p

  if ( p.isType<std::string>("QP Temperature Name") ) {
    Teuchos::RCP<PHX::DataLayout> scalar_dl =
      p.get< Teuchos::RCP<PHX::DataLayout> >("QP Scalar Data Layout");
    PHX::MDField<ScalarT,Cell,QuadPoint>
      tmp(p.get<std::string>("QP Temperature Name"), scalar_dl);
    Temperature = tmp;
    this->addDependentField(Temperature);
    isThermoElastic = true;
    dnudT_value = pr_list->get("dnudT Value", 0.0);
    refTemp = p.get<RealType>("Reference Temperature", 0.0);
    new Sacado::ParameterRegistration<EvalT, SPL_Traits>(
                                "dnudT Value", this, paramLib);
  }
  else {
    isThermoElastic=false;
    dnudT_value=0.0;
  }


  this->addEvaluatedField(poissonsRatio);
  this->setName("Poissons Ratio"+PHX::TypeString<EvalT>::value);
}

// **********************************************************************
template<typename EvalT, typename Traits>
void PoissonsRatio<EvalT, Traits>::
postRegistrationSetup(typename Traits::SetupData d,
                      PHX::FieldManager<Traits>& fm)
{
  this->utils.setFieldData(poissonsRatio,fm);
  if (!is_constant) this->utils.setFieldData(coordVec,fm);
  if (isThermoElastic) this->utils.setFieldData(Temperature,fm);
}

// **********************************************************************
template<typename EvalT, typename Traits>
void PoissonsRatio<EvalT, Traits>::
evaluateFields(typename Traits::EvalData workset)
{
  std::size_t numCells = workset.numCells;

  if (is_constant) {
    for (std::size_t cell=0; cell < numCells; ++cell) {
      for (std::size_t qp=0; qp < numQPs; ++qp) {
	poissonsRatio(cell,qp) = constant_value;
      }
    }
  }
  else {
    for (std::size_t cell=0; cell < numCells; ++cell) {
      for (std::size_t qp=0; qp < numQPs; ++qp) {
	Teuchos::Array<MeshScalarT> point(numDims);
	for (std::size_t i=0; i<numDims; i++)
	  point[i] = Sacado::ScalarValue<MeshScalarT>::eval(coordVec(cell,qp,i));
	poissonsRatio(cell,qp) = exp_rf_kl->evaluate(point, rv);
      }
    }
  }
  if (isThermoElastic) {
    for (std::size_t cell=0; cell < numCells; ++cell) {
      for (std::size_t qp=0; qp < numQPs; ++qp) {
        poissonsRatio(cell,qp) += dnudT_value * (Temperature(cell,qp) - refTemp);;
      }
    }
  }
}

// **********************************************************************
template<typename EvalT,typename Traits>
typename PoissonsRatio<EvalT,Traits>::ScalarT& 
PoissonsRatio<EvalT,Traits>::getValue(const std::string &n)
{
  if (n=="Poissons Ratio")
    return constant_value;
  else if (n == "dnudT Value")
    return dnudT_value;
  for (int i=0; i<rv.size(); i++) {
    if (n == Albany::strint("Poissons Ratio KL Random Variable",i))
      return rv[i];
  }
  TEUCHOS_TEST_FOR_EXCEPTION(true, Teuchos::Exceptions::InvalidParameter,
			     std::endl <<
			     "Error! Logic error in getting paramter " << n
			     << " in PoissonsRatio::getValue()" << std::endl);
  return constant_value;
}

// **********************************************************************
// **********************************************************************
}

