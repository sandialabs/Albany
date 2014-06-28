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
#include <typeinfo>

namespace LCM {

template<typename EvalT, typename Traits>
EquivalentInclusionConductivity<EvalT, Traits>::
EquivalentInclusionConductivity(Teuchos::ParameterList& p) :
  effectiveK(p.get<std::string>("QP Variable Name"),
		 p.get<Teuchos::RCP<PHX::DataLayout> >("QP Scalar Data Layout"))
{
  Teuchos::ParameterList* elmd_list = 
    p.get<Teuchos::ParameterList*>("Parameter List");

  Teuchos::RCP<PHX::DataLayout> vector_dl =
    p.get< Teuchos::RCP<PHX::DataLayout> >("QP Vector Data Layout");
  std::vector<PHX::DataLayout::size_type> dims;
  vector_dl->dimensions(dims);
  numQPs  = dims[1];
  numDims = dims[2];

  Teuchos::RCP<ParamLib> paramLib = 
    p.get< Teuchos::RCP<ParamLib> >("Parameter Library", Teuchos::null);

  std::string type = elmd_list->get("Effective Thermal Conductivity Type", "Constant");
  if (type == "Constant") {
    is_constant = true;
    constant_value = elmd_list->get("Value", 1.0);

    // Add effective thermal conductivity as a Sacado-ized parameter
    new Sacado::ParameterRegistration<EvalT, SPL_Traits>(
	"Effective Thermal COnductivity", this, paramLib);
  }
  else if (type == "Truncated KL Expansion") {
    is_constant = false;
    PHX::MDField<MeshScalarT,Cell,QuadPoint,Dim>
      fx(p.get<std::string>("QP Coordinate Vector Name"), vector_dl);
    coordVec = fx;
    this->addDependentField(coordVec);

    exp_rf_kl = 
      Teuchos::rcp(new Stokhos::KL::ExponentialRandomField<MeshScalarT>(*elmd_list));
    int num_KL = exp_rf_kl->stochasticDimension();

    // Add KL random variables as Sacado-ized parameters
    rv.resize(num_KL);
    for (int i=0; i<num_KL; i++) {
      std::string ss = Albany::strint("Effective Thermal Conductivity KL Random Variable",i);
      new Sacado::ParameterRegistration<EvalT, SPL_Traits>(ss, this, paramLib);
      rv[i] = elmd_list->get(ss, 1.0);
    }
  }
  else {
    TEUCHOS_TEST_FOR_EXCEPTION(true, Teuchos::Exceptions::InvalidParameter,
			       "Invalid effective thermal conductivity type " << type);
  } 


  if ( p.isType<std::string>("Porosity Name") ) {
      Teuchos::RCP<PHX::DataLayout> scalar_dl =
        p.get< Teuchos::RCP<PHX::DataLayout> >("QP Scalar Data Layout");
      PHX::MDField<ScalarT,Cell,QuadPoint>
        tporo(p.get<std::string>("Porosity Name"), scalar_dl);
      porosity = tporo;
      this->addDependentField(porosity);
      isPoroElastic = true;
    }
    else {
      isPoroElastic= false;
    }

  if ( p.isType<std::string>("Jacobian Name") ) {
      Teuchos::RCP<PHX::DataLayout> scalar_dl =
        p.get< Teuchos::RCP<PHX::DataLayout> >("QP Scalar Data Layout");
      PHX::MDField<ScalarT,Cell,QuadPoint>
        detf(p.get<std::string>("Jacobian Name"), scalar_dl);
      J = detf;
      this->addDependentField(J);
      isPoroElastic = true;
    }
  else {
    isPoroElastic= false;
  }

  condKs = elmd_list->get("Solid Thermal Conducutivity Value", 1.0);
  new Sacado::ParameterRegistration<EvalT, SPL_Traits>(
                          "Solid Thermal Conductivity Value", this, paramLib);
  condKf = elmd_list->get("Fluid Thermal Conductivity Value", 100.0);
  new Sacado::ParameterRegistration<EvalT, SPL_Traits>(
                          "Fluid Thermal Conductivity Value", this, paramLib);

  this->addEvaluatedField(effectiveK);
  this->setName("Effective Thermal Conductivity"+PHX::TypeString<EvalT>::value);
}

// **********************************************************************
template<typename EvalT, typename Traits>
void EquivalentInclusionConductivity<EvalT, Traits>::
postRegistrationSetup(typename Traits::SetupData d,
                      PHX::FieldManager<Traits>& fm)
{
  this->utils.setFieldData(effectiveK,fm);
  if (!is_constant)    this->utils.setFieldData(coordVec,fm);
  if (isPoroElastic)   this->utils.setFieldData(porosity,fm);
  if (isPoroElastic)   this->utils.setFieldData(J,fm);
}

// **********************************************************************
template<typename EvalT, typename Traits>
void EquivalentInclusionConductivity<EvalT, Traits>::
evaluateFields(typename Traits::EvalData workset)
{
  std::size_t numCells = workset.numCells;

  if (is_constant) {
    for (std::size_t cell=0; cell < numCells; ++cell) {
      for (std::size_t qp=0; qp < numQPs; ++qp) {
    	  effectiveK(cell,qp) = constant_value;
      }
    }
  }
  else {
    for (std::size_t cell=0; cell < numCells; ++cell) {
      for (std::size_t qp=0; qp < numQPs; ++qp) {
	Teuchos::Array<MeshScalarT> point(numDims);
	for (std::size_t i=0; i<numDims; i++)
	  point[i] = Sacado::ScalarValue<MeshScalarT>::eval(coordVec(cell,qp,i));
	effectiveK(cell,qp) = exp_rf_kl->evaluate(point, rv);
      }
    }
  }


  if (isPoroElastic) {
      for (std::size_t cell=0; cell < numCells; ++cell) {
        for (std::size_t qp=0; qp < numQPs; ++qp) {
        	effectiveK(cell,qp) = porosity(cell,qp)*condKf +
        			              (J(cell,qp)-porosity(cell,qp))*condKs;
       // 	effectiveK(cell,qp) = condKf
       //  	                + ( J(cell,qp) - porosity(cell,qp))*
        //			              (condKs - condKf)*condKf/
        //			              ((condKs - condKf)*porosity(cell,qp)
        //			            	+ J(cell,qp)*condKf);
        }
      }
    }

}

// **********************************************************************
template<typename EvalT,typename Traits>
typename EquivalentInclusionConductivity<EvalT,Traits>::ScalarT&
EquivalentInclusionConductivity<EvalT,Traits>::getValue(const std::string &n)
{
  if (n == "Effective Thermal Conductivity")
    return constant_value;
  else if (n == "Fluid Thermal Conducutivity Value")
    return condKf;
  else if (n == "Solid Thermal Conducutivity Value")
    return condKs;
  for (int i=0; i<rv.size(); i++) {
    if (n == Albany::strint("Effective Thermal Conductivity KL Random Variable",i))
      return rv[i];
  }
  TEUCHOS_TEST_FOR_EXCEPTION(true, Teuchos::Exceptions::InvalidParameter,
			     std::endl <<
			     "Error! Logic error in getting paramter " << n
			     << " in EquivalentInclusionConductivity::getValue()" << std::endl);
  return constant_value;
}

// **********************************************************************
// **********************************************************************
}

