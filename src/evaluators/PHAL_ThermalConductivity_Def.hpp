/********************************************************************\
*            Albany, Copyright (2010) Sandia Corporation             *
*                                                                    *
* Notice: This computer software was prepared by Sandia Corporation, *
* hereinafter the Contractor, under Contract DE-AC04-94AL85000 with  *
* the Department of Energy (DOE). All rights in the computer software*
* are reserved by DOE on behalf of the United States Government and  *
* the Contractor as provided in the Contract. You are authorized to  *
* use this computer software for Governmental purposes but it is not *
* to be released or distributed to the public. NEITHER THE GOVERNMENT*
* NOR THE CONTRACTOR MAKES ANY WARRANTY, EXPRESS OR IMPLIED, OR      *
* ASSUMES ANY LIABILITY FOR THE USE OF THIS SOFTWARE. This notice    *
* including this sentence must appear on any copies of this software.*
*    Questions to Andy Salinger, agsalin@sandia.gov                  *
\********************************************************************/


#include <fstream>
#include "Teuchos_TestForException.hpp"
#include "Phalanx_DataLayout.hpp"
#include "Sacado_ParameterRegistration.hpp"

template<typename EvalT, typename Traits>
ThermalConductivity<EvalT, Traits>::
ThermalConductivity(Teuchos::ParameterList& p) :
  thermalCond(p.get<std::string>("QP Variable Name"),
	      p.get<Teuchos::RCP<PHX::DataLayout> >("QP Scalar Data Layout"))
{
  Teuchos::ParameterList* cond_list = 
    p.get<Teuchos::ParameterList*>("Parameter List");

  Teuchos::RCP<PHX::DataLayout> vector_dl =
    p.get< Teuchos::RCP<PHX::DataLayout> >("QP Vector Data Layout");
  std::vector<PHX::DataLayout::size_type> dims;
  vector_dl->dimensions(dims);
  numQPs  = dims[1];
  numDims = dims[2];

  std::string type = cond_list->get("Thermal Conductivity Type", "Constant");
  if (type == "Constant") {
    is_constant = true;
    constant_value = cond_list->get("Value", 1.0);

    // Add thermal conductivity as a Sacado-ized parameter
    Teuchos::RCP<ParamLib> paramLib = 
      p.get< Teuchos::RCP<ParamLib> >("Parameter Library", Teuchos::null);
      new Sacado::ParameterRegistration<EvalT, SPL_Traits>(
	"Thermal Conductivity", this, paramLib);
  }
  else if (type == "Truncated KL Expansion") {
    is_constant = false;
    Teuchos::RCP<PHX::DataLayout> scalar_dl =
      p.get< Teuchos::RCP<PHX::DataLayout> >("QP Scalar Data Layout");
    PHX::MDField<MeshScalarT,Cell,QuadPoint,Dim>
      fx(p.get<string>("QP Coordinate Vector Name"), vector_dl);
    coordVec = fx;
    this->addDependentField(coordVec);

    exp_rf_kl = 
      Teuchos::rcp(new Stokhos::KL::ExponentialRandomField<MeshScalarT>(*cond_list));
    int num_KL = exp_rf_kl->stochasticDimension();

    // Add KL random variables as Sacado-ized parameters
    rv.resize(num_KL);
    Teuchos::RCP<ParamLib> paramLib = 
      p.get< Teuchos::RCP<ParamLib> >("Parameter Library", Teuchos::null);
    for (int i=0; i<num_KL; i++) {
      std::stringstream ss;
      ss << "Thermal Conductivity KL Random Variable " << i;
      new Sacado::ParameterRegistration<EvalT, SPL_Traits>(ss.str(), this, 
							   paramLib);
      rv[i] = cond_list->get(ss.str(), 0.0);
    }
  }
  else {
    TEST_FOR_EXCEPTION(true, Teuchos::Exceptions::InvalidParameter,
		       "Invalid thermal conductivity type " << type);
  } 

  this->addEvaluatedField(thermalCond);
  this->setName("Thermal Conductivity");
}

// **********************************************************************
template<typename EvalT, typename Traits>
void ThermalConductivity<EvalT, Traits>::
postRegistrationSetup(typename Traits::SetupData d,
                      PHX::FieldManager<Traits>& fm)
{
  this->utils.setFieldData(thermalCond,fm);
  if (!is_constant) this->utils.setFieldData(coordVec,fm);
}

// **********************************************************************
template<typename EvalT, typename Traits>
void ThermalConductivity<EvalT, Traits>::
evaluateFields(typename Traits::EvalData workset)
{
  int numCells = workset.numCells;

  if (is_constant) {
    for (std::size_t cell=0; cell < numCells; ++cell) {
      for (std::size_t qp=0; qp < numQPs; ++qp) {
	thermalCond(cell,qp) = constant_value;
      }
    }
  }
  else {
    for (std::size_t cell=0; cell < numCells; ++cell) {
      for (std::size_t qp=0; qp < numQPs; ++qp) {
	Teuchos::Array<MeshScalarT> point(numDims);
	for (std::size_t i=0; i<numDims; i++)
	  point[i] = Sacado::ScalarValue<MeshScalarT>::eval(coordVec(cell,qp,i));
	thermalCond(cell,qp) = exp_rf_kl->evaluate(point, rv);
      }
    }
  }
}

// **********************************************************************
template<typename EvalT,typename Traits>
typename ThermalConductivity<EvalT,Traits>::ScalarT& 
ThermalConductivity<EvalT,Traits>::getValue(const std::string &n)
{
  if (is_constant)
    return constant_value;
  for (int i=0; i<rv.size(); i++) {
    std::stringstream ss;
    ss << "Thermal Conductivity KL Random Variable " << i;
    if (n == ss.str())
      return rv[i];
  }
  TEST_FOR_EXCEPTION(true, Teuchos::Exceptions::InvalidParameter,
		     std::endl <<
		     "Error! Logic error in getting paramter " << n
		     << " in ThermalConductivity::getValue()" << std::endl);
  return constant_value;
}

// **********************************************************************
// **********************************************************************
