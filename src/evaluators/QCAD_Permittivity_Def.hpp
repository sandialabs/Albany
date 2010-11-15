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
QCAD::Permittivity<EvalT, Traits>::
Permittivity(Teuchos::ParameterList& p) :
  permittivity(p.get<std::string>("QP Variable Name"),
	      p.get<Teuchos::RCP<PHX::DataLayout> >("QP Scalar Data Layout")),
  is_constant(false), temp_dependent(false)
{
	  Teuchos::ParameterList* perm_list = 
    p.get<Teuchos::ParameterList*>("Parameter List");

  Teuchos::RCP<PHX::DataLayout> scalar_dl =
    p.get< Teuchos::RCP<PHX::DataLayout> >("QP Scalar Data Layout");
  Teuchos::RCP<PHX::DataLayout> vector_dl =
    p.get< Teuchos::RCP<PHX::DataLayout> >("QP Vector Data Layout");
  std::vector<PHX::DataLayout::size_type> dims;
  vector_dl->dimensions(dims);
  numQPs  = dims[1];
  numDims = dims[2];

  std::string type = perm_list->get("Permittivity Type", "Constant");
  if (type == "Constant") {
    is_constant = true;
    constant_value = perm_list->get("Value", 1.0);

    // Add Permittivity as a Sacado-ized parameter
    Teuchos::RCP<ParamLib> paramLib = 
      p.get< Teuchos::RCP<ParamLib> >("Parameter Library", Teuchos::null);
      new Sacado::ParameterRegistration<EvalT, SPL_Traits>(
	"Permittivity", this, paramLib);
  }
  else if (type == "Temperature Dependent") {
    temp_dependent = true;
    is_constant = false;
    constant_value = perm_list->get("Value", 1.0);
    factor = perm_list->get("Factor", 1.0);

    PHX::MDField<ScalarT,Cell,QuadPoint>
      tmp(p.get<string>("Temperature Variable Name"), scalar_dl);
    Temp = tmp;
    this->addDependentField(Temp);

    // Add Permittivity as a Sacado-ized parameter
    Teuchos::RCP<ParamLib> paramLib = 
      p.get< Teuchos::RCP<ParamLib> >("Parameter Library", Teuchos::null);
      new Sacado::ParameterRegistration<EvalT, SPL_Traits>(
	"Permittivity", this, paramLib);
      new Sacado::ParameterRegistration<EvalT, SPL_Traits>(
	"Permittivity Factor", this, paramLib);
  }
  // Parse for other functional form for permittivity variation here
  else {
    TEST_FOR_EXCEPTION(true, Teuchos::Exceptions::InvalidParameter,
		       "Invalid Permittivity type " << type);
  } 

  this->addEvaluatedField(permittivity);
  this->setName("Permittivity"+PHX::TypeString<EvalT>::value);
}

// **********************************************************************
template<typename EvalT, typename Traits>
void QCAD::Permittivity<EvalT, Traits>::
postRegistrationSetup(typename Traits::SetupData d,
                      PHX::FieldManager<Traits>& fm)
{
  this->utils.setFieldData(permittivity,fm);
  //if (!is_constant) this->utils.setFieldData(coordVec,fm);
  if (temp_dependent) this->utils.setFieldData(Temp,fm);
  //if (!is_constant && temp_dependent) this->utils.setFieldData(coordVec,fm);
}

// **********************************************************************
template<typename EvalT, typename Traits>
void QCAD::Permittivity<EvalT, Traits>::
evaluateFields(typename Traits::EvalData workset)
{
  int numCells = workset.numCells;

  if (is_constant) {
    for (std::size_t cell=0; cell < numCells; ++cell) {
      for (std::size_t qp=0; qp < numQPs; ++qp) {
	permittivity(cell,qp) = constant_value;
      }
    }
  }
  else if (temp_dependent) {
    for (std::size_t cell=0; cell < numCells; ++cell) {
      for (std::size_t qp=0; qp < numQPs; ++qp) {
        ScalarT denom = 1.0 + factor * Temp(cell,qp);
	permittivity(cell,qp) = constant_value / denom;;
      }
    }
    cout << " TTTDDD  " << permittivity(5,1) << endl;
  }
}

// **********************************************************************
template<typename EvalT,typename Traits>
typename QCAD::Permittivity<EvalT,Traits>::ScalarT& 
QCAD::Permittivity<EvalT,Traits>::getValue(const std::string &n)
{
  if (n=="Permittivity")
    return constant_value;
  else if (n=="Permittivity Factor")
    return factor;
  else {
    TEST_FOR_EXCEPTION(true, Teuchos::Exceptions::InvalidParameter,
  		       std::endl <<
  		       "Error! Logic error in getting paramter " << n
		       << " in Permittivity::getValue()" << std::endl);
    return constant_value;
  }
}

// **********************************************************************
// **********************************************************************

