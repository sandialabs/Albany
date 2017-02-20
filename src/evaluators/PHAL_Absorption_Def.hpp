//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

//IK, 9/12/14: no Epetra

#include <fstream>
#include "Teuchos_TestForException.hpp"
#include "Phalanx_DataLayout.hpp"
#include "Sacado_ParameterRegistration.hpp"
#include "Albany_Utils.hpp"
namespace PHAL {

template<typename EvalT, typename Traits>
Absorption<EvalT, Traits>::
Absorption(Teuchos::ParameterList& p) :
  absorption(p.get<std::string>("QP Variable Name"),
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

  std::string type = cond_list->get("Absorption Type", "Constant");
  if (type == "Constant") {
    is_constant = true;
    constant_value = cond_list->get("Value", 1.0);

    // Add absorption as a Sacado-ized parameter
    Teuchos::RCP<ParamLib> paramLib = 
      p.get< Teuchos::RCP<ParamLib> >("Parameter Library", Teuchos::null);
      this->registerSacadoParameter("Absorption", paramLib);
  }
#ifdef ALBANY_STOKHOS
  else if (type == "Truncated KL Expansion") {
    is_constant = false;
    Teuchos::RCP<PHX::DataLayout> scalar_dl =
      p.get< Teuchos::RCP<PHX::DataLayout> >("QP Scalar Data Layout");
    PHX::MDField<MeshScalarT,Cell,QuadPoint,Dim>
      fx(p.get<std::string>("QP Coordinate Vector Name"), vector_dl);
    coordVec = fx;
    this->addDependentField(coordVec);

    exp_rf_kl = 
      Teuchos::rcp(new Stokhos::KL::ExponentialRandomField<RealType>(*cond_list));
    int num_KL = exp_rf_kl->stochasticDimension();

    // Add KL random variables as Sacado-ized parameters
    rv.resize(num_KL);
    Teuchos::RCP<ParamLib> paramLib = 
      p.get< Teuchos::RCP<ParamLib> >("Parameter Library", Teuchos::null);
    for (int i=0; i<num_KL; i++) {
      std::string ss = Albany::strint("Absorption KL Random Variable",i);
      this->registerSacadoParameter(ss, paramLib);
      rv[i] = cond_list->get(ss, 0.0);
    }
  }
#endif
  else {
    TEUCHOS_TEST_FOR_EXCEPTION(true, Teuchos::Exceptions::InvalidParameter,
		       "Invalid absorption type " << type);
  } 

  this->addEvaluatedField(absorption);
  this->setName("Absorption" );
}

// **********************************************************************
template<typename EvalT, typename Traits>
void Absorption<EvalT, Traits>::
postRegistrationSetup(typename Traits::SetupData d,
                      PHX::FieldManager<Traits>& fm)
{
  this->utils.setFieldData(absorption,fm);
  if (!is_constant) this->utils.setFieldData(coordVec,fm);
}

// **********************************************************************
template<typename EvalT, typename Traits>
void Absorption<EvalT, Traits>::
evaluateFields(typename Traits::EvalData workset)
{
  if (is_constant) {
    for (std::size_t cell=0; cell < workset.numCells; ++cell) {
      for (std::size_t qp=0; qp < numQPs; ++qp) {
	absorption(cell,qp) = constant_value;
      }
    }
  }
  /*else {
    for (std::size_t cell=0; cell < workset.numCells; ++cell) {
      for (std::size_t qp=0; qp < numQPs; ++qp) {
	Teuchos::Array<MeshScalarT> point(numDims);
	for (std::size_t i=0; i<numDims; i++)
	  point[i] = Sacado::ScalarValue<MeshScalarT>::eval(coordVec(cell,qp,i));
	thermalCond(cell,qp) = exp_rf_kl->evaluate(point, rv);
      }
    }
  }*/
}

// **********************************************************************
template<typename EvalT,typename Traits>
typename Absorption<EvalT,Traits>::ScalarT& 
Absorption<EvalT,Traits>::getValue(const std::string &n)
{
  if (is_constant)
    return constant_value;
  /*for (int i=0; i<rv.size(); i++) {
    if (n == Albany::strint("Thermal Conductivity KL Random Variable",i))
      return rv[i];
  }*/
  TEUCHOS_TEST_FOR_EXCEPTION(true, Teuchos::Exceptions::InvalidParameter,
		     std::endl <<
		     "Error! Logic error in getting paramter " << n
		     << " in Absorption::getValue()" << std::endl);
  return constant_value;
}

// **********************************************************************
// **********************************************************************
}

