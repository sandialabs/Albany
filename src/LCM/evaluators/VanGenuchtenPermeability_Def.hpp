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
VanGenuchtenPermeability<EvalT, Traits>::
VanGenuchtenPermeability(Teuchos::ParameterList& p) :
  vgPermeability(p.get<std::string>("Van Genuchten Permeability Name"),
		 p.get<Teuchos::RCP<PHX::DataLayout>>("QP Scalar Data Layout"))
{
  Teuchos::ParameterList* elmd_list = 
    p.get<Teuchos::ParameterList*>("Parameter List");

  Teuchos::RCP<PHX::DataLayout> vector_dl =
    p.get< Teuchos::RCP<PHX::DataLayout>>("QP Vector Data Layout");
  std::vector<PHX::DataLayout::size_type> dims;
  vector_dl->dimensions(dims);
  numQPs  = dims[1];
  numDims = dims[2];

  Teuchos::RCP<ParamLib> paramLib = 
    p.get< Teuchos::RCP<ParamLib>>("Parameter Library", Teuchos::null);

  std::string type = elmd_list->get("Van Genuchten Permeability Type", "Constant");
  if (type == "Constant") {
    is_constant = true;
    constant_value = elmd_list->get("Value", 1.0e-5); // default value=1, identical to Terzaghi stress

    // Add Van Genuchten Permeability as a Sacado-ized parameter
    this->registerSacadoParameter("Van Genuchten Permeability", paramLib);
  }
  else if (type == "Truncated KL Expansion") {
    is_constant = false;
    PHX::MDField<MeshScalarT,Cell,QuadPoint,Dim>
      fx(p.get<std::string>("QP Coordinate Vector Name"), vector_dl);
    coordVec = fx;
    this->addDependentField(coordVec);

    exp_rf_kl = 
      Teuchos::rcp(new Stokhos::KL::ExponentialRandomField<RealType>(*elmd_list));
    int num_KL = exp_rf_kl->stochasticDimension();

    // Add KL random variables as Sacado-ized parameters
    rv.resize(num_KL);
    for (int i=0; i<num_KL; i++) {
      std::string ss = Albany::strint("Van Genuchten Permeability KL Random Variable",i);
      this->registerSacadoParameter(ss, paramLib);
      rv[i] = elmd_list->get(ss, 0.0);
    }
  }
  else {
	  TEUCHOS_TEST_FOR_EXCEPTION(true, Teuchos::Exceptions::InvalidParameter,
		       "Invalid Van Genuchten Permeability type " << type);
  } 

  // Optional dependence on Temperature (E = E_ + dEdT * T)
  // Switched ON by sending Temperature field in p

  if ( p.isType<std::string>("Porosity Name") ) {
    Teuchos::RCP<PHX::DataLayout> scalar_dl =
      p.get< Teuchos::RCP<PHX::DataLayout>>("QP Scalar Data Layout");
    PHX::MDField<ScalarT,Cell,QuadPoint>
      tp(p.get<std::string>("Porosity Name"), scalar_dl);
    porosity = tp;
    this->addDependentField(porosity);
    isPoroElastic = true;

  }
  else {
    isPoroElastic = false;

  }

  if ( p.isType<std::string>("QP Pore Pressure Name") ) {
         Teuchos::RCP<PHX::DataLayout> scalar_dl =
           p.get< Teuchos::RCP<PHX::DataLayout>>("QP Scalar Data Layout");
         PHX::MDField<ScalarT,Cell,QuadPoint>
           ppn(p.get<std::string>("QP Pore Pressure Name"), scalar_dl);
         porePressure = ppn;
         isPoroElastic = true;
         this->addDependentField(porePressure);

         waterUnitWeight = elmd_list->get("Water Unit Weight Value", 9810.0); // typically Kgrain >> Kskeleton
         this->registerSacadoParameter("Water Unit Weight Value", paramLib);
  }


  this->addEvaluatedField(vgPermeability);
  this->setName("Van Genuchten Permeability"+PHX::typeAsString<EvalT>());
}

// **********************************************************************
template<typename EvalT, typename Traits>
void VanGenuchtenPermeability<EvalT, Traits>::
postRegistrationSetup(typename Traits::SetupData d,
                      PHX::FieldManager<Traits>& fm)
{
  this->utils.setFieldData(vgPermeability,fm);
  if (!is_constant) this->utils.setFieldData(coordVec,fm);
  if (isPoroElastic) this->utils.setFieldData(porosity,fm);
  if (isPoroElastic) this->utils.setFieldData(porePressure,fm);
}

// **********************************************************************
template<typename EvalT, typename Traits>
void VanGenuchtenPermeability<EvalT, Traits>::
evaluateFields(typename Traits::EvalData workset)
{
  int numCells = workset.numCells;

  if (is_constant) {
    for (int cell=0; cell < numCells; ++cell) {
      for (int qp=0; qp < numQPs; ++qp) {
    	  vgPermeability(cell,qp) = constant_value;
      }
    }
  }
  else {
    for (int cell=0; cell < numCells; ++cell) {
      for (int qp=0; qp < numQPs; ++qp) {
	Teuchos::Array<MeshScalarT> point(numDims);
	for (int i=0; i<numDims; i++)
	  point[i] = Sacado::ScalarValue<MeshScalarT>::eval(coordVec(cell,qp,i));
		  vgPermeability(cell,qp) = exp_rf_kl->evaluate(point, rv);
      }
    }
  }
  if (isPoroElastic) {
    for (int cell=0; cell < numCells; ++cell) {
      for (int qp=0; qp < numQPs; ++qp) {
    	  // van Genuchten permeability equation
    	  vgPermeability(cell,qp) = constant_value*porosity(cell,qp)*porosity(cell,qp)*porosity(cell,qp)/
    			                    (  1.0-porosity(cell,qp)*porosity(cell,qp) )/
    			                    std::pow( 1.0 + std::pow(50.0*porePressure(cell,qp)/waterUnitWeight, 4.0),0.9);
      }
    }
  }
}

// **********************************************************************
template<typename EvalT,typename Traits>
typename VanGenuchtenPermeability<EvalT,Traits>::ScalarT&
VanGenuchtenPermeability<EvalT,Traits>::getValue(const std::string &n)
{
  if (n == "Van Genuchten Permeability")
    return constant_value;
  else if (n == "Water Unit Weight Value")
      return waterUnitWeight;
  for (int i=0; i<rv.size(); i++) {
    if (n == Albany::strint("Van Genuchten Permeability KL Random Variable",i))
      return rv[i];
  }
  TEUCHOS_TEST_FOR_EXCEPTION(true, Teuchos::Exceptions::InvalidParameter,
		     std::endl <<
		     "Error! Logic error in getting parameter " << n
		     << " in VanGenuchtenPermeability::getValue()" << std::endl);
  return constant_value;
}

// **********************************************************************
// **********************************************************************
}

