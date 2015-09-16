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
VanGenuchtenSaturation<EvalT, Traits>::
VanGenuchtenSaturation(Teuchos::ParameterList& p) :
  vgSaturation(p.get<std::string>("Van Genuchten Saturation Name"),
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

  std::string type = elmd_list->get("Van Genuchten Saturation Type", "Constant");
  if (type == "Constant") {
    is_constant = true;
    constant_value = elmd_list->get("Value", 1.0); // default value=1, identical to Terzaghi stress

    // Add Van Genuchten Saturation as a Sacado-ized parameter
    this->registerSacadoParameter("Van Genuchten Saturation", paramLib);
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
      std::string ss = Albany::strint("Van Genuchten Saturation KL Random Variable",i);
      this->registerSacadoParameter(ss, paramLib);
      rv[i] = elmd_list->get(ss, 0.0);
    }
  }
  else {
	  TEUCHOS_TEST_FOR_EXCEPTION(true, Teuchos::Exceptions::InvalidParameter,
		       "Invalid Van Genuchten Saturation type " << type);
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

         // typically bulk modulus of solid grain is larger than
         // bulk modulus of solid skeleton.

         waterUnitWeight = elmd_list->get("Water Unit Weight Value", 9810.0);
         this->registerSacadoParameter("Water Unit Weight Value", paramLib);
  }

  this->addEvaluatedField(vgSaturation);
  this->setName("Van Genuchten Saturation"+PHX::typeAsString<EvalT>());
}

// **********************************************************************
template<typename EvalT, typename Traits>
void VanGenuchtenSaturation<EvalT, Traits>::
postRegistrationSetup(typename Traits::SetupData d,
                      PHX::FieldManager<Traits>& fm)
{
  this->utils.setFieldData(vgSaturation,fm);
  if (!is_constant) this->utils.setFieldData(coordVec,fm);
  if (isPoroElastic) this->utils.setFieldData(porosity,fm);
  if (isPoroElastic) this->utils.setFieldData(porePressure,fm);
}

// **********************************************************************
template<typename EvalT, typename Traits>
void VanGenuchtenSaturation<EvalT, Traits>::
evaluateFields(typename Traits::EvalData workset)
{

  ScalarT tempTerm;
  int numCells = workset.numCells;

  if (is_constant) {
    for (int cell=0; cell < numCells; ++cell) {
      for (int qp=0; qp < numQPs; ++qp) {
    	  vgSaturation(cell,qp) = constant_value;
      }
    }
  }
  else {
    for (int cell=0; cell < numCells; ++cell) {
      for (int qp=0; qp < numQPs; ++qp) {
	Teuchos::Array<MeshScalarT> point(numDims);
	for (int i=0; i<numDims; i++)
	  point[i] = Sacado::ScalarValue<MeshScalarT>::eval(coordVec(cell,qp,i));
		  vgSaturation(cell,qp) = exp_rf_kl->evaluate(point, rv);
      }
    }
  }
  if (isPoroElastic) {
    for (int cell=0; cell < numCells; ++cell) {
      for (int qp=0; qp < numQPs; ++qp) {
    	  // van Genuchten equation
    	  vgSaturation(cell,qp) = 0.084 + 0.916
    			       /std::pow((1.0 + 7.0*porePressure(cell,qp)/waterUnitWeight),2.0);


      }
    }
  }
}

// **********************************************************************
template<typename EvalT,typename Traits>
typename VanGenuchtenSaturation<EvalT,Traits>::ScalarT&
VanGenuchtenSaturation<EvalT,Traits>::getValue(const std::string &n)
{
  if (n == "Van Genuchten Saturation")
    return constant_value;
  else if (n == "Water Unit Weight Value")
      return waterUnitWeight;
  for (int i=0; i<rv.size(); i++) {
    if (n == Albany::strint("Van Genuchten Saturation KL Random Variable",i))
      return rv[i];
  }
  TEUCHOS_TEST_FOR_EXCEPTION(true, Teuchos::Exceptions::InvalidParameter,
		     std::endl <<
		     "Error! Logic error in getting parameter " << n
		     << " in VanGenuchtenSaturation::getValue()" << std::endl);
  return constant_value;
}

// **********************************************************************
// **********************************************************************
}

