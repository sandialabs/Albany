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
#include "Albany_Utils.hpp"

namespace LCM {

template<typename EvalT, typename Traits>
VanGenuchtenSaturation<EvalT, Traits>::
VanGenuchtenSaturation(Teuchos::ParameterList& p) :
  vgSaturation(p.get<std::string>("Van Genuchten Saturation Name"),
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

  std::string type = elmd_list->get("Van Genuchten Saturation Type", "Constant");
  if (type == "Constant") {
    is_constant = true;
    constant_value = elmd_list->get("Value", 1.0); // default value=1, identical to Terzaghi stress

    // Add Van Genuchten Saturation as a Sacado-ized parameter
    new Sacado::ParameterRegistration<EvalT, SPL_Traits>(
	"Van Genuchten Saturation", this, paramLib);
  }
  else if (type == "Truncated KL Expansion") {
    is_constant = false;
    PHX::MDField<MeshScalarT,Cell,QuadPoint,Dim>
      fx(p.get<string>("QP Coordinate Vector Name"), vector_dl);
    coordVec = fx;
    this->addDependentField(coordVec);

    exp_rf_kl = 
      Teuchos::rcp(new Stokhos::KL::ExponentialRandomField<MeshScalarT>(*elmd_list));
    int num_KL = exp_rf_kl->stochasticDimension();

    // Add KL random variables as Sacado-ized parameters
    rv.resize(num_KL);
    for (int i=0; i<num_KL; i++) {
      std::string ss = Albany::strint("Van Genuchten Saturation KL Random Variable",i);
      new Sacado::ParameterRegistration<EvalT, SPL_Traits>(ss, this, paramLib);
      rv[i] = elmd_list->get(ss, 0.0);
    }
  }
  else {
	  TEUCHOS_TEST_FOR_EXCEPTION(true, Teuchos::Exceptions::InvalidParameter,
		       "Invalid Van Genuchten Saturation type " << type);
  } 

  // Optional dependence on Temperature (E = E_ + dEdT * T)
  // Switched ON by sending Temperature field in p

  if ( p.isType<string>("Porosity Name") ) {
    Teuchos::RCP<PHX::DataLayout> scalar_dl =
      p.get< Teuchos::RCP<PHX::DataLayout> >("QP Scalar Data Layout");
    PHX::MDField<ScalarT,Cell,QuadPoint>
      tp(p.get<string>("Porosity Name"), scalar_dl);
    porosity = tp;
    this->addDependentField(porosity);
    isPoroElastic = true;

  }
  else {
    isPoroElastic = false;

  }

  if ( p.isType<string>("QP Pore Pressure Name") ) {
         Teuchos::RCP<PHX::DataLayout> scalar_dl =
           p.get< Teuchos::RCP<PHX::DataLayout> >("QP Scalar Data Layout");
         PHX::MDField<ScalarT,Cell,QuadPoint>
           ppn(p.get<string>("QP Pore Pressure Name"), scalar_dl);
         porePressure = ppn;
         isPoroElastic = true;
         this->addDependentField(porePressure);

         waterUnitWeight = elmd_list->get("Water Unit Weight Value", 9810.0); // typically Kgrain >> Kskeleton
                    new Sacado::ParameterRegistration<EvalT, SPL_Traits>(
          "Water Unit Weight Value", this, paramLib);
  }


  this->addEvaluatedField(vgSaturation);
  this->setName("Van Genuchten Saturation"+PHX::TypeString<EvalT>::value);
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
  std::size_t numCells = workset.numCells;

  if (is_constant) {
    for (std::size_t cell=0; cell < numCells; ++cell) {
      for (std::size_t qp=0; qp < numQPs; ++qp) {
    	  vgSaturation(cell,qp) = constant_value;
      }
    }
  }
  else {
    for (std::size_t cell=0; cell < numCells; ++cell) {
      for (std::size_t qp=0; qp < numQPs; ++qp) {
	Teuchos::Array<MeshScalarT> point(numDims);
	for (std::size_t i=0; i<numDims; i++)
	  point[i] = Sacado::ScalarValue<MeshScalarT>::eval(coordVec(cell,qp,i));
		  vgSaturation(cell,qp) = exp_rf_kl->evaluate(point, rv);
      }
    }
  }
  if (isPoroElastic) {
    for (std::size_t cell=0; cell < numCells; ++cell) {
      for (std::size_t qp=0; qp < numQPs; ++qp) {
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

