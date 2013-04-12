//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#include <fstream>
#include <Sacado_MathFunctions.hpp>
#include "Teuchos_TestForException.hpp"
#include "Phalanx_DataLayout.hpp"
#include "Sacado_ParameterRegistration.hpp"
#include "Albany_Utils.hpp"

namespace LCM {

template<typename EvalT, typename Traits>
TauContribution<EvalT, Traits>::
TauContribution(Teuchos::ParameterList& p) :
tauFactor(p.get<std::string>("Tau Contribution Name"),
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

  std::string type = elmd_list->get("Tau Contribution Type", "Constant");
  if (type == "Constant") {
    is_constant = true;
    constant_value = elmd_list->get("Value", 0.0);

    // Add Trapped Solvent as a Sacado-ized parameter
    new Sacado::ParameterRegistration<EvalT, SPL_Traits>(
	"Tau Contribution", this, paramLib);
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
      std::string ss = Albany::strint("Trapped Solvent KL Random Variable",i);
      new Sacado::ParameterRegistration<EvalT, SPL_Traits>(ss, this, paramLib);
      rv[i] = elmd_list->get(ss, 0.0);
    }
  }
  else {
	  TEUCHOS_TEST_FOR_EXCEPTION(true, Teuchos::Exceptions::InvalidParameter,
		       "Invalid Trapped Solvent type " << type);
  } 

  if ( p.isType<string>("QP Variable Name") ) {
     Teuchos::RCP<PHX::DataLayout> scalar_dl =
       p.get< Teuchos::RCP<PHX::DataLayout> >("QP Scalar Data Layout");
     PHX::MDField<ScalarT,Cell,QuadPoint>
       tp(p.get<string>("QP Variable Name"), scalar_dl);
     Clattice = tp;
     this->addDependentField(Clattice);
     VmPartial = elmd_list->get("Partial Molar Volume Value", 0.0);
     new Sacado::ParameterRegistration<EvalT, SPL_Traits>(
                                 "Partial Molar Volume Value", this, paramLib);

   }
   else {
     VmPartial = 2.0e-6;
   }

  if ( p.isType<string>("Diffusion Coefficient Name") ) {
       Teuchos::RCP<PHX::DataLayout> scalar_dl =
         p.get< Teuchos::RCP<PHX::DataLayout> >("QP Scalar Data Layout");
       PHX::MDField<ScalarT,Cell,QuadPoint>
         ap(p.get<string>("Diffusion Coefficient Name"), scalar_dl);
       DL = ap;
       this->addDependentField(DL);

  }

 /*
  if ( p.isType<string>("Ideal Gas Constant Name") ) {
           Teuchos::RCP<PHX::DataLayout> scalar_dl =
             p.get< Teuchos::RCP<PHX::DataLayout> >("QP Scalar Data Layout");
           PHX::MDField<ScalarT,Cell,QuadPoint>
             idg(p.get<string>("Ideal Gas Constant Name"), scalar_dl);
           Rideal = idg;
           this->addDependentField(Rideal);

      }
  */

  if ( p.isType<string>("Temperature Name") ) {
         Teuchos::RCP<PHX::DataLayout> scalar_dl =
           p.get< Teuchos::RCP<PHX::DataLayout> >("QP Scalar Data Layout");
         PHX::MDField<ScalarT,Cell,QuadPoint>
           tep(p.get<string>("Temperature Name"), scalar_dl);
         temperature = tep;

         Rideal = p.get<RealType>("Ideal Gas Constant", 8.3144621);
         this->addDependentField(temperature);

    }




  this->addEvaluatedField(tauFactor);
  this->setName("Tau Contribution"+PHX::TypeString<EvalT>::value);
}

// **********************************************************************
template<typename EvalT, typename Traits>
void TauContribution<EvalT, Traits>::
postRegistrationSetup(typename Traits::SetupData d,
                      PHX::FieldManager<Traits>& fm)
{
  this->utils.setFieldData(tauFactor,fm);
  if (!is_constant) this->utils.setFieldData(coordVec,fm);
  this->utils.setFieldData(temperature,fm);
  this->utils.setFieldData(DL,fm);
  this->utils.setFieldData(Clattice,fm);
}

// **********************************************************************
template<typename EvalT, typename Traits>
void TauContribution<EvalT, Traits>::
evaluateFields(typename Traits::EvalData workset)
{
  std::size_t numCells = workset.numCells;

  if (is_constant) {
    for (std::size_t cell=0; cell < numCells; ++cell) {
      for (std::size_t qp=0; qp < numQPs; ++qp) {
    	  tauFactor(cell,qp) = constant_value;
      }
    }
  }
  else {
    for (std::size_t cell=0; cell < numCells; ++cell) {
      for (std::size_t qp=0; qp < numQPs; ++qp) {
	Teuchos::Array<MeshScalarT> point(numDims);
	for (std::size_t i=0; i<numDims; i++)
	  point[i] = Sacado::ScalarValue<MeshScalarT>::eval(coordVec(cell,qp,i));
		tauFactor(cell,qp) = exp_rf_kl->evaluate(point, rv);
      }
    }
  }
  for (std::size_t cell=0; cell < numCells; ++cell) {
      for (std::size_t qp=0; qp < numQPs; ++qp) {
    	  tauFactor(cell,qp) = DL(cell,qp)*Clattice(cell,qp)*VmPartial/
    			               ( Rideal*temperature(cell,qp) );


      }
   }
}


// **********************************************************************
template<typename EvalT,typename Traits>
typename TauContribution<EvalT,Traits>::ScalarT&
TauContribution<EvalT,Traits>::getValue(const std::string &n)
{
  if (n == "Tau Contribution")
    return constant_value;
  else if (n == "Partial Molar Volume Value")
     return VmPartial;
  for (int i=0; i<rv.size(); i++) {
    if (n == Albany::strint("Tau Contribution KL Random Variable",i))
      return rv[i];
  }
  TEUCHOS_TEST_FOR_EXCEPTION(true, Teuchos::Exceptions::InvalidParameter,
		     std::endl <<
		     "Error! Logic error in getting parameter " << n
		     << " in TauContribution::getValue()" << std::endl);
  return constant_value;
}

// **********************************************************************
}

