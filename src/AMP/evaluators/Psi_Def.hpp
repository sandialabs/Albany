//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#include <fstream>
#include "Sacado_ParameterRegistration.hpp"
#include "Albany_Utils.hpp"
 
#include "Teuchos_TestForException.hpp"
#include "Phalanx_DataLayout.hpp"

#include "Intrepid2_FunctionSpaceTools.hpp"


namespace AMP {

//**********************************************************************
template<typename EvalT, typename Traits>
Psi<EvalT, Traits>::
Psi(Teuchos::ParameterList& p,
    const Teuchos::RCP<Albany::Layouts>& dl) :
  T_          (p.get<std::string>("Temperature Name"),
               dl->qp_scalar),
  phi_        (p.get<std::string>("Phi Name"),
               dl->qp_scalar),
  psi_        (p.get<std::string>("Psi Name"),
               dl->qp_scalar)
{

  this->addDependentField(T_);
  this->addDependentField(phi_);
  this->addEvaluatedField(psi_);
 
  Teuchos::RCP<PHX::DataLayout> scalar_dl = dl->qp_scalar;
  std::vector<PHX::Device::size_type> dims;
  scalar_dl->dimensions(dims);
  workset_size_ = dims[0];
  num_qps_      = dims[1];

  Teuchos::ParameterList* cond_list =
    p.get<Teuchos::ParameterList*>("Parameter List");

  Teuchos::RCP<const Teuchos::ParameterList> reflist = 
    this->getValidPsiParameters(); 

  cond_list->validateParameters(*reflist, 0, 
      Teuchos::VALIDATE_USED_ENABLED, Teuchos::VALIDATE_DEFAULTS_DISABLED); 

  std::string typ = cond_list->get("Psi Type", "Constant"); 
  constant_value_ = cond_list->get("Psi", 0.0); 
  

  psi_Name_ = p.get<std::string>("Psi Name")+"_old";
  this->setName("Psi"+PHX::typeAsString<EvalT>());
}

//**********************************************************************
template<typename EvalT, typename Traits>
void Psi<EvalT, Traits>::
postRegistrationSetup(typename Traits::SetupData d,
                      PHX::FieldManager<Traits>& fm)
{
  this->utils.setFieldData(T_,fm);
  this->utils.setFieldData(phi_,fm);
  this->utils.setFieldData(psi_,fm);
}

//**********************************************************************
int iteration = 0; 
template<typename EvalT, typename Traits>
void Psi<EvalT, Traits>::
evaluateFields(typename Traits::EvalData workset)
{
  //grab old psi value
  Albany::MDArray psi_old = (*workset.stateArrayPtr)[psi_Name_];

  // current time
  const RealType t = workset.current_time;

  // do this only at the begining
  //if ( t == 0.0 )
  //  {
      // initializing psi_old values:
      for (std::size_t cell = 0; cell < workset.numCells; ++cell) 
	{
	  for (std::size_t qp = 0; qp < num_qps_; ++qp) 
	    {
	      psi_(cell,qp) = constant_value_;
	    }
	}
      //  }

  // defining psi_
  for (std::size_t cell = 0; cell < workset.numCells; ++cell) {
    for (std::size_t qp = 0; qp < num_qps_; ++qp) {
      psi_(cell,qp) = std::max(phi_(cell,qp), psi_old(cell,qp));
	//for 11dec folder  
	// if (iteration ==84 || iteration ==169 || iteration ==254 || iteration ==339 || iteration ==424 || iteration ==509 || iteration ==594){     
	//for 12nov folder 
	if (iteration ==0 || iteration ==59 || iteration ==89 || iteration ==119 || iteration ==149)      
 	std::cout<<iteration<<": @Cell = "<<cell<<"; qp = "<<qp<<"; Psi_Value = "<<psi_(cell,qp)<<" Phi_Value = "<<phi_(cell,qp)<<"; Temp = "<<T_(cell,qp)<<"; Psi_Old = "<<psi_old(cell,qp)<<std::endl;
     }
  }
 std::cout<<"===iteration:==== "<<iteration<<" ==== "<<std::endl;
 iteration++;
}
//**********************************************************************
template<typename EvalT, typename Traits>
Teuchos::RCP<const Teuchos::ParameterList>
Psi<EvalT, Traits>::
getValidPsiParameters() const
{
  Teuchos::RCP<Teuchos::ParameterList> valid_pl =
    rcp(new Teuchos::ParameterList("Valid Psi Params"));;

  valid_pl->set<std::string>("Psi Type", "Constant",
      "Constant psi across the element block");

  valid_pl->set<double>("Psi", 1.0, "Constant psi value");
  return valid_pl;
}
//**********************************************************************

}
