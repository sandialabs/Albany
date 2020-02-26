//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#include <MiniTensor.h>
#include <MiniTensor_Mechanics.h>
#include <Phalanx_DataLayout.hpp>
#include <Sacado_ParameterRegistration.hpp>
#include <Teuchos_TestForException.hpp>

#ifdef ALBANY_TIMER
#include <chrono>
#endif

//IKT: uncomment to turn on debug output
//#define DEBUG_OUTPUT

//This is a direct modification of the MechanicsResidual_Def.hpp evaluator
namespace TDM {

//------------------------------------------------------------------------------
template<typename EvalT, typename Traits>
ConsolidationResidual<EvalT, Traits>::ConsolidationResidual(
    Teuchos::ParameterList&              p,
    const Teuchos::RCP<Albany::Layouts>& dl) : 
      w_bf_			(p.get<std::string>("Weighted BF Name"), dl->node_qp_scalar),
      w_grad_bf_	(p.get<std::string>("Weighted Gradient BF Name"), dl->node_qp_vector),
	  psi_			(p.get<std::string>("Psi Name"), dl->qp_scalar),
	  porosity_		(p.get<std::string>("Porosity Name"),dl->qp_scalar),
	  GradU			(p.get<std::string>("Gradient QP Variable Name"),dl->qp_tensor),	  
      residual_		(p.get<std::string>("Residual Name"), dl->node_vector),
{
  Teuchos::RCP<Teuchos::FancyOStream> out = Teuchos::VerboseObjectBase::getDefaultOStream();
  this->addDependentField(w_bf_);
  this->addDependentField(w_grad_bf_);
  this->addDependentField(psi_);
  this->addDependentField(porosity_); 
  this->addDependentField(GradU);  

  this->addEvaluatedField(residual_);

  std::vector<PHX::DataLayout::size_type> dims;
  w_grad_bf_.fieldTag().dataLayout().dimensions(dims);
  workset_size_ = dims[0];
  num_nodes_ 	= dims[1];
  num_pts_   	= dims[2];
  num_dims_  	= dims[3];
  
  
  Teuchos::ParameterList* cond_list = p.get<Teuchos::ParameterList*>("Porosity Parameter List");
  Initial_porosity = cond_list->get("Value", 0.0);

  //not sure if needed for consolidation evaluator
  Teuchos::RCP<ParamLib> paramLib = p.get<Teuchos::RCP<ParamLib>>("Parameter Library");
  this->setName("ConsolidationResidual" + PHX::print<EvalT>());
}

//------------------------------------------------------------------------------
template<typename EvalT, typename Traits>
void
ConsolidationResidual<EvalT, Traits>::postRegistrationSetup(
    typename Traits::SetupData d,
    PHX::FieldManager<Traits>& fm)
{
  this->utils.setFieldData(w_bf_,fm);
  this->utils.setFieldData(w_grad_bf_,fm);
  this->utils.setFieldData(psi_,fm);
  this->utils.setFieldData(porosity_,fm);
  this->utils.setFieldData(GradU,fm);
  
  //this->utils.setFieldData(residual_, fm);
  
//this line is in the AMP PhaseResidual_Def.hpp evaluator, not sure what it does or if it is needed here
//term1_ = Kokkos::createDynRankView(k_.get_view(), "term1_", workset_size_,num_qps_,num_dims_);
}

//Removed all the Kokkos kernels for body force, stress, and acceleration


// ***************************************************************************
template<typename EvalT, typename Traits>
void
ConsolidationResidual<EvalT, Traits>::evaluateFields(
    typename Traits::EvalData workset)
{
  std::cout << "consolidation residual started\n" ; 

  //zero out residual
  for (int cell = 0; cell < workset.numCells; ++cell) {
    for (int node = 0; node < num_nodes_; ++node)
      for (int dim = 0; dim < num_dims_; ++dim)
        residual_(cell, node, dim) = ScalarT(0);
    for (int pt = 0; pt < num_pts_; ++pt) {
	  //Compute the local volume ratio for this cell
	  DetF = (1.0 - Initial_porosity) / (1.0 - porosity_(cell, pt));
      for (int node = 0; node < num_nodes_; ++node) {
        for (int i = 0; i < num_dims_; ++i)
          for (int j = 0; j < num_dims_; ++j)
            residual_(cell, node, i) +=
                GradU(cell, pt, i, j) * w_grad_bf_(cell, node, pt, j);
      }
    }
  }
std::cout << "consolidation residual has been finished\n" ; 
}


//------------------------------------------------------------------------------
}
