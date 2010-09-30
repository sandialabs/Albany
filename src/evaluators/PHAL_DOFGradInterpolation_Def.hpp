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


#include "Teuchos_TestForException.hpp"
#include "Phalanx_DataLayout.hpp"

#include "Intrepid_FunctionSpaceTools.hpp"


//**********************************************************************
template<typename EvalT, typename Traits>
DOFGradInterpolation<EvalT, Traits>::
DOFGradInterpolation(const Teuchos::ParameterList& p) :
  val_node    (p.get<std::string>                   ("Variable Name"),
	       p.get<Teuchos::RCP<PHX::DataLayout> >("Node Data Layout") ),
  GBF         (p.get<std::string>                   ("Gradient BF Name"),
	       p.get<Teuchos::RCP<PHX::DataLayout> >("Node QP Vector Data Layout") ),
  grad_val_qp (p.get<std::string>                   ("Gradient Variable Name"),
	       p.get<Teuchos::RCP<PHX::DataLayout> >("QP Vector Data Layout") )
{
  this->addDependentField(val_node);
  this->addDependentField(GBF);
  this->addEvaluatedField(grad_val_qp);

  this->setName("DOFGradInterpolation");
}

//**********************************************************************
template<typename EvalT, typename Traits>
void DOFGradInterpolation<EvalT, Traits>::
postRegistrationSetup(typename Traits::SetupData d,
                      PHX::FieldManager<Traits>& fm)
{
  this->utils.setFieldData(val_node,fm);
  this->utils.setFieldData(GBF,fm);
  this->utils.setFieldData(grad_val_qp,fm);
}

//**********************************************************************
template<typename EvalT, typename Traits>
void DOFGradInterpolation<EvalT, Traits>::
evaluateFields(typename Traits::EvalData workset)
{
  // This is needed, since evaluate currently sums into
  for (int i=0; i < grad_val_qp.size() ; i++) grad_val_qp[i] = 0.0;

  Intrepid::FunctionSpaceTools::
      evaluate<ScalarT>(grad_val_qp, val_node, GBF);
}

//**********************************************************************
