//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#include "Albany_Utils.hpp"

namespace SEE
{

//*********************************************************************
template<typename EvalT, typename Traits>
LinearAdjointSolveBase<EvalT, Traits>::
LinearAdjointSolveBase(Teuchos::ParameterList& p,
                       const Teuchos::RCP<Albany::Layouts>& dl) :
  wBF (p.get<std::string>("Weighted BF Name"),
       dl->node_qp_scalar),
  BF  (p.get<std::string>("BF Name"),
       dl->node_qp_scalar)
{

  this->addDependentField(wBF);
  this->addDependentField(BF);

  Teuchos::ParameterList* plist =
    p.get<Teuchos::ParameterList*>("Parameter List");
  Teuchos::RCP<const Teuchos::ParameterList> reflist =
    this->getValidLinearAdjointSolveParameters();
  plist->validateParameters(*reflist,0);

  field_tag_ = Teuchos::rcp(
      new PHX::Tag<ScalarT>("Linear Adjoint Solve", dl->dummy));

  this->addEvaluatedField(*field_tag_);

}

//*********************************************************************
template<typename EvalT, typename Traits>
void LinearAdjointSolveBase<EvalT, Traits>::
postRegistrationSetup(typename Traits::SetupData d,
                      PHX::FieldManager<Traits>& fm)
{

  this->utils.setFieldData(BF,fm);
  this->utils.setFieldData(wBF,fm);

}

//*********************************************************************
template<typename Traits>
LinearAdjointSolve<PHAL::AlbanyTraits::Residual, Traits>::
LinearAdjointSolve(Teuchos::ParameterList& p, const Teuchos::RCP<Albany::Layouts>& dl) :
  LinearAdjointSolveBase<PHAL::AlbanyTraits::Residual, Traits>(p, dl)
{
}

//*********************************************************************
template<typename Traits>
void LinearAdjointSolve<PHAL::AlbanyTraits::Residual, Traits>::
preEvaluate(typename Traits::PreEvalData workset)
{
  std::cout << "preEvaluate called" << std::endl;
}

//*********************************************************************
template<typename Traits>
void LinearAdjointSolve<PHAL::AlbanyTraits::Residual, Traits>::
evaluateFields(typename Traits::EvalData workset)
{
  std::cout << "evaluateFields called" << std::endl;
}

//*********************************************************************
template<typename Traits>
void LinearAdjointSolve<PHAL::AlbanyTraits::Residual, Traits>::
postEvaluate(typename Traits::PostEvalData workset)
{
  std::cout << "postEvaluate called" << std::endl;
}

//*********************************************************************
template<typename EvalT, typename Traits>
Teuchos::RCP<const Teuchos::ParameterList>
LinearAdjointSolveBase<EvalT,Traits>::getValidLinearAdjointSolveParameters() const
{
  Teuchos::RCP<Teuchos::ParameterList> validPL =
    rcp(new Teuchos::ParameterList("Valid LinearAdjointSolve Params"));;

  validPL->set<std::string>("Name", "", "Name of linear adjoint solve evaluator");
  validPL->set<bool>("Estimate Error",false,"Use adjoint to estimate error");
  validPL->set<bool>("Write Adjoint",false,"Write adjoint solution to file");

  return validPL;
}

//*********************************************************************
}

