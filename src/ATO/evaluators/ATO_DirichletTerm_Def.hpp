//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//
//
// JR todo:  implement for qp_scalar

#include "Teuchos_TestForException.hpp"
#include "Phalanx_DataLayout.hpp"
#include "Albany_Layouts.hpp"

#include "Intrepid2_FunctionSpaceTools.hpp"

namespace ATO {

//**********************************************************************
template<typename EvalT, typename Traits>
DirichletTerm<EvalT, Traits>::
DirichletTerm(const Teuchos::ParameterList& p) :
  outVector        (p.get<std::string>                   ("Dirichlet Force Name"),
	            p.get<Teuchos::RCP<PHX::DataLayout> >("Data Layout") ),
  dirVector        (p.get<std::string>                   ("Variable Name"),
	            p.get<Teuchos::RCP<PHX::DataLayout> >("Data Layout") )
{
  // Pull out numQPs and numDims from a Layout
  Teuchos::RCP<PHX::DataLayout> vector_dl =
    p.get< Teuchos::RCP<PHX::DataLayout> >("Data Layout");
  std::vector<PHX::Device::size_type> dims;
  vector_dl->dimensions(dims);
  numQPs  = dims[1];
  numDims = dims[2];

  double penaltyConstant = p.get<double>("Penalty Coefficient");

  penaltyVector.resize(numDims);
  constraintVector.resize(numDims);
  for(int i=0; i<numDims; i++){
    penaltyVector[i] = 0.0;
    constraintVector[i] = 0.0;
  }

  if(p.isType<double>("X") ){
    constraintVector[0] = p.get<double>("X");
    penaltyVector[0] = penaltyConstant;
  }
  if(p.isType<double>("Y") ){
    constraintVector[1] = p.get<double>("Y");
    penaltyVector[1] = penaltyConstant;
  }
  if(p.isType<double>("Z") ){
    constraintVector[2] = p.get<double>("Z");
    penaltyVector[2] = penaltyConstant;
  }

  this->addEvaluatedField(outVector);
  this->addDependentField(dirVector);

  this->setName("DirichletTerm"+PHX::typeAsString<EvalT>());

}

//**********************************************************************
template<typename EvalT, typename Traits>
void DirichletTerm<EvalT, Traits>::
postRegistrationSetup(typename Traits::SetupData d,
                      PHX::FieldManager<Traits>& fm)
{
  this->utils.setFieldData(outVector,fm);
  this->utils.setFieldData(dirVector,fm);
}

//**********************************************************************
template<typename EvalT, typename Traits>
void DirichletTerm<EvalT, Traits>::
evaluateFields(typename Traits::EvalData workset)
{
  for (std::size_t cell=0; cell < workset.numCells; ++cell)
    for (std::size_t qp=0; qp < numQPs; ++qp)
      for (std::size_t i=0; i < numDims; ++i)
        outVector(cell,qp,i) = penaltyVector[i]*(constraintVector[i] - dirVector(cell,qp,i));
}

//**********************************************************************
}

