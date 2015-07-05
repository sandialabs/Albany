//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#include "Teuchos_TestForException.hpp"
#include "Teuchos_VerboseObject.hpp"
#include "Phalanx_DataLayout.hpp"
#include "Phalanx_TypeStrings.hpp"
#include "Intrepid_FunctionSpaceTools.hpp"

//uncomment the following line if you want debug output to be printed to screen
#define OUTPUT_TO_SCREEN



namespace FELIX {

//**********************************************************************
template<typename EvalT, typename Traits>
StokesFOThicknessResid<EvalT, Traits>::
StokesFOThicknessResid(const Teuchos::ParameterList& p,
              const Teuchos::RCP<Albany::Layouts>& dl) :
  wBF      (p.get<std::string> ("Weighted BF Name"), dl->node_qp_scalar),
  wGradBF  (p.get<std::string> ("Weighted Gradient BF Name"),dl->node_qp_gradient),
  Ugrad    (p.get<std::string> ("Gradient QP Variable Name"), dl->qp_vecgradient),
  gradH0    (p.get<std::string> ("H0 Gradient QP Name"), dl->qp_gradient),
  Residual (p.get<std::string> ("Residual Name"), dl->node_vector)
{

  Teuchos::ParameterList* list = 
    p.get<Teuchos::ParameterList*>("Parameter List");

  std::string type = list->get("Type", "FELIX");

  Teuchos::RCP<Teuchos::FancyOStream> out(Teuchos::VerboseObjectBase::getDefaultOStream());


  this->addDependentField(Ugrad);
  this->addDependentField(gradH0);
  this->addDependentField(wBF);
  this->addDependentField(wGradBF);

  this->addEvaluatedField(Residual);


  this->setName("StokesFOThicknessResid"+PHX::typeAsString<EvalT>());

  std::vector<PHX::DataLayout::size_type> dims;
  wGradBF.fieldTag().dataLayout().dimensions(dims);
  numNodes = dims[1];
  numQPs   = dims[2];
  numDims  = dims[3];

#ifdef OUTPUT_TO_SCREEN
*out << " in FELIX Prolongate H residual! " << std::endl;
*out << " numDims = " << numDims << std::endl;
*out << " numQPs = " << numQPs << std::endl; 
*out << " numNodes = " << numNodes << std::endl; 
#endif
}

//**********************************************************************
template<typename EvalT, typename Traits>
void StokesFOThicknessResid<EvalT, Traits>::
postRegistrationSetup(typename Traits::SetupData d,
                      PHX::FieldManager<Traits>& fm)
{
  this->utils.setFieldData(Ugrad,fm);
  this->utils.setFieldData(gradH0,fm);
  this->utils.setFieldData(wBF,fm);
  this->utils.setFieldData(wGradBF,fm);

  this->utils.setFieldData(Residual,fm);
}
//**********************************************************************
//Kokkos functors


//**********************************************************************
template<typename EvalT, typename Traits>
void StokesFOThicknessResid<EvalT, Traits>::
evaluateFields(typename Traits::EvalData workset)
{
  typedef Intrepid::FunctionSpaceTools FST; 

  // Initialize residual to 0.0
  Kokkos::deep_copy(Residual.get_kokkos_view(), ScalarT(0.0));

  Intrepid::FieldContainer<ScalarT> res(numNodes,3);

  for (std::size_t cell=0; cell < workset.numCells; ++cell) {
    for (int i = 0; i < res.size(); i++) res(i) = 0.0;
    for (std::size_t qp=0; qp < numQPs; ++qp) {
      ScalarT dHdx = Ugrad(cell,qp,2,0);
      ScalarT dHdy = Ugrad(cell,qp,2,1);
      ScalarT dHdz = Ugrad(cell,qp,2,2);
      double rho_g=910.0*9.8;
      for (std::size_t node=0; node < numNodes; ++node) {
           res(node,0) += rho_g*(dHdx - gradH0(cell,qp,0))*wBF(cell,node,qp);
           res(node,1) += rho_g*(dHdy - gradH0(cell,qp,1))*wBF(cell,node,qp);
           res(node,2) += dHdz*wGradBF(cell,node,qp,2);
      }
    }
    for (std::size_t node=0; node < numNodes; ++node) {
       Residual(cell,node,0) = res(node,0);
       Residual(cell,node,1) = res(node,1);
       Residual(cell,node,2) = res(node,2);
    }
  }
}

//**********************************************************************
}

