//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#include "Teuchos_TestForException.hpp"
#include "Teuchos_VerboseObject.hpp"
#include "Phalanx_DataLayout.hpp"
#include "Phalanx_TypeStrings.hpp"
#include "PHAL_Utilities.hpp"

//uncomment the following line if you want debug output to be printed to screen
//#define OUTPUT_TO_SCREEN



namespace FELIX {

//**********************************************************************
template<typename EvalT, typename Traits>
StokesFOImplicitThicknessUpdateResid<EvalT, Traits>::
StokesFOImplicitThicknessUpdateResid(const Teuchos::ParameterList& p,
              const Teuchos::RCP<Albany::Layouts>& dl_full,
              const Teuchos::RCP<Albany::Layouts>& dl_ice) :
  wBF      (p.get<std::string> ("Weighted BF Name"), dl_full->node_qp_scalar),
  gradBF  (p.get<std::string> ("Gradient BF Name"),dl_full->node_qp_gradient),
  dH    (p.get<std::string> ("Thickness Increment Variable Name"), dl_full->node_scalar),
  InputResidual    (p.get<std::string> ("Input Residual Name"), dl_ice->node_vector),
  Residual (p.get<std::string> ("Residual Name"), dl_full->node_vector)
{

  Teuchos::ParameterList* p_list =
      p.get<Teuchos::ParameterList*>("Physical Parameter List");

  g = p_list->get<double>("Gravity Acceleration");
  rho = p_list->get<double>("Ice Density");

  Teuchos::RCP<Teuchos::FancyOStream> out(Teuchos::VerboseObjectBase::getDefaultOStream());

  this->addDependentField(dH);
  this->addDependentField(wBF);
  this->addDependentField(gradBF);
  this->addDependentField(InputResidual);
  this->addEvaluatedField(Residual);


  this->setName("StokesFOImplicitThicknessUpdateResid"+PHX::typeAsString<EvalT>());

  std::vector<PHX::DataLayout::size_type> dims;
  gradBF.fieldTag().dataLayout().dimensions(dims);
  numNodes = dims[1];
  numQPs   = dims[2];
  numCells = dims[0] ;

  dl_full->node_vector->dimensions(dims);
  numVecDims  =dims[2];

#ifdef OUTPUT_TO_SCREEN
*out << " in FELIX StokesFOImplicitThicknessUpdate residual! " << std::endl;
*out << " numQPs = " << numQPs << std::endl;
*out << " numNodes = " << numNodes << std::endl;
#endif
}

//**********************************************************************
template<typename EvalT, typename Traits>
void StokesFOImplicitThicknessUpdateResid<EvalT, Traits>::
postRegistrationSetup(typename Traits::SetupData d,
                      PHX::FieldManager<Traits>& fm)
{
  this->utils.setFieldData(dH,fm);
  this->utils.setFieldData(wBF,fm);
  this->utils.setFieldData(gradBF,fm);
  this->utils.setFieldData(InputResidual,fm);
  this->utils.setFieldData(Residual,fm);

  Res = createDynRankView(Residual.get_view(), "Residual", numCells, numNodes,2);
}
//**********************************************************************
//Kokkos functors
#ifdef ALBANY_KOKKOS_UNDER_DEVELOPMENT
template<typename EvalT, typename Traits>
KOKKOS_INLINE_FUNCTION
void StokesFOImplicitThicknessUpdateResid<EvalT, Traits>::
operator() (const StokesFOImplicitThicknessUpdateResid_Tag& tag, const int& cell) const 
{
  double rho_g=rho*g;

  for (int node=0; node < numNodes; ++node){
    Res(cell,node,0)=0.0;
    Res(cell,node,1)=0.0;
  }

  for (int qp=0; qp < numQPs; ++qp) {
    ScalarT dHdiffdx = 0;//Ugrad(cell,qp,2,0);
    ScalarT dHdiffdy = 0;//Ugrad(cell,qp,2,1);
    for (int node=0; node < numNodes; ++node) {
      dHdiffdx += dH(cell,node) * gradBF(cell,node, qp,0);
      dHdiffdy += dH(cell,node) * gradBF(cell,node, qp,1);
    }
    for (int node=0; node < numNodes; ++node) {
      Res(cell,node,0) += rho_g*dHdiffdx*wBF(cell,node,qp);
      Res(cell,node,1) += rho_g*dHdiffdy*wBF(cell,node,qp);
    }
  }
  for (int node=0; node < numNodes; ++node) {
    Residual(cell,node,0) = InputResidual(cell,node,0)+Res(cell,node,0);
    Residual(cell,node,1) = InputResidual(cell,node,1)+Res(cell,node,1);
    if(numVecDims==3)
      Residual(cell,node,2) = InputResidual(cell,node,2);
  }
}
#endif

//**********************************************************************
template<typename EvalT, typename Traits>
void StokesFOImplicitThicknessUpdateResid<EvalT, Traits>::
evaluateFields(typename Traits::EvalData workset)
{
#ifndef ALBANY_KOKKOS_UNDER_DEVELOPMENT
  // Initialize residual to 0.0
  Kokkos::DynRankView<ScalarT, PHX::Device> res= createDynRankView(Residual.get_view(), "XXX", numNodes,2);

  double rho_g=rho*g;

  for (std::size_t cell=0; cell < workset.numCells; ++cell) {
    for (std::size_t node=0; node < numNodes; ++node) {res(node,0)=0.0; res(node,1)=0.0;}
    for (std::size_t qp=0; qp < numQPs; ++qp) {
      ScalarT dHdiffdx = 0;//Ugrad(cell,qp,2,0);
      ScalarT dHdiffdy = 0;//Ugrad(cell,qp,2,1);
      for (std::size_t node=0; node < numNodes; ++node) {
        dHdiffdx += dH(cell,node) * gradBF(cell,node, qp,0);
        dHdiffdy += dH(cell,node) * gradBF(cell,node, qp,1);
      }

      for (std::size_t node=0; node < numNodes; ++node) {
           res(node,0) += rho_g*dHdiffdx*wBF(cell,node,qp);
           res(node,1) += rho_g*dHdiffdy*wBF(cell,node,qp);
      }
    }
    for (std::size_t node=0; node < numNodes; ++node) {
       Residual(cell,node,0) = InputResidual(cell,node,0)+res(node,0);
       Residual(cell,node,1) = InputResidual(cell,node,1)+res(node,1);
       if(numVecDims==3)
         Residual(cell,node,2) = InputResidual(cell,node,2);
    }
  }

#else

  Kokkos::parallel_for(StokesFOImplicitThicknessUpdateResid_Policy(0,workset.numCells),*this);


#endif
}

//**********************************************************************
}

