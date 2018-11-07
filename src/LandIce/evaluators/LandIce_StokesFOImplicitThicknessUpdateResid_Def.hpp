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

#include "LandIce_StokesFOImplicitThicknessUpdateResid.hpp"

//uncomment the following line if you want debug output to be printed to screen
//#define OUTPUT_TO_SCREEN

namespace LandIce {

//**********************************************************************
template<typename EvalT, typename Traits>
StokesFOImplicitThicknessUpdateResid<EvalT, Traits>::
StokesFOImplicitThicknessUpdateResid(const Teuchos::ParameterList& p,
                                     const Teuchos::RCP<Albany::Layouts>& dl) :
  wBF           (p.get<std::string> ("Weighted BF Name"), dl->node_qp_scalar),
  gradBF        (p.get<std::string> ("Gradient BF Name"),dl->node_qp_gradient),
  dH            (p.get<std::string> ("Thickness Increment Variable Name"), dl->node_scalar),
  Residual      (p.get<std::string> ("Residual Name"), dl->node_vector)
{
  Teuchos::ParameterList* p_list = p.get<Teuchos::ParameterList*>("Physical Parameter List");

  double g = p_list->get<double>("Gravity Acceleration");
  double rho = p_list->get<double>("Ice Density");
  rho_g = rho*g;

  Teuchos::RCP<Teuchos::FancyOStream> out(Teuchos::VerboseObjectBase::getDefaultOStream());

  this->addDependentField(dH);
  this->addDependentField(wBF);
  this->addDependentField(gradBF);
  this->addContributedField(Residual);


  this->setName("StokesFOImplicitThicknessUpdateResid"+PHX::typeAsString<EvalT>());

  std::vector<PHX::DataLayout::size_type> dims;
  gradBF.fieldTag().dataLayout().dimensions(dims);
  numNodes = dims[1];
  numQPs   = dims[2];
  numCells = dims[0] ;

  numVecDims  = Residual.fieldTag().dataLayout().dimension(2);

#ifdef OUTPUT_TO_SCREEN
*out << " in LandIce StokesFOImplicitThicknessUpdate residual! " << std::endl;
*out << " numQPs = " << numQPs << std::endl;
*out << " numNodes = " << numNodes << std::endl;
#endif
}

//**********************************************************************
template<typename EvalT, typename Traits>
void StokesFOImplicitThicknessUpdateResid<EvalT, Traits>::
postRegistrationSetup(typename Traits::SetupData /* d */,
                      PHX::FieldManager<Traits>& fm)
{
  this->utils.setFieldData(dH,fm);
  this->utils.setFieldData(wBF,fm);
  this->utils.setFieldData(gradBF,fm);
  this->utils.setFieldData(Residual,fm);

  Res = createDynRankView(Residual.get_view(), "Residual", numCells, numNodes,2);
}
//**********************************************************************
//Kokkos functors
template<typename EvalT, typename Traits>
KOKKOS_INLINE_FUNCTION
void StokesFOImplicitThicknessUpdateResid<EvalT, Traits>::
operator() (const StokesFOImplicitThicknessUpdateResid_Tag&, const int& cell) const
{
  for (int node=0; node < numNodes; ++node){
    Res(cell,node,0)=0.0;
    Res(cell,node,1)=0.0;
  }

  for (int qp=0; qp < numQPs; ++qp) {
    ScalarT dHdiffdx = 0;
    ScalarT dHdiffdy = 0;
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
    Residual(cell,node,0) += Res(cell,node,0);
    Residual(cell,node,1) += Res(cell,node,1);
  }
}

//**********************************************************************
template<typename EvalT, typename Traits>
void StokesFOImplicitThicknessUpdateResid<EvalT, Traits>::
evaluateFields(typename Traits::EvalData workset)
{
  Kokkos::parallel_for(StokesFOImplicitThicknessUpdateResid_Policy(0,workset.numCells),*this);
}

} // namespace LandIce
