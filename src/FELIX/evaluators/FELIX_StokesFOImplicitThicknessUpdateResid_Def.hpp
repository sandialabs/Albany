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
StokesFOImplicitThicknessUpdateResid<EvalT, Traits>::
StokesFOImplicitThicknessUpdateResid(const Teuchos::ParameterList& p,
              const Teuchos::RCP<Albany::Layouts>& dl) :
  wBF      (p.get<std::string> ("Weighted BF Name"), dl->node_qp_scalar),
  gradBF  (p.get<std::string> ("Gradient BF Name"),dl->node_qp_gradient),
  H0    (p.get<std::string> ("H0 Name"), dl->node_scalar),
  H    (p.get<std::string> ("H Variable Name"), dl->node_scalar),
  Residual (p.get<std::string> ("Residual Name"), dl->node_vector)
{

  Teuchos::ParameterList* p_list =
      p.get<Teuchos::ParameterList*>("Physical Parameter List");

  g = p_list->get("Gravity", 9.8);
  rho = p_list->get("Ice Density", 910.0);

  Teuchos::RCP<Teuchos::FancyOStream> out(Teuchos::VerboseObjectBase::getDefaultOStream());


  this->addDependentField(H0);
  this->addDependentField(H);
  this->addDependentField(wBF);
  this->addDependentField(gradBF);
  this->addEvaluatedField(Residual);


  this->setName("StokesFOImplicitThicknessUpdateResid"+PHX::typeAsString<EvalT>());

  std::vector<PHX::DataLayout::size_type> dims;
  gradBF.fieldTag().dataLayout().dimensions(dims);
  numNodes = dims[1];
  numQPs   = dims[2];

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
  this->utils.setFieldData(H0,fm);
  this->utils.setFieldData(H,fm);
  this->utils.setFieldData(wBF,fm);
  this->utils.setFieldData(gradBF,fm);
  this->utils.setFieldData(Residual,fm);
}
//**********************************************************************
//Kokkos functors


//**********************************************************************
template<typename EvalT, typename Traits>
void StokesFOImplicitThicknessUpdateResid<EvalT, Traits>::
evaluateFields(typename Traits::EvalData workset)
{
  typedef Intrepid::FunctionSpaceTools FST; 

  // Initialize residual to 0.0
  Kokkos::deep_copy(Residual.get_kokkos_view(), ScalarT(0.0));

  Intrepid::FieldContainer<ScalarT> res(numNodes,3);

  double rho_g=rho*g;

  for (std::size_t cell=0; cell < workset.numCells; ++cell) {
    res.initialize();
    for (std::size_t qp=0; qp < numQPs; ++qp) {
      ScalarT dHdiffdx = 0;//Ugrad(cell,qp,2,0);
      ScalarT dHdiffdy = 0;//Ugrad(cell,qp,2,1);
      for (std::size_t node=0; node < numNodes; ++node) {
        dHdiffdx += (H(cell,node)-H0(cell,node)) * gradBF(cell,node, qp,0);
        dHdiffdy += (H(cell,node)-H0(cell,node)) * gradBF(cell,node, qp,1);
      }

      for (std::size_t node=0; node < numNodes; ++node) {
           res(node,0) += rho_g*dHdiffdx*wBF(cell,node,qp);
           res(node,1) += rho_g*dHdiffdy*wBF(cell,node,qp);
      }
    }
    for (std::size_t node=0; node < numNodes; ++node) {
       Residual(cell,node,0) = res(node,0);
       Residual(cell,node,1) = res(node,1);
    }
  }
}

//**********************************************************************
}

