//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#include "Phalanx_DataLayout.hpp"
#include "Phalanx_Print.hpp"
#include "Shards_CellTopology.hpp"

#include "PHAL_SideLaplacianResidual.hpp"

//uncomment the following line if you want debug output to be printed to screen
#define OUTPUT_TO_SCREEN

namespace PHAL {

//**********************************************************************
template<typename EvalT, typename Traits>
SideLaplacianResidual<EvalT, Traits>::
SideLaplacianResidual (const Teuchos::ParameterList& p,
                       const Teuchos::RCP<Albany::Layouts>& dl)
 : u         (p.get<std::string> ("Solution QP Variable Name"), dl->qp_scalar)
 , grad_u    (p.get<std::string> ("Solution Gradient QP Variable Name"), dl->qp_gradient)
 , BF        (p.get<std::string> ("BF Variable Name"), dl->node_qp_scalar)
 , GradBF    (p.get<std::string> ("Gradient BF Variable Name"), dl->node_qp_gradient)
 , w_measure (p.get<std::string> ("Weighted Measure Variable Name"), dl->qp_scalar)
 , residual  (p.get<std::string> ("Residual Variable Name"),    dl->node_scalar)
{
  sideSetEquation = p.get<bool>("Side Equation");
  TEUCHOS_TEST_FOR_EXCEPTION (
      sideSetEquation!=dl->isSideLayouts, std::runtime_error,
     "Error! The layout structure does match what's expected:\n"
     "  sideSetEquation: " << sideSetEquation << "\n"
     "  isSideLayouts  : " << dl->isSideLayouts << "\n");

  if (sideSetEquation) {
    sideSetName = p.get<std::string>("Side Set Name");

    metric     = PHX::MDField<MeshScalarT>(p.get<std::string> ("Metric Name"), dl->qp_tensor);
    this->addDependentField(metric.fieldTag());
  }

  numNodes   = dl->node_scalar->extent(1);
  numQPs     = dl->qp_scalar->extent(1);

  gradDim = 2;

  this->addDependentField(u);
  this->addDependentField(grad_u);
  this->addDependentField(BF);
  this->addDependentField(GradBF);
  this->addDependentField(w_measure);

  this->addEvaluatedField(residual);

  this->setName("SideLaplacianResidual"+PHX::print<EvalT>());
}

//**********************************************************************
template<typename EvalT, typename Traits>
void SideLaplacianResidual<EvalT, Traits>::
postRegistrationSetup(typename Traits::SetupData d,
                      PHX::FieldManager<Traits>& fm)
{
  if (sideSetEquation) {
    this->utils.setFieldData(metric,fm);
  }

  this->utils.setFieldData(u,fm);
  this->utils.setFieldData(grad_u,fm);
  this->utils.setFieldData(BF,fm);
  this->utils.setFieldData(GradBF,fm);
  this->utils.setFieldData(w_measure,fm);
  this->utils.setFieldData(residual,fm);
}

//**********************************************************************
//Kokkos functor
template<typename EvalT, typename Traits>
KOKKOS_INLINE_FUNCTION
void SideLaplacianResidual<EvalT,Traits>::
operator() (const SideLaplacianResidual_Side_Tag& tag, const int& side) const {

  // Assembling the residual of -\Delta u = 1
  for (int node=0; node<numNodes; ++node) {
    ScalarT res = 0;
    for (int qp=0; qp<numQPs; ++qp) {
      // Laplace part
      for (int idim(0); idim<gradDim; ++idim) {
        for (int jdim(0); jdim<gradDim; ++jdim) {
          res -= grad_u(side,qp,idim)
               * metric(side,qp,idim,jdim)
               * GradBF(side,node,qp,jdim)
               * w_measure(side,qp);
        }
      }
      // Source term part
      res += BF(side,node,qp) * w_measure(side,qp);
    }
    residual(side,node) = res;
  }
}

template<typename EvalT, typename Traits>
KOKKOS_INLINE_FUNCTION
void SideLaplacianResidual<EvalT,Traits>::
operator() (const SideLaplacianResidual_Cell_Tag& tag, const int& cell) const {

  // Assembling the residual of -\Delta u = 1
  for (int node(0); node<numNodes; ++node) {
    ScalarT res = 0;
    for (int qp=0; qp<numQPs; ++qp) {
      // Laplace part
      for (int idim(0); idim<gradDim; ++idim) {
        res -= grad_u(cell,qp,idim)*GradBF(cell,node,qp,idim)*w_measure(cell,qp);
      }
      // Source term part
      res += BF(cell,node,qp)*w_measure(cell,qp);
    }
    residual(cell,node) = res;
  }

}

//**********************************************************************
template<typename EvalT, typename Traits>
void SideLaplacianResidual<EvalT, Traits>::evaluateFields (typename Traits::EvalData workset)
{
  residual.deep_copy (ScalarT(0));
  if (sideSetEquation)
    evaluateFieldsSide (workset);
  else
    evaluateFieldsCell (workset);
}

template<typename EvalT, typename Traits>
void SideLaplacianResidual<EvalT, Traits>::evaluateFieldsSide (typename Traits::EvalData workset)
{
  if (workset.sideSets->find(sideSetName)==workset.sideSets->end())
    return;

  auto sideSet = workset.sideSetViews->at(sideSetName);
  
  Kokkos::parallel_for(SideLaplacianResidual_Side_Policy(0, sideSet.size), *this);
}

template<typename EvalT, typename Traits>
void SideLaplacianResidual<EvalT,Traits>::evaluateFieldsCell (typename Traits::EvalData workset)
{
  Kokkos::parallel_for(SideLaplacianResidual_Cell_Policy(0, workset.numCells), *this);
}

} // namespace PHAL
