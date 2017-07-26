//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#include "Phalanx_DataLayout.hpp"
#include "Phalanx_TypeStrings.hpp"

//uncomment the following line if you want debug output to be printed to screen
#define OUTPUT_TO_SCREEN

namespace PHAL {

//**********************************************************************
template<typename EvalT, typename Traits>
SideLaplacianResidual<EvalT, Traits>::SideLaplacianResidual (const Teuchos::ParameterList& p,
                                                             const Teuchos::RCP<Albany::Layouts>& dl) :
  residual (p.get<std::string> ("Residual Variable Name"),dl->node_scalar)
{
  sideSetEquation = p.get<bool>("Side Equation");
  if (sideSetEquation)
  {
    sideSetName = p.get<std::string>("Side Set Name");

    TEUCHOS_TEST_FOR_EXCEPTION (dl->side_layouts.find(sideSetName)==dl->side_layouts.end(), std::runtime_error,
                                "Error! The layout structure does not appear to store the layout for side set " << sideSetName << "\n");

    auto dl_side = dl->side_layouts.at(sideSetName);

    u          = PHX::MDField<ScalarT>(p.get<std::string> ("Solution QP Variable Name"), dl_side->qp_scalar);
    grad_u     = PHX::MDField<ScalarT>(p.get<std::string> ("Solution Gradient QP Variable Name"), dl_side->qp_gradient);
    BF         = PHX::MDField<RealType>(p.get<std::string> ("BF Variable Name"), dl_side->node_qp_scalar);
    GradBF     = PHX::MDField<RealType>(p.get<std::string> ("Gradient BF Variable Name"), dl_side->node_qp_gradient);
    w_measure  = PHX::MDField<RealType>(p.get<std::string> ("Weighted Measure Variable Name"), dl_side->qp_scalar);
    metric     = PHX::MDField<RealType,Cell,Side,QuadPoint,Dim,Dim>(p.get<std::string> ("Metric Name"), dl_side->qp_tensor);
    this->addDependentField(metric.fieldTag());

    int numSides = dl_side->cell_gradient->dimension(1);
    numNodes     = dl_side->node_scalar->dimension(2);
    numQPs       = dl_side->qp_scalar->dimension(2);
    int sideDim  = dl_side->cell_gradient->dimension(2);

    // Index of the nodes on the sides in the numeration of the cell
    Teuchos::RCP<shards::CellTopology> cellType;
    cellType = p.get<Teuchos::RCP <shards::CellTopology> > ("Cell Type");
    sideNodes.resize(numSides);
    for (int side=0; side<numSides; ++side)
    {
      // Need to get the subcell exact count, since different sides may have different number of nodes (e.g., Wedge)
      int thisSideNodes = cellType->getNodeCount(sideDim,side);
      sideNodes[side].resize(thisSideNodes);
      for (int node=0; node<thisSideNodes; ++node)
        sideNodes[side][node] = cellType->getNodeMap(sideDim,side,node);
    }
  }
  else
  {
    u          = PHX::MDField<ScalarT>(p.get<std::string> ("Solution QP Variable Name"), dl->qp_scalar);
    grad_u     = PHX::MDField<ScalarT>(p.get<std::string> ("Solution Gradient QP Variable Name"), dl->qp_gradient);
    BF         = PHX::MDField<RealType>(p.get<std::string> ("BF Variable Name"), dl->node_qp_scalar);
    GradBF     = PHX::MDField<RealType>(p.get<std::string> ("Gradient BF Variable Name"), dl->node_qp_gradient);
    w_measure  = PHX::MDField<RealType>(p.get<std::string> ("Weighted Measure Variable Name"), dl->qp_scalar);

    numNodes   = dl->node_scalar->dimension(1);
    numQPs     = dl->qp_scalar->dimension(1);
  }

  spaceDim = 3;
  gradDim  = 2;

  this->addDependentField(u.fieldTag());
  this->addDependentField(grad_u.fieldTag());
  this->addDependentField(BF.fieldTag());
  this->addDependentField(GradBF.fieldTag());

  this->addEvaluatedField(residual);

  this->setName("SideLaplacianResidual"+PHX::typeAsString<EvalT>());
}

//**********************************************************************
template<typename EvalT, typename Traits>
void SideLaplacianResidual<EvalT, Traits>::
postRegistrationSetup(typename Traits::SetupData d,
                      PHX::FieldManager<Traits>& fm)
{
  if (sideSetEquation)
  {
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

  const std::vector<Albany::SideStruct>& sideSet = workset.sideSets->at(sideSetName);
  for (auto const& it_side : sideSet)
  {
    // Get the local data of side and cell
    const int cell = it_side.elem_LID;
    const int side = it_side.side_local_id;

    // Assembling the residual of -\Delta u + u = f
    for (int node=0; node<numNodes; ++node)
    {
      for (int qp=0; qp<numQPs; ++qp)
      {
        for (int idim(0); idim<gradDim; ++idim)
        {
          for (int jdim(0); jdim<gradDim; ++jdim)
          {
            residual(cell,sideNodes[side][node]) -= grad_u(cell,side,qp,idim)
                                                  * metric(cell,side,qp,idim,jdim)
                                                  * GradBF(cell,side,node,qp,jdim)
                                                  * w_measure(cell,side,qp);
          }
        }
        residual(cell,sideNodes[side][node]) += 1.0 * BF(cell,side,node,qp) * w_measure(cell,side,qp);
      }
    }
  }
}

template<typename EvalT, typename Traits>
void SideLaplacianResidual<EvalT,Traits>::evaluateFieldsCell (typename Traits::EvalData workset)
{
  for (int cell(0); cell<workset.numCells; ++cell)
  {
    // Assembling the residual of -\Delta u + u = f
    for (int node(0); node<numNodes; ++node)
    {
      residual(cell,node) = 0;
      for (int qp=0; qp<numQPs; ++qp)
      {
        for (int idim(0); idim<gradDim; ++idim)
        {
          residual(cell,node) -= grad_u(cell,qp,idim)*GradBF(cell,node,qp,idim)*w_measure(cell,qp);
        }
        residual(cell,node) += 1.0*BF(cell,node,qp)*w_measure(cell,qp);
      }
    }
  }
}

} // namespace PHAL
