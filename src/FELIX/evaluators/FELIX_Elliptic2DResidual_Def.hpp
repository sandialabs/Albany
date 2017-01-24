//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#include "Phalanx_DataLayout.hpp"
#include "Phalanx_TypeStrings.hpp"

//uncomment the following line if you want debug output to be printed to screen
#define OUTPUT_TO_SCREEN

namespace FELIX {

//**********************************************************************
template<typename EvalT, typename Traits>
Elliptic2DResidual<EvalT, Traits>::Elliptic2DResidual (const Teuchos::ParameterList& p,
                                                       const Teuchos::RCP<Albany::Layouts>& dl) :
  u         (p.get<std::string> ("Solution QP Variable Name"), dl->qp_scalar),
  grad_u    (p.get<std::string> ("Solution Gradient QP Variable Name"), dl->qp_gradient),
  BF        (p.get<std::string> ("BF Variable Name"), dl->node_qp_scalar),
  GradBF    (p.get<std::string> ("Gradient BF Variable Name"), dl->node_qp_gradient),
  coords    (p.get<std::string> ("Coordinate Vector Variable Name"),dl->vertices_vector),
  residual  (p.get<std::string> ("Residual Variable Name"),dl->node_scalar),
  w_measure (p.get<std::string> ("Weighted Measure Variable Name"), dl->qp_scalar)
{
  sideSetEquation = p.get<bool>("Side Equation");
  if (sideSetEquation)
  {
    TEUCHOS_TEST_FOR_EXCEPTION (!dl->isSideLayouts, Teuchos::Exceptions::InvalidParameter,
                                "Error! The layout structure does not appear to be that of a side set.\n");

    inv_metric   = PHX::MDField<RealType,Cell,Side,QuadPoint,Dim,Dim>(p.get<std::string> ("Inverse Metric Name"), dl->qp_tensor);
    this->addDependentField(inv_metric.fieldTag());

    sideSetName = p.get<std::string>("Side Set Name");

    int numSides = dl->cell_gradient->dimension(1);
    numNodes     = dl->node_scalar->dimension(2);
    numQPs       = dl->qp_scalar->dimension(2);
    int sideDim  = dl->cell_gradient->dimension(2);

    // Index of the nodes on the sides in the numeration of the cell
    Teuchos::RCP<shards::CellTopology> cellType;
    cellType = p.get<Teuchos::RCP <shards::CellTopology> > ("Cell Type");
    sideNodes.resize(numSides);
    for (int side=0; side<numSides; ++side)
    {
      sideNodes[side].resize(numNodes);
      for (int node=0; node<numNodes; ++node)
        sideNodes[side][node] = cellType->getNodeMap(sideDim,side,node);
    }
  }
  else
  {
    numNodes     = dl->node_scalar->dimension(1);
    numQPs       = dl->qp_scalar->dimension(1);
  }

  gradDim = 2;

  this->addDependentField(u.fieldTag());
  this->addDependentField(grad_u.fieldTag());
  this->addDependentField(BF.fieldTag());
  this->addDependentField(GradBF.fieldTag());
  this->addDependentField(coords.fieldTag());
  this->addEvaluatedField(residual);

  this->setName("Elliptic2DResidual"+PHX::typeAsString<EvalT>());
}

//**********************************************************************
template<typename EvalT, typename Traits>
void Elliptic2DResidual<EvalT, Traits>::
postRegistrationSetup(typename Traits::SetupData d,
                      PHX::FieldManager<Traits>& fm)
{
  if (sideSetEquation)
  {
    this->utils.setFieldData(inv_metric,fm);
  }

  this->utils.setFieldData(coords,fm);
  this->utils.setFieldData(u,fm);
  this->utils.setFieldData(grad_u,fm);
  this->utils.setFieldData(BF,fm);
  this->utils.setFieldData(GradBF,fm);
  this->utils.setFieldData(w_measure,fm);
  this->utils.setFieldData(residual,fm);
}

//**********************************************************************
template<typename EvalT, typename Traits>
void Elliptic2DResidual<EvalT, Traits>::evaluateFields (typename Traits::EvalData workset)
{
  for (int cell(0); cell<workset.numCells; ++cell)
  {
    for (int node(0); node<numNodes; ++node)
    {
      residual(cell,node) = 0;
    }
  }

  if (sideSetEquation)
  {
    const Albany::SideSetList& ssList = *(workset.sideSets);
    Albany::SideSetList::const_iterator it_ss = ssList.find(sideSetName);

    if (it_ss==ssList.end())
      return;

    const std::vector<Albany::SideStruct>& sideSet = it_ss->second;
    std::vector<Albany::SideStruct>::const_iterator iter_s;
    for (iter_s=sideSet.begin(); iter_s!=sideSet.end(); ++iter_s)
    {
      // Get the local data of side and cell
      const int cell = iter_s->elem_LID;
      const int side = iter_s->side_local_id;

      std::vector<ScalarT> f_qp(numQPs,0);
      for (int qp=0; qp<numQPs; ++qp)
      {
        for (int idim=0; idim<gradDim; ++idim)
        {
          MeshScalarT x_qp = 0;
          for (int node=0; node<numNodes; ++node)
            x_qp += coords(cell,sideNodes[side][node],idim)*BF(cell,side,node,qp);

          f_qp[qp] += std::pow(x_qp-0.5,2);
        }
        f_qp[qp] -= 4.0;
      }

      for (int node=0; node<numNodes; ++node)
      {
        for (int qp=0; qp<numQPs; ++qp)
        {
          for (int idim(0); idim<gradDim; ++idim)
          {
            for (int jdim(0); jdim<gradDim; ++jdim)
            {
              residual(cell,sideNodes[side][node]) -= grad_u(cell,side,qp,idim)
                                                    * inv_metric(cell,side,qp,idim,jdim)
                                                    * GradBF(cell,side,node,qp,jdim)
                                                    * w_measure(cell,side,qp);
            }
          }
          residual(cell,sideNodes[side][node]) -= u(cell,side,qp) * BF(cell,side,node,qp) * w_measure(cell,side,qp);
          residual(cell,sideNodes[side][node]) += f_qp[qp] * BF(cell,side,node,qp) * w_measure(cell,side,qp);
        }
      }
    }
  }
  else
  {
    for (int cell(0); cell<workset.numCells; ++cell)
    {
      std::vector<ScalarT> f_qp(numQPs,0);

      for (int qp=0; qp<numQPs; ++qp)
      {
        for (int idim=0; idim<gradDim; ++idim)
        {
          MeshScalarT x_qp = 0;
          for (int node=0; node<numNodes; ++node)
            x_qp += coords(cell,node,idim)*BF(cell,node,qp);

          f_qp[qp] += std::pow(x_qp-0.5,2);
        }
        f_qp[qp] -= 4.0;
      }

      for (int node(0); node<numNodes; ++node)
      {
        residual(cell,node) = 0;
        for (int qp=0; qp<numQPs; ++qp)
        {
          for (int idim(0); idim<gradDim; ++idim)
          {
            residual(cell,node) -= grad_u(cell,qp,idim)*GradBF(cell,node,qp,idim)*w_measure(cell,qp);
          }
          residual(cell,node) -= u(cell,qp)*BF(cell,node,qp)*w_measure(cell,qp);
          residual(cell,node) += f_qp[qp]*BF(cell,node,qp)*w_measure(cell,qp);
        }
      }
    }
  }
}

} // Namespace FELIX
