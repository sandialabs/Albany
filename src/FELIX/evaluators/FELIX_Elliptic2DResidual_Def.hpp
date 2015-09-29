//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#include "Phalanx_DataLayout.hpp"
#include "Phalanx_TypeStrings.hpp"

#include "FELIX_HomotopyParameter.hpp"

//uncomment the following line if you want debug output to be printed to screen
#define OUTPUT_TO_SCREEN

namespace FELIX {

//**********************************************************************
template<typename EvalT, typename Traits>
Elliptic2DResidual<EvalT, Traits>::Elliptic2DResidual (const Teuchos::ParameterList& p,
                                                       const Teuchos::RCP<Albany::Layouts>& dl) :
  coords   (p.get<std::string> ("Coordinate Vector Variable Name"),dl->vertices_vector),
  residual (p.get<std::string> ("Residual Variable Name"),dl->node_scalar)
{
  sideSetEquation = p.get<bool>("Side Equation");
  if (sideSetEquation)
  {
    u_node       = PHX::MDField<ScalarT,Cell,Node>(p.get<std::string> ("Solution Variable Name"), dl->node_scalar);
    u_side       = PHX::MDField<ScalarT,Cell,Side,QuadPoint>(p.get<std::string> ("Solution QP Variable Name"), dl->side_qp_scalar);
    grad_u_side  = PHX::MDField<ScalarT,Cell,Side,QuadPoint,Dim>(p.get<std::string> ("Solution Gradient QP Variable Name"), dl->side_qp_gradient);
    inv_metric   = PHX::MDField<RealType,Cell,Side,QuadPoint,Dim,Dim>(p.get<std::string> ("Inverse Metric Name"), dl->side_qp_tensor);
    BF_side      = PHX::MDField<RealType,Cell,Side,Node,QuadPoint>(p.get<std::string> ("BF Variable Name"), dl->side_node_qp_scalar);
    GradBF_side  = PHX::MDField<RealType,Cell,Side,Node,QuadPoint,Dim>(p.get<std::string> ("Gradient BF Variable Name"), dl->side_node_qp_gradient);
    w_measure    = PHX::MDField<RealType,Cell,Side,QuadPoint>(p.get<std::string> ("Weighted Measure Variable Name"), dl->side_qp_scalar);

    this->addDependentField(u_node);
    this->addDependentField(u_side);
    this->addDependentField(BF_side);
    this->addDependentField(inv_metric);
    this->addDependentField(grad_u_side);
    this->addDependentField(w_measure);
  }
  else
  {
    u       = PHX::MDField<ScalarT,Cell,QuadPoint>(p.get<std::string> ("Solution QP Variable Name"), dl->qp_scalar);
    grad_u  = PHX::MDField<ScalarT,Cell,QuadPoint,Dim>(p.get<std::string> ("Solution Gradient QP Variable Name"), dl->qp_gradient);
    BF      = PHX::MDField<RealType,Cell,Node,QuadPoint>(p.get<std::string> ("BF Variable Name"), dl->node_qp_scalar);
    wBF     = PHX::MDField<RealType,Cell,Node,QuadPoint>(p.get<std::string> ("Weighted BF Variable Name"), dl->node_qp_scalar);
    wGradBF = PHX::MDField<RealType,Cell,Node,QuadPoint,Dim>(p.get<std::string> ("Weighted Gradient BF Variable Name"), dl->node_qp_gradient);

    this->addDependentField(u);
    this->addDependentField(grad_u);
    this->addDependentField(BF);
    this->addDependentField(wBF);
    this->addDependentField(wGradBF);
  }

  this->addDependentField(coords);
  this->addEvaluatedField(residual);

  std::vector<PHX::DataLayout::size_type> dims;
  dl->side_node_qp_gradient->dimensions(dims);
  int numSides = dims[1];
  numSideNodes = dims[2];
  numSideQPs   = dims[3];
  int sideDim  = dims[4];

  gradDim = 2;

  if (sideSetEquation)
  {
    sideSetName = p.get<std::string>("Side Set Name");

    // Index of the nodes on the sides in the numeration of the cell
    Teuchos::RCP<shards::CellTopology> cellType;
    cellType = p.get<Teuchos::RCP <shards::CellTopology> > ("Cell Type");
    sideNodes.resize(numSides);
    for (int side=0; side<numSides; ++side)
    {
      sideNodes[side].resize(numSideNodes);
      for (int node=0; node<numSideNodes; ++node)
        sideNodes[side][node] = cellType->getNodeMap(sideDim,side,node);
    }
  }

  dl->node_qp_scalar->dimensions(dims);
  numNodes     = dims[1];
  numQPs       = dims[2];

  this->setName("Elliptic2DResidual"+PHX::typeAsString<EvalT>());
}

//**********************************************************************
template<typename EvalT, typename Traits>
void Elliptic2DResidual<EvalT, Traits>::
postRegistrationSetup(typename Traits::SetupData d,
                      PHX::FieldManager<Traits>& fm)
{
  this->utils.setFieldData(residual,fm);

  this->utils.setFieldData(coords,fm);

  if (sideSetEquation)
  {
    this->utils.setFieldData(u_node,fm);
    this->utils.setFieldData(u_side,fm);
    this->utils.setFieldData(BF_side,fm);
    this->utils.setFieldData(GradBF_side,fm);
    this->utils.setFieldData(inv_metric,fm);
    this->utils.setFieldData(grad_u_side,fm);
    this->utils.setFieldData(w_measure,fm);
  }
  else
  {
    this->utils.setFieldData(u,fm);
    this->utils.setFieldData(grad_u,fm);
    this->utils.setFieldData(wBF,fm);
    this->utils.setFieldData(BF,fm);
    this->utils.setFieldData(wGradBF,fm);
  }
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

  std::vector<std::vector<bool> > done (workset.numCells,std::vector<bool>(numNodes,false));

  std::set<int> dones;
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

      std::vector<ScalarT> f_qp(numSideQPs,0);
      for (int qp=0; qp<numSideQPs; ++qp)
      {
        for (int idim=0; idim<gradDim; ++idim)
        {
          MeshScalarT x_qp = 0;
          for (int node=0; node<numSideNodes; ++node)
            x_qp += coords(cell,sideNodes[side][node],idim)*BF_side(cell,side,node,qp);

          f_qp[qp] += std::pow(x_qp-0.5,2);
        }
        f_qp[qp] -= 4.0;
      }

      for (int node=0; node<numSideNodes; ++node)
      {
        for (int qp=0; qp<numSideQPs; ++qp)
        {
          for (int idim(0); idim<gradDim; ++idim)
          {
            for (int jdim(0); jdim<gradDim; ++jdim)
            {
              residual(cell,sideNodes[side][node]) -= grad_u_side(cell,side,qp,idim)
                                                    * inv_metric(cell,side,qp,idim,jdim)
                                                    * GradBF_side(cell,side,node,qp,jdim)
                                                    * w_measure(cell,side,qp);
            }
          }
          residual(cell,sideNodes[side][node]) -= u_side(cell,side,qp) * BF_side(cell,side,node,qp) * w_measure(cell,side,qp);
          residual(cell,sideNodes[side][node]) += f_qp[qp] * BF_side(cell,side,node,qp) * w_measure(cell,side,qp);
        }
        done [cell][sideNodes[side][node]] = true;
        dones.insert(workset.wsElNodeEqID[cell][sideNodes[side][node]][0]);
      }
    }
/*
    for (int cell(0); cell<workset.numCells; ++cell)
    {
      for (int node(0); node<numNodes; ++node)
      {
//        if (!done[cell][node])
        if (dones.find (workset.wsElNodeEqID[cell][node][0])==dones.end())
        {
          residual(cell,node) = u_node(cell,node);
        }
      }
    }
*/
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
            residual(cell,node) -= grad_u(cell,qp,idim)*wGradBF(cell,node,qp,idim);
          }
          residual(cell,node) -= u(cell,qp)*wBF(cell,node,qp);
          residual(cell,node) += f_qp[qp]*wBF(cell,node,qp);
        }
      }
    }
  }
}

} // Namespace FELIX
