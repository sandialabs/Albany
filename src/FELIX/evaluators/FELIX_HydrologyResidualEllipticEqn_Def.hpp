//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#include "Phalanx_DataLayout.hpp"
#include "Phalanx_TypeStrings.hpp"

namespace FELIX {

//**********************************************************************
template<typename EvalT, typename Traits>
HydrologyResidualEllipticEqn<EvalT, Traits>::HydrologyResidualEllipticEqn (const Teuchos::ParameterList& p,
                                                     const Teuchos::RCP<Albany::Layouts>& dl)
{
  if (p.isParameter("Stokes Coupling"))
  {
    stokes_coupling = p.get<bool>("Stokes Coupling");
  }
  else
  {
    stokes_coupling = false;
  }

  if (stokes_coupling)
  {
    BF          = PHX::MDField<RealType>(p.get<std::string> ("BF Name"), dl->side_node_qp_scalar);
    GradBF      = PHX::MDField<RealType>(p.get<std::string> ("Gradient BF Name"), dl->side_node_qp_gradient);
    w_measure   = PHX::MDField<MeshScalarT>(p.get<std::string> ("Weighted Measure Name"), dl->side_qp_scalar);

    q           = PHX::MDField<ScalarT>(p.get<std::string> ("Water Discharge QP Variable Name"), dl->side_qp_gradient);
    N           = PHX::MDField<ScalarT>(p.get<std::string> ("Effective Pressure QP Variable Name"), dl->side_qp_scalar);
    h           = PHX::MDField<ScalarT>(p.get<std::string> ("Drainage Sheet Depth QP Variable Name"), dl->side_qp_scalar);
    m           = PHX::MDField<ScalarT>(p.get<std::string> ("Melting Rate QP Variable Name"), dl->side_qp_scalar);
    omega       = PHX::MDField<ScalarT>(p.get<std::string> ("Surface Water Input QP Variable Name"), dl->side_qp_scalar);
    u_b         = PHX::MDField<ScalarT>(p.get<std::string> ("Sliding Velocity QP Variable Name"), dl->side_qp_scalar);

    numNodes = dl->side_node_scalar->dimension(2);
    numQPs   = dl->side_qp_scalar->dimension(2);
    numDims  = dl->side_qp_gradient->dimension(3);

    sideSetName = p.get<std::string>("Side Set Name");

    // Index of the nodes on the sides in the numeration of the cell
    int numSides = dl->side_node_scalar->dimension(1);
    int sideDim  = dl->side_qp_gradient->dimension(3);

    Teuchos::RCP<shards::CellTopology> cellType;
    cellType = p.get<Teuchos::RCP <shards::CellTopology> > ("Cell Type");
    sideNodes.resize(numSides);
    for (int side=0; side<numSides; ++side)
    {
      // Need to get the subcell exact count, since different sides may have different number of nodes (e.g., Wedge)
      int thisSideNodes = cellType->getNodeCount(sideDims,side);
      sideNodes[side].resize(thisSideNodes);
      for (int node=0; node<thisSideNodes; ++node)
      {
        sideNodes[side][node] = cellType->getNodeMap(sideDims,side,node);
      }
    }
  }
  else
  {
    BF          = PHX::MDField<RealType>(p.get<std::string> ("BF Name"), dl->node_qp_scalar);
    GradBF      = PHX::MDField<RealType>(p.get<std::string> ("Gradient BF Name"), dl->node_qp_gradient);
    w_measure   = PHX::MDField<MeshScalarT>(p.get<std::string> ("Weighted Measure Name"), dl->qp_scalar);

    q           = PHX::MDField<ScalarT>(p.get<std::string> ("Water Discharge QP Variable Name"), dl->qp_gradient);
    N           = PHX::MDField<ScalarT>(p.get<std::string> ("Effective Pressure QP Variable Name"), dl->qp_scalar);
    h           = PHX::MDField<ScalarT>(p.get<std::string> ("Drainage Sheet Depth QP Variable Name"), dl->qp_scalar);
    m           = PHX::MDField<ScalarT>(p.get<std::string> ("Melting Rate QP Variable Name"), dl->qp_scalar);
    omega       = PHX::MDField<ScalarT>(p.get<std::string> ("Surface Water Input QP Variable Name"), dl->qp_scalar);
    u_b         = PHX::MDField<ScalarT>(p.get<std::string> ("Sliding Velocity QP Variable Name"), dl->qp_scalar);

    numNodes = dl->node_scalar->dimension(1);
    numQPs   = dl->qp_scalar->dimension(1);
    numDims  = dl->qp_gradient->dimension(2);
  }

  this->addDependentField(BF);
  this->addDependentField(GradBF);
  this->addDependentField(w_measure);
  this->addDependentField(q);
  this->addDependentField(N);
  this->addDependentField(h);
  this->addDependentField(m);
  this->addDependentField(omega);
  this->addDependentField(u_b);

  residual = PHX::MDField<ScalarT,Cell,Node>(p.get<std::string> ("Hydrology Elliptic Eqn Residual Name"),dl->node_scalar);

  this->addEvaluatedField(residual);

  // Setting parameters
  Teuchos::ParameterList& hydrology_params = *p.get<Teuchos::ParameterList*>("FELIX Hydrology Parameters");
  Teuchos::ParameterList& physical_params  = *p.get<Teuchos::ParameterList*>("FELIX Physical Parameters");

  double rho_i = physical_params.get<double>("Ice Density", 910.0);
  double rho_w = physical_params.get<double>("Water Density", 1028.0);

  rho_combo = 1./rho_w - 1./rho_i;
  mu_w      = physical_params.get<double>("Water Viscosity");
  h_r       = hydrology_params.get<double>("Bed Bumps Height");
  l_r       = hydrology_params.get<double>("Bed Bumps Length");
  n         = hydrology_params.get<double>("Elliptic Eqn Mass Term Exponent");
  A         = hydrology_params.get<double>("Flow Factor Constant");

  this->setName("HydrologyResidualEllipticEqn"+PHX::typeAsString<EvalT>());
}

//**********************************************************************
template<typename EvalT, typename Traits>
void HydrologyResidualEllipticEqn<EvalT, Traits>::
postRegistrationSetup(typename Traits::SetupData d,
                      PHX::FieldManager<Traits>& fm)
{
  this->utils.setFieldData(BF,fm);
  this->utils.setFieldData(GradBF,fm);
  this->utils.setFieldData(w_measure,fm);
  this->utils.setFieldData(q,fm);
  this->utils.setFieldData(N,fm);
  this->utils.setFieldData(h,fm);
  this->utils.setFieldData(m,fm);
  this->utils.setFieldData(omega,fm);
  this->utils.setFieldData(u_b,fm);

  this->utils.setFieldData(residual,fm);
}

//**********************************************************************
template<typename EvalT, typename Traits>
void HydrologyResidualEllipticEqn<EvalT, Traits>::evaluateFields (typename Traits::EvalData workset)
{
  if (stokes_coupling)
  {
    // Zero out, to avoid leaving stuff from previous workset!
    const int numCellNodes = residual.fieldTag().dataLayout().dimension(1);
    for (int cell=0; cell<workset.numCells; ++cell)
      for (int node=0; node<numCellNodes; ++node)
        residual(cell,node) = 0;

    if (workset.sideSets->find(sideSetName)==workset.sideSets->end())
      return;

    ScalarT res_qp, res_node;
    const std::vector<Albany::SideStruct>& sideSet = workset.sideSets->at(sideSetName);
    for (auto const& it_side : sideSet)
    {
      // Get the local data of side and cell
      const int cell = it_side.elem_LID;
      const int side = it_side.side_local_id;

      for (int node=0; node < numNodes; ++node)
      {
        res_node = 0;
        for (int qp=0; qp < numQPs; ++qp)
        {
          res_qp = rho_combo*m(cell,side,qp) + omega(cell,side,qp)
                 - (h_r -h(cell,side,qp))*u_b(cell,side,qp)/l_r
                 + h(cell,side,qp)*A*std::pow(N(cell,side,qp),n);

          res_qp *= BF(cell,side,node,qp);

          for (int dim=0; dim<numDims; ++dim)
          {
            res_qp += q(cell,side,qp,dim) * GradBF(cell,side,node,qp,dim);
          }

          res_node += res_qp * w_measure(cell,side,qp);
        }
        residual (cell,sideNodes[side][node]) += res_node;
      }
    }
  }
  else
  {
    ScalarT res_qp, res_node;
    for (int cell=0; cell < workset.numCells; ++cell)
    {
      for (int node=0; node < numNodes; ++node)
      {
        res_node = 0;
        for (int qp=0; qp < numQPs; ++qp)
        {
          res_qp = rho_combo*m(cell,qp) + omega(cell,qp)
                 - (h_r -h(cell,qp))*u_b(cell,qp)/l_r
                 + h(cell,qp)*A*std::pow(N(cell,qp),n);

          res_qp *= BF(cell,node,qp);

          for (int dim=0; dim<numDims; ++dim)
          {
            res_qp += q(cell,qp,dim) * GradBF(cell,node,qp,dim);
          }

          res_node += res_qp * w_measure(cell,qp);
        }

        residual (cell,node) += res_node;
      }
    }
  }
}

} // Namespace FELIX
