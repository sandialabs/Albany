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
HydrologyResidualEvolutionEqn<EvalT, Traits>::HydrologyResidualEvolutionEqn (const Teuchos::ParameterList& p,
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
    w_measure   = PHX::MDField<MeshScalarT>(p.get<std::string> ("Weighted Measure Name"), dl->side_qp_scalar);

    h           = PHX::MDField<ScalarT>(p.get<std::string> ("Drainage Sheet Depth QP Variable Name"), dl->side_qp_scalar);
    h_dot       = PHX::MDField<ScalarT>(p.get<std::string> ("Drainage Sheet Depth Dot QP Variable Name"), dl->side_qp_scalar);
    N           = PHX::MDField<ScalarT>(p.get<std::string> ("Effective Pressure QP Variable Name"), dl->side_qp_scalar);
    m           = PHX::MDField<ScalarT>(p.get<std::string> ("Melting Rate QP Variable Name"), dl->side_qp_scalar);
    u_b         = PHX::MDField<ScalarT>(p.get<std::string> ("Sliding Velocity QP Name"), dl->side_qp_scalar);

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
    w_measure   = PHX::MDField<MeshScalarT>(p.get<std::string> ("Weighted Measure Name"), dl->side_qp_scalar);

    h           = PHX::MDField<ScalarT>(p.get<std::string> ("Drainage Sheet Depth QP Variable Name"), dl->qp_scalar);
    h_dot       = PHX::MDField<ScalarT>(p.get<std::string> ("Drainage Sheet Depth Dot QP Variable Name"), dl->qp_scalar);
    N           = PHX::MDField<ScalarT>(p.get<std::string> ("Effective Pressure QP Variable Name"), dl->qp_scalar);
    m           = PHX::MDField<ScalarT>(p.get<std::string> ("Melting Rate QP Variable Name"), dl->qp_scalar);
    u_b         = PHX::MDField<ScalarT>(p.get<std::string> ("Sliding Velocity QP Name"), dl->qp_scalar);

    numNodes = dl->node_scalar->dimension(1);
    numQPs   = dl->qp_scalar->dimension(1);
    numDims  = dl->qp_gradient->dimension(2);
  }

  this->addDependentField(BF);
  this->addDependentField(w_measure);
  this->addDependentField(h);
  this->addDependentField(h_dot);
  this->addDependentField(N);
  this->addDependentField(m);
  this->addDependentField(u_b);

  residual = PHX::MDField<ScalarT,Cell,Node>(p.get<std::string> ("Hydrology Evolution Eqn Residual Name"),dl->node_scalar);

  this->addEvaluatedField(residual);

  // Setting parameters
  Teuchos::ParameterList& hydrology_params = *p.get<Teuchos::ParameterList*>("FELIX Hydrology");
  Teuchos::ParameterList& physical_params  = *p.get<Teuchos::ParameterList*>("FELIX Physical Parameters");

  rho_i = physical_params.get<double>("Ice Density");
  h_r = hydrology_params.get<double>("Bed Bumps Height");
  l_r = hydrology_params.get<double>("Bed Bumps Length");
  n   = hydrology_params.get<double>("Elliptic Eqn Mass Term Exponent");
  A   = hydrology_params.get<double>("Flow Factor Constant");

  this->setName("HydrologyResidualEvolutionEqn"+PHX::typeAsString<EvalT>());
}

template<typename EvalT, typename Traits>
void HydrologyResidualEvolutionEqn<EvalT, Traits>::
postRegistrationSetup(typename Traits::SetupData d,
                      PHX::FieldManager<Traits>& fm)
{
  this->utils.setFieldData(BF,fm);
  this->utils.setFieldData(w_measure,fm);
  this->utils.setFieldData(h,fm);
  this->utils.setFieldData(h_dot,fm);
  this->utils.setFieldData(N,fm);
  this->utils.setFieldData(m,fm);
  this->utils.setFieldData(u_b,fm);

  this->utils.setFieldData(residual,fm);
}

template<typename EvalT, typename Traits>
void HydrologyResidualEvolutionEqn<EvalT, Traits>::evaluateFields (typename Traits::EvalData workset)
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
          res_qp = m(cell,side,qp)/rho_i - (h_r -h(cell,side,qp))*u_b(cell,side,qp)/l_r
                 + h(cell,side,qp)*A*std::pow(N(cell,side,qp),n) - h_dot(cell,side,qp);

          res_node += res_qp * BF(cell,side,node,qp) * w_measure(cell,side,qp);
        }

        residual (cell,sideNodes[side][node]) = res_node;
      }
    }
  }
  else
  {
    ScalarT res_node, res_qp;

    // h' = W_O - W_C = (m/rho_i + u_b*(h_b-h)/l_b) - AhN^n
    for (int cell=0; cell < workset.numCells; ++cell)
    {
      for (int node=0; node < numNodes; ++node)
      {
        res_node = 0;
        for (int qp=0; qp < numQPs; ++qp)
        {
          res_qp = m(cell,qp)/rho_i - (h_r -h(cell,qp))*u_b(cell,qp)/l_r
                 + h(cell,qp)*A*std::pow(N(cell,qp),n) - h_dot(cell,qp);

          res_node += res_qp * BF(cell,node,qp) * w_measure(cell,qp);
        }

        residual (cell,node) = res_node;
      }
    }
  }
}

} // Namespace FELIX
