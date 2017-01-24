//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#include "Phalanx_DataLayout.hpp"
#include "Phalanx_TypeStrings.hpp"

namespace FELIX {

template<typename EvalT, typename Traits, bool IsStokes>
HydrologyResidualThicknessEqn<EvalT, Traits, IsStokes>::
HydrologyResidualThicknessEqn (const Teuchos::ParameterList& p,
                               const Teuchos::RCP<Albany::Layouts>& dl) :
  BF        (p.get<std::string> ("BF Name"), dl->node_qp_scalar),
  w_measure (p.get<std::string> ("Weighted Measure Name"), dl->qp_scalar),
  h         (p.get<std::string> ("Water Thickness QP Variable Name"), dl->qp_scalar),
  h_dot     (p.get<std::string> ("Water Thickness Dot QP Variable Name"), dl->qp_scalar),
  N         (p.get<std::string> ("Effective Pressure QP Variable Name"), dl->qp_scalar),
  m         (p.get<std::string> ("Melting Rate QP Variable Name"), dl->qp_scalar),
  u_b       (p.get<std::string> ("Sliding Velocity QP Variable Name"), dl->qp_scalar),
  residual  (p.get<std::string> ("Thickness Eqn Residual Name"),dl->node_scalar)
{
  if (IsStokes)
  {
    TEUCHOS_TEST_FOR_EXCEPTION (!dl->isSideLayouts, Teuchos::Exceptions::InvalidParameter,
                                "Error! The layout structure does not appear to be that of a side set.\n");

    numNodes = dl->node_scalar->dimension(2);
    numQPs   = dl->qp_scalar->dimension(2);

    sideSetName = p.get<std::string>("Side Set Name");
  }
  else
  {
    TEUCHOS_TEST_FOR_EXCEPTION (dl->isSideLayouts, Teuchos::Exceptions::InvalidParameter,
                                "Error! The layout structure appears to be that of a side set.\n");

    numNodes = dl->node_scalar->dimension(1);
    numQPs   = dl->qp_scalar->dimension(1);
  }

  this->addDependentField(BF.fieldTag());
  this->addDependentField(w_measure.fieldTag());
  this->addDependentField(h.fieldTag());
  this->addDependentField(N.fieldTag());
  this->addDependentField(m.fieldTag());
  this->addDependentField(u_b.fieldTag());

  unsteady = p.get<bool>("Unsteady");
  if (unsteady)
    this->addDependentField(h_dot.fieldTag());
  else
    this->addEvaluatedField(h_dot); // Will be set to zero

  this->addEvaluatedField(residual);

  // Setting parameters
  Teuchos::ParameterList& hydrology_params = *p.get<Teuchos::ParameterList*>("FELIX Hydrology");
  Teuchos::ParameterList& physical_params  = *p.get<Teuchos::ParameterList*>("FELIX Physical Parameters");

  double rho_i = physical_params.get<double>("Ice Density");
  h_r = hydrology_params.get<double>("Bed Bumps Height");
  l_r = hydrology_params.get<double>("Bed Bumps Length");
  A   = hydrology_params.get<double>("Flow Factor Constant");

  bool melting_cav = hydrology_params.get<bool>("Use Melting In Cavities Equation", false);
  use_eff_cav = (hydrology_params.get<bool>("Use Effective Cavities Height", true) ? 1.0 : 0.0);
  if (melting_cav)
    rho_i_inv = 1./rho_i;
  else
    rho_i_inv = 0;

  // Scalings, needed to account for different units: ice velocity
  // is in m/yr rather than m/s, while all other quantities are in SI units.
  double yr_to_s = 365.25*24*3600;
  A   *= 1./(1000*yr_to_s);    // Need to adjust A, which is given in k [kPa]^-n yr^-1, to [kPa]^-n s^-1.
  l_r *= yr_to_s;  // Since u_b is always divided by l_r, we simply scale l_r

  this->setName("HydrologyResidualThicknessEqn"+PHX::typeAsString<EvalT>());
}

template<typename EvalT, typename Traits, bool IsStokes>
void HydrologyResidualThicknessEqn<EvalT, Traits, IsStokes>::
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

  if (!unsteady)
    h_dot.deep_copy(ScalarT(0.0));
}

template<typename EvalT, typename Traits, bool IsStokes>
void HydrologyResidualThicknessEqn<EvalT, Traits, IsStokes>::
evaluateFields (typename Traits::EvalData workset)
{
  // h' = W_O - W_C = (m/rho_i + u_b*(h_b-h)/l_b) - AhN^n

  ScalarT res_node, res_qp;

  if (IsStokes)
  {
    // Zero out, to avoid leaving stuff from previous workset!
    residual.deep_copy(ScalarT(0.0));

    if (workset.sideSets->find(sideSetName)==workset.sideSets->end())
      return;

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
          res_qp = rho_i_inv*m(cell,side,qp) - (h_r - use_eff_cav*h(cell,side,qp))*u_b(cell,side,qp)/l_r
                 + h(cell,side,qp)*A*std::pow(N(cell,side,qp),3) - h_dot(cell,side,qp);

          res_node += res_qp * BF(cell,side,node,qp) * w_measure(cell,side,qp);
        }

        residual (cell,side,node) += res_node;
      }
    }
  }
  else
  {
    for (int cell=0; cell < workset.numCells; ++cell)
    {
      for (int node=0; node < numNodes; ++node)
      {
        res_node = 0;
        for (int qp=0; qp < numQPs; ++qp)
        {
          res_qp = rho_i_inv*m(cell,qp) + (h_r - use_eff_cav*h(cell,qp))*u_b(cell,qp)/l_r
                 - h(cell,qp)*A*std::pow(N(cell,qp),3) - h_dot(cell,qp);

          res_node += res_qp * BF(cell,node,qp) * w_measure(cell,qp);
        }
        residual (cell,node) += res_node;
      }
    }
  }
}

} // Namespace FELIX
