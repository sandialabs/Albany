//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#include "Phalanx_DataLayout.hpp"
#include "Phalanx_TypeStrings.hpp"

namespace FELIX {

//**********************************************************************
template<typename EvalT, typename Traits, bool IsStokesCoupling, bool ThermoCoupled>
HydrologyResidualMassEqn<EvalT, Traits, IsStokesCoupling, ThermoCoupled>::
HydrologyResidualMassEqn (const Teuchos::ParameterList& p,
                          const Teuchos::RCP<Albany::Layouts>& dl) :
  BF        (p.get<std::string> ("BF Name"), dl->node_qp_scalar),
  GradBF    (p.get<std::string> ("Gradient BF Name"), dl->node_qp_gradient),
  w_measure (p.get<std::string> ("Weighted Measure Name"), dl->qp_scalar),
  q         (p.get<std::string> ("Water Discharge Variable Name"), dl->qp_gradient),
  omega     (p.get<std::string> ("Surface Water Input Variable Name"), dl->qp_scalar),
  residual  (p.get<std::string> ("Mass Eqn Residual Name"),dl->node_scalar)
{
  /*
   *  The mass conserbation equation has the following (strong) form
   *
   *     dh/dt + div(q) = m/rho_w + omega
   *
   *  where q is the water discharge, h the water thickness, rho_w is the water density,
   *  m the melting rate of the ice (due to geothermal flow and sliding), and omega
   *  is  the water source (water reaching the bed from the surface, through crevasses)
   */

  if (IsStokesCoupling)
  {
    TEUCHOS_TEST_FOR_EXCEPTION (!dl->isSideLayouts, Teuchos::Exceptions::InvalidParameter,
                                "Error! The layout structure does not appear to be that of a side set.\n");

    numNodes = dl->node_scalar->dimension(2);
    numQPs   = dl->qp_scalar->dimension(2);
    numDims  = dl->qp_gradient->dimension(3);

    sideSetName = p.get<std::string>("Side Set Name");

    metric = PHX::MDField<const MeshScalarT,Cell,Side,QuadPoint,Dim,Dim>(p.get<std::string>("Metric Name"),dl->qp_tensor);
    this->addDependentField(metric);
  }
  else
  {
    TEUCHOS_TEST_FOR_EXCEPTION (dl->isSideLayouts, Teuchos::Exceptions::InvalidParameter,
                                "Error! The layout structure appears to be that of a side set.\n");

    numNodes = dl->node_scalar->dimension(1);
    numQPs   = dl->qp_scalar->dimension(1);
    numDims  = dl->qp_gradient->dimension(2);
  }

  this->addEvaluatedField(residual);

  // Setting parameters
  Teuchos::ParameterList& hydrology_params = *p.get<Teuchos::ParameterList*>("FELIX Hydrology Parameters");
  Teuchos::ParameterList& physical_params  = *p.get<Teuchos::ParameterList*>("FELIX Physical Parameters");

  rho_w       = physical_params.get<double>("Water Density", 1028.0);
  use_melting = hydrology_params.get<bool>("Use Melting In Conservation Of Mass", false);

  unsteady = p.get<bool>("Unsteady");
  if (unsteady)
  {
    h_dot = PHX::MDField<const ScalarT>(p.get<std::string> ("Water Thickness Dot Variable Name"), dl->qp_scalar);
    this->addDependentField(h_dot);
  }

  penalization_coeff = hydrology_params.isParameter("Water Pressure Bounds Penalization Coefficient")
                     ? hydrology_params.get<double>("Water Pressure Bounds Penalization Coefficient") : 0.0;
  TEUCHOS_TEST_FOR_EXCEPTION (penalization_coeff<0.0, Teuchos::Exceptions::InvalidParameter, "Error! Penalization coefficient must be positive.\n");
  penalization = (penalization_coeff!=0.0);
  if (penalization) {
    P_w = PHX::MDField<const ScalarT>(p.get<std::string> ("Water Pressure Variable Name"), dl->node_scalar);
    P_o = PHX::MDField<const ParamScalarT>(p.get<std::string>("Ice Overburden Variable Name"),dl->node_scalar);
    this->addDependentField(P_w);
    this->addDependentField(P_o);
  }

  Teuchos::RCP<PHX::DataLayout> layout;
  if (use_melting) {
    mass_lumping = hydrology_params.isParameter("Mass Lumping") ? hydrology_params.get<bool>("Mass Lumping") : false;
  } else {
    mass_lumping = false;
  }

  if (mass_lumping) {
    layout = dl->node_scalar;
  } else {
    layout = dl->qp_scalar;
  }

  if (use_melting) {
    m = PHX::MDField<const ScalarT>(p.get<std::string> ("Melting Rate Variable Name"), layout);
    this->addDependentField(m);
  }

  this->addDependentField(BF);
  this->addDependentField(GradBF);
  this->addDependentField(w_measure);
  this->addDependentField(q);
  this->addDependentField(omega);

  /*
   * Scalings, needed to account for different units: ice velocity
   * is in m/yr, the mesh is in km, and hydrology space/time unit are m/s.
   *
   * The residual has 4 terms (here in strong form, without signs), with the following units:
   *
   *  1) dh/dt            [m  s^-1  ]
   *  2) div(q)           [mm s^-1  ]
   *  3) m/rho_w          [m  yr^-1 ]
   *  4) omega            [mm day^-1]
   *
   * where q=k*h^alpha*|gradPhi|^(beta-2)*gradPhi.
   * We decide to uniform all terms to have units [m yr^-1].
   * Where possible, we do this by rescaling some constants. Otherwise,
   * we simply introduce a new scaling factor
   *
   *  1) dh/dt                  scaling_h_dot = yr_to_s
   *  2) scaling_q*div(q)       scaling_q     = 1e-3*yr_to_s
   *  3) m/rho_w                (no scaling)
   *  4) scaling_omega*omega    scaling_omega = 1e-3*yr_to_d
   *
   * where yr_to_s=365.25*24*3600 (the number of seconds in a year)
   * and   yr_to_d=365.25 (the number of days in a year)
   */
  double yr_to_d  = 365.25;
  double yr_to_s  = yr_to_d*24*3600;
  scaling_omega   = 1e-3*yr_to_d;
  scaling_h_dot   = yr_to_s;
  scaling_q       = 1e-3*yr_to_s;

  this->setName("HydrologyResidualMassEqn"+PHX::typeAsString<EvalT>());
}

//**********************************************************************
template<typename EvalT, typename Traits, bool IsStokesCoupling, bool ThermoCoupled>
void HydrologyResidualMassEqn<EvalT, Traits, IsStokesCoupling, ThermoCoupled>::
postRegistrationSetup(typename Traits::SetupData d,
                      PHX::FieldManager<Traits>& fm)
{
  this->utils.setFieldData(BF,fm);
  this->utils.setFieldData(GradBF,fm);
  this->utils.setFieldData(w_measure,fm);
  this->utils.setFieldData(q,fm);
  this->utils.setFieldData(omega,fm);

  if (IsStokesCoupling)
    this->utils.setFieldData(metric,fm);

  if (penalization) {
    this->utils.setFieldData(P_w,fm);
    this->utils.setFieldData(P_o,fm);
  }
  if (unsteady) {
    this->utils.setFieldData(h_dot,fm);
  }

  if (use_melting) {
    this->utils.setFieldData(m,fm);
  }

  this->utils.setFieldData(residual,fm);
}

//**********************************************************************
template<typename EvalT, typename Traits, bool IsStokesCoupling, bool ThermoCoupled>
void HydrologyResidualMassEqn<EvalT, Traits, IsStokesCoupling, ThermoCoupled>::
evaluateFields (typename Traits::EvalData workset)
{
  if (IsStokesCoupling) {
    evaluateFieldsSide(workset);
  } else {
    evaluateFieldsCell(workset);
  }
}

template<typename EvalT, typename Traits, bool IsStokesCoupling, bool ThermoCoupled>
void HydrologyResidualMassEqn<EvalT, Traits, IsStokesCoupling, ThermoCoupled>::
evaluateFieldsSide (typename Traits::EvalData workset)
{
  // Zero out, to avoid leaving stuff from previous workset!
  residual.deep_copy(ScalarT(0.));

  if (workset.sideSets->find(sideSetName)==workset.sideSets->end())
    return;

  ScalarT res_qp, res_node, zero(0);
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
        res_qp = scaling_omega*omega(cell,side,qp) - (unsteady ? scaling_h_dot*h_dot(cell,side,qp) : zero);

        if (use_melting && !mass_lumping) {
          res_qp += m(cell,side,qp)/rho_w;
        }

        res_qp *= BF(cell,side,node,qp);

        for (int idim=0; idim<numDims; ++idim)
        {
          for (int jdim=0; jdim<numDims; ++jdim)
          {
            res_qp += scaling_q*q(cell,side,qp,idim) * metric(cell,side,qp,idim,jdim) * GradBF(cell,side,node,qp,jdim);
          }
        }

        res_node += res_qp * w_measure(cell,side,qp);
      }

      if (use_melting && mass_lumping) {
        res_node += m(cell,side,node)/rho_w;
      }

      if (penalization) {
        res_node += penalization_coeff*std::pow(std::min(ScalarT(0.0),P_o(cell,side,node)-P_w(cell,side,node)),2);
        res_node += penalization_coeff*std::pow(std::min(ScalarT(0.0),P_w(cell,side,node)),2);
      }


      residual (cell,side,node) = res_node;
    }
  }
}

template<typename EvalT, typename Traits, bool IsStokesCoupling, bool ThermoCoupled>
void HydrologyResidualMassEqn<EvalT, Traits, IsStokesCoupling, ThermoCoupled>::
evaluateFieldsCell (typename Traits::EvalData workset)
{
  ScalarT res_qp, res_node, zero(0);
  for (int cell=0; cell < workset.numCells; ++cell)
  {
    for (int node=0; node < numNodes; ++node)
    {
      res_node = 0;
      for (int qp=0; qp < numQPs; ++qp)
      {
        res_qp = scaling_omega*omega(cell,qp) - (unsteady ? scaling_h_dot*h_dot(cell,qp) : zero);

        if (use_melting && !mass_lumping) {
          res_qp += m(cell,qp)/rho_w;
        }

        res_qp *= BF(cell,node,qp);

        for (int dim=0; dim<numDims; ++dim)
        {
          res_qp += scaling_q*q(cell,qp,dim) * GradBF(cell,node,qp,dim);
        }

        res_node += res_qp * w_measure(cell,qp);
      }

      if (use_melting && mass_lumping) {
        res_node += m(cell,node)/rho_w;
      }

      if (penalization) {
        res_node += penalization_coeff*std::min(ScalarT(0.0),P_o(cell,node)-P_w(cell,node));
        res_node += penalization_coeff*std::min(ScalarT(0.0),P_w(cell,node));
      }


      residual (cell,node) = res_node;
    }
  }
}

} // Namespace FELIX
