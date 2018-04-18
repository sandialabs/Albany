//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#include "Phalanx_DataLayout.hpp"
#include "Phalanx_TypeStrings.hpp"

namespace FELIX
{

template<typename EvalT, typename Traits, bool IsStokes, bool ThermoCoupled>
HydrologyWaterThickness<EvalT, Traits, IsStokes, ThermoCoupled>::
HydrologyWaterThickness (const Teuchos::ParameterList& p,
                         const Teuchos::RCP<Albany::Layouts>& dl)
{
  /*
   *  The (water) thickness equation has the following (strong) form
   *
   *     dh/dt = m/rho_i + (h_r-k*h)*|u_b|/l_r - AhN^3
   *
   *  where h is the water thickness, m the melting rate of the ice,
   *  h_r/l_r typical height/length of bed bumps, u_b the sliding
   *  velocity of the ice, A the Glen's law flow factor, and N is
   *  the effective pressure. k=0,1 allows to consider effective or
   *  total cavities height in the model
   *  NOTE: if the term m/rho_i is present, there is no proof of well posedness
   *        for the hydrology equations. Therefore, we only turn this term
   *        on upon request.
   *
   *  In the steady case, we can solve for h, leading to the expression
   *
   *     h = ( m/rho_i + |u_b|*h_r/l_r ) / ( AN^3 + k*|u_b|/l_r )
   */

  bool nodal = p.get<bool>("Nodal");
  Teuchos::RCP<PHX::DataLayout> layout = nodal ? dl->node_scalar : dl->qp_scalar;

  if (IsStokes) {
    TEUCHOS_TEST_FOR_EXCEPTION (!dl->isSideLayouts, std::logic_error,
                                "Error! For coupling with StokesFO, the Layouts structure must be that of the basal side.\n");

    sideSetName = p.get<std::string>("Side Set Name");

    numPts = layout->dimension(2);
  } else {
    numPts = layout->dimension(1);
  }

  u_b = PHX::MDField<const IceScalarT>(p.get<std::string> ("Sliding Velocity Variable Name"), layout);
  N   = PHX::MDField<const ScalarT>(p.get<std::string> ("Effective Pressure Variable Name"), layout);
  m   = PHX::MDField<const ScalarT>(p.get<std::string> ("Melting Rate Variable Name"), layout);
  A   = PHX::MDField<const TempScalarT>(p.get<std::string> ("Flow Factor A Variable Name"), dl->cell_scalar2);
  h   = PHX::MDField<ScalarT>(p.get<std::string> ("Water Thickness Variable Name"), layout);

  this->addDependentField(u_b);
  this->addDependentField(N);
  this->addDependentField(m);
  this->addDependentField(A);

  this->addEvaluatedField(h);

  // Setting parameters
  Teuchos::ParameterList& hydrology = *p.get<Teuchos::ParameterList*>("FELIX Hydrology");
  Teuchos::ParameterList& physics   = *p.get<Teuchos::ParameterList*>("FELIX Physical Parameters");

  double rho_i = physics.get<double>("Ice Density");
  h_r = hydrology.get<double>("Bed Bumps Height");
  l_r = hydrology.get<double>("Bed Bumps Length");

  bool melting_cav = hydrology.get<bool>("Use Melting In Cavities Equation", false);
  use_eff_cav = (hydrology.get<bool>("Use Effective Cavities Height", true) ? 1.0 : 0.0);
  if (melting_cav) {
    rho_i_inv = 1./rho_i;
  } else {
    rho_i_inv = 0;
  }

  /*
   * Scalings, needed to account for different units: ice velocity
   * is in m/yr, the mesh is in km, and hydrology time unit is s.
   *
   * The steady cavity equation has 3 terms (forget about signs), with the following
   * units (including the km^2 from dx):
   *
   *  1) \int rho_i_inv*m*v*dx      [m km^2 yr^-1]
   *  2) \int 2/(n^n)*A*h*N^3*v*dx          [1000 m km^2 yr^-1]
   *  3) \int (h_r-h)*|u|/l_r*v*dx  [m km^2 yr^-1]
   *
   * where q=k*h^3*gradPhi/mu_w, n as in Glen's law, and v is the test function.
   * We decide to uniform all terms to have units [m km^2 yr^-1].
   * Where possible, we do this by rescaling some constants. Otherwise,
   * we simply introduce a new scaling factor
   *
   *  1) rho_i_inv_m          (no scaling)
   *  2) scaling_A*A*h*N^3    scaling_A = 1.0/1000
   *  3) (h_r-h)*|u|/l_r      (no scaling)
   *
   * where yr_to_s=365.25*24*3600 (the number of seconds in a year)
   */

  // Scalings, needed to account for different units: ice velocity
  // is in m/yr rather than m/s, while all other quantities are in SI units.
  scaling_A   = 1.0/1000;

  this->setName("HydrologyWaterThickness"+PHX::typeAsString<EvalT>());
}

//**********************************************************************
template<typename EvalT, typename Traits, bool IsStokes, bool ThermoCoupled>
void HydrologyWaterThickness<EvalT, Traits, IsStokes, ThermoCoupled>::
postRegistrationSetup(typename Traits::SetupData d,
                      PHX::FieldManager<Traits>& fm)
{
  this->utils.setFieldData(u_b,fm);
  this->utils.setFieldData(N,fm);
  this->utils.setFieldData(m,fm);

  this->utils.setFieldData(h,fm);
}

//**********************************************************************
template<typename EvalT, typename Traits, bool IsStokes, bool ThermoCoupled>
void HydrologyWaterThickness<EvalT, Traits, IsStokes, ThermoCoupled>::evaluateFields (typename Traits::EvalData workset)
{
  if (IsStokes) {
    evaluateFieldsSide(workset);
  } else {
    evaluateFieldsCell(workset);
  }
}

template<typename EvalT, typename Traits, bool IsStokes, bool ThermoCoupled>
void HydrologyWaterThickness<EvalT, Traits, IsStokes, ThermoCoupled>::evaluateFieldsCell (typename Traits::EvalData workset)
{
  for (int cell=0; cell < workset.numCells; ++cell)
  {
    for (int ipt=0; ipt < numPts; ++ipt)
    {
      h(cell,ipt)  = rho_i_inv*m(cell,ipt) + u_b(cell,ipt)*h_r/l_r;
      h(cell,ipt) /= (2.0/9.0)*scaling_A*A(cell)*std::pow(N(cell,ipt),3) + use_eff_cav*u_b(cell,ipt)/l_r;
    }
  }
}

template<typename EvalT, typename Traits, bool IsStokes, bool ThermoCoupled>
void HydrologyWaterThickness<EvalT, Traits, IsStokes, ThermoCoupled>::
evaluateFieldsSide (typename Traits::EvalData workset)
{
  if (workset.sideSets->find(sideSetName)==workset.sideSets->end())
    return;

  const std::vector<Albany::SideStruct>& sideSet = workset.sideSets->at(sideSetName);
  for (auto const& it_side : sideSet)
  {
    // Get the local data of side and cell
    const int cell = it_side.elem_LID;
    const int side = it_side.side_local_id;

    for (int ipt=0; ipt < numPts; ++ipt)
    {
      h(cell,ipt)  = rho_i_inv*m(cell,side,ipt) + u_b(cell,side,ipt)*h_r/l_r;
      h(cell,ipt) /= (2.0/9.0)*scaling_A*A(cell)*std::pow(N(cell,side,ipt),3) + use_eff_cav*u_b(cell,side,ipt)/l_r;
    }
  }
}

} // Namespace FELIX
