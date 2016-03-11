//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#include "Teuchos_TestForException.hpp"
#include "Teuchos_VerboseObject.hpp"
#include "Phalanx_DataLayout.hpp"
#include "Intrepid2_FunctionSpaceTools.hpp"
#include "Albany_Layouts.hpp"

#include "FELIX_HomotopyParameter.hpp"

//uncomment the following line if you want debug output to be printed to screen
//#define OUTPUT_TO_SCREEN

namespace FELIX
{

template<typename EvalT, typename Traits>
BasalFrictionCoefficient<EvalT, Traits>::BasalFrictionCoefficient (const Teuchos::ParameterList& p,
                                                                   const Teuchos::RCP<Albany::Layouts>& dl)
{
#ifdef OUTPUT_TO_SCREEN
  Teuchos::RCP<Teuchos::FancyOStream> output(Teuchos::VerboseObjectBase::getDefaultOStream());
#endif

  Teuchos::ParameterList& beta_list = *p.get<Teuchos::ParameterList*>("Parameter List");

  std::string betaType = (beta_list.isParameter("Type") ? beta_list.get<std::string>("Type") : "Given Field");
  is_hydrology         = (p.isParameter("Hydrology") ? p.get<bool>("Hydrology") : false);

  if (is_hydrology)
  {
    beta = PHX::MDField<ScalarT>(p.get<std::string> ("Basal Friction Coefficient Variable Name"), dl->qp_scalar);

    numQPs   = dl->qp_scalar->dimension(1);
    numNodes = dl->node_scalar->dimension(1);
  }
  else
  {
    beta = PHX::MDField<ScalarT>(p.get<std::string> ("Basal Friction Coefficient Variable Name"), dl->side_qp_scalar);
    basalSideName = p.get<std::string>("Side Set Name");

    numQPs   = dl->side_qp_scalar->dimension(2);
    numNodes = dl->side_node_scalar->dimension(2);
  }

  this->addEvaluatedField(beta);

  if (betaType == "Given Constant")
  {
#ifdef OUTPUT_TO_SCREEN
    *output << "Given constant and uniform beta, value loaded from xml input file.\n";
#endif
    beta_type = GIVEN_CONSTANT;
    beta_given_val = beta_list.get<double>("Constant Given Beta Value");
  }
  else if ((betaType == "Given Field")|| (betaType == "Exponent of Given Field"))
  {
#ifdef OUTPUT_TO_SCREEN
    *output << "Given constant beta field, loaded from mesh or file.\n";
#endif
    if (betaType == "Given Field")
      beta_type = GIVEN_FIELD;
    else
      beta_type = EXP_GIVEN_FIELD;

    if (is_hydrology)
    {
      BF = PHX::MDField<RealType>(p.get<std::string> ("BF Variable Name"), dl->node_qp_scalar);
      beta_given_field = PHX::MDField<ParamScalarT>(p.get<std::string> ("Basal Friction Coefficient Variable Name"), dl->node_scalar);
    }
    else
    {
      BF = PHX::MDField<RealType>(p.get<std::string> ("BF Side Variable Name"), dl->side_node_qp_scalar);
      beta_given_field = PHX::MDField<ParamScalarT>(p.get<std::string> ("Basal Friction Coefficient Variable Name"), dl->side_node_scalar);
    }

    this->addDependentField (BF);
    this->addDependentField (beta_given_field);
  }
  else if (betaType == "Power Law")
  {
    beta_type = POWER_LAW;

    mu    = beta_list.get<double>("Coulomb Friction Coefficient");
    power = beta_list.get("Power Exponent",1.0);
    TEUCHOS_TEST_FOR_EXCEPTION(power<-1.0, Teuchos::Exceptions::InvalidParameter,
        std::endl << "Error in FELIX::BasalFrictionCoefficient: \"Power Exponent\" must be greater than (or equal to) -1.\n");

#ifdef OUTPUT_TO_SCREEN
    *output << "Velocity-dependent beta (power law):\n\n"
            << "      beta = mu * N * |u|^p \n\n"
            << "  with N being the effective pressure, |u| the sliding velocity, and\n"
            << "    - mu (Coulomb Friction Coefficient): " << mu << "\n"
            << "    - p  (Power Exponent): " << power << "\n";
#endif

    if (is_hydrology)
    {
      N      = PHX::MDField<ParamScalarT>(p.get<std::string> ("Effective Pressure QP Variable Name"), dl->qp_scalar);
      u_norm = PHX::MDField<ScalarT>(p.get<std::string> ("Sliding Velocity QP Variable Name"), dl->qp_scalar);
    }
    else
    {
      N      = PHX::MDField<ParamScalarT>(p.get<std::string> ("Effective Pressure Side QP Variable Name"), dl->side_qp_scalar);
      u_norm = PHX::MDField<ScalarT>(p.get<std::string> ("Sliding Velocity Side QP Variable Name"), dl->side_qp_scalar);
    }
    this->addDependentField (u_norm);
    this->addDependentField (N);
  }
  else if (betaType == "Regularized Coulomb")
  {
    beta_type = REGULARIZED_COULOMB;

    mu     = beta_list.get<double>("Coulomb Friction Coefficient");
    power  = beta_list.get("Power Exponent",1.0);
    lambda = beta_list.get("Bed Roughness",1e-4);
    if (beta_list.isParameter("Constant Flow Factor A"))
    {
      A = beta_list.get<double>("Constant Flow Factor A");
    }
    else
    {
      TEUCHOS_TEST_FOR_EXCEPTION (true, std::logic_error, "Error! The case with variable flow factor has not been implemented yet.\n");
    }
#ifdef OUTPUT_TO_SCREEN
    *output << "Velocity-dependent beta (regularized coulomb law):\n\n"
            << "      beta = mu * N * |u|^{p-1} / [|u| + lambda*A*N^(1/p)]^p\n\n"
            << "  with N being the effective pressure, |u| the sliding velocity, and\n"
            << "    - mu (Coulomb Friction Coefficient): " << mu << "\n"
            << "    - lambda (Bed Roughness or Regularization Parameter): " << lambda << "\n"
            << "    - A (Flow Factor A): " << A << "\n"
            << "    - p  (Power Exponent): " << power << "\n";
#endif

    if (is_hydrology)
    {
      N      = PHX::MDField<ParamScalarT>(p.get<std::string> ("Effective Pressure QP Variable Name"), dl->qp_scalar);
      u_norm = PHX::MDField<ScalarT>(p.get<std::string> ("Sliding Velocity QP Variable Name"), dl->qp_scalar);
    }
    else
    {
      N      = PHX::MDField<ParamScalarT>(p.get<std::string> ("Effective Pressure Side QP Variable Name"), dl->side_qp_scalar);
      u_norm = PHX::MDField<ScalarT>(p.get<std::string> ("Sliding Velocity Side QP Variable Name"), dl->side_qp_scalar);
    }

    this->addDependentField (u_norm);
    this->addDependentField (N);
  }
  else
  {
    TEUCHOS_TEST_FOR_EXCEPTION(true, Teuchos::Exceptions::InvalidParameter,
        std::endl << "Error in FELIX::BasalFrictionCoefficient:  \"" << betaType << "\" is not a valid parameter for Beta Type\n");
  }

  auto& stereographicMapList = p.get<Teuchos::ParameterList*>("Stereographic Map");
  use_stereographic_map = stereographicMapList->get("Use Stereographic Map", false);
  if(use_stereographic_map)
  {
    if (is_hydrology)
      coordVec = PHX::MDField<MeshScalarT>(p.get<std::string>("Coordinate Vector Variable Name"), dl->qp_gradient);
    else
      coordVec = PHX::MDField<MeshScalarT>(p.get<std::string>("Coordinate Vector Variable Name"), dl->side_qp_coords);

    double R = stereographicMapList->get<double>("Earth Radius", 6371);
    x_0 = stereographicMapList->get<double>("X_0", 0);//-136);
    y_0 = stereographicMapList->get<double>("Y_0", 0);//-2040);
    R2 = std::pow(R,2);

    this->addDependentField(coordVec);
  }

  this->setName("BasalFrictionCoefficient"+PHX::typeAsString<EvalT>());
}

//**********************************************************************
template<typename EvalT, typename Traits>
void BasalFrictionCoefficient<EvalT, Traits>::
postRegistrationSetup (typename Traits::SetupData d,
                       PHX::FieldManager<Traits>& fm)
{
  this->utils.setFieldData(beta,fm);

  switch (beta_type)
  {
    case GIVEN_CONSTANT:
      if (is_hydrology)
      {
        for (int cell=0; cell<beta.fieldTag().dataLayout().dimension(0); ++cell)
          for (int qp=0; qp<numQPs; ++qp)
            beta(cell,qp) = beta_given_val;
      }
      else
      {
        for (int cell=0; cell<beta.fieldTag().dataLayout().dimension(0); ++cell)
          for (int side=0; side<beta.fieldTag().dataLayout().dimension(1); ++side)
            for (int qp=0; qp<numQPs; ++qp)
              beta(cell,side,qp) = beta_given_val;
      }
      break;
    case GIVEN_FIELD:
    case EXP_GIVEN_FIELD:
      this->utils.setFieldData(BF,fm);
      this->utils.setFieldData(beta_given_field,fm);
      break;
    case POWER_LAW:
    case REGULARIZED_COULOMB:
      this->utils.setFieldData(N,fm);
      this->utils.setFieldData(u_norm,fm);
  }

  if (use_stereographic_map)
    this->utils.setFieldData(coordVec,fm);
}

//**********************************************************************
template<typename EvalT, typename Traits>
void BasalFrictionCoefficient<EvalT, Traits>::evaluateFields (typename Traits::EvalData workset)
{
  if (is_hydrology)
  {
    ScalarT homotopyParam = FELIX::HomotopyParameter<EvalT>::value;
    ScalarT ff = 0;
    if (homotopyParam!=0)
      ff = pow(10.0, -10.0*homotopyParam);

    switch (beta_type)
    {
      case GIVEN_CONSTANT:
        break;   // We don't have anything to do

      case GIVEN_FIELD:
        for (int cell=0; cell<workset.numCells; ++cell)
          for (int qp=0; qp<numQPs; ++qp)
          {
            beta(cell,qp) = 0.;
            for (int node=0; node<numNodes; ++node)
            {
              beta(cell,qp) += BF(cell,node,qp)*beta_given_field(cell,node);
            }
          }
        break;

      case POWER_LAW:
        for (int cell=0; cell<workset.numCells; ++cell)
          for (int qp=0; qp<numQPs; ++qp)
          {
            beta(cell,qp) = mu * N(cell,qp) * std::pow (u_norm(cell,qp), power);
          }
        break;

      case REGULARIZED_COULOMB:
        for (int cell=0; cell<workset.numCells; ++cell)
          for (int qp=0; qp<numQPs; ++qp)
          {
            beta(cell,qp) = mu * N(cell,qp) * std::pow (u_norm(cell,qp),power-1)
                          / std::pow( std::pow(u_norm(cell,qp),1-ff) + lambda*A*std::pow(N(cell,qp),1./power), power);
          }
        break;

      case EXP_GIVEN_FIELD:
        for (int cell=0; cell<workset.numCells; ++cell)
          for (int qp=0; qp<numQPs; ++qp)
          {
            beta(cell,qp) = 0.;
            for (int node=0; node<numNodes; ++node)
            {
              beta(cell,qp) += BF(cell,node,qp)*beta_given_field(cell,node);
            }
            beta(cell,qp) = std::exp(beta(cell,qp));
          }
        break;
    }

    // Correct the value if we are using a stereographic map
    if (use_stereographic_map)
    {
      for (int cell=0; cell<workset.numCells; ++cell)
      {
        for (int qp=0; qp<numQPs; ++qp)
        {
          MeshScalarT x = coordVec(cell,qp,0) - x_0;
          MeshScalarT y = coordVec(cell,qp,1) - y_0;
          MeshScalarT h = 4.0*R2/(4.0*R2 + x*x + y*y);
          beta(cell,qp) *= h*h;
        }
      }
    }
  }
  else
  {
    if (workset.sideSets->find(basalSideName)==workset.sideSets->end())
      return;

    ScalarT homotopyParam = FELIX::HomotopyParameter<EvalT>::value;
    ScalarT ff = 0;
    if (homotopyParam!=0)
      ff = pow(10.0, -10.0*homotopyParam);

    const std::vector<Albany::SideStruct>& sideSet = workset.sideSets->at(basalSideName);
    for (auto const& it_side : sideSet)
    {
      // Get the local data of side and cell
      const int cell = it_side.elem_LID;
      const int side = it_side.side_local_id;

      switch (beta_type)
      {
        case GIVEN_CONSTANT:
          return;   // We can save ourself some useless iterations

        case GIVEN_FIELD:
          for (int qp=0; qp<numQPs; ++qp)
          {
            beta(cell,side,qp) = 0.;
            for (int node=0; node<numNodes; ++node)
            {
              beta(cell,side,qp) += BF(cell,side,node,qp)*beta_given_field(cell,side,node);
            }
          }
          break;

        case POWER_LAW:
          for (int qp=0; qp<numQPs; ++qp)
          {
            beta(cell,side,qp) = mu * N(cell,side,qp) * std::pow (u_norm(cell,side,qp), power);
          }
          break;

        case REGULARIZED_COULOMB:
          for (int qp=0; qp<numQPs; ++qp)
          {
            beta(cell,side,qp) = mu * N(cell,side,qp) * std::pow (u_norm(cell,side,qp),power-1)
                               / std::pow( std::pow(u_norm(cell,side,qp),1-ff) + lambda*A*std::pow(N(cell,side,qp),1./power), power);
          }
          break;

        case EXP_GIVEN_FIELD:
          for (int qp=0; qp<numQPs; ++qp)
          {
            beta(cell,side,qp) = 0.;
            for (int node=0; node<numNodes; ++node)
            {
              beta(cell,side,qp) += BF(cell,side,node,qp)*beta_given_field(cell,side,node);
            }
            beta(cell,side,qp) = std::exp(beta(cell,side,qp));
          }
          break;
      }

      // Correct the value if we are using a stereographic map
      if (use_stereographic_map)
      {
        for (int qp=0; qp<numQPs; ++qp)
        {
          MeshScalarT x = coordVec(cell,side,qp,0) - x_0;
          MeshScalarT y = coordVec(cell,side,qp,1) - y_0;
          MeshScalarT h = 4.0*R2/(4.0*R2 + x*x + y*y);
          beta(cell,side,qp) *= h*h;
        }
      }
    }
  }
}

} // Namespace FELIX
