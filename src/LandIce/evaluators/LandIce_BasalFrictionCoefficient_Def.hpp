//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#include "Teuchos_GlobalMPISession.hpp"
#include "Teuchos_TestForException.hpp"
#include "Teuchos_VerboseObject.hpp"
#include "Phalanx_DataLayout.hpp"

#include "Albany_DiscretizationUtils.hpp"
#include "Albany_Layouts.hpp"
#include "Albany_KokkosUtils.hpp"

#include "LandIce_BasalFrictionCoefficient.hpp"

#include <string.hpp> // for 'upper_case' (comes from src/utility; not to be confused with <string>)

//uncomment the following line if you want debug output to be printed to screen
//#define OUTPUT_TO_SCREEN

namespace LandIce
{

template<typename EvalT, typename Traits, typename EffPressureST, typename VelocityST, typename TemperatureST>
BasalFrictionCoefficient<EvalT, Traits, EffPressureST, VelocityST, TemperatureST>::
BasalFrictionCoefficient (const Teuchos::ParameterList& p,
                          const Teuchos::RCP<Albany::Layouts>& dl)
{
#ifdef OUTPUT_TO_SCREEN
  Teuchos::RCP<Teuchos::FancyOStream> output(Teuchos::VerboseObjectBase::getDefaultOStream());

  int procRank = Teuchos::GlobalMPISession::getRank();
  int numProcs = Teuchos::GlobalMPISession::getNProc();
  output->setProcRankAndSize (procRank, numProcs);
  output->setOutputToRootOnly (0);
#endif

  n = -1; //dummy value
  Teuchos::ParameterList& beta_list = *p.get<Teuchos::ParameterList*>("Parameter List");
  zero_on_floating = beta_list.get<bool> ("Zero Beta On Floating Ice", false);

  //whether to first interpolate the given field and then exponetiate it (on quad points) or the other way around.
  interpolate_then_exponentiate = beta_list.get<bool> ("Interpolate Then Exponentiate Given Field", true); 

  std::string betaType = util::upper_case((beta_list.isParameter("Type") ? beta_list.get<std::string>("Type") : "Given Field"));

  is_side_equation = p.isParameter("Side Set Name");

  if (is_side_equation) {
    TEUCHOS_TEST_FOR_EXCEPTION (!dl->isSideLayouts, Teuchos::Exceptions::InvalidParameter,
                                "Error! The layout structure does not appear to be that of a side set.\n");

    basalSideName = p.get<std::string>("Side Set Name");
    numQPs        = dl->qp_scalar->extent(2);
    numNodes      = dl->node_scalar->extent(2);
  } else {
    TEUCHOS_TEST_FOR_EXCEPTION (dl->isSideLayouts, Teuchos::Exceptions::InvalidParameter,
                                "Error! The layout structure appears to be that of a side set.\n");

    numQPs    = dl->qp_scalar->extent(1);
    numNodes  = dl->node_scalar->extent(1);
  }

  nodal = p.isParameter("Nodal") ? p.get<bool>("Nodal") : false;
  Teuchos::RCP<PHX::DataLayout> layout;
  if (is_side_equation) {
    layout = nodal ? dl->node_scalar_sideset : dl->qp_scalar_sideset;
  } else {
    layout = nodal ? dl->node_scalar : dl->qp_scalar;
  }

  beta = PHX::MDField<ScalarT>(p.get<std::string> ("Basal Friction Coefficient Variable Name"), layout);
  this->addEvaluatedField(beta);

  printedMu      = -9999.999;
  printedLambda  = -9999.999;
  printedQ       = -9999.999;

  auto is_dist_param = p.isParameter("Dist Param Query Map") ? p.get<Teuchos::RCP<std::map<std::string,bool>>>("Dist Param Query Map") : Teuchos::null;
  if (betaType == "GIVEN CONSTANT") {
    beta_type = GIVEN_CONSTANT;
    given_val = beta_list.get<double>("Constant Given Beta Value");
#ifdef OUTPUT_TO_SCREEN
    *output << "Given constant and uniform beta, value = " << given_val << " (loaded from xml input file).\n";
#endif
  } else if ((betaType == "GIVEN FIELD")|| (betaType == "EXPONENT OF GIVEN FIELD")) {
#ifdef OUTPUT_TO_SCREEN
    *output << "Given constant beta field, loaded from mesh or file.\n";
#endif
    if (betaType == "GIVEN FIELD") {
      beta_type = GIVEN_FIELD;
    } else {
      beta_type = EXP_GIVEN_FIELD;
    }

    auto layout_given_field = layout;
    std::string given_field_name = beta_list.get<std::string> ("Given Field Variable Name");
    is_given_field_param = is_dist_param.is_null() ? false : (*is_dist_param)[given_field_name];
    if (is_side_equation) {
      given_field_name += "_" + basalSideName;
    }
    if( (beta_type == EXP_GIVEN_FIELD) && !interpolate_then_exponentiate ) {
      BF = PHX::MDField<const RealType>(p.get<std::string> ("BF Variable Name"), is_side_equation ? dl->node_qp_scalar_sideset : dl->node_qp_scalar);
      layout_given_field = is_side_equation ? dl->node_scalar_sideset : dl->node_scalar;
      this->addDependentField (BF);
    }
    if (is_given_field_param) {
      given_field_param = PHX::MDField<const ParamScalarT>(given_field_name, layout_given_field);
      this->addDependentField (given_field_param);
    } else {
      given_field = PHX::MDField<const RealType>(given_field_name, layout_given_field);
      this->addDependentField (given_field);
    }
  } else if (betaType == "POWER LAW") {
    beta_type = POWER_LAW;

#ifdef OUTPUT_TO_SCREEN
    *output << "Velocity-dependent beta (power law):\n\n"
            << "      beta = mu * N * |u|^{q-1} \n\n"
            << "  with N being the effective pressure, |u| the sliding velocity\n";
#endif

    N              = PHX::MDField<const EffPressureST>(p.get<std::string> ("Effective Pressure Variable Name"), layout);
    u_norm         = PHX::MDField<const VelocityST>(p.get<std::string> ("Sliding Velocity Variable Name"), layout);
    powerParam     = PHX::MDField<const ScalarT,Dim>("Power Exponent", dl->shared_param);

    this->addDependentField (powerParam);
    this->addDependentField (u_norm);
    this->addDependentField (N);

    distributedMu = beta_list.get<bool>("Distributed Mu",false);
    if (distributedMu) {
      muPowerLawField = PHX::MDField<const ParamScalarT>(p.get<std::string> ("Power Law Coefficient Variable Name"), layout);
      this->addDependentField (muPowerLawField);
    } else {
      muPowerLaw     = PHX::MDField<const ScalarT,Dim>("Power Law Coefficient", dl->shared_param);
      this->addDependentField (muPowerLaw);
    }
  } else if (betaType == "REGULARIZED COULOMB") {
    beta_type = REGULARIZED_COULOMB;

#ifdef OUTPUT_TO_SCREEN
    *output << "Velocity-dependent beta (regularized coulomb law):\n\n"
            << "      beta = mu * N * |u|^{q-1} / [|u| + lambda*A*N^n]^q\n\n"
            << "  with N being the effective pressure, |u| the sliding velocity\n";
#endif

    n            = p.get<Teuchos::ParameterList*>("Viscosity Parameter List")->get<double>("Glen's Law n");
    N            = PHX::MDField<const EffPressureST>(p.get<std::string> ("Effective Pressure Variable Name"), layout);
    u_norm       = PHX::MDField<const VelocityST>(p.get<std::string> ("Sliding Velocity Variable Name"), layout);
    powerParam   = PHX::MDField<const ScalarT,Dim>("Power Exponent", dl->shared_param);
    ice_softness = PHX::MDField<const TemperatureST>(p.get<std::string>("Ice Softness Variable Name"), is_side_equation ? dl->cell_scalar2_sideset : dl->cell_scalar2);

    this->addDependentField (powerParam);
    this->addDependentField (N);
    this->addDependentField (u_norm);
    this->addDependentField (ice_softness);

    distributedLambda = beta_list.get<bool>("Distributed Lambda",false);
    if (distributedLambda) {
      lambdaField = PHX::MDField<const ParamScalarT>(p.get<std::string> ("Bed Roughness Variable Name"), layout);
      this->addDependentField (lambdaField);
    } else {
      lambdaParam    = PHX::MDField<ScalarT,Dim>("Bed Roughness", dl->shared_param);
      this->addDependentField (lambdaParam);
    }

    distributedMu = beta_list.get<bool>("Distributed Mu",false);
    if (distributedMu) {
      muCoulombField = PHX::MDField<const ParamScalarT>(p.get<std::string> ("Coulomb Friction Coefficient Variable Name"), layout);
      this->addDependentField (muCoulombField);
    } else {
      muCoulomb = PHX::MDField<const ScalarT,Dim>("Coulomb Friction Coefficient", dl->shared_param);
      this->addDependentField (muCoulomb);
    }
  } else {
    TEUCHOS_TEST_FOR_EXCEPTION(true, Teuchos::Exceptions::InvalidParameter,
        std::endl << "Error in LandIce::BasalFrictionCoefficient:  \"" << betaType << "\" is not a valid parameter for Beta Type\n");
  }

  if(zero_on_floating) {
    is_thickness_param = is_dist_param.is_null() ? false : (*is_dist_param)[p.get<std::string>("Ice Thickness Variable Name")];
    if (is_thickness_param) {
      bed_topo_field_mst = decltype(bed_topo_field_mst)(p.get<std::string> ("Bed Topography Variable Name"), layout);
      this->addDependentField (bed_topo_field_mst);
      thickness_param_field = decltype(thickness_param_field)(p.get<std::string> ("Ice Thickness Variable Name"), layout);
      this->addDependentField (thickness_param_field);
    } else {
      bed_topo_field = decltype(bed_topo_field)(p.get<std::string> ("Bed Topography Variable Name"), layout);
      this->addDependentField (bed_topo_field);
      thickness_field = decltype(thickness_field)(p.get<std::string> ("Ice Thickness Variable Name"), layout);
      this->addDependentField (thickness_field);
    }
    Teuchos::ParameterList& phys_param_list = *p.get<Teuchos::ParameterList*>("Physical Parameter List");
    rho_i = phys_param_list.get<double> ("Ice Density");
    rho_w = phys_param_list.get<double> ("Water Density");
  }

  auto& stereographicMapList = p.get<Teuchos::ParameterList*>("Stereographic Map");
  use_stereographic_map = stereographicMapList->get("Use Stereographic Map", false);
  if(use_stereographic_map) {
    if (is_side_equation) { 
      layout = nodal ? dl->node_vector_sideset : dl->qp_coords_sideset;
    } else {
      layout = nodal ? dl->node_vector: dl->qp_coords;
    }
    coordVec = PHX::MDField<MeshScalarT>(p.get<std::string>("Coordinate Vector Variable Name"), layout);

    double R = stereographicMapList->get<double>("Earth Radius", 6371);
    x_0 = stereographicMapList->get<double>("X_0", 0);//-136);
    y_0 = stereographicMapList->get<double>("Y_0", 0);//-2040);
    R2 = std::pow(R,2);

    this->addDependentField(coordVec);
  }

  logParameters = beta_list.get<bool>("Use log scalar parameters",false);

  this->setName("BasalFrictionCoefficient"+PHX::print<EvalT>());
}

//**********************************************************************
template<typename EvalT, typename Traits, typename EffPressureST, typename VelocityST, typename TemperatureST>
void BasalFrictionCoefficient<EvalT, Traits, EffPressureST, VelocityST, TemperatureST>::
postRegistrationSetup (typename Traits::SetupData d,
                       PHX::FieldManager<Traits>&)
{
  if (beta_type == GIVEN_CONSTANT)
    beta.deep_copy(ScalarT(given_val));

  d.fill_field_dependencies(this->dependentFields(),this->evaluatedFields());
  if (d.memoizer_active())
    memoizer.enable_memoizer();
}

// *********************************************************************
// Kokkos functor
template<typename EvalT, typename Traits, typename EffPressureST, typename VelocityST, typename TemperatureST>
KOKKOS_INLINE_FUNCTION
void BasalFrictionCoefficient<EvalT, Traits, EffPressureST, VelocityST, TemperatureST>::
operator() (const BasalFrictionCoefficient_Tag& tag, const int& cell_or_side_idx) const {

  switch(beta_type) {
    case GIVEN_CONSTANT:
    break;

    case GIVEN_FIELD:
      if (is_given_field_param) {
        for (unsigned int ipt=0; ipt<dim; ++ipt)
          beta(cell_or_side_idx,ipt) = given_field_param(cell_or_side_idx,ipt);
      } else {
        for (unsigned int ipt=0; ipt<dim; ++ipt)
          beta(cell_or_side_idx,ipt) = given_field(cell_or_side_idx,ipt);
      }
    break;
    
    case POWER_LAW:
      if (distributedMu) {
        for (unsigned int ipt=0; ipt<dim; ++ipt) {
          ScalarT Nval = N(cell_or_side_idx,ipt);
          if (is_side_equation) Nval = KU::max(Nval, 0.0);
          beta(cell_or_side_idx,ipt) = muPowerLawField(cell_or_side_idx,ipt) * Nval * std::pow (u_norm(cell_or_side_idx,ipt), power-1);
        }
      } else {
        for (unsigned int ipt=0; ipt<dim; ++ipt) {
          ScalarT Nval = N(cell_or_side_idx,ipt);
          if (is_side_equation) Nval = KU::max(Nval, 0.0);
          beta(cell_or_side_idx,ipt) = mu * Nval * std::pow (u_norm(cell_or_side_idx,ipt), power-1);
        }
      }
    break;

    case REGULARIZED_COULOMB:
      if (distributedLambda) {
        if (distributedMu) {
          for (unsigned int ipt=0; ipt<dim; ++ipt) {
            ScalarT Nval = N(cell_or_side_idx,ipt);
            if (is_side_equation) Nval = KU::max(Nval, 0.0);
            ScalarT q = u_norm(cell_or_side_idx,ipt) / ( u_norm(cell_or_side_idx,ipt) + lambdaField(cell_or_side_idx,ipt)*ice_softness(cell_or_side_idx)*std::pow(Nval,n) );
            beta(cell_or_side_idx,ipt) = muCoulombField(cell_or_side_idx,ipt) * Nval * std::pow( q, power) / u_norm(cell_or_side_idx,ipt);
          }
        } else {
          for (unsigned int ipt=0; ipt<dim; ++ipt) {
            ScalarT Nval = N(cell_or_side_idx,ipt);
            if (is_side_equation) Nval = KU::max(Nval, 0.0);
            ScalarT q = u_norm(cell_or_side_idx,ipt) / ( u_norm(cell_or_side_idx,ipt) + lambdaField(cell_or_side_idx,ipt)*ice_softness(cell_or_side_idx)*std::pow(Nval,n) );
            beta(cell_or_side_idx,ipt) = mu * Nval * std::pow( q, power) / u_norm(cell_or_side_idx,ipt);
          }
        }
      } else {
        if (distributedMu) {
          for (unsigned int ipt=0; ipt<dim; ++ipt) {
            ScalarT Nval = N(cell_or_side_idx,ipt);
            if (is_side_equation) Nval = KU::max(Nval, 0.0);
            ScalarT q = u_norm(cell_or_side_idx,ipt) / ( u_norm(cell_or_side_idx,ipt) + lambda*ice_softness(cell_or_side_idx)*std::pow(Nval,n) );
            beta(cell_or_side_idx,ipt) = muCoulombField(cell_or_side_idx,ipt) * Nval * std::pow( q, power) / u_norm(cell_or_side_idx,ipt);
          }
        } else {
          for (unsigned int ipt=0; ipt<dim; ++ipt) {
            ScalarT Nval = N(cell_or_side_idx,ipt);
            if (is_side_equation) Nval = KU::max(Nval, 0.0);
            ScalarT q = u_norm(cell_or_side_idx,ipt) / ( u_norm(cell_or_side_idx,ipt) + lambda*ice_softness(cell_or_side_idx)*std::pow(Nval,n) );
            beta(cell_or_side_idx,ipt) = mu * Nval * std::pow( q, power) / u_norm(cell_or_side_idx,ipt);
          }
        }
      }
    break;

    case EXP_GIVEN_FIELD:
      if(nodal || interpolate_then_exponentiate) {
        if (is_given_field_param) {
          for (unsigned int ipt=0; ipt<dim; ++ipt)
            beta(cell_or_side_idx,ipt) = std::exp(given_field_param(cell_or_side_idx,ipt));
        } else {
          for (unsigned int ipt=0; ipt<dim; ++ipt)
            beta(cell_or_side_idx,ipt) = std::exp(given_field(cell_or_side_idx,ipt));
        }
      } else {
        if (is_given_field_param) {
          unsigned int exp_dim = is_side_equation ? numQPs : dim;
          for (unsigned int ipt=0; ipt<exp_dim; ++ipt) {
            beta(cell_or_side_idx,ipt) = 0;
            for (unsigned int node=0; node<numNodes; ++node)
              beta(cell_or_side_idx,ipt) += std::exp(given_field_param(cell_or_side_idx,node))*BF(cell_or_side_idx,node,ipt);  
          }
        } else {
          unsigned int exp_dim = is_side_equation ? numQPs : dim;
          for (unsigned int ipt=0; ipt<exp_dim; ++ipt) {
            beta(cell_or_side_idx,ipt) = 0;
            for (unsigned int node=0; node<numNodes; ++node)
              beta(cell_or_side_idx,ipt) += std::exp(given_field(cell_or_side_idx,node))*BF(cell_or_side_idx,node,ipt);
          }
        }
      }
    break;

    default:
    break;
  }

  if (is_side_equation && zero_on_floating) {
    if (is_thickness_param) {
      for (unsigned int ipt=0; ipt<dim; ++ipt) {
        ParamScalarT isGrounded = rho_i*thickness_param_field(cell_or_side_idx,ipt) > -rho_w*bed_topo_field_mst(cell_or_side_idx,ipt);
        beta(cell_or_side_idx,ipt) *=  isGrounded;
      }
    } else {
      for (unsigned int ipt=0; ipt<dim; ++ipt) {
        ParamScalarT isGrounded = rho_i*thickness_field(cell_or_side_idx,ipt) > -rho_w*bed_topo_field(cell_or_side_idx,ipt);
        beta(cell_or_side_idx,ipt) *=  isGrounded;
      }
    }
  }

  if (use_stereographic_map) {
    for (unsigned int ipt=0; ipt<dim; ++ipt) {
      MeshScalarT x = coordVec(cell_or_side_idx,ipt,0) - x_0;
      MeshScalarT y = coordVec(cell_or_side_idx,ipt,1) - y_0;
      MeshScalarT h = 4.0*R2/(4.0*R2 + x*x + y*y);
      beta(cell_or_side_idx,ipt) *= h*h;
    }
  }

}

//**********************************************************************
template<typename EvalT, typename Traits, typename EffPressureST, typename VelocityST, typename TemperatureST>
void BasalFrictionCoefficient<EvalT, Traits, EffPressureST, VelocityST, TemperatureST>::
evaluateFields (typename Traits::EvalData workset)
{
  if (memoizer.have_saved_data(workset,this->evaluatedFields()))
    return;

  if (beta_type == POWER_LAW) {
    if (logParameters) {
      power = std::exp(Albany::convertScalar<const ParamScalarT>(powerParam(0)));
      if (!distributedMu)
        mu = std::exp(Albany::convertScalar<const ParamScalarT>(muPowerLaw(0)));
    } else {
      power = Albany::convertScalar<const ParamScalarT>(powerParam(0));
      if (!distributedMu)
        mu = Albany::convertScalar<const ParamScalarT>(muPowerLaw(0));
    }
#ifdef OUTPUT_TO_SCREEN
    Teuchos::RCP<Teuchos::FancyOStream> output(Teuchos::VerboseObjectBase::getDefaultOStream());
    int procRank = Teuchos::GlobalMPISession::getRank();
    int numProcs = Teuchos::GlobalMPISession::getNProc();
    output->setProcRankAndSize (procRank, numProcs);
    output->setOutputToRootOnly (0);

    if (!distributedMu && printedMu!=mu) {
      *output << "[Basal Friction Coefficient" << PHX::print<EvalT>() << "] mu = " << mu << " [kPa yr^q m^{-q}]\n";
      printedMu = mu;
    }

    if (printedQ!=power) {
      *output << "[Basal Friction Coefficient" << PHX::print<EvalT>() << "] power = " << power << "\n";
      printedQ = power;
    }
#endif

    TEUCHOS_TEST_FOR_EXCEPTION (power<0, Teuchos::Exceptions::InvalidParameter,
                                "\nError in LandIce::BasalFrictionCoefficient: 'Power Exponent' must be >= 0.\n");
    TEUCHOS_TEST_FOR_EXCEPTION (!distributedMu && mu<0, Teuchos::Exceptions::InvalidParameter,
                                "\nError in LandIce::BasalFrictionCoefficient: 'Coulomb Friction Coefficient' must be >= 0.\n");
  }

  if (beta_type==REGULARIZED_COULOMB) {
    if (logParameters) {
      power = std::exp(Albany::convertScalar<const ParamScalarT>(powerParam(0)));

      if (!distributedLambda)
        lambda = std::exp(Albany::convertScalar<const ParamScalarT>(lambdaParam(0)));
      if (!distributedMu) 
        mu     = std::exp(Albany::convertScalar<const ParamScalarT>(muCoulomb(0)));
    } else {
      power = Albany::convertScalar<const ParamScalarT>(powerParam(0));

      if (!distributedLambda)
        lambda = Albany::convertScalar<const ParamScalarT>(lambdaParam(0));
      if (!distributedMu) 
        mu     = Albany::convertScalar<const ParamScalarT>(muCoulomb(0));
    }
#ifdef OUTPUT_TO_SCREEN
    Teuchos::RCP<Teuchos::FancyOStream> output(Teuchos::VerboseObjectBase::getDefaultOStream());
    int procRank = Teuchos::GlobalMPISession::getRank();
    int numProcs = Teuchos::GlobalMPISession::getNProc();
    output->setProcRankAndSize (procRank, numProcs);
    output->setOutputToRootOnly (0);

    if (!distributedLambda && printedLambda!=lambda) {
      *output << "[Basal Friction Coefficient" << PHX::print<EvalT>() << "] lambda = " << lambda << "\n";
      printedLambda = lambda;
    }

    if (!distributedMu && printedMu!=mu) {
      *output << "[Basal Friction Coefficient" << PHX::print<EvalT>() << "] mu = " << mu << "\n";
      printedMu = mu;
    }

    if (printedQ!=power) {
      *output << "[Basal Friction Coefficient" << PHX::print<EvalT>() << "] power = " << power << "\n";
      printedQ = power;
    }
#endif

    TEUCHOS_TEST_FOR_EXCEPTION (power<0, Teuchos::Exceptions::InvalidParameter,
                                "\nError in LandIce::BasalFrictionCoefficient: 'Power Exponent' must be >= 0.\n");
    TEUCHOS_TEST_FOR_EXCEPTION (!distributedMu && mu<0, Teuchos::Exceptions::InvalidParameter,
                                "\nError in LandIce::BasalFrictionCoefficient: 'Coulomb Friction Coefficient' must be >= 0.\n");
    TEUCHOS_TEST_FOR_EXCEPTION (!distributedLambda && lambda<0, Teuchos::Exceptions::InvalidParameter,
                                "\nError in LandIce::BasalFrictionCoefficient: \"Bed Roughness\" must be >= 0.\n");
  }

  dim = nodal ? numNodes : numQPs;

  if (is_side_equation) {
    if (workset.sideSetViews->find(basalSideName)==workset.sideSetViews->end()) return;
    sideSet = workset.sideSetViews->at(basalSideName);
    Kokkos::parallel_for(BasalFrictionCoefficient_Policy(0, sideSet.size), *this);
  } else {
    Kokkos::parallel_for(BasalFrictionCoefficient_Policy(0, workset.numCells), *this);
  }
}

} // Namespace LandIce
