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

#include "Albany_StringUtils.hpp" // for 'upper_case'

#include "Albany_Utils.hpp"

//uncomment the following line if you want debug output to be printed to screen
// #define OUTPUT_TO_SCREEN

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
  dim = -1;
  worksetSize = -1;
  is_power_parameter = false;
  use_pressurized_bed = false;
  save_pressure_field = false;
  overburden_fraction = 0.0;
  pressure_smoothing_length_scale = 1.0;
  Teuchos::ParameterList beta_list = *p.get<Teuchos::ParameterList*>("Parameter List");

  //Validate Parameters
  Teuchos::ParameterList validPL;
  validPL.set<std::string>("Type", "Power Law", "Type Of Beta: Constant, Power Law, Regularized Coulomb, Debris Friction");
  validPL.set<double>("Power Exponent", 1.0, "Exponent of Power Law or Regularized Coulomb Law");
  validPL.set<std::string>("Mu Type", "Field", "Mu Type: Constant,Field, Exponent Of Field, Exponent Of Field At Nodes");
  validPL.set<std::string>("Mu Field Name", "", "Name of the Field Mu");
  validPL.set<double>("Mu", 1.0, "Constant value for Mu");
  validPL.set<std::string>("Bed Roughness Type", "Field", "Bed Roughness Type: Field, Exponent Of Field, Exponent Of Field At Nodes");
  validPL.set<double>("Bed Roughness", 1.0, "Constant value for Bed Roughness");
  validPL.set<std::string>("Flow Rate Type", "Viscosity Flow Rate", "Type of Flow Rate: Constant, Flow Rate Type");
  validPL.set<double>("Flow Rate", 3.17e-24, "Constant Value for Flow Rate");
  validPL.set<std::string>("Effective Pressure Type", "Field", "Type of N: One, Field, Hydrostatic, Hydrostatic Computed At Nodes");
  validPL.set<double>("Effective Pressure", 1.0, "Effective Pressure [kPa]");
  validPL.set<double>("Effective Pressure Regularization", 1000.0, "Effective Pressure Regularization [kPa]");
  validPL.set<double>("Sliding Velocity Regularization", 500.0, "Sliding Velocity Regularization [m yr^{-1}]");
  validPL.set<double>("Minimum Fraction Overburden Pressure", 1.0, "Minimum Fraction Overburden Pressure");
  validPL.set<double>("Length Scale Factor", 1.0, "Length Scale Factor [km]");
  validPL.set<std::string>("Beta Field Name", "", "Name of the Field Mu");
  validPL.set<double>("Beta", 1.0, "Constant value for beta");
  validPL.set<bool>("Zero Effective Pressure On Floating Ice At Nodes", false, "Whether to zero the effective pressure on floating ice at nodes");
  validPL.set<bool>("Zero Beta On Floating Ice", false, "Whether to zero beta on floating ice");
  validPL.set<bool>("Exponentiate Scalar Parameters", false, "Whether the scalar parameters needs to be exponentiate");
  validPL.set<bool>("Use Pressurized Bed Above Sea Level", false, "Whether to use a Downs & Johnson (2022) type parameterization for basal water pressure");
  validPL.set<std::string>("Bulk Friction Coefficient Type", "Constant", "Bulk Friction Coefficient Type: Field");
  validPL.set<double>("Bulk Friction Coefficient", 0.0, "Constant value for Bulk Friction Coefficient (for debris friction slip)");
  validPL.set<std::string>("Basal Debris Factor Type", "Constant", "Basal Debris Factor Type: Field");
  validPL.set<double>("Basal Debris Factor", 0.0, "Constant value for Basal Debris Factor (for debris friction slip)"); 
  beta_list.validateParameters(validPL,0);

  zero_on_floating = beta_list.get<bool> ("Zero Beta On Floating Ice", false);
  zero_N_on_floating_at_nodes = beta_list.get<bool> ("Zero Effective Pressure On Floating Ice At Nodes", false);

  //whether to first interpolate the given field and then exponetiate it (on quad points) or the other way around.
  logParameters = beta_list.get<bool>("Exponentiate Scalar Parameters",false);

  std::string betaType = util::upper_case(beta_list.get<std::string>("Type"));  

  is_side_equation = p.isParameter("Side Set Name");

  if (is_side_equation) {
    TEUCHOS_TEST_FOR_EXCEPTION (!dl->isSideLayouts, Teuchos::Exceptions::InvalidParameter,
                                "Error! The layout structure does not appear to be that of a side set.\n");

    basalSideName = p.get<std::string>("Side Set Name");
    numQPs        = dl->qp_scalar->extent(1);
    numNodes      = dl->node_scalar->extent(1);
  } else {
    TEUCHOS_TEST_FOR_EXCEPTION (dl->isSideLayouts, Teuchos::Exceptions::InvalidParameter,
                                "Error! The layout structure appears to be that of a side set.\n");

    numQPs    = dl->qp_scalar->extent(1);
    numNodes  = dl->node_scalar->extent(1);
  }

  nodal = p.isParameter("Nodal") ? p.get<bool>("Nodal") : false;
  Teuchos::RCP<PHX::DataLayout> layout, nodal_layout;
  layout = nodal ? dl->node_scalar : dl->qp_scalar;
  nodal_layout = dl->node_scalar;

  beta = PHX::MDField<ScalarT>(p.get<std::string> ("Basal Friction Coefficient Variable Name"), layout);
  this->addEvaluatedField(beta);

  printedMu            = -9999.999;
  printedBedRoughness  = -9999.999;
  printedQ             = -9999.999;
  printedBulkFriction  = -9999.999;
  printedBasalDebris   = -9999.999;

  if (betaType == "CONSTANT") {
    beta_type = BETA_TYPE::CONSTANT;
    beta_val = beta_list.get<double>("Beta");
#ifdef OUTPUT_TO_SCREEN
    *output << "LandIce::BasalFrictionCoefficient: Constant beta value = " << beta_val << " (loaded from yaml input file).\n";
#endif
  } else if (betaType == "FIELD") {
#ifdef OUTPUT_TO_SCREEN
    *output << "LandIce::BasalFrictionCoefficient: Prescribed beta field.\n";
#endif

      //turning this into a Power Law BC
      beta_type = BETA_TYPE::POWER_LAW;
      bool forbidden_parameter = beta_list.isParameter("Power Exponent") || beta_list.isParameter("Mu Type") ||
          beta_list.isParameter("Mu Field Name") || beta_list.isParameter("Effective Pressure Type") || beta_list.isParameter("Effective Pressure");
      TEUCHOS_TEST_FOR_EXCEPTION(forbidden_parameter, Teuchos::Exceptions::InvalidParameter,
          std::endl << "Error in LandIce::BasalFrictionCoefficient: invalid parameter for Beta Type == Field. Use Power Law Type if you need to set that.\n");
      beta_list.set<std::string>("Mu Field Name", beta_list.get<std::string>("Beta Field Name"));
      beta_list.set<double>("Power Exponent", 1.0);
      beta_list.set<std::string>("Effective Pressure Type", "Constant");
      beta_list.set<double>("Effective Pressure", 1.0);
      beta_list.set<std::string>("Mu Type", "Field");
      N_val = 1;
  } else if (betaType == "POWER LAW") {
    beta_type = BETA_TYPE::POWER_LAW;
#ifdef OUTPUT_TO_SCREEN
    *output << "Velocity-dependent beta (power law):\n\n"
            << "      beta = mu * N * |u|^{q-1} \n\n"
            << "  with N being the effective pressure, |u| the sliding velocity\n";
#endif
  } else if (betaType == "REGULARIZED COULOMB") {
    beta_type = BETA_TYPE::REGULARIZED_COULOMB;
#ifdef OUTPUT_TO_SCREEN
    *output << "Velocity-dependent beta (regularized coulomb law):\n\n"
            << "      beta = mu * N * |u|^{q-1} / [|u| + bedRoughness*A*N^n]^q\n\n"
            << "  with N being the effective pressure, |u| the sliding velocity\n";
#endif

    std::string flowRateType = util::upper_case(beta_list.get<std::string>("Flow Rate Type", "Viscosity Flow Rate"));
    n = p.get<Teuchos::ParameterList*>("Viscosity Parameter List")->get<double>("Glen's Law n");
    if(flowRateType == "VISCOSITY FLOW RATE") {
      flowRate_type = FLOW_RATE_TYPE::VISCOSITY_FLOW_RATE;
      flowRate = PHX::MDField<const TemperatureST>(p.get<std::string>("Flow Rate Variable Name"), dl->cell_scalar2);
      this->addDependentField (flowRate);
    } else if (flowRateType == "CONSTANT") {  //"Constant"
      flowRate_type = FLOW_RATE_TYPE::CONSTANT;
      flowRate_val = beta_list.get<double>("Flow Rate");
#ifdef OUTPUT_TO_SCREEN
      *output << "LandIce::BasalFrictionCoefficient: Constant Flow Rate value = " << flowRate_val << " (loaded from yaml input file).\n";
#endif
    } else {
      TEUCHOS_TEST_FOR_EXCEPTION(true, Teuchos::Exceptions::InvalidParameter,
        std::endl << "Error in LandIce::BasalFrictionCoefficient:  \"" << flowRateType << "\" is not a valid parameter for Is not a valid parameter for Flow Rate Type\n");
    }

  } else if (betaType == "DEBRIS FRICTION") {
    beta_type = BETA_TYPE::DEBRIS_FRICTION;
#ifdef OUTPUT_TO_SCREEN
    *output << "Velocity-dependent beta (debris friction law):\n\n"
	    << "    beta = mu * N * |u|^{q-1} / [|u| + bedRoughness*A*N^n]^q + BulkFrictionCoefficient*N/(1+N/N0)/|u| + BasalDebrisFactor/(1+|u|/u0)\n\n"
	    << " with N being the effective pressure, |u| the sliding velocity\n";
#endif

    std::string flowRateType = util::upper_case(beta_list.get<std::string>("Flow Rate Type", "Viscosity Flow Rate"));
    n = p.get<Teuchos::ParameterList*>("Viscosity Parameter List")->get<double>("Glen's Law n");
    if(flowRateType == "VISCOSITY FLOW RATE") {
      flowRate_type = FLOW_RATE_TYPE::VISCOSITY_FLOW_RATE;
      flowRate = PHX::MDField<const TemperatureST>(p.get<std::string>("Flow Rate Variable Name"), dl->cell_scalar2);
      this->addDependentField (flowRate);
    } else if (flowRateType == "CONSTANT") {  //"Constant"
      flowRate_type = FLOW_RATE_TYPE::CONSTANT;
      flowRate_val = beta_list.get<double>("Flow Rate");
#ifdef OUTPUT_TO_SCREEN
      *output << "LandIce::BasalFrictionCoefficient: Constant Flow Rate value = " << flowRate_val << " (loaded from yaml input file).\n";
#endif
    } else {
      TEUCHOS_TEST_FOR_EXCEPTION(true, Teuchos::Exceptions::InvalidParameter,
        std::endl << "Error in LandIce::BasalFrictionCoefficient:  \"" << flowRateType << "\" is not a valid parameter for Is not a valid parameter for Flow Rate Type\n");
    }

  } else {
    TEUCHOS_TEST_FOR_EXCEPTION(true, Teuchos::Exceptions::InvalidParameter,
        std::endl << "Error in LandIce::BasalFrictionCoefficient:  \"" << betaType << "\" is not a valid parameter for Beta Type\n");
  }


  if (beta_type != BETA_TYPE::CONSTANT) {

    std::string effectivePressureType = util::upper_case(beta_list.get<std::string>("Effective Pressure Type"));
    if (effectivePressureType == "CONSTANT") {
      effectivePressure_type = EFFECTIVE_PRESSURE_TYPE::CONSTANT;
      N_val = beta_list.get<double>("Effective Pressure");
    } else if (effectivePressureType == "FIELD") {
      effectivePressure_type = EFFECTIVE_PRESSURE_TYPE::FIELD;
      if(zero_N_on_floating_at_nodes)
        N = PHX::MDField<const EffPressureST>(p.get<std::string> ("Effective Pressure Variable Name"), nodal_layout);
      else
        N = PHX::MDField<const EffPressureST>(p.get<std::string> ("Effective Pressure Variable Name"), layout);
      this->addDependentField (N);
    } else if (effectivePressureType == "HYDROSTATIC") {
      effectivePressure_type = EFFECTIVE_PRESSURE_TYPE::HYDROSTATIC;
      use_pressurized_bed = beta_list.get<bool>("Use Pressurized Bed Above Sea Level", false);
      save_pressure_field = p.isParameter("Effective Pressure Output Variable Name");
      if(save_pressure_field && nodal) {
        outN = PHX::MDField<EffPressureST>(p.get<std::string> ("Effective Pressure Output Variable Name"), nodal_layout);
        this->addEvaluatedField (outN);
      }
    } else if (effectivePressureType == "HYDROSTATIC COMPUTED AT NODES") {
      effectivePressure_type = EFFECTIVE_PRESSURE_TYPE::HYDROSTATIC_AT_NODES;
      use_pressurized_bed = beta_list.get<bool>("Use Pressurized Bed Above Sea Level", false);
      save_pressure_field = p.isParameter("Effective Pressure Output Variable Name");
      if(save_pressure_field && nodal) {
        outN = PHX::MDField<EffPressureST>(p.get<std::string> ("Effective Pressure Output Variable Name"), nodal_layout);
        this->addEvaluatedField (outN);
      }
    } else {
      TEUCHOS_TEST_FOR_EXCEPTION(true, Teuchos::Exceptions::InvalidParameter,
        std::endl << "Error in LandIce::BasalFrictionCoefficient:  \"" << effectivePressureType << "\" is not a valid parameter for Effective Pressure Type\n");
    }

    if(use_pressurized_bed) {
      overburden_fraction = beta_list.get<double>("Minimum Fraction Overburden Pressure");
      pressure_smoothing_length_scale = beta_list.get<double>("Length Scale Factor");
      TEUCHOS_TEST_FOR_EXCEPTION(pressure_smoothing_length_scale <= 0.0, Teuchos::Exceptions::InvalidParameter,
        std::endl << "Error in LandIce::BasalFrictionCoefficient:  \"Length Scale Factor\" should be positive\n");
    }

    if(zero_on_floating || zero_N_on_floating_at_nodes || (effectivePressure_type == EFFECTIVE_PRESSURE_TYPE::HYDROSTATIC_AT_NODES) || (effectivePressure_type == EFFECTIVE_PRESSURE_TYPE::HYDROSTATIC) ) {
      bed_topo_field = PHX::MDField<const MeshScalarT>(p.get<std::string> ("Bed Topography Variable Name"), nodal_layout);
      this->addDependentField (bed_topo_field);
      thickness_field = PHX::MDField<const MeshScalarT>(p.get<std::string> ("Ice Thickness Variable Name"), nodal_layout);
      this->addDependentField (thickness_field);
      if(!nodal) {
        BF = PHX::MDField<const RealType>(p.get<std::string> ("BF Variable Name"), dl->node_qp_scalar);
        this->addDependentField (BF);
      }
      Teuchos::ParameterList& phys_param_list = *p.get<Teuchos::ParameterList*>("Physical Parameter List");
      rho_i = phys_param_list.get<double> ("Ice Density");
      rho_w = phys_param_list.get<double> ("Water Density");
      g = phys_param_list.get<double> ("Gravity Acceleration");
    }

    if(!nodal && zero_N_on_floating_at_nodes && (effectivePressure_type == EFFECTIVE_PRESSURE_TYPE::HYDROSTATIC)) {
      TEUCHOS_TEST_FOR_EXCEPTION(true, Teuchos::Exceptions::InvalidParameter,
        std::endl << "Error in LandIce::BasalFrictionCoefficient: Cannot set the effective pressure to zero at nodes when effective pressure type is Hydrostatic\n");
    }

    std::string muType = util::upper_case((beta_list.isParameter("Mu Type") ? beta_list.get<std::string>("Mu Type") : "Field"));

    if(muType == "CONSTANT") {
      mu_type = FIELD_TYPE::CONSTANT;
    } else if (muType == "FIELD") {
      mu_type = FIELD_TYPE::FIELD;
    } else if (muType == "EXPONENT OF FIELD AT NODES") {
      mu_type = FIELD_TYPE::EXPONENT_OF_FIELD_AT_NODES;
    } else if (muType == "EXPONENT OF FIELD") {
      mu_type = FIELD_TYPE::EXPONENT_OF_FIELD;
    } else {
      TEUCHOS_TEST_FOR_EXCEPTION(true, Teuchos::Exceptions::InvalidParameter,
        std::endl << "Error in LandIce::BasalFrictionCoefficient:  \"" << muType << "\" is not a valid parameter for Mu Type\n");
    }

    if(mu_type == FIELD_TYPE::CONSTANT) {
      muParam     = PHX::MDField<const ScalarT,Dim>("Mu", dl->shared_param);
      this->addDependentField (muParam);
    } else {
      std::string mu_field_name;
      mu_field_name = beta_list.get<std::string> ("Mu Field Name");
      if (is_side_equation) {
        mu_field_name += "_" + basalSideName;
      }

      auto layout_mu_field = layout;
      if(!nodal && (mu_type == FIELD_TYPE::EXPONENT_OF_FIELD_AT_NODES)) {
        BF = PHX::MDField<const RealType>(p.get<std::string> ("BF Variable Name"), dl->node_qp_scalar);
        layout_mu_field = nodal_layout;
        this->addDependentField (BF);
      }

      muField = PHX::MDField<const ParamScalarT>(mu_field_name, layout_mu_field);
      this->addDependentField (muField);
    }

    powerParam     = PHX::MDField<const ScalarT,Dim>("Power Exponent", dl->shared_param);
    this->addDependentField (powerParam);

    if (beta_type == BETA_TYPE::REGULARIZED_COULOMB) {

      std::string bedRoughnessType = util::upper_case((beta_list.isParameter("Bed Roughness Type") ? beta_list.get<std::string>("Bed Roughness Type") : "Field"));

      auto layout_bedRoughness_field = layout;
      if(bedRoughnessType == "CONSTANT") {
        bedRoughness_type = FIELD_TYPE::CONSTANT;
      } else if (bedRoughnessType == "FIELD") {
        bedRoughness_type = FIELD_TYPE::FIELD;
      } else if (bedRoughnessType == "EXPONENT OF FIELD AT NODES") {
        bedRoughness_type = FIELD_TYPE::EXPONENT_OF_FIELD_AT_NODES;
        if(!nodal) {
          BF = PHX::MDField<const RealType>(p.get<std::string> ("BF Variable Name"), dl->node_qp_scalar);
          layout_bedRoughness_field = nodal_layout;
          this->addDependentField (BF);
        }
      } else if (bedRoughnessType == "EXPONENT OF FIELD") {
        bedRoughness_type = FIELD_TYPE::EXPONENT_OF_FIELD;
      } else {
        TEUCHOS_TEST_FOR_EXCEPTION(true, Teuchos::Exceptions::InvalidParameter,
          std::endl << "Error in LandIce::BasalFrictionCoefficient:  \"" << bedRoughnessType << "\" is not a valid parameter for Bed Roughness Type\n");
      }

      if(bedRoughness_type == FIELD_TYPE::CONSTANT) {
        bedRoughnessParam    = PHX::MDField<ScalarT,Dim>("Bed Roughness", dl->shared_param);
        this->addDependentField (bedRoughnessParam);
      } else {
        bedRoughnessField = PHX::MDField<const ParamScalarT>(p.get<std::string> ("Bed Roughness Variable Name"), layout_bedRoughness_field);
        this->addDependentField (bedRoughnessField);
      }
      u_norm = PHX::MDField<const VelocityST>(p.get<std::string> ("Sliding Velocity Variable Name"), layout);
      this->addDependentField (u_norm);

    } else if (beta_type == BETA_TYPE::DEBRIS_FRICTION) {

      std::string bedRoughnessType = util::upper_case((beta_list.isParameter("Bed Roughness Type") ? beta_list.get<std::string>("Bed Roughness Type") : "Field"));

      N0 = beta_list.get<double>("Effective Pressure Regularization");
      u0 = beta_list.get<double>("Sliding Velocity Regularization");
      //        N0    = 1.0e3;                 // Regularization value for effective pressure [kPa]
      //  u0    = 500.0;                 // Regularization value for sliding velocity [m yr^-1]

      auto layout_bedRoughness_field = layout;
      if(bedRoughnessType == "CONSTANT") {
        bedRoughness_type = FIELD_TYPE::CONSTANT;
      } else if (bedRoughnessType == "FIELD") {
        bedRoughness_type = FIELD_TYPE::FIELD;
      } else if (bedRoughnessType == "EXPONENT OF FIELD AT NODES") {
        bedRoughness_type = FIELD_TYPE::EXPONENT_OF_FIELD_AT_NODES;
        if(!nodal) {
          BF = PHX::MDField<const RealType>(p.get<std::string> ("BF Variable Name"), dl->node_qp_scalar);
          layout_bedRoughness_field = nodal_layout;
          this->addDependentField (BF);
        }
      } else if (bedRoughnessType == "EXPONENT OF FIELD") {
        bedRoughness_type = FIELD_TYPE::EXPONENT_OF_FIELD;
      } else {
        TEUCHOS_TEST_FOR_EXCEPTION(true, Teuchos::Exceptions::InvalidParameter,
          std::endl << "Error in LandIce::BasalFrictionCoefficient:  \"" << bedRoughnessType << "\" is not a valid parameter for Bed Roughness Type\n");
      }

      if(bedRoughness_type == FIELD_TYPE::CONSTANT) {
        bedRoughnessParam    = PHX::MDField<ScalarT,Dim>("Bed Roughness", dl->shared_param);
        this->addDependentField (bedRoughnessParam);
      } else {
        bedRoughnessField = PHX::MDField<const ParamScalarT>(p.get<std::string> ("Bed Roughness Variable Name"), layout_bedRoughness_field);
        this->addDependentField (bedRoughnessField);
      }

      std::string bulkFrictionType = util::upper_case((beta_list.isParameter("Bulk Friction Coefficient Type") ? beta_list.get<std::string>("Bulk Friction Coefficient Type") : "Field"));

      auto layout_bulkFriction_field = layout;
      if (bulkFrictionType == "CONSTANT") {
	bulkFriction_type = FIELD_TYPE::CONSTANT;
      } else if (bulkFrictionType == "FIELD") {
        bulkFriction_type = FIELD_TYPE::FIELD;
      } else {
        TEUCHOS_TEST_FOR_EXCEPTION(true, Teuchos::Exceptions::InvalidParameter,
	  std::endl << "Error in LandIce::BasalFrictionCoefficient: \"" << bulkFrictionType << "\" is not a valid parameter for Bulk Friction Coefficient Type\n");
      }

      if (bulkFriction_type == FIELD_TYPE::CONSTANT) {
        bulkFrictionParam = PHX::MDField<ScalarT,Dim>("Bulk Friction Coefficient", dl->shared_param);
	this->addDependentField (bulkFrictionParam);
      } else {
	bulkFrictionField = PHX::MDField<const ParamScalarT>(p.get<std::string> ("Bulk Friction Coefficient Variable Name"), layout_bulkFriction_field);
	this->addDependentField (bulkFrictionField);
      }

      std::string basalDebrisType = util::upper_case((beta_list.isParameter("Basal Debris Factor Type") ? beta_list.get<std::string>("Basal Debris Factor Type") : "Field"));

      auto layout_basalDebris_field = layout;
      if (basalDebrisType == "CONSTANT") {
        basalDebris_type = FIELD_TYPE::CONSTANT;
      } else if (basalDebrisType == "FIELD") {
        basalDebris_type = FIELD_TYPE::FIELD;
      } else {
      	TEUCHOS_TEST_FOR_EXCEPTION(true, Teuchos::Exceptions::InvalidParameter,
	  std::endl << "Error in LandIce:BasalFrictionCoefficient: \"" << basalDebrisType << "\" is not a valid parameter for Basal Debris Factor Type\n");
      } 

      if (basalDebris_type == FIELD_TYPE::CONSTANT) {
        basalDebrisParam = PHX::MDField<ScalarT,Dim>("Basal Debris Factor", dl->shared_param);
	this->addDependentField (basalDebrisParam);
      } else {
      	basalDebrisField = PHX::MDField<const ParamScalarT>(p.get<std::string> ("Basal Debris Factor Variable Name"), layout_basalDebris_field);
	this->addDependentField (basalDebrisField);
      } 

      u_norm = PHX::MDField<const VelocityST>(p.get<std::string> ("Sliding Velocity Variable Name"), layout);
      this->addDependentField (u_norm);

    } else if (beta_type == BETA_TYPE::POWER_LAW) {
      auto paramLib = p.get<Teuchos::RCP<ParamLib> >("Parameter Library");
      is_power_parameter = paramLib->isParameter("Power Exponent");
      if(p.isParameter("Random Parameters")) {
        Teuchos::ParameterList rparams = *p.get<Teuchos::ParameterList*>("Random Parameters");
        if (!is_power_parameter) {
          int nrparams = rparams.get<int>("Number Of Parameters");
          for (int i_rparams=0; i_rparams<nrparams; ++i_rparams) {
            auto rparams_i = rparams.sublist(util::strint("Parameter",i_rparams));
            if (rparams_i.get<std::string>("Name") == "Power Exponent") {
              is_power_parameter = true;
              break;
            }
          }
        }
      }
      if (is_power_parameter) {
        u_norm = PHX::MDField<const VelocityST>(p.get<std::string> ("Sliding Velocity Variable Name"), layout);
        this->addDependentField (u_norm);
      } else {
        double q = beta_list.get<double>("Power Exponent");
        bool linearLaw = (!logParameters && q==1.0)||(logParameters && q==0.0);
        if(!linearLaw) {
          u_norm = PHX::MDField<const VelocityST>(p.get<std::string> ("Sliding Velocity Variable Name"), layout);
          this->addDependentField (u_norm);
        }
      }
    }
  }

  auto& stereographicMapList = p.get<Teuchos::ParameterList*>("Stereographic Map");
  use_stereographic_map = stereographicMapList->get("Use Stereographic Map", false);
  if(use_stereographic_map) {
    layout = nodal ? dl->node_vector: dl->qp_coords;
    coordVec = PHX::MDField<MeshScalarT>(p.get<std::string>("Coordinate Vector Variable Name"), layout);

    double R = stereographicMapList->get<double>("Earth Radius", 6371);
    x_0 = stereographicMapList->get<double>("X_0", 0);//-136);
    y_0 = stereographicMapList->get<double>("Y_0", 0);//-2040);
    R2 = std::pow(R,2);

    this->addDependentField(coordVec);
  }

  this->setName("BasalFrictionCoefficient"+PHX::print<EvalT>());
}

//**********************************************************************
template<typename EvalT, typename Traits, typename EffPressureST, typename VelocityST, typename TemperatureST>
void BasalFrictionCoefficient<EvalT, Traits, EffPressureST, VelocityST, TemperatureST>::
postRegistrationSetup (typename Traits::SetupData d,
                       PHX::FieldManager<Traits>&)
{
  if (beta_type == BETA_TYPE::CONSTANT)
    beta.deep_copy(ScalarT(beta_val));

  d.fill_field_dependencies(this->dependentFields(),this->evaluatedFields());
  if (d.memoizer_active())
    memoizer.enable_memoizer();
}

// *********************************************************************
// Kokkos functor
template<typename EvalT, typename Traits, typename EffPressureST, typename VelocityST, typename TemperatureST>
KOKKOS_INLINE_FUNCTION
void BasalFrictionCoefficient<EvalT, Traits, EffPressureST, VelocityST, TemperatureST>::
operator() (const BasalFrictionCoefficient_Tag& tag, const int& cell) const {

  ParamScalarT mu, bedRoughness, power, bulkFriction, basalDebris;

  if ((beta_type == BETA_TYPE::POWER_LAW) || (beta_type==BETA_TYPE::REGULARIZED_COULOMB) || (beta_type==BETA_TYPE::DEBRIS_FRICTION) ) {
    if (logParameters) {
      power = std::exp(Albany::convertScalar<const ParamScalarT>(powerParam(0)));
      if (mu_type == FIELD_TYPE::CONSTANT)
        mu = std::exp(Albany::convertScalar<const ParamScalarT>(muParam(0)));
    } else {
      power = Albany::convertScalar<const ParamScalarT>(powerParam(0));
      if (mu_type == FIELD_TYPE::CONSTANT)
        mu = Albany::convertScalar<const ParamScalarT>(muParam(0));
    }
  }

  if ( (beta_type== BETA_TYPE::REGULARIZED_COULOMB) || (beta_type == BETA_TYPE::DEBRIS_FRICTION) ) {
    if (logParameters) {
      if (bedRoughness_type == FIELD_TYPE::CONSTANT)
        bedRoughness = std::exp(Albany::convertScalar<const ParamScalarT>(bedRoughnessParam(0)));
    } else {
      if (bedRoughness_type == FIELD_TYPE::CONSTANT)
        bedRoughness = Albany::convertScalar<const ParamScalarT>(bedRoughnessParam(0));
    }
  }

  if (beta_type == BETA_TYPE::DEBRIS_FRICTION) {
    if (bulkFriction_type == FIELD_TYPE::CONSTANT)
      bulkFriction = Albany::convertScalar<const ParamScalarT>(bulkFrictionParam(0));
    if (basalDebris_type == FIELD_TYPE::CONSTANT)
      basalDebris = Albany::convertScalar<const ParamScalarT>(basalDebrisParam(0));
  } 

  ParamScalarT muValue = 1.0;
  typename Albany::StrongestScalarType<EffPressureST,MeshScalarT>::type NVal = N_val;

  if(beta_type != BETA_TYPE::CONSTANT) {
    for (int ipt=0; ipt<dim; ++ipt) {

      switch (effectivePressure_type) {
      case EFFECTIVE_PRESSURE_TYPE::FIELD:
        if(zero_N_on_floating_at_nodes) {
          NVal = 0.0;
          if(nodal) {
            if (rho_i*thickness_field(cell,ipt)+rho_w*bed_topo_field(cell,ipt) > 0)
              NVal = N(cell,ipt);
          } else {
            for (int node=0; node<numNodes; ++node)
              if (rho_i*thickness_field(cell,node)+rho_w*bed_topo_field(cell,node) > 0)
                NVal += N(cell,node)*BF(cell,node,ipt);
          }
        } else
          NVal = N(cell,ipt);
        break;
      case EFFECTIVE_PRESSURE_TYPE::CONSTANT:
        if(zero_N_on_floating_at_nodes) {
          NVal = 0;
          if(nodal) {
            if (rho_i*thickness_field(cell,ipt)+rho_w*bed_topo_field(cell,ipt) > 0)
              NVal =  1.0;
          } else {
            for (int node=0; node<numNodes; ++node)
              if (rho_i*thickness_field(cell,node)+rho_w*bed_topo_field(cell,node) > 0)
                NVal += BF(cell,node,ipt);
          }
        }
        break;
      case EFFECTIVE_PRESSURE_TYPE::HYDROSTATIC:
        if(nodal) {
          auto f_p = use_pressurized_bed ? MeshScalarT(1.0 / (1.0 + std::exp(-bed_topo_field(cell,ipt)/pressure_smoothing_length_scale))) : MeshScalarT(0.0);
          NVal = g* KU::max(rho_i*thickness_field(cell,ipt) - ( (overburden_fraction*rho_i*
                    thickness_field(cell,ipt)*f_p) + (1.0 - f_p)*
                    KU::max(-1.0 * rho_w*bed_topo_field(cell,ipt),0.0) ),0.0);
          if(save_pressure_field) {
            outN(cell,ipt) = Albany::convertScalar<EffPressureST>(NVal);
          }
        } else {
          MeshScalarT thickness(0), bed_topo(0);
          for (int node=0; node<numNodes; ++node) {
            thickness += thickness_field(cell,node)*BF(cell,node,ipt);
            bed_topo += bed_topo_field(cell,node)*BF(cell,node,ipt);
          }
          auto f_p = use_pressurized_bed ?  MeshScalarT(1.0 / (1.0 + std::exp(-bed_topo/pressure_smoothing_length_scale))) : MeshScalarT(0.0);
          NVal = g* KU::max(rho_i*thickness - ( (overburden_fraction*rho_i*
                    thickness*f_p) + (1.0 - f_p)*
                    KU::max(-1.0 * rho_w*bed_topo,0.0) ),0.0);
        }
        break;
      case EFFECTIVE_PRESSURE_TYPE::HYDROSTATIC_AT_NODES:
        if(nodal) {
          auto f_p = use_pressurized_bed ?  MeshScalarT(1.0 / (1.0 + std::exp(-bed_topo_field(cell,ipt)/pressure_smoothing_length_scale))) :  MeshScalarT(0.0);
          NVal = g* KU::max(rho_i*thickness_field(cell,ipt) - ( (overburden_fraction*rho_i*
                    thickness_field(cell,ipt)*f_p) + (1.0 - f_p)*
                    KU::max(-1.0 * rho_w*bed_topo_field(cell,ipt),0.0) ),0.0);
          if(save_pressure_field) {
            outN(cell,ipt) = Albany::convertScalar<EffPressureST>(NVal);
          }
        } else {
          NVal = 0;
          for (int node=0; node<numNodes; ++node) {
            auto f_p =use_pressurized_bed ?  MeshScalarT(1.0 / (1.0 + std::exp(-bed_topo_field(cell,node)/pressure_smoothing_length_scale))) :  MeshScalarT(0.0);
            NVal += g* KU::max(rho_i*thickness_field(cell,node) - ( (overburden_fraction*rho_i*
                    thickness_field(cell,node)*f_p) + (1.0 - f_p)*
                    KU::max(-1.0 * rho_w*bed_topo_field(cell,node),0.0) ),0.0)*BF(cell,node,ipt);
          }
        }
        break;
      }

      switch (mu_type) {
      case FIELD_TYPE::FIELD:
        muValue = muField(cell,ipt);
        break;
      case FIELD_TYPE::EXPONENT_OF_FIELD_AT_NODES:
        if(nodal)
          muValue = std::exp(muField(cell,ipt));
        else {
          muValue = 0;
          for (int node=0; node<numNodes; ++node)
            muValue += std::exp(muField(cell,node))*BF(cell,node,ipt);
        }
        break;
      case FIELD_TYPE::EXPONENT_OF_FIELD:
        muValue = std::exp(muField(cell,ipt));
        break;
      case FIELD_TYPE::CONSTANT:
        muValue = mu;
        break;
      }

      if (!is_power_parameter && power == 1.0)
        beta(cell,ipt) = muValue * NVal;
      else
        beta(cell,ipt) = muValue * NVal * std::pow (u_norm(cell,ipt), power-1.0);

      if (beta_type == BETA_TYPE::REGULARIZED_COULOMB) {
        ParamScalarT bedRoughnessValue;
        switch (bedRoughness_type) {
        case FIELD_TYPE::FIELD:
          bedRoughnessValue = bedRoughnessField(cell,ipt);
          break;
        case FIELD_TYPE::EXPONENT_OF_FIELD_AT_NODES:
          if(nodal)
            bedRoughnessValue = std::exp(bedRoughnessField(cell,ipt));
          else {
            bedRoughnessValue = 0;
            for (int node=0; node<numNodes; ++node)
              bedRoughnessValue += std::exp(bedRoughnessField(cell,node))*BF(cell,node,ipt);
          }
          break;
        case FIELD_TYPE::EXPONENT_OF_FIELD:
          bedRoughnessValue = std::exp(bedRoughnessField(cell,ipt));
          break;
        case FIELD_TYPE::CONSTANT:
          bedRoughnessValue = bedRoughness;
          break;
        }

        const double secsInYr = 365*24*3600;
        const double scaling = secsInYr*pow(1000,n+1); //turns flow rate from [Pa^{-n} s^{-1}] to [k^{-1} kPa^{-n} yr^{-1}]
        switch (flowRate_type) {
        case FLOW_RATE_TYPE::CONSTANT: {
          beta(cell,ipt) /=  std::pow ( u_norm(cell,ipt) + bedRoughnessValue*scaling*flowRate_val*std::pow(NVal,n),  power); //bedRoughness in km
          }
          break;
        case FLOW_RATE_TYPE::VISCOSITY_FLOW_RATE:
          beta(cell,ipt) /=  std::pow ( u_norm(cell,ipt) + bedRoughnessValue*scaling*flowRate(cell)*std::pow(NVal,n),  power); //bedRoughness in km
          break;
        }
      }
      if (beta_type == BETA_TYPE::DEBRIS_FRICTION) {
        ParamScalarT bedRoughnessValue;
	      ParamScalarT bulkFrictionValue;
	      ParamScalarT basalDebrisValue;
        switch (bedRoughness_type) {
        case FIELD_TYPE::FIELD:
          bedRoughnessValue = bedRoughnessField(cell,ipt);
          break;
        case FIELD_TYPE::EXPONENT_OF_FIELD_AT_NODES:
          if(nodal)
            bedRoughnessValue = std::exp(bedRoughnessField(cell,ipt));
          else {
            bedRoughnessValue = 0;
            for (int node=0; node<numNodes; ++node)
              bedRoughnessValue += std::exp(bedRoughnessField(cell,node))*BF(cell,node,ipt);
          }
          break;
        case FIELD_TYPE::EXPONENT_OF_FIELD:
          bedRoughnessValue = std::exp(bedRoughnessField(cell,ipt));
          break;
        case FIELD_TYPE::CONSTANT:
          bedRoughnessValue = bedRoughness;
          break;
        }

        switch (bulkFriction_type) {
        case FIELD_TYPE::FIELD:
          bulkFrictionValue = bulkFrictionField(cell,ipt);
          break;
        case FIELD_TYPE::CONSTANT:
          bulkFrictionValue = bulkFriction;
          break;
        }

        switch (basalDebris_type) {
        case FIELD_TYPE::FIELD:
          basalDebrisValue = basalDebrisField(cell,ipt);
          break;
        case FIELD_TYPE::CONSTANT:
          basalDebrisValue = basalDebris;
          break;
        } 
        const double secsInYr = 365*24*3600;
        const double scaling = secsInYr*pow(1000,n+1); //turns flow rate from [Pa^{-n} s^{-1}] to [k^{-1} kPa^{-n} yr^{-1}]
        switch (flowRate_type) {
        case FLOW_RATE_TYPE::CONSTANT: { 
          beta(cell,ipt) /=  std::pow ( u_norm(cell,ipt) + bedRoughnessValue*scaling*flowRate_val*std::pow(NVal,n),  power); //bedRoughness in km
          beta(cell,ipt) += bulkFrictionValue * NVal / (1.0 + NVal / N0) / u_norm(cell,ipt) + basalDebrisValue / (1.0 + u_norm(cell,ipt) / u0);
	      }
	      break;
        case FLOW_RATE_TYPE::VISCOSITY_FLOW_RATE:
          beta(cell,ipt) /=  std::pow ( u_norm(cell,ipt) + bedRoughnessValue*scaling*flowRate(cell)*std::pow(NVal,n),  power); //bedRoughness in km
          beta(cell,ipt) += bulkFrictionValue * NVal / (1.0 + NVal / N0) / u_norm(cell,ipt) + basalDebrisValue / (1.0 + u_norm(cell,ipt) / u0);
          break;
        }
      }
    }
  }

  if (is_side_equation && zero_on_floating) {
    for (int ipt=0; ipt<dim; ++ipt) {
      bool isGrounded;
      if(nodal)
        isGrounded = rho_i*thickness_field(cell,ipt) > -rho_w*bed_topo_field(cell,ipt);
      else {
        MeshScalarT thickness(0), bed_topo(0);
        for (int node=0; node<numNodes; ++node) {
          thickness += thickness_field(cell,node)*BF(cell,node,ipt);
          bed_topo += bed_topo_field(cell,node)*BF(cell,node,ipt);
        }
        isGrounded = rho_i*thickness > -rho_w*bed_topo;
      }
      if(!isGrounded)
        beta(cell,ipt) =  0;
    }
  }

  if (use_stereographic_map) {
    for (int ipt=0; ipt<dim; ++ipt) {
      MeshScalarT x = coordVec(cell,ipt,0) - x_0;
      MeshScalarT y = coordVec(cell,ipt,1) - y_0;
      MeshScalarT h = 4.0*R2/(4.0*R2 + x*x + y*y);
      beta(cell,ipt) *= h*h;
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

  if ( (beta_type == BETA_TYPE::POWER_LAW) || (beta_type == BETA_TYPE::REGULARIZED_COULOMB) || (beta_type == BETA_TYPE::DEBRIS_FRICTION) ) {
    if (logParameters) {
        auto hostPower_view = Kokkos::create_mirror_view(powerParam.get_view());
        Kokkos::deep_copy(hostPower_view, powerParam.get_view());
        ScalarT hostPower = hostPower_view(0);
        power = std::exp(Albany::convertScalar<ParamScalarT>(hostPower));
        if (mu_type == FIELD_TYPE::CONSTANT) {
          auto hostMu_view = Kokkos::create_mirror_view(muParam.get_view());
          Kokkos::deep_copy(hostMu_view, muParam.get_view());
          ScalarT hostMu = hostMu_view(0);
          mu = std::exp(Albany::convertScalar<ParamScalarT>(hostMu));
        }
      } else {
        auto hostPower_view = Kokkos::create_mirror_view(powerParam.get_view());
        Kokkos::deep_copy(hostPower_view, powerParam.get_view());
        ScalarT hostPower = hostPower_view(0);
        power = std::exp(Albany::convertScalar<ParamScalarT>(hostPower));
        if (mu_type == FIELD_TYPE::CONSTANT) {
          auto hostMu_view = Kokkos::create_mirror_view(muParam.get_view());
          Kokkos::deep_copy(hostMu_view, muParam.get_view());
          ScalarT hostMu = hostMu_view(0);
          mu = Albany::convertScalar<ParamScalarT>(hostMu);
        }
      }
  #ifdef OUTPUT_TO_SCREEN
      Teuchos::RCP<Teuchos::FancyOStream> output(Teuchos::VerboseObjectBase::getDefaultOStream());
      int procRank = Teuchos::GlobalMPISession::getRank();
      int numProcs = Teuchos::GlobalMPISession::getNProc();
      output->setProcRankAndSize (procRank, numProcs);
      output->setOutputToRootOnly (0);

      if (mu_type==FIELD_TYPE::CONSTANT && printedMu!=mu) {
        *output << "[Basal Friction Coefficient" << PHX::print<EvalT>() << "] mu = " << mu << " [kPa yr^q m^{-q}]\n";
        printedMu = mu;
      }

      if (printedQ!=power) {
        *output << "[Basal Friction Coefficient" << PHX::print<EvalT>() << "] power = " << power << "\n";
        printedQ = power;
      }
#endif
      TEUCHOS_TEST_FOR_EXCEPTION (
          power<0, Teuchos::Exceptions::InvalidParameter,
          "Error in LandIce::BasalFrictionCoefficient: 'Power Exponent' must be >= 0.\n"
          "   Input value: " + std::to_string(Albany::ADValue(mu)) + "\n");
      TEUCHOS_TEST_FOR_EXCEPTION (
          mu_type==FIELD_TYPE::CONSTANT && mu<0, Teuchos::Exceptions::InvalidParameter,
          "Error in LandIce::BasalFrictionCoefficient: 'Coulomb Friction Coefficient' must be >= 0.\n"
          "   Input value: " + std::to_string(Albany::ADValue(mu)) + "\n");
  }

  if ( (beta_type== BETA_TYPE::REGULARIZED_COULOMB) || (beta_type == BETA_TYPE::DEBRIS_FRICTION) ) {
      if (logParameters) {
        if (bedRoughness_type == FIELD_TYPE::CONSTANT) {
          auto hostBedRoughness_view = Kokkos::create_mirror_view(bedRoughnessParam.get_view());
          Kokkos::deep_copy(hostBedRoughness_view, bedRoughnessParam.get_view());
          ScalarT hostBedRoughness = hostBedRoughness_view(0);
          bedRoughness = std::exp(Albany::convertScalar<ParamScalarT>(hostBedRoughness));
        }
      } else {
        if (bedRoughness_type == FIELD_TYPE::CONSTANT) {
          auto hostBedRoughness_view = Kokkos::create_mirror_view(bedRoughnessParam.get_view());
          Kokkos::deep_copy(hostBedRoughness_view, bedRoughnessParam.get_view());
          ScalarT hostBedRoughness = hostBedRoughness_view(0);
          bedRoughness = Albany::convertScalar<ParamScalarT>(hostBedRoughness);
        }
      }
#ifdef OUTPUT_TO_SCREEN
      Teuchos::RCP<Teuchos::FancyOStream> output(Teuchos::VerboseObjectBase::getDefaultOStream());
      int procRank = Teuchos::GlobalMPISession::getRank();
      int numProcs = Teuchos::GlobalMPISession::getNProc();
      output->setProcRankAndSize (procRank, numProcs);
      output->setOutputToRootOnly (0);

      if (bedRoughness_type==FIELD_TYPE::CONSTANT && printedBedRoughness!=bedRoughness) {
        *output << "[Basal Friction Coefficient" << PHX::print<EvalT>() << "] bedRoughness = " << bedRoughness << "\n";
        printedBedRoughness = bedRoughness;
      }
#endif
      TEUCHOS_TEST_FOR_EXCEPTION ((bedRoughness_type == FIELD_TYPE::CONSTANT) && bedRoughness<0, Teuchos::Exceptions::InvalidParameter,
                                  "\nError in LandIce::BasalFrictionCoefficient: \"Bed Roughness\" must be >= 0.\n");
  } 

  if (beta_type == BETA_TYPE::DEBRIS_FRICTION) {
    if (bulkFriction_type == FIELD_TYPE::CONSTANT) {
      auto hostBulkFriction_view = Kokkos::create_mirror_view(bulkFrictionParam.get_view());
      Kokkos::deep_copy(hostBulkFriction_view, bulkFrictionParam.get_view());
      ScalarT hostBulkFriction = hostBulkFriction_view(0);
      bulkFriction = Albany::convertScalar<ParamScalarT>(hostBulkFriction);
    } 
    if (basalDebris_type == FIELD_TYPE::CONSTANT) {
      auto hostBasalDebris_view = Kokkos::create_mirror_view(basalDebrisParam.get_view());
      Kokkos::deep_copy(hostBasalDebris_view, basalDebrisParam.get_view());
      ScalarT hostBasalDebris = hostBasalDebris_view(0);
      basalDebris = Albany::convertScalar<ParamScalarT>(hostBasalDebris);
    } 
  } 

  dim = nodal ? numNodes : numQPs;

  if (is_side_equation) {
    if (workset.sideSetViews->find(basalSideName)==workset.sideSetViews->end()) return;
    sideSet = workset.sideSetViews->at(basalSideName);
    worksetSize = sideSet.size;
  } else {
    worksetSize = workset.numCells;
  }

  TEUCHOS_TEST_FOR_EXCEPTION ((save_pressure_field && !std::is_constructible<EffPressureST,MeshScalarT>::value), std::logic_error,
                                "Error! BasalFrictionCoefficient: Trying to convert a FAD type (NVal) into a double (outN).\n");
  Kokkos::parallel_for(BasalFrictionCoefficient_Policy(0, worksetSize), *this);
}

} // Namespace LandIce
