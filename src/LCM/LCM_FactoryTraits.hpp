/********************************************************************\
*            Albany, Copyright (2010) Sandia Corporation             *
*                                                                    *
* Notice: This computer software was prepared by Sandia Corporation, *
* hereinafter the Contractor, under Contract DE-AC04-94AL85000 with  *
* the Department of Energy (DOE). All rights in the computer software*
* are reserved by DOE on behalf of the United States Government and  *
* the Contractor as provided in the Contract. You are authorized to  *
* use this computer software for Governmental purposes but it is not *
* to be released or distributed to the public. NEITHER THE GOVERNMENT*
* NOR THE CONTRACTOR MAKES ANY WARRANTY, EXPRESS OR IMPLIED, OR      *
* ASSUMES ANY LIABILITY FOR THE USE OF THIS SOFTWARE. This notice    *
* including this sentence must appear on any copies of this software.*
*    Questions to Andy Salinger, agsalin@sandia.gov                  *
\********************************************************************/


#ifndef LCM_FACTORY_TRAITS_HPP
#define LCM_FACTORY_TRAITS_HPP

// User Defined Evaluator Types
#include "PHAL_Constant.hpp"
#include "PHAL_GatherSolution.hpp"
#include "PHAL_ScatterResidual.hpp"
#include "PHAL_Source.hpp"
#include "PHAL_ThermalConductivity.hpp"
#include "PHAL_Absorption.hpp"
#include "PHAL_ComputeBasisFunctions.hpp"
#include "PHAL_DOFInterpolation.hpp"
#include "PHAL_DOFGradInterpolation.hpp"
#include "PHAL_DOFVecInterpolation.hpp"
#include "PHAL_DOFVecGradInterpolation.hpp"
#include "PHAL_MapToPhysicalFrame.hpp"
#include "PHAL_HelmholtzResid.hpp"
#include "PHAL_HeatEqResid.hpp"
#include "PHAL_GatherCoordinateVector.hpp"
#include "PHAL_JouleHeating.hpp"
#include "PHAL_SaveStateField.hpp"
#include "QCAD_ResponseFieldIntegral.hpp"
#include "QCAD_ResponseFieldValue.hpp"
#include "QCAD_ResponseSaveField.hpp"

#include "LCM/evaluators/Stress.hpp"
#ifdef ALBANY_LAME
#include "LCM/evaluators/LameStress.hpp"
#endif
#include "LCM/evaluators/Strain.hpp"
#include "LCM/evaluators/ElasticModulus.hpp"
#include "LCM/evaluators/ElasticityResid.hpp"
#include "LCM/evaluators/PoissonsRatio.hpp"
#include "LCM/evaluators/DefGrad.hpp"
#include "LCM/evaluators/RCG.hpp"
#include "LCM/evaluators/LCG.hpp"
#include "LCM/evaluators/Neohookean.hpp"
#include "LCM/evaluators/J2Stress.hpp"
#include "LCM/evaluators/TLElasResid.hpp"
#include "LCM/evaluators/EnergyPotential.hpp"
#include "LCM/evaluators/HardeningModulus.hpp"
#include "LCM/evaluators/YieldStrength.hpp"
#include "LCM/evaluators/PisdWdF.hpp"
#include "LCM/evaluators/DamageResid.hpp"
#include "LCM/evaluators/J2Damage.hpp"
#include "LCM/evaluators/DamageLS.hpp"
#include "LCM/evaluators/SaturationModulus.hpp"
#include "LCM/evaluators/SaturationExponent.hpp"
#include "LCM/evaluators/Localization.hpp"
#include "LCM/evaluators/DamageSource.hpp"
#include "LCM/evaluators/ShearModulus.hpp"
#include "LCM/evaluators/BulkModulus.hpp"
#include "LCM/evaluators/DislocationDensity.hpp"
#include "LCM/evaluators/TotalStress.hpp"
#include "LCM/evaluators/PoroElasticityResidMomentum.hpp"
#include "LCM/evaluators/PoroElasticityResidMass.hpp"
#include "LCM/evaluators/Porosity.hpp"
#include "LCM/evaluators/BiotCoefficient.hpp"
#include "LCM/evaluators/BiotModulus.hpp"


#include "boost/mpl/vector/vector50.hpp"
#include "boost/mpl/placeholders.hpp"
// \cond  Have doxygern ignore this namespace   
using namespace boost::mpl::placeholders;
// \endcond

//! Code Base for LCM Project
namespace LCM {

/*! \brief Struct to define Evaluator objects for the EvaluatorFactory.
    
    Preconditions:
    - You must provide a boost::mpl::vector named EvaluatorTypes that contain all 
    Evaluator objects that you wish the factory to build.  Do not confuse evaluator types 
    (concrete instances of evaluator objects) with evaluation types (types of evaluations 
    to perform, i.e., Residual, Jacobian). 

*/

template<typename Traits>
struct FactoryTraits {
  
  static const int id_gather_solution           =  0;
  static const int id_gather_coordinate_vector  =  1;
  static const int id_scatter_residual          =  2;
  static const int id_compute_basis_functions   =  3;
  static const int id_dof_interpolation         =  4;
  static const int id_dof_grad_interpolation    =  5;
  static const int id_dofvec_interpolation      =  6;
  static const int id_dofvec_grad_interpolation =  7;
  static const int id_map_to_physical_frame     =  8;
  static const int id_qcad_response_fieldintegral = 9;
  static const int id_qcad_response_fieldvalue    = 10;
  static const int id_qcad_response_savefield     = 11;
  static const int id_source                    = 12;
  static const int id_thermal_conductivity      = 13;
  static const int id_helmholtzresid            = 14;
  static const int id_heateqresid               = 15;
  static const int id_constant                  = 16;
  static const int id_jouleheating              = 17;
  static const int id_elastic_modulus           = 18;
  static const int id_stress                    = 19;
  static const int id_strain                    = 20;
  static const int id_elasticityresid           = 21;
  static const int id_poissons_ratio            = 22;
  static const int id_defgrad                   = 23;
  static const int id_rcg                       = 24;
  static const int id_lcg                       = 25;
  static const int id_neohookean_stress         = 26;
  static const int id_tl_elas_resid             = 27;
  static const int id_j2_stress                 = 28;
  static const int id_energy_potential          = 29;
  static const int id_hardening_modulus         = 30;
  static const int id_yield_strength            = 31;
  static const int id_pisdwdf_stress            = 32;
  static const int id_damage_resid              = 33;
  static const int id_j2_damage                 = 34;
  static const int id_damage_ls                 = 35;
  static const int id_sat_mod                   = 36;
  static const int id_sat_exp                   = 37;
  static const int id_localization              = 38;
  static const int id_damage_source             = 39;
  static const int id_bulk_modulus              = 40;
  static const int id_shear_modulus             = 41;
  static const int id_savestatefield            = 42;
  static const int id_dislocation_density       = 43;
  static const int id_total_stress              = 44;
  static const int id_poroelasticityresidmomentum=45;
  static const int id_porosty                   = 46;
  static const int id_biotcoefficient           = 47;
  static const int id_biotmodulus               = 48;
  static const int id_poroelasticityresidmass   = 49;
  // JTO - leave lame stress at the bottom for the convention below to be most effective
  static const int id_lame_stress               = 50;


#ifndef ALBANY_LAME
  typedef boost::mpl::vector50<
#else
  typedef boost::mpl::vector51<
#endif  

    PHAL::GatherSolution<_,Traits>,           //  0
    PHAL::GatherCoordinateVector<_,Traits>,   //  1
    PHAL::ScatterResidual<_,Traits>,          //  2
    PHAL::ComputeBasisFunctions<_,Traits>,    //  3
    PHAL::DOFInterpolation<_,Traits>,         //  4
    PHAL::DOFGradInterpolation<_,Traits>,     //  5
    PHAL::DOFVecInterpolation<_,Traits>,      //  6
    PHAL::DOFVecGradInterpolation<_,Traits>,  //  7
    PHAL::MapToPhysicalFrame<_,Traits>,       //  8
    QCAD::ResponseFieldIntegral<_,Traits>,    //  9
    QCAD::ResponseFieldValue<_,Traits>,       // 10
    QCAD::ResponseSaveField<_,Traits>,        // 11
    PHAL::Source<_,Traits>,                   // 12
    PHAL::ThermalConductivity<_,Traits>,      // 13
    PHAL::HelmholtzResid<_,Traits>,           // 14
    PHAL::HeatEqResid<_,Traits>,              // 15
    PHAL::Constant<_,Traits>,                 // 16
    PHAL::JouleHeating<_,Traits>,             // 17
    LCM::ElasticModulus<_,Traits>,            // 18
    LCM::Stress<_,Traits>,                    // 19
    LCM::Strain<_,Traits>,                    // 20
    LCM::ElasticityResid<_,Traits>,           // 21
    LCM::PoissonsRatio<_,Traits>,             // 22
    LCM::DefGrad<_,Traits>,                   // 23
    LCM::RCG<_,Traits>,                       // 24
    LCM::LCG<_,Traits>,                       // 25
    LCM::Neohookean<_,Traits>,                // 26
    LCM::TLElasResid<_,Traits>,               // 27
    LCM::J2Stress<_,Traits>,                  // 28
    LCM::EnergyPotential<_,Traits>,           // 29
    LCM::HardeningModulus<_,Traits>,          // 30
    LCM::YieldStrength<_,Traits>,             // 31
    LCM::PisdWdF<_,Traits>,                   // 32
    LCM::DamageResid<_,Traits>,               // 33
    LCM::J2Damage<_,Traits>,                  // 34
    LCM::DamageLS<_,Traits>,                  // 35
    LCM::SaturationModulus<_,Traits>,         // 36
    LCM::SaturationExponent<_,Traits>,        // 37
    LCM::Localization<_,Traits>,              // 38
    LCM::DamageSource<_,Traits>,              // 39
    LCM::BulkModulus<_,Traits>,               // 40
    LCM::ShearModulus<_,Traits>,              // 41
    PHAL::SaveStateField<_,Traits>,           // 42
    LCM::DislocationDensity<_,Traits>,        // 43
    LCM::TotalStress<_,Traits>,               // 44
    LCM::PoroElasticityResidMomentum<_,Traits>,// 45
    LCM::Porosity<_, Traits>,                   // 46
    LCM::BiotCoefficient<_,Traits>,            // 47
    LCM::BiotModulus<_,Traits>,                // 48
    LCM::PoroElasticityResidMass<_,Traits>   // 49
#ifdef ALBANY_LAME
    ,LCM::LameStress<_,Traits>                // 50
#endif
    > EvaluatorTypes;
};
}

#endif

