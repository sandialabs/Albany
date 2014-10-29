//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#include "FieldNameMap.hpp"

namespace LCM {

  //----------------------------------------------------------------------------
  FieldNameMap::FieldNameMap(bool surface_flag)
  {
    Teuchos::RCP<std::map<std::string, std::string> > name_map =
      Teuchos::rcp( new std::map<std::string, std::string> );

    name_map->insert( std::make_pair("Cauchy_Stress","Cauchy_Stress") );
    name_map->insert( std::make_pair("FirstPK","FirstPK") );
    name_map->insert( std::make_pair("Fp","Fp") );
    name_map->insert( std::make_pair("logFp","logFp") );
    name_map->insert( std::make_pair("eqps","eqps") );
    name_map->insert( std::make_pair("Yield_Surface","Yield_Surface") );
    name_map->insert( std::make_pair("Matrix_Energy","Matrix_Energy") );
    name_map->insert( std::make_pair("F1_Energy","F1_Energy") );
    name_map->insert( std::make_pair("F2_Energy","F2_Energy") );
    name_map->insert( std::make_pair("Matrix_Damage","Matrix_Damage") );
    name_map->insert( std::make_pair("F1_Damage","F1_Damage") );
    name_map->insert( std::make_pair("F2_Damage","F2_Damage") );
    name_map->insert( std::make_pair("Void_Volume","Void_Volume") );
    name_map->insert( std::make_pair("isotropic_hardening","isotropic_hardening") );
    name_map->insert( std::make_pair("eps_ss","eps_ss") );
    name_map->insert( std::make_pair("Energy","Energy") );
    name_map->insert( std::make_pair("Material Tangent","Material Tangent") );
    name_map->insert( std::make_pair("Temperature","Temperature") );
    name_map->insert( std::make_pair("Pressure","Pressure") );
    name_map->insert( std::make_pair("Mechanical_Source","Mechanical_Source") );
    name_map->insert( std::make_pair("F","F") );
    name_map->insert( std::make_pair("J","J") );
    name_map->insert( std::make_pair("Velocity_Gradient","Velocity_Gradient") );
    // Poroplasticity model
    name_map->insert( std::make_pair("Total_Stress","Total_Stress") );
    name_map->insert( std::make_pair("KCPermeability","KCPermeability") );
    name_map->insert( std::make_pair("Biot_Modulus","Biot_Modulus") );
    name_map->insert( std::make_pair("Biot_Coefficient","Biot_Coefficient") );
    name_map->insert( std::make_pair("Porosity","Porosity") );
    name_map->insert( std::make_pair("Pore_Pressure","Pore_Pressure") );
    // hydrogen transport model
    name_map->insert( std::make_pair("Transport","Transport") );
    name_map->insert( std::make_pair("HydroStress","HydroStress") );
    name_map->insert( std::make_pair("Diffusion_Coefficient","Diffusion_Coefficient") );
    name_map->insert( std::make_pair("Tau_Contribution","Tau_Contribution") );
    name_map->insert( std::make_pair("Trapped_Concentration","Trapped_Concentration") );
    name_map->insert( std::make_pair("Total_Concentration","Total_Concentration") );
    name_map->insert( std::make_pair("Effective_Diffusivity","Effective_Diffusivity") );
    name_map->insert( std::make_pair("Trapped_Solvent","Trapped_Solvent") );
    name_map->insert( std::make_pair("Strain_Rate_Factor","Strain_Rate_Factor") );
    name_map->insert( std::make_pair("Concentration_Equilibrium_Parameter",
    	"Concentration_Equilibrium_Parameter") );
    name_map->insert( std::make_pair("Gradient_Element_Length",
    	"Gradient_Element_Length") );
    // helium ODEs
    name_map->insert( std::make_pair("He_Concentration","He_Concentration") );
    name_map->insert( std::make_pair("Total_Bubble_Density","Total_Bubble_Density") );
    name_map->insert( std::make_pair("Bubble_Volume_Fraction","Bubble_Volume_Fraction") );
    // geo-models
    name_map->insert( std::make_pair("Back_Stress","Back_Stress") );
    name_map->insert( std::make_pair("Cap_Parameter","Cap_Parameter") );
    name_map->insert( std::make_pair("volPlastic_Strain","volPlastic_Strain") );
    name_map->insert( std::make_pair("Strain","Strain") );
    name_map->insert( std::make_pair("Friction_Parameter","Friction_Parameter") );

    if ( surface_flag ) {
      std::map<std::string, std::string>::iterator it;
      for (it = name_map->begin(); it != name_map->end(); ++it) {
        it->second = "surf_"+it->second;
      }
    }
    field_name_map_ = name_map;
  }

  //----------------------------------------------------------------------------
  FieldNameMap::~FieldNameMap()
  {
  }
}
