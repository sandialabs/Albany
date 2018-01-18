//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#include "FieldNameMap.hpp"
#include "Albany_Utils.hpp"

namespace LCM {

  //----------------------------------------------------------------------------
  FieldNameMap::FieldNameMap(bool surface_flag)
  {
    Teuchos::RCP<std::map<std::string, std::string>> name_map =
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
    name_map->insert( std::make_pair("void_volume_fraction","void_volume_fraction") );
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
    name_map->insert( std::make_pair("Velocity_Gradient_Plastic","Velocity_Gradient_Plastic") );

    //
    // Crystal plasticity model
    //
    int const
    max_slip_systems = 48;

    // field names for slip on each slip system
    for (int sys{0}; sys < max_slip_systems; ++sys) {

      std::string const
      g = Albany::strint("gamma", sys + 1, '_');

      name_map->insert(std::make_pair(g, g));
    }

    // field names for slip rate on each slip system
    for (int sys{0}; sys < max_slip_systems; ++sys) {

      std::string const
      g_dot = Albany::strint("gamma_dot", sys + 1,'_');

      name_map->insert(std::make_pair(g_dot, g_dot));
    }

    // field names for the hardening on each slip system
    for (int sys{0}; sys < max_slip_systems; ++sys) {

      std::string const
      t_h = Albany::strint("tau_hard", sys + 1, '_');

      name_map->insert(std::make_pair(t_h, t_h));
    }

    // field names for shear stress on each slip system
    for (int sys{0}; sys < max_slip_systems; ++sys) {

      std::string const
      tau = Albany::strint("tau", sys + 1, '_');

      name_map->insert(std::make_pair(tau, tau));
    }

    name_map->insert(std::make_pair("CP_Residual","CP_Residual"));
    name_map->insert(std::make_pair("CP_Residual_Iter","CP_Residual_Iter"));
    // field name for crystallographic rotation tensor
    name_map->insert(std::make_pair("Re","Re"));

    //
    // ViscoElastic model
    //
    name_map->insert(std::make_pair("H_1","H_1"));
    name_map->insert(std::make_pair("H_2","H_2"));
    name_map->insert(std::make_pair("H_3","H_3"));
    name_map->insert(std::make_pair("Instantaneous Stress","Instantaneous Stress"));

    //
    // Poroplasticity model
    //
    name_map->insert( std::make_pair("Total_Stress","Total_Stress") );
    name_map->insert( std::make_pair("KCPermeability","KCPermeability") );
    name_map->insert( std::make_pair("Biot_Modulus","Biot_Modulus") );
    name_map->insert( std::make_pair("Biot_Coefficient","Biot_Coefficient") );
    name_map->insert( std::make_pair("Porosity","Porosity") );
    name_map->insert( std::make_pair("Pore_Pressure","Pore_Pressure") );

    //
    // Hydrogen transport model
    //
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

    //
    // Helium ODEs
    //
    name_map->insert( std::make_pair("He_Concentration","He_Concentration") );
    name_map->insert( std::make_pair("Total_Bubble_Density","Total_Bubble_Density") );
    name_map->insert( std::make_pair("Bubble_Volume_Fraction","Bubble_Volume_Fraction") );

    //
    // Geo-models
    //
    name_map->insert( std::make_pair("Back_Stress","Back_Stress") );
    name_map->insert( std::make_pair("Cap_Parameter","Cap_Parameter") );
    name_map->insert( std::make_pair("volPlastic_Strain","volPlastic_Strain") );
    name_map->insert( std::make_pair("Strain","Strain") );
    name_map->insert( std::make_pair("Friction_Parameter","Friction_Parameter") );

    if (surface_flag) {

      std::map<std::string, std::string>::iterator
      it;

      for (it = name_map->begin(); it != name_map->end(); ++it) {
        it->second = "surf_" + it->second;
      }
    }

    field_name_map_ = name_map;
  }

  //----------------------------------------------------------------------------
  FieldNameMap::~FieldNameMap()
  {
  }
}
