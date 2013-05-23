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
    name_map->insert( std::make_pair("Fp","Fp") );
    name_map->insert( std::make_pair("logFp","logFp") );
    name_map->insert( std::make_pair("eqps","eqps") );
    name_map->insert( std::make_pair("Total_Stress","Total_Stress") );
    name_map->insert( std::make_pair("KCPermeability","KCPermeability") );
    name_map->insert( std::make_pair("Biot_Modulus","Biot_Modulus") );
    name_map->insert( std::make_pair("Biot_Coefficient","Biot_Coefficient") );
    name_map->insert( std::make_pair("Porosity","Porosity") );
    name_map->insert( std::make_pair("Pore_Pressure","Pore_Pressure") );
    name_map->insert( std::make_pair("Matrix_Energy","Matrix_Energy") );
    name_map->insert( std::make_pair("F1_Energy","F1_Energy") );
    name_map->insert( std::make_pair("F2_Energy","F2_Energy") );
    name_map->insert( std::make_pair("Matrix_Damage","Matrix_Damage") );
    name_map->insert( std::make_pair("F1_Damage","F1_Damage") );
    name_map->insert( std::make_pair("F2_Damage","F2_Damage") );
    name_map->insert( std::make_pair("Void_Volume","Void_Volume") );
    name_map->insert( std::make_pair("isoHardening","isoHardening") );
    name_map->insert( std::make_pair("ess","ess") );
    name_map->insert( std::make_pair("Energy","Energy") );
    name_map->insert( std::make_pair("Material Tangent","Material Tangent") );

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
