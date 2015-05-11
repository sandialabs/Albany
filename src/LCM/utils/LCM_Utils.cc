//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//
#include "LCM_Utils.h"

namespace LCM {

RCP<MaterialDatabase>
createMaterialDatabase(
    RCP<ParameterList> const & params,
    RCP<Teuchos_Comm const> & commT)
{
  bool
  is_valid_material_db = params->isType<std::string>("MaterialDB Filename");

  TEUCHOS_TEST_FOR_EXCEPTION(
      is_valid_material_db == false,
      std::logic_error,
      "A required material database cannot be found.");

  std::string
  filename = params->get<std::string>("MaterialDB Filename");

  return Teuchos::rcp(new MaterialDatabase(filename, commT));
}

} // namespace LCM
