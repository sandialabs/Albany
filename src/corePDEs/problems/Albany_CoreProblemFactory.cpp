//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#include "Albany_CoreProblemFactory.hpp"

#include "Albany_HeatProblem.hpp"
#include "Albany_AdvDiffProblem.hpp"
#include "Albany_PopulateMesh.hpp"
#include "Albany_SideLaplacianProblem.hpp"

namespace Albany
{

bool CoreProblemFactory::provides (const std::string& key) const 
{
  return key == "Heat 1D" ||
         key == "Heat 2D" ||
         key == "Heat 3D" ||
         key == "AdvDiff 1D" ||
         key == "AdvDiff 2D" ||
         key == "Populate Mesh" ||
         key == "Side Laplacian 3D";
}

CoreProblemFactory::obj_ptr_type
CoreProblemFactory::
create (const std::string& key,
        const Teuchos::RCP<const Teuchos_Comm>&     comm,
        const Teuchos::RCP<Teuchos::ParameterList>& topLevelParams,
        const Teuchos::RCP<ParamLib>&               paramLib) const
{
  obj_ptr_type problem;

  auto problemParams = Teuchos::sublist(topLevelParams, "Problem", true);
  auto discParams = Teuchos::sublist(topLevelParams, "Discretization");

  if (key == "Heat 1D") {
    problem = Teuchos::rcp(new HeatProblem(problemParams, paramLib, 1, comm));
  } else if (key == "Heat 2D") {
    problem = Teuchos::rcp(new HeatProblem(problemParams, paramLib, 2, comm));
  } else if (key == "Heat 3D") {
    problem = Teuchos::rcp(new HeatProblem(problemParams, paramLib, 3, comm));
  } else if (key == "AdvDiff 1D") {
    problem = Teuchos::rcp(new AdvDiffProblem(problemParams, paramLib, 1));
  } else if (key == "AdvDiff 2D") {
    problem = Teuchos::rcp(new AdvDiffProblem(problemParams, paramLib, 2));
  } else if (key == "Populate Mesh") {
    problem = Teuchos::rcp(new PopulateMesh(problemParams, discParams, paramLib));
  } else if (key == "Side Laplacian 3D") {
    problem = Teuchos::rcp(new SideLaplacian(problemParams, paramLib));
  } else {
    TEUCHOS_TEST_FOR_EXCEPTION (true, std::logic_error,
      "Error! Unrecognized key in BasicProblemFactory. Did you forget to check with 'provides(key)' first?\n");
  }

  return problem;
}

} // namespace Albany
