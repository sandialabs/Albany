//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#include "GOAL_BCUtils.hpp"
#include "Albany_Application.hpp"
#include "Albany_GOALDiscretization.hpp"

using Teuchos::RCP;
using Teuchos::ParameterList;

namespace GOAL {

class BCManager
{
  public:
    BCManager(
        Albany::Application const& application,
        RCP<Tpetra_Vector> const& residual,
        RCP<Tpetra_CrsMatrix> const& jacobian);
    void run();
  private:
    Albany::Application const& app;
    RCP<Tpetra_Vector> const& res;
    RCP<Tpetra_CrsMatrix> const& jac;
    ParameterList bcParams;
};

BCManager::BCManager(
    Albany::Application const& application,
    RCP<Tpetra_Vector> const& residual,
    RCP<Tpetra_CrsMatrix> const& jacobian) :
  app(application),
  res(residual),
  jac(jacobian)
{
  // get the parameterlist of bcs
  RCP<ParameterList> pl = app.getProblemPL();
  bcParams = pl->sublist("Hierarchic Boundary Conditions");
}

void BCManager::run()
{
}

void computeHierarchicBCs(
    Albany::Application const& app,
    RCP<Tpetra_Vector> const& res,
    RCP<Tpetra_CrsMatrix> const& jac)
{
  RCP<ParameterList> pl = app.getProblemPL();
  std::string name = pl->get<std::string>("Name","");
  if (name.find("GOAL") != 0)
    return;
  BCManager bcm(app, res, jac);
  bcm.run();
}

}
