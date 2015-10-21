//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#include "GOAL_BCUtils.hpp"
#include "Albany_Application.hpp"
#include "Albany_GOALDiscretization.hpp"
#include "GOAL_MechanicsProblem.hpp"

using Teuchos::RCP;
using Teuchos::ArrayRCP;
using Teuchos::ParameterList;

namespace GOAL {

class BCManager
{
  public:
    BCManager(
        Albany::Application const& application,
        RCP<const Tpetra_Vector> const& solution,
        RCP<Tpetra_Vector> const& residual,
        RCP<Tpetra_CrsMatrix> const& jacobian);
    void run();
  private:
    void applyBC(Teuchos::ParameterList const& p);
    void modifyLinearSystem(double v, int offset, std::string set);
    Albany::Application const& app;
    RCP<const Tpetra_Vector> const& sol;
    RCP<Tpetra_Vector> const& res;
    RCP<Tpetra_CrsMatrix> const& jac;
    RCP<Albany::GOALDiscretization> disc;
    RCP<Albany::GOALMeshStruct> meshStruct;
    RCP<Albany::GOALMechanicsProblem> problem;
    Albany::GOALNodeSets ns;
    ParameterList bcParams;
};

BCManager::BCManager(
    Albany::Application const& application,
    RCP<const Tpetra_Vector> const& solution,
    RCP<Tpetra_Vector> const& residual,
    RCP<Tpetra_CrsMatrix> const& jacobian) :
  app(application),
  sol(solution),
  res(residual),
  jac(jacobian)
{
  RCP<const ParameterList> pl = app.getProblemPL();
  bcParams = pl->sublist("Hierarchic Boundary Conditions");
  RCP<Albany::AbstractDiscretization> ad = app.getDiscretization();
  disc = Teuchos::rcp_dynamic_cast<Albany::GOALDiscretization>(ad);
  meshStruct = disc->getGOALMeshStruct();
  RCP<Albany::AbstractProblem> ap = app.getProblem();
  problem = Teuchos::rcp_dynamic_cast<Albany::GOALMechanicsProblem>(ap);
  ns = disc->getGOALNodeSets();
}

void BCManager::run()
{
  typedef ParameterList::ConstIterator ParamIter;
  for (ParamIter i = bcParams.begin(); i != bcParams.end(); ++i)
  {
    std::string const& name = bcParams.name(i);
    Teuchos::ParameterEntry const& entry = bcParams.entry(i);
    assert(entry.isList());
    applyBC(Teuchos::getValue<ParameterList>(entry));
  }
}

static RCP<ParameterList> getValidBCParameters()
{
  RCP<ParameterList> p = rcp(new ParameterList("Valid Hierarchic BC Params"));
  p->set<std::string>("DOF", "", "Degree of freedom to which BC is applied");
  p->set<double>("Value", 0.0, "Value of the BC as function of t");
  p->set<std::string>("Node Set", "", "Node Set to apply the BC to");
  return p;
}

void BCManager::applyBC(ParameterList const& p)
{
  // validate parameters
  RCP<ParameterList> vp = getValidBCParameters();
  p.validateParameters(*vp,0);

  // get the input parameters
  double v = p.get<double>("Value");
  std::string set = p.get<std::string>("Node Set");
  std::string dof = p.get<std::string>("DOF");

  // does this node set actually exist?
  assert(ns.count(set) == 1);

  // does this dof actually exist?
  int offset = problem->getOffset(dof);

  modifyLinearSystem(v, offset, set);
}

void BCManager::modifyLinearSystem(double v, int offset, std::string set)
{
  // should we fill in BC info?
  bool fillRes = (res != Teuchos::null);
  bool fillJac = (jac != Teuchos::null);
  if ((!fillRes) && (!fillJac)) return;

  // get views of the solution and residual vectors
  ArrayRCP<const ST> x_const_view;
  ArrayRCP<ST> f_nonconst_view;
  if (fillRes) {
    x_const_view = sol->get1dView();
    f_nonconst_view = res->get1dViewNonConst();
  }

  // set up some data for replacing jacobian values
  Teuchos::Array<LO> index(1);
  Teuchos::Array<ST> value(1);
  value[0] = 1.0;
  size_t numEntries;
  Teuchos::Array<ST> matrixEntries;
  Teuchos::Array<LO> matrixIndices;

  // loop over all of the nodes in this node set
  std::vector<Albany::GOALNode> nodes = ns[set];
  for (int i=0; i < nodes.size(); ++i)
  {
    Albany::GOALNode node = nodes[i];
    int lunk = disc->getDOF(node.lid, offset);

    // if the node is higher order, we set the value of the DBC to be 0
    // note: this assumes that bcs are either constant or linear in space
    // anything else would require a linear solve to find coefficients v
    if (node.higherOrder)
      v = 0.0;

    // modify the residual if necessary
    if (fillRes)
      f_nonconst_view[lunk] = x_const_view[lunk] - v;

    // modify the jacobian if necessary
    if (fillJac)
    {
      index[0] = lunk;
      numEntries = jac->getNumEntriesInLocalRow(lunk);
      matrixEntries.resize(numEntries);
      matrixIndices.resize(numEntries);
      jac->getLocalRowCopy(lunk, matrixIndices(), matrixEntries(), numEntries);
      for (int i=0; i < numEntries; ++i)
        matrixEntries[i] = 0.0;
      jac->replaceLocalValues(lunk, matrixIndices(), matrixEntries());
      jac->replaceLocalValues(lunk, index(), value());
    }
  }
}

void computeHierarchicBCs(
    Albany::Application const& app,
    RCP<const Tpetra_Vector> const& sol,
    RCP<Tpetra_Vector> const& res,
    RCP<Tpetra_CrsMatrix> const& jac)
{
  RCP<const ParameterList> pl = app.getProblemPL();
  std::string name = pl->get<std::string>("Name");
  if (name.find("GOAL") != 0)
    return;
  BCManager bcm(app, sol, res, jac);
  bcm.run();
}

}
