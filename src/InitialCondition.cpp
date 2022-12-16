//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#include "InitialCondition.hpp"
#include "AnalyticFunction.hpp"
#include "Albany_Utils.hpp"
#include "Albany_ThyraUtils.hpp"

#include <Teuchos_CommHelpers.hpp>

#include <cmath>

namespace Albany {

static const double pi = 4.0 * std::atan(4.0);

Teuchos::ParameterList
getValidInitialConditionParameters(const Teuchos::ArrayRCP<std::string>& wsEBNames)
{
  Teuchos::ParameterList validPL ("ValidInitialConditionParams");
  validPL.set<std::string>("Function", "", "");
  Teuchos::Array<double> defaultData;
  validPL.set<Teuchos::Array<double> >("Function Data", defaultData, "");
  validPL.set<std::string >("Function Expression for DOF X", "None", "");
  validPL.set<std::string >("Function Expression for DOF Y", "None", "");
  validPL.set<std::string >("Function Expression for DOF Z", "None", "");
  Teuchos::Array<std::string> expr;
  validPL.set<Teuchos::Array<std::string>>("Function Expressions", expr);

  // Validate element block constant data

  for(int i = 0; i < wsEBNames.size(); i++)
    validPL.set<Teuchos::Array<double> >(wsEBNames[i], defaultData, "");

  // For EBConstant data, we can optionally randomly perturb the IC on each variable some amount

  validPL.set<Teuchos::Array<double> >("Perturb IC", defaultData, "");

  return validPL;
}

void InitialConditions (const Teuchos::RCP<Thyra_Vector>& soln,
                        const Teuchos::RCP<AbstractDiscretization>& disc,
                        Teuchos::ParameterList& icParams)
{
  auto soln_data = Albany::getNonconstLocalData(soln);
  const auto& dof_mgr = disc->getNewDOFManager();
  const auto& elem_lids = disc->getWsElementLIDs().host();
  const auto& ws_sizes = disc->getWorksetsSizes();
  const auto& elem_dof_lids = dof_mgr->elem_dof_lids().host();
  const auto& wsEBNames = disc->getWsEBNames();
  const auto& coords = disc->getCoords();
  const auto  numDim = disc->getNumDim();
  const auto  neq = dof_mgr->getNumFields();
  const auto  hasRestartSolution = disc->hasRestartSolution();

  constexpr auto ALL = Kokkos::ALL();

  // TODO: this won't be correct if/when we allow multiple solution fields with
  //       different discretization order/type
  const auto numNodes = dof_mgr->elem_dof_lids().host().extent_int(1) / neq;

  // Called three times, with x, xdot, and xdotdot. Different param lists are sent in.
  icParams.validateParameters(getValidInitialConditionParameters(wsEBNames), 0);

  // Default function is Constant, unless a Restart solution vector
  // was used, in which case the Init COnd defaults to Restart.
  std::string name;
  if (!hasRestartSolution) {
    name = icParams.get("Function","Constant");
  } else {
    name = icParams.get("Function","Restart");
  }

  if (name=="Restart") {
    return;
  }
  // Handle element block specific constant data
  if(name == "EBPerturb" || name == "EBPerturbGaussian" || name == "EBConstant"){

    bool perturb_values = false;

    Teuchos::Array<double> defaultData(neq);
    Teuchos::Array<double> perturb_mag;

    // Only perturb if the user has told us by how much to perturb
    if(name != "EBConstant" && icParams.isParameter("Perturb IC")){

      perturb_values = true;

      perturb_mag = icParams.get("Perturb IC", defaultData);

    }

/* The element block-based IC specification here is currently a hack. It assumes the initial value is constant
 * within each element across the element block (or optionally perturbed somewhat element by element). The
 * proper way to do this would be to project the element integration point values to the nodes using the basis
 * functions and a consistent mass matrix.
 *
 * The current implementation uses a single integration point per element - this integration point value for this
 * element within the element block is specified in the input file (and optionally perturbed). An approximation
 * of the load vector is obtained by accumulating the resulting (possibly perturbed) value into the nodes. Then,
 * a lumped version of the mass matrix is inverted and used to solve for the approximate nodal point initial
 * conditions.
 */

    // Use a Tpetra_Vector to hold the lumped mass matrix (has entries only on the diagonal). Zero-ed out.
    Teuchos::RCP<Thyra_Vector> lumpedMMT = Thyra::createMember(soln->space());
    lumpedMMT->assign(0.0);

    //get non-const view of lumpedMMT
    auto lumpedMMT_data = Albany::getNonconstLocalData(lumpedMMT);

    // Make sure soln is zeroed - we are accumulating into it
    for(int i = 0; i < soln_data.size(); ++i) {
      soln_data[i] = 0.0;
    }

    // Loop over all worksets, elements, all local nodes: compute soln as a function of coord and wsEBName
    std::vector<double> x(neq);

    Teuchos::RCP<AnalyticFunction> initFunc;

    for (int ws=0; ws<elem_lids.extent_int(0); ++ws) {

      Teuchos::Array<double> data = icParams.get(wsEBNames[ws], defaultData);
      // Call factory method from library of initial condition functions

      if(perturb_values){

        if(name == "EBPerturb") {
          initFunc = Teuchos::rcp(new ConstantFunctionPerturbed(neq, numDim, data, perturb_mag));

        } else { // name == EBGaussianPerturb

          initFunc = Teuchos::rcp(new
            ConstantFunctionGaussianPerturbed(neq, numDim, data, perturb_mag));
        }
      } else {
        initFunc = Teuchos::rcp(new ConstantFunction(neq, numDim, data));
      }

      std::vector<double> X(neq);

      for (int ielem=0; ielem<ws_sizes[ws]; ++ielem) {
        for (int eq=0; eq<neq; eq++)
          X[eq] = 0;

        for (int inode=0; inode<numNodes; ++inode)
          for (int eq=0; eq<neq; eq++)
            X[eq] += coords[ws][ielem][inode][eq]; // nodal coords

        for (int eq=0; eq<neq; eq++)
          X[eq] /= (double)neq;

        initFunc->compute(&x[0], &X[0]);

        const auto elem_LID = elem_lids(ws,ielem);
        const auto& dof_lids = Kokkos::subview(elem_dof_lids,elem_LID,ALL);
        for (int eq=0; eq<neq; eq++){
          const auto& offsets = dof_mgr->getGIDFieldOffsets(eq);
          for (int inode=0; inode<numNodes; ++inode) {
             soln_data[dof_lids[offsets[inode]]] += x[eq];
             lumpedMMT_data[dof_lids[offsets[inode]]] += 1.0;
          }
    }}}

//  Apply the inverted lumped mass matrix to get the final nodal projection

    for(int i = 0; i < soln_data.size(); ++i) {
      soln_data[i] /= lumpedMMT_data[i];
    }
  } else if(name == "Coordinates") {
    // Place the coordinate locations of the nodes into the solution vector for an initial guess
    for (int ws=0; ws<elem_lids.extent_int(0); ++ws) {
      for (int ielem=0; ielem<ws_sizes[ws]; ++ielem) {
        const auto elem_LID = elem_lids(ws,ielem);
        const auto& dof_lids = Kokkos::subview(elem_dof_lids,elem_LID,ALL);
        for (int eq=0; eq<neq; ++eq) {
          const auto& offsets = dof_mgr->getGIDFieldOffsets(eq);
          const int idim = eq % numDim;
          for (int inode=0; inode<numNodes; ++inode) {
            const double* X = coords[ws][ielem][inode];
            soln_data[dof_lids[offsets[inode]]] += X[idim];
        }}
      }
    }
  } else if(name == "Expression Parser") {

    std::string defaultExpression = "value = 0.0";

    std::string expressionX = icParams.get("Function Expression for DOF X", defaultExpression);
    std::string expressionY = icParams.get("Function Expression for DOF Y", defaultExpression);
    std::string expressionZ = icParams.get("Function Expression for DOF Z", defaultExpression);

    Teuchos::RCP<AnalyticFunction> initFunc = Teuchos::rcp(new ExpressionParser(neq, numDim, expressionX, expressionY, expressionZ));

    // Loop over all worksets, elements, all local nodes: compute soln as a function of coord
    std::vector<double> x(neq);
    for (int ws=0; ws<elem_lids.extent_int(0); ++ws) {
      for (int ielem=0; ielem<ws_sizes[ws]; ++ielem) {
        const auto elem_LID = elem_lids(ws,ielem);
        const auto& dof_lids = Kokkos::subview(elem_dof_lids,elem_LID,ALL);
        for (int inode=0; inode<numNodes; ++inode) {
          const double* X = coords[ws][ielem][inode];
          for (int eq=0; eq<neq; ++eq) {
            const auto& offsets = dof_mgr->getGIDFieldOffsets(eq);
            x[eq] = soln_data[dof_lids[offsets[inode]]];
          }
          initFunc->compute(&x[0],X);
          for (int eq=0; eq<neq; eq++) {
            const auto& offsets = dof_mgr->getGIDFieldOffsets(eq);
            soln_data[dof_lids[offsets[inode]]] = x[eq];
          }
        }
      }
    }
  }
#ifdef ALBANY_STK_EXPR_EVAL
  else if (name == "Expression Parser All DOFs") {
    Teuchos::Array<std::string> default_expr(neq);
    for (auto i = 0; i < default_expr.size(); ++i) { default_expr[i] = "0.0"; }
    Teuchos::Array<std::string> expr =
        icParams.get("Function Expressions", default_expr);

    Teuchos::RCP<AnalyticFunction> initFunc =
        Teuchos::rcp(new ExpressionParserAllDOFs(neq, numDim, expr));

    // Loop over all worksets, elements, all local nodes: compute soln as a
    // function of coord
    std::vector<double> x(neq);
    for (int ws=0; ws<elem_lids.extent_int(0); ++ws) {
      for (int ielem=0; ielem<ws_sizes[ws]; ++ielem) {
        const auto elem_LID = elem_lids(ws,ielem);
        const auto& dof_lids = Kokkos::subview(elem_dof_lids,elem_LID,ALL);
        for (int inode=0; inode<numNodes; ++inode) {
          double const* X = coords[ws][ielem][inode];
          for (int eq=0; eq<neq; ++eq) {
            const auto& offsets = dof_mgr->getGIDFieldOffsets(eq);
            x[eq] = soln_data[dof_lids[offsets[inode]]];
          }
          initFunc->compute(&x[0], X);
          for (int eq=0; eq<neq; ++eq) {
            const auto& offsets = dof_mgr->getGIDFieldOffsets(eq);
            soln_data[dof_lids[offsets[inode]]] = x[eq];
          }
        }
      }
    }
  }
#endif
  else {
    Teuchos::Array<double> defaultData(neq);
    Teuchos::Array<double> data = icParams.get("Function Data", defaultData);

    // Call factory method from library of initial condition functions
    Teuchos::RCP<AnalyticFunction> initFunc
      = createAnalyticFunction(name, neq, numDim, data);

    // Loop over all worksets, elements, all local nodes: compute soln as a function of coord
    std::vector<double> x(neq);
    for (int ws=0; ws<elem_lids.extent_int(0); ++ws) {
      for (int ielem=0; ielem<ws_sizes[ws]; ++ielem) {
        const auto elem_LID = elem_lids(ws,ielem);
        const auto& dof_lids = Kokkos::subview(elem_dof_lids,elem_LID,ALL);
        for (int inode=0; inode<numNodes; ++inode) {
          const double* X = coords[ws][ielem][inode];
          for (int eq=0; eq<neq; eq++) {
            const auto& offsets = dof_mgr->getGIDFieldOffsets(eq);
            x[eq] = soln_data[dof_lids[offsets[inode]]];
          }
          initFunc->compute(&x[0],X);
          for (int eq=0; eq<neq; eq++) {
            const auto& offsets = dof_mgr->getGIDFieldOffsets(eq);
            soln_data[dof_lids[offsets[inode]]] = x[eq];
          }
        }
      }
    }
  }
}

} // namespace Albany
