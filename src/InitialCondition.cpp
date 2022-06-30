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
getValidInitialConditionParameters(const Teuchos::ArrayRCP<std::string>& wsEBNames) {
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
                        const Teuchos::RCP<Albany::AbstractDiscretization>& disc,
                        Teuchos::ParameterList& icParams)
{
  auto soln_data = Albany::getNonconstDeviceData(soln);

  const auto& coords    = disc->getCoords();
  const auto& wsEBNames = disc->getWsEBNames();
  const int   numDim    = disc->getNumDim();
  const int   neq       = disc->getNumEq();

  // Called three times, with x, xdot, and xdotdot. Different param lists are sent in.
  icParams.validateParameters(getValidInitialConditionParameters(wsEBNames), 0);

  const auto& wsSizes = disc->getWorksetSizes();
  const auto& sol_dof_mgr = disc->getSolutionDOF().dof_mgr;

  // Default function is Constant, unless a Restart solution vector
  // was used, in which case the Init COnd defaults to Restart.
  std::string name;
  if (!disc->hasRestartSolution()) {
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
    auto lumpedMMT_data = Albany::getNonconstDeviceData(lumpedMMT);

    // Make sure soln is zeroed - we are accumulating into it
    Kokkos::deep_copy(soln_data,0.0);

    // Loop over all worksets, elements, all local nodes: compute soln as a function of coord and wsEBName
    DualView<double*> X("x",neq);
    DualView<double*> x("sol",neq);

    Teuchos::RCP<AnalyticFunction> initFunc;

    for (int ws=0; ws<disc->getNumWorksets(); ++ws) {

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

      for (int ielem=0; ielem<wsSizes[ws]; ++ielem) {
        for (int idim=0; idim<numDim; ++idim) {
            X.host()[idim] = 0;
        }

        const auto ALL = Kokkos::ALL();
        const auto& elem_coords = Kokkos::subview(coords.host(),ws,ielem,ALL,ALL);
        for (int inode=0; inode<elem_coords.size(); ++inode) {
          for (int idim=0; idim<numDim; ++idim) {
            X.host()[idim] += elem_coords(inode,idim);
        }}

        for (int idim=0; idim<numDim; ++idim) {
          X.host()[idim] /= elem_coords.size();
        }

        initFunc->compute(x.host().data(), X.host().data());

        x.sync_to_dev();
        auto x_dev = x.dev();
        const auto elem_lids = sol_dof_mgr->getElementLIDs(ielem);
        Kokkos::parallel_for(Kokkos::RangePolicy<>(0,elem_lids.size()),
                             KOKKOS_LAMBDA(const int idx) {
          const int icomp = idx % neq;
          const auto dof = elem_lids(idx);

          soln_data(dof) += x_dev[icomp];
          lumpedMMT_data(dof) += 1.0;
        });
    }}

    // Apply the inverted lumped mass matrix to get the final nodal projection
    Kokkos::parallel_for(Kokkos::RangePolicy<>(0,soln_data.size()),
                         KOKKOS_LAMBDA(const int i) {
      soln_data[i] /= lumpedMMT_data[i];
    });

    return;
  }

  if(name == "Coordinates") {
    // Place the coordinate locations of the nodes into the solution vector for an initial guess

    int numDOFsPerDim = neq / numDim;

    const auto ALL = Kokkos::ALL();
    for (int ws=0; ws<disc->getNumWorksets(); ++ws) {
      for (int ielem=0; ielem<wsSizes[ws]; ++ielem) {
        const auto elem_lids = sol_dof_mgr->getElementLIDs(ielem);

        const auto& elem_coords = Kokkos::subview(coords.dev(),ws,ielem,ALL,ALL);
        Kokkos::parallel_for(Kokkos::RangePolicy<>(0,elem_lids.size()),
                             KOKKOS_LAMBDA(const int idx) {
          const int inode = idx / neq;
          const int icomp = idx % neq;
          const int ieq   = icomp / numDim;
          const int idim  = icomp % numDim;
          const auto dof = elem_lids(idx);

          soln_data(dof) = elem_coords(inode,idim);
        });
      }
    }
  } else if(name == "Expression Parser") {

    std::string defaultExpression = "value = 0.0";

    std::string expressionX = icParams.get("Function Expression for DOF X", defaultExpression);
    std::string expressionY = icParams.get("Function Expression for DOF Y", defaultExpression);
    std::string expressionZ = icParams.get("Function Expression for DOF Z", defaultExpression);

    Teuchos::RCP<AnalyticFunction> initFunc = Teuchos::rcp(new ExpressionParser(neq, numDim, expressionX, expressionY, expressionZ));

    // Loop over all worksets, elements, all local nodes: compute soln as a function of coord
    auto soln_data = Albany::getNonconstLocalData(soln);
    std::vector<double> x(neq);
    const auto ALL = Kokkos::ALL();
    const int num_nodes = coords.host().extent_int(2);
    for (int ws=0; ws<disc->getNumWorksets(); ++ws) {
      for (int ielem=0; ielem<wsSizes[ws]; ++ielem) {
        const auto elem_lids = sol_dof_mgr->getElementLIDs(ielem);
        const auto elem_lids_h = Kokkos::create_mirror_view(elem_lids);
        Kokkos::deep_copy(elem_lids_h,elem_lids);
        for (int inode=0; inode<num_nodes; ++inode) {
          const auto X_node = Kokkos::subview(coords.host(),ws,ielem,inode,ALL);
          for (int i=0; i<neq; i++) {
            const int dof = elem_lids_h(inode*neq+i);
            x[i] = soln_data[dof];
          }
          initFunc->compute(&x[0],X_node.data());
          for (int i=0; i<neq; i++) {
            const int dof = elem_lids_h(inode*neq+i);
            soln_data[dof] = x[i];
          }
        }
      }
    }
  }
#ifdef ALBANY_STK_EXPR_EVAL
  else if (name == "Expression Parser All DOFs") {
    Teuchos::Array<std::string> default_expr(neq);
    for (auto i = 0; i < default_expr.size(); ++i) {
      default_expr[i] = "0.0";
    }

    auto expr = icParams.get("Function Expressions", default_expr);
    auto initFunc = Teuchos::rcp(new ExpressionParserAllDOFs(neq, numDim, expr));

    // Loop over all worksets, elements, all local nodes: compute soln as a
    // function of coord
    std::vector<double> x(neq);
    const int num_nodes = coords.host().extent_int(2);
    const auto ALL = Kokkos::ALL();
    for (int ws=0; ws<disc->getNumWorksets(); ++ws) {
      for (int ielem=0; ielem<wsSizes[ws]; ++ielem) {
        const auto elem_lids = sol_dof_mgr->getElementLIDs(ielem);
        const auto elem_lids_h = Kokkos::create_mirror_view(elem_lids);
        Kokkos::deep_copy(elem_lids_h,elem_lids);
        for (int inode=0; inode<num_nodes; ++inode) {
          const auto X_node = Kokkos::subview(coords.host(),ws,ielem,inode,ALL);
          for (int i = 0; i < neq; i++) {
            const int dof = elem_lids_h(inode*neq+i);
            x[i] = soln_data[dof];
          }
          initFunc->compute(&x[0], X_node.data());
          for (int i = 0; i < neq; i++) {
            const int dof = elem_lids_h(inode*neq+i);
            soln_data[dof] = x[i];
    }}}}
  }
#endif
  else {
    Teuchos::Array<double> defaultData(neq);
    Teuchos::Array<double> data = icParams.get("Function Data", defaultData);

    // Call factory method from library of initial condition functions
    auto initFunc = createAnalyticFunction(name, neq, numDim, data);

    // Loop over all worksets, elements, all local nodes: compute soln as a function of coord
    std::vector<double> x(neq);
    const auto ALL = Kokkos::ALL();
    const int num_nodes = coords.host().extent_int(2);
    for (int ws=0; ws<disc->getNumWorksets(); ++ws) {
      for (int ielem=0; ielem<wsSizes[ws]; ++ielem) {
        const auto elem_lids = sol_dof_mgr->getElementLIDs(ielem);
        const auto elem_lids_h = Kokkos::create_mirror_view(elem_lids);
        Kokkos::deep_copy(elem_lids_h,elem_lids);
        for (int inode=0; inode<num_nodes; ++inode) {
          const auto X_node = Kokkos::subview(coords.host(),ws,ielem,inode,ALL);
          for (int i=0; i<neq; i++) {
            const int dof = elem_lids_h(inode*neq+i);
            x[i] = soln_data[dof];
          }
          initFunc->compute(&x[0],X_node.data());
          for (int i=0; i<neq; i++) {
            const int dof = elem_lids_h(inode*neq+i);
            soln_data[dof] = x[i];
          }
        }
      }
    }
  }
}

} // namespace Albany
