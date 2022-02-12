//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#include "InitialCondition.hpp"

#include "AnalyticFunction.hpp"
#include "Albany_Utils.hpp"
#include "Albany_ThyraUtils.hpp"

// #include <Teuchos_CommHelpers.hpp>

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
                        const Albany::AbstractDiscretization& disc,
                        Teuchos::ParameterList& icParams,
                        const bool hasRestartSolution)
{
  auto soln_data = Albany::getNonconstLocalData(soln);

  const auto& wsEBNames    = disc.getWsEBNames();
  const auto& solDofMgr    = disc.getSolutionOverlapDOFManager();
  const auto& wsElNodeLID  = disc.getWsElNodeLID();
  const auto& coords       = disc.getCoords();
  const int neq            = solDofMgr.numComponents();
  const int numDim         = disc.getNumDim();
  const int numWorksets    = disc.getNumWorksets();

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

    // Loop over all worksets, elements, all local nodes.
    // Compute soln as a function of coord and wsEBName
    std::vector<double> x(neq);

    Teuchos::RCP<AnalyticFunction> initFunc;

    // loop over worksets
    for (int ws=0; ws < numWorksets; ++ws) {
      const auto& ElNodeID = wsElNodeLID[ws].host();

      Teuchos::Array<double> data = icParams.get(wsEBNames[ws], defaultData);

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

      // loop over elements in workset
      const int numElements = ElNodeID.extent(0);
      const int numNodes    = ElNodeID.extent(1);
      for (int el=0; el<numElements; ++el) {

        for (int i=0; i<neq; i++)
            X[i] = 0;
        // loop over node local to the element
        for (int ln=0; ln<numNodes; ++ln)
          for (int i=0; i<neq; i++)
            X[i] += coords[ws][el][ln][i]; // nodal coords

        for (int i=0; i<neq; i++)
          X[i] /= neq;

        initFunc->compute(&x[0], &X[0]);

        // loop over node local to the element
        for (unsigned ln=0; ln < ElNodeID.extent(1); ln++) {
          const LO nodeID = ElNodeID(el,ln);
          for (int eq=0; eq<neq; eq++){
            const LO dofID = solDofMgr.getLocalDOF(nodeID,eq);
            soln_data[dofID] += x[eq];
            lumpedMMT_data[dofID] += 1.0;
          }
        }
      }
    }

    //  Apply the inverted lumped mass matrix to get the final nodal projection
    for(int i = 0; i < soln_data.size(); ++i) {
      soln_data[i] /= lumpedMMT_data[i];
    }
  } else if(name == "Coordinates") {
    // Place the coordinate locations of the nodes into the solution vector for an initial guess
    int numDOFsPerDim = neq / numDim;

    for(int ws=0; ws<numWorksets; ++ws) {
      const auto& ElNodeID = wsElNodeLID[ws].host();

      const int numElements = ElNodeID.extent(0);
      const int numNodes    = ElNodeID.extent(1);
      for (int el=0; el<numElements; ++el) {
        for (int ln=0; ln<numNodes; ++ln) {
          const LO nodeID = ElNodeID(el,ln);

          const double* X = coords[ws][el][ln];
          for(int j = 0; j < numDOFsPerDim; j++) {
            for(int i = 0; i < numDim; i++) {
              const LO dofID = solDofMgr.getLocalDOF(nodeID,j*numDim+i);
              soln_data[dofID] = X[i];
            }
          }
        }
      }
    }
  } else if(name == "Expression Parser") {

    std::string defaultExpression = "value = 0.0";

    std::string expressionX = icParams.get("Function Expression for DOF X", defaultExpression);
    std::string expressionY = icParams.get("Function Expression for DOF Y", defaultExpression);
    std::string expressionZ = icParams.get("Function Expression for DOF Z", defaultExpression);

    Teuchos::RCP<AnalyticFunction> initFunc = Teuchos::rcp(new ExpressionParser(neq, numDim, expressionX, expressionY, expressionZ));

    // Loop over all worksets, elements, all local nodes: compute soln as a function of coord
    std::vector<double> x; x.resize(neq);
    for(int ws=0; ws<numWorksets; ++ws) {
      const auto& ElNodeID = wsElNodeLID[ws].host();

      const int numElements = ElNodeID.extent(0);
      const int numNodes    = ElNodeID.extent(1);
      for (int el=0; el<numElements; ++el) {
        for (int ln=0; ln<numNodes; ++ln) {
          const LO nodeID = ElNodeID(el,ln);
          const double* X = coords[ws][el][ln];
          for (int eq=0; eq<neq; eq++) {
            const LO dofID = solDofMgr.getLocalDOF(nodeID,eq);
            x[eq] = soln_data[dofID];
          }
          initFunc->compute(&x[0],X);
          for (int eq=0; eq<neq; eq++) {
            const LO dofID = solDofMgr.getLocalDOF(nodeID,eq);
            soln_data[dofID] = x[eq];
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
    std::vector<double> x;
    x.resize(neq);
    for(int ws=0; ws<numWorksets; ++ws) {
      const auto& ElNodeID = wsElNodeLID[ws].host();

      const int numElements = ElNodeID.extent(0);
      const int numNodes    = ElNodeID.extent(1);
      for (int el=0; el<numElements; ++el) {
        for (int ln=0; ln<numNodes; ++ln) {
          const LO nodeID = ElNodeID(el,ln);
          double const* X = coords[ws][el][ln];
          for (int eq=0; eq<neq; eq++) {
            const LO dofID = solDofMgr.getLocalDOF(nodeID,eq);
            x[eq] = soln_data[dofID];
          }
          initFunc->compute(&x[0], X);
          for (int eq=0; eq<neq; eq++) {
            const LO dofID = solDofMgr.getLocalDOF(nodeID,eq);
            soln_data[dofID] = x[eq];
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
    std::vector<double> x; x.resize(neq);
    for(int ws=0; ws<numWorksets; ++ws) {
      const auto& ElNodeID = wsElNodeLID[ws].host();

      const int numElements = ElNodeID.extent(0);
      const int numNodes    = ElNodeID.extent(1);
      for (int el=0; el<numElements; ++el) {
        for (int ln=0; ln<numNodes; ++ln) {
          const LO nodeID = ElNodeID(el,ln);
          const double* X = coords[ws][el][ln];
          for (int eq=0; eq<neq; eq++) {
            const LO dofID = solDofMgr.getLocalDOF(nodeID,eq);
            x[eq] = soln_data[dofID];
          }
          initFunc->compute(&x[0],X);
          for (int eq=0; eq<neq; eq++) {
            const LO dofID = solDofMgr.getLocalDOF(nodeID,eq);
            soln_data[dofID] = x[eq];
          }
        }
      }
    }
  }
}

} // namespace Albany
