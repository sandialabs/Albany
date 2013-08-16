//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#include <cmath>

#include <Teuchos_CommHelpers.hpp>

#include "AAdapt_InitialCondition.hpp"
#include "AAdapt_AnalyticFunction.hpp"
#include "Epetra_Comm.h"
#include "Albany_Utils.hpp"

namespace AAdapt {

const double pi = 3.141592653589793;


Teuchos::RCP<const Teuchos::ParameterList>
getValidInitialConditionParameters(const Teuchos::ArrayRCP<std::string>& wsEBNames) {
  Teuchos::RCP<Teuchos::ParameterList> validPL =
    rcp(new Teuchos::ParameterList("ValidInitialConditionParams"));;
  validPL->set<std::string>("Function", "", "");
  Teuchos::Array<double> defaultData;
  validPL->set<Teuchos::Array<double> >("Function Data", defaultData, "");

  // Validate element block constant data

  for(int i = 0; i < wsEBNames.size(); i++)

    validPL->set<Teuchos::Array<double> >(wsEBNames[i], defaultData, "");

  // For EBConstant data, we can optionally randomly perturb the IC on each variable some amount

  validPL->set<Teuchos::Array<double> >("Perturb IC", defaultData, "");

  return validPL;
}

void InitialConditions(const Teuchos::RCP<Epetra_Vector>& soln,
                       const Teuchos::ArrayRCP<Teuchos::ArrayRCP<Teuchos::ArrayRCP<Teuchos::ArrayRCP<int> > > >& wsElNodeEqID,
                       const Teuchos::ArrayRCP<std::string>& wsEBNames,
                       const Teuchos::ArrayRCP<Teuchos::ArrayRCP<Teuchos::ArrayRCP<double*> > > coords,
                       const int neq, const int numDim,
                       Teuchos::ParameterList& icParams, const bool hasRestartSolution) {
  // Called twice, with x and xdot. Different param lists are sent in.
  icParams.validateParameters(*AAdapt::getValidInitialConditionParameters(wsEBNames), 0);

  // Default function is Constant, unless a Restart solution vector
  // was used, in which case the Init COnd defaults to Restart.
  std::string name;

  if(!hasRestartSolution) name = icParams.get("Function", "Constant");

  else                     name = icParams.get("Function", "Restart");

  if(name == "Restart") return;

  // Handle element block specific constant data

  if(name == "EBPerturb" || name == "EBPerturbGaussian" || name == "EBConstant") {

    bool perturb_values = false;

    Teuchos::Array<double> defaultData(neq);
    Teuchos::Array<double> perturb_mag;

    // Only perturb if the user has told us by how much to perturb
    if(name != "EBConstant" && icParams.isParameter("Perturb IC")) {

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

    // Use an Epetra_Vector to hold the lumped mass matrix (has entries only on the diagonal). Zero-ed out.

    Epetra_Vector lumpedMM(soln->Map(), true);

    // Make sure soln is zeroed - we are accumulating into it

    for(int i = 0; i < soln->MyLength(); i++)

      (*soln)[i] = 0;

    // Loop over all worksets, elements, all local nodes: compute soln as a function of coord and wsEBName


    Teuchos::RCP<AAdapt::AnalyticFunction> initFunc;

    for(int ws = 0; ws < wsElNodeEqID.size(); ws++) { // loop over worksets

      Teuchos::Array<double> data = icParams.get(wsEBNames[ws], defaultData);
      // Call factory method from library of initial condition functions

      if(perturb_values) {

        if(name == "EBPerturb")

          initFunc = Teuchos::rcp(new AAdapt::ConstantFunctionPerturbed(neq, numDim, ws, data, perturb_mag));

        else // name == EBGaussianPerturb

          initFunc = Teuchos::rcp(new
                                  AAdapt::ConstantFunctionGaussianPerturbed(neq, numDim, ws, data, perturb_mag));
      }

      else

        initFunc = Teuchos::rcp(new AAdapt::ConstantFunction(neq, numDim, data));

      std::vector<double> X(neq);
      std::vector<double> x(neq);

      for(int el = 0; el < wsElNodeEqID[ws].size(); el++) { // loop over elements in workset

        for(int i = 0; i < neq; i++)
          X[i] = 0;

        for(int ln = 0; ln < wsElNodeEqID[ws][el].size(); ln++) // loop over node local to the element
          for(int i = 0; i < neq; i++)
            X[i] += coords[ws][el][ln][i]; // nodal coords

        for(int i = 0; i < neq; i++)
          X[i] /= (double)neq;

        initFunc->compute(&x[0], &X[0]);

        for(int ln = 0; ln < wsElNodeEqID[ws][el].size(); ln++) { // loop over node local to the element
          Teuchos::ArrayRCP<int> lid = wsElNodeEqID[ws][el][ln]; // local node ids

          for(int i = 0; i < neq; i++) {

            (*soln)[lid[i]] += x[i];
            //             (*soln)[lid[i]] += X[i]; // Test with coord values
            lumpedMM[lid[i]] += 1.0;

          }

        }
      }
    }

    //  Apply the inverted lumped mass matrix to get the final nodal projection

    for(int i = 0; i < soln->MyLength(); i++)

      (*soln)[i] /= lumpedMM[i];

    return;

  }

  if(name == "Coordinates") {

    // Place the coordinate locations of the nodes into the solution vector for an initial guess

    int numDOFsPerDim = neq / numDim;

    for(int ws = 0; ws < wsElNodeEqID.size(); ws++) {
      for(int el = 0; el < wsElNodeEqID[ws].size(); el++) {
        for(int ln = 0; ln < wsElNodeEqID[ws][el].size(); ln++) {

          const double* X = coords[ws][el][ln];
          Teuchos::ArrayRCP<int> lid = wsElNodeEqID[ws][el][ln];

          int cnt = 0;

          for(int i = 0; i < numDim; i++)
            for(int j = 0; j < numDOFsPerDim; j++){
              (*soln)[lid[cnt]] = X[i];
              cnt++;
            }

        }
      }
    }

  }

  else {

    Teuchos::Array<double> defaultData(neq);
    Teuchos::Array<double> data = icParams.get("Function Data", defaultData);

    // Call factory method from library of initial condition functions
    Teuchos::RCP<AAdapt::AnalyticFunction> initFunc
      = createAnalyticFunction(name, neq, numDim, data);

    // Loop over all worksets, elements, all local nodes: compute soln as a function of coord
    std::vector<double> x(neq);

    for(int ws = 0; ws < wsElNodeEqID.size(); ws++) {
      for(int el = 0; el < wsElNodeEqID[ws].size(); el++) {
        for(int ln = 0; ln < wsElNodeEqID[ws][el].size(); ln++) {
          const double* X = coords[ws][el][ln];
          Teuchos::ArrayRCP<int> lid = wsElNodeEqID[ws][el][ln];

          for(int i = 0; i < neq; i++) x[i] = (*soln)[lid[i]];

          initFunc->compute(&x[0], X);

          for(int i = 0; i < neq; i++)(*soln)[lid[i]] = x[i];
        }
      }
    }

  }

}

}
