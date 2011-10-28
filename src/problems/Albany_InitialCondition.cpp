/********************************************************************\
*            Albany, Copyright (2010) Sandia Corporation             *
*                                                                    *
* Notice: This computer software was prepared by Sandia Corporation, *
* hereinafter the Contractor, under Contract DE-AC04-94AL85000 with  *
* the Department of Energy (DOE). All rights in the computer software*
* are reserved by DOE on behalf of the United States Government and  *
* the Contractor as provided in the Contract. You are authorized to  *
* use this computer software for Governmental purposes but it is not *
* to be released or distributed to the public. NEITHER THE GOVERNMENT*
* NOR THE CONTRACTOR MAKES ANY WARRANTY, EXPRESS OR IMPLIED, OR      *
* ASSUMES ANY LIABILITY FOR THE USE OF THIS SOFTWARE. This notice    *
* including this sentence must appear on any copies of this software.*
*    Questions to Andy Salinger, agsalin@sandia.gov                  *
\********************************************************************/


#include <cmath>

#include <Teuchos_CommHelpers.hpp>

#include "Albany_InitialCondition.hpp"
#include "Albany_AnalyticFunction.hpp"
#include "Epetra_Comm.h"
#include "Albany_Utils.hpp"

namespace Albany {

const double pi=3.141592653589793;


Teuchos::RCP<const Teuchos::ParameterList>
getValidInitialConditionParameters()
{
  Teuchos::RCP<Teuchos::ParameterList> validPL =
     rcp(new Teuchos::ParameterList("ValidInitialConditionParams"));;
  validPL->set<std::string>("Function", "", "");
  Teuchos::Array<double> defaultData;
  validPL->set<Teuchos::Array<double> >("Function Data", defaultData, "");

  return validPL;
}

void InitialConditions(const Teuchos::RCP<Epetra_Vector>& soln,
                       const Teuchos::ArrayRCP<Teuchos::ArrayRCP<Teuchos::ArrayRCP<Teuchos::ArrayRCP<int> > > >& wsElNodeEqID,
                       const Teuchos::ArrayRCP<Teuchos::ArrayRCP<Teuchos::ArrayRCP<double*> > > coords,
                       const int neq, const int numDim,
                       Teuchos::ParameterList& icParams)
{
  // Called twice, with x and xdot. Different param lists are sent in.
  icParams.validateParameters(*Albany::getValidInitialConditionParameters(),0);

  const std::string name = icParams.get("Function","Constant");
  if (name=="Restart") return;

  Teuchos::Array<double> defaultData;
  Teuchos::Array<double> data = icParams.get("Function Data",defaultData);

  // Call factory method from library of initial condition functions
  Teuchos::RCP<Albany::AnalyticFunction> initFunc
    = createAnalyticFunction(name, neq, numDim, data);

  cout << "*****\nI am in Initial conditions\n*****\n";
  cout << "Data: " << data << endl;
  cout << "neq : " << neq << endl;

  // Loop over all worksets, elements, all local nodes: compute soln as a function of coord
  std::vector<double> x; x.resize(neq);
  for (int ws=0; ws < wsElNodeEqID.size(); ws++) {
    for (int el=0; el < wsElNodeEqID[ws].size(); el++) {
      for (int ln=0; ln < wsElNodeEqID[ws][el].size(); ln++) {
        const double* X = coords[ws][el][ln];
        Teuchos::ArrayRCP<int> lid = wsElNodeEqID[ws][el][ln];
        for (int i=0; i<neq; i++) x[i] = (*soln)[lid[i]];
        initFunc->compute(&x[0],X);
        for (int i=0; i<neq; i++) (*soln)[lid[i]] = x[i];
  } } }

  std::cout << "solution: " << *soln << std::endl;
}

}
