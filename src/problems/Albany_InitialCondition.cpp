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

Teuchos::RCP<const Teuchos::ParameterList>
getValidInitialConditionParameters()
{
  Teuchos::RCP<Teuchos::ParameterList> validPL =
     rcp(new Teuchos::ParameterList("ValidInitialConditionParams"));;
  validPL->set<double>("Nonlinear Factor", 0.0, "");
  validPL->set<double>("Beta", 0.0, "");
  validPL->set<std::string>("Name", "", "");

  return validPL;
}

void InitialCondition(const Teuchos::RCP<Epetra_Vector>& u,
                      const unsigned int local_len_x,
                      const unsigned int local_len_y,
                      Teuchos::ParameterList& p)
{
  p.validateParameters(*Albany::getValidInitialConditionParameters(),0);

  int  global_len_x = 0 ;
  int  global_len_y = 0 ;
  int lx = local_len_x;
  int ly = local_len_y;

  u->Comm().SumAll(&lx, &global_len_x, 1);
  u->Comm().SumAll(&ly, &global_len_y, 1);

  const std::string name = p.get<std::string>("Name");

  // All other sources are names of Analytic functions
  const double factor    = p.get("Nonlinear Factor", 0.0);
  const double beta      = p.get("Beta",             0.0);
  InitialCondition(u, global_len_x, global_len_y, factor, beta, name);
}

void InitialCondition(const Teuchos::RCP<Epetra_Vector>& u,
                      const unsigned int global_len_x,
                      const unsigned int global_len_y,
                      const double alpha, 
                      const double beta, 
                      const std::string &name)
{
  AnalyticFunction(*u, global_len_x, global_len_y, 0.0, alpha, beta, name);
}

}
