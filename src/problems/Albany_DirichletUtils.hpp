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


#ifndef ALBANY_DIRICHLETUTILS_HPP
#define ALBANY_DIRICHLETUTILS_HPP

#include <vector>
#include <string>

#include "Teuchos_RCP.hpp"
#include "Teuchos_ParameterList.hpp"

#include "Albany_DataTypes.hpp"
#include "PHAL_AlbanyTraits.hpp"
#include "Phalanx.hpp"


namespace Albany {
  /*!
   * \brief Generic Functions to help define Dirichlet Field Manager
   */
  class DirichletUtils {

   public:

    DirichletUtils() {};

    //! Generic implementation of Field Manager for Dirichlet BCs
    Teuchos::RCP<PHX::FieldManager<PHAL::AlbanyTraits> > 
    constructDirichletEvaluators(
       const std::vector<std::string>& nodeSetIDs,
       const std::vector<std::string>& dirichletNames,
       Teuchos::RCP<Teuchos::ParameterList> params,
       Teuchos::RCP<ParamLib> paramLib);

    //! Function to return valid list of parameters in Dirichlet section of input file
    Teuchos::RCP<const Teuchos::ParameterList> getValidDirichletBCParameters(
                 const std::vector<std::string>& nodeSetIDs,
                 const std::vector<std::string>& dirichletNames) const;

  private:

    //! Local utility function to construct unique string from Nodeset name and dof name
    std::string constructDBCName(const std::string ns, const std::string dof) const;
  };
}

#endif 
