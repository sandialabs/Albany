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


#ifndef ALBANY_PROBLEMFACTORY_HPP
#define ALBANY_PROBLEMFACTORY_HPP

#include "Teuchos_ParameterList.hpp"
#include "Teuchos_RCP.hpp"

#include "Albany_AbstractProblem.hpp"

namespace Albany {

  /*!
   * \brief A factory class to instantiate AbstractProblem objects
   */
  class ProblemFactory {
  public:

    //! Default constructor
    ProblemFactory(const Teuchos::RCP<Teuchos::ParameterList>& problemParams,
                   const Teuchos::RCP<ParamLib>& paramLib);

    //! Destructor
    virtual ~ProblemFactory() {}

    virtual Teuchos::RCP<Albany::AbstractProblem>
    create();

  private:

    //! Private to prohibit copying
    ProblemFactory(const ProblemFactory&);

    //! Private to prohibit copying
    ProblemFactory& operator=(const ProblemFactory&);

  protected:

    //! Parameter list specifying what problem to create
    Teuchos::RCP<Teuchos::ParameterList> problemParams;

    //! Parameter library
    Teuchos::RCP<ParamLib> paramLib;

  };

}

#endif // ALBANY_PROBLEMFACTORY_HPP
