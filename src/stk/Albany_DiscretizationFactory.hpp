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


#ifndef ALBANY_DISCRETIZATIONFACTORY_HPP
#define ALBANY_DISCRETIZATIONFACTORY_HPP

#include <vector>

#include "Teuchos_ParameterList.hpp"
#include "Teuchos_RCP.hpp"
#include "Epetra_Comm.h"

#include "Albany_AbstractDiscretization.hpp"

#ifdef ALBANY_CUTR
#include "CUTR_CubitMeshMover.hpp"
#endif

namespace Albany {

  /*!
   * \brief A factory class to instantiate AbstractDiscretization objects
   */
  class DiscretizationFactory {
  public:

    //! Default constructor
    DiscretizationFactory(
	      const Teuchos::RCP<Teuchos::ParameterList>& discParams);

    //! Destructor
    ~DiscretizationFactory() {}

    //! Method to inject cubit dependence.
#ifdef ALBANY_CUTR
    void setMeshMover(const Teuchos::RCP<CUTR::CubitMeshMover>& meshMover_);
#endif

    Teuchos::RCP<Albany::AbstractDiscretization>
    create(unsigned int num_equations,unsigned int num_states,
           const Teuchos::RCP<const Epetra_Comm>& epetra_comm);

  private:

    //! Private to prohibit copying
    DiscretizationFactory(const DiscretizationFactory&);

    //! Private to prohibit copying
    DiscretizationFactory& operator=(const DiscretizationFactory&);

  protected:

    //! Parameter list specifying what element to create
    Teuchos::RCP<Teuchos::ParameterList> discParams;

#ifdef ALBANY_CUTR
    Teuchos::RCP<CUTR::CubitMeshMover> meshMover;
#endif

  };

}

#endif // ALBANY_DISCRETIZATIONFACTORY_HPP
