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


#ifndef ALBANY_HEATPROBLEM_HPP
#define ALBANY_HEATPROBLEM_HPP

#include "Teuchos_RCP.hpp"
#include "Teuchos_ParameterList.hpp"

#include "Albany_AbstractProblem.hpp"

#include "Phalanx.hpp"
#include "PHAL_Workset.hpp"
#include "PHAL_Dimension.hpp"

namespace Albany {

  /*!
   * \brief Abstract interface for representing a 1-D finite element
   * problem.
   */
  class HeatProblem : public AbstractProblem {
  public:
  
    //! Default constructor
    HeatProblem(
                         const Teuchos::RCP<Teuchos::ParameterList>& params,
                         const Teuchos::RCP<ParamLib>& paramLib,
                         const int numDim_);

    //! Destructor
    ~HeatProblem();

    //! Build the PDE instantiations, boundary conditions, and initial solution
    void buildProblem(
       const int worksetSize,
       StateManager& stateMgr,
       const Albany::AbstractDiscretization& disc,
       std::vector< Teuchos::RCP<Albany::AbstractResponseFunction> >& responses,
       const Teuchos::RCP<Epetra_Vector>& u);

    //! Each problem must generate it's list of valide parameters
    Teuchos::RCP<const Teuchos::ParameterList> getValidProblemParameters() const;

  private:

    //! Private to prohibit copying
    HeatProblem(const HeatProblem&);
    
    //! Private to prohibit copying
    HeatProblem& operator=(const HeatProblem&);

    void constructEvaluators(const int worksetSize, const int cubDegree);

  protected:

    //! Boundary conditions on source term
    bool periodic;
    bool haveIC;
    bool haveSource;
    int numDim;

  };

}

#endif // ALBANY_HEATNONLINEARSOURCEPROBLEM_HPP
