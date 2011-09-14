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


#ifndef ALBANY_ODEPROBLEM_HPP
#define ALBANY_ODEPROBLEM_HPP

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
  class ODEProblem : public AbstractProblem {
  public:
  
    //! Default constructor
    ODEProblem(
               const Teuchos::RCP<Teuchos::ParameterList>& params,
               const Teuchos::RCP<ParamLib>& paramLib,
               const int numDim_);

    //! Destructor
    ~ODEProblem();

    //! Build the PDE instantiations, boundary conditions, and initial solution
    void buildProblem(
       const Albany::MeshSpecsStruct& meshSpecs,
       StateManager& stateMgr,
       std::vector< Teuchos::RCP<Albany::AbstractResponseFunction> >& responses);

    //! Each problem must generate it's list of valide parameters
    Teuchos::RCP<const Teuchos::ParameterList> getValidProblemParameters() const;

  private:

    //! Private to prohibit copying
    ODEProblem(const ODEProblem&);
    
    //! Private to prohibit copying
    ODEProblem& operator=(const ODEProblem&);

    void constructEvaluators(const Albany::MeshSpecsStruct& meshSpecs,
                             Albany::StateManager& stateMgr,
                             std::vector< Teuchos::RCP<Albany::AbstractResponseFunction> >& responses);

    void constructDirichletEvaluators(const Albany::MeshSpecsStruct& meshSpecs);

  protected:
    int numDim;

  };

}

#endif 
