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


#ifndef LAMEPROBLEM_HPP
#define LAMEPROBLEM_HPP

#include "Teuchos_RCP.hpp"
#include "Teuchos_ParameterList.hpp"

#include "Albany_AbstractProblem.hpp"

#include "Phalanx.hpp"
#include "PHAL_Workset.hpp"
#include "PHAL_Dimension.hpp"

namespace Albany {

  /*!
   * \brief Abstract interface for representing a 2-D finite element
   * problem.
   */
  class LameProblem : public Albany::AbstractProblem {
  public:
  
    //! Default constructor
    LameProblem(const Teuchos::RCP<Teuchos::ParameterList>& params,
                const Teuchos::RCP<ParamLib>& paramLib,
                const int numEq);

    //! Destructor
    virtual ~LameProblem();

    //! Build the PDE instantiations, boundary conditions, and initial solution
    virtual void 
    buildProblem(const Albany::MeshSpecsStruct& meshSpecs,
                 StateManager& stateMgr,
                 std::vector< Teuchos::RCP<Albany::AbstractResponseFunction> >& responses);

    //! Each problem must generate it's list of valid parameters
    Teuchos::RCP<const Teuchos::ParameterList> getValidProblemParameters() const;

  private:

    //! Private to prohibit copying
    LameProblem(const LameProblem&);
    
    //! Private to prohibit copying
    LameProblem& operator=(const LameProblem&);

    void constructEvaluators(const Albany::MeshSpecsStruct& meshSpecs,
                             Albany::StateManager& stateMgr);
    void constructDirichletEvaluators(const Albany::MeshSpecsStruct& meshSpecs);
  protected:

    //! Boundary conditions on source term
    bool haveSource;

  };

}

#endif // ALBANY_LAMEPROBLEM_HPP
