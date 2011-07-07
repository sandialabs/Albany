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


#ifndef QCAD_POISSONPROBLEM_HPP
#define QCAD_POISSONPROBLEM_HPP

#include "Teuchos_RCP.hpp"
#include "Teuchos_ParameterList.hpp"

#include "Albany_AbstractProblem.hpp"

#include "Phalanx.hpp"
#include "PHAL_Workset.hpp"
#include "PHAL_Dimension.hpp"

namespace QCAD {

  /*!
   * \brief Abstract interface for representing a 1-D finite element
   * problem.
   */
  class PoissonProblem : public Albany::AbstractProblem 
  {
  public:
  
    //! Default constructor
    PoissonProblem(
                         const Teuchos::RCP<Teuchos::ParameterList>& params,
                         const Teuchos::RCP<ParamLib>& paramLib,
                         const int numDim_,
			 const Teuchos::RCP<const Epetra_Comm>& comm_);

    //! Destructor
    ~PoissonProblem();

    //! Build the PDE instantiations, boundary conditions, and initial solution
    void buildProblem(
       const Albany::MeshSpecsStruct& meshSpecs,
       Albany::StateManager& stateMgr,
       std::vector< Teuchos::RCP<Albany::AbstractResponseFunction> >& responses);

    //! Each problem must generate it's list of valide parameters
    Teuchos::RCP<const Teuchos::ParameterList> getValidProblemParameters() const;

  private:

    //! Private to prohibit copying
    PoissonProblem(const PoissonProblem&);
    
    //! Private to prohibit copying
    PoissonProblem& operator=(const PoissonProblem&);

    void constructEvaluators(const Albany::MeshSpecsStruct& meshSpecs,
                             Albany::StateManager& stateMgr,
			     std::vector< Teuchos::RCP<Albany::AbstractResponseFunction> >& responses);

    void constructDirichletEvaluators(const std::vector<std::string>& nodeSetIDs);

  protected:

    //! Boundary conditions on source term
    bool periodic;

    //! Parameters to use when constructing evaluators
    Teuchos::RCP<const Epetra_Comm> comm;
    bool haveSource;
    int numDim;
    double length_unit_in_m;
    double temperature;
    std::string mtrlDbFilename;

    //! Parameters for coupling to Schrodinger
    bool bUseSchrodingerSource;
    int nEigenvectors;    
  };

}

#endif
