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


#ifndef ALBANY_ABSTRACTPROBLEM_HPP
#define ALBANY_ABSTRACTPROBLEM_HPP

#include <vector>
#include <string>

#include "Teuchos_RCP.hpp"
#include "Teuchos_ParameterList.hpp"
#include "Epetra_Map.h"
#include "Epetra_Vector.h"

#include "Albany_AbstractResponseFunction.hpp"
#include "Albany_StateManager.hpp"

#include "Phalanx.hpp"
#include "Albany_DataTypes.hpp"
#include "PHAL_AlbanyTraits.hpp"
#include "PHAL_Workset.hpp"
#include "PHAL_Dimension.hpp"

#include "Teuchos_VerboseObject.hpp"
#include <Intrepid_FieldContainer.hpp>

// Includes needed by most prolems
#include "Intrepid_HGRAD_LINE_C1_FEM.hpp"
#include "Intrepid_HGRAD_QUAD_C1_FEM.hpp"
#include "Intrepid_HGRAD_TRI_C1_FEM.hpp"
#include "Intrepid_HGRAD_HEX_C1_FEM.hpp"
#include "Intrepid_FieldContainer.hpp"
#include "Intrepid_DefaultCubatureFactory.hpp"
#include "Shards_CellTopology.hpp"
#include "PHAL_FactoryTraits.hpp"


namespace Albany {

  /*!
   * \brief Abstract interface for representing a 1-D finite element
   * problem.
   */
  class AbstractProblem {
  public:
  
    //! Only constructor
    AbstractProblem( const Teuchos::RCP<Teuchos::ParameterList>& params_,
                     const Teuchos::RCP<ParamLib>& paramLib_,
                     const int neq_);

    //! Destructor
    virtual ~AbstractProblem() {};

    //! Get the number of equations
    unsigned int numEquations() const;
    unsigned int numStates() const;

    //! Build the PDE instantiations, boundary conditions, and initial solution
    //! And construct the evaluators and field managers
    virtual void buildProblem(
       const int worksetSize,
       StateManager& stateMgr,
       const Albany::AbstractDiscretization& disc,
       std::vector< Teuchos::RCP<Albany::AbstractResponseFunction> >& responses,
       const Teuchos::RCP<Epetra_Vector>& u) = 0;

    Teuchos::RCP<PHX::FieldManager<PHAL::AlbanyTraits> > getFieldManager();
    Teuchos::RCP<PHX::FieldManager<PHAL::AlbanyTraits> > getDirichletFieldManager() ;

    //! Each problem must generate it's list of valide parameters
    virtual Teuchos::RCP<const Teuchos::ParameterList> getValidProblemParameters() const 
      {return getGenericProblemParams("Generic Problem List");};

    virtual void
      getAllocatedStates(
         Teuchos::ArrayRCP<Teuchos::ArrayRCP<Teuchos::RCP<Intrepid::FieldContainer<RealType> > > > oldState_,
         Teuchos::ArrayRCP<Teuchos::ArrayRCP<Teuchos::RCP<Intrepid::FieldContainer<RealType> > > > newState_
         ) const  {};

  protected:

    //! List of valid problem params common to all problems, as 
    //! a starting point for the specific  getValidProblemParameters
    Teuchos::RCP<Teuchos::ParameterList>
      getGenericProblemParams(std::string listname = "ProblemList") const;

    //! Configurable output stream, defaults to printing on proc=0
    Teuchos::RCP<Teuchos::FancyOStream> out;

    //! Generic implementation, can be made virtual if specialization is desired
    void constructDirichletEvaluators(
       const std::vector<std::string>& nodeSetIDs);

    Teuchos::RCP<const Teuchos::ParameterList> getValidDirichletBCParameters(
                 const std::vector<std::string>& nodeSetIDs) const;

    //! Number of equations per node being solved
    unsigned int neq;
    unsigned int nstates;

    //! Problem parameters
    Teuchos::RCP<Teuchos::ParameterList> params;
    Teuchos::ParameterList& DBCparams;

    //! Parameter library
    Teuchos::RCP<ParamLib> paramLib;

    //! Field manager for Volumettric Fill
    Teuchos::RCP<PHX::FieldManager<PHAL::AlbanyTraits> > fm;

    //! Field manager for Dirchlet Conditions Fill
    Teuchos::RCP<PHX::FieldManager<PHAL::AlbanyTraits> > dfm;

    //! Names of scalar variables in this problem, used in specifying BCs
    std::vector<std::string> dofNames;

  private:

    //! Private to prohibit default or copy constructor
    AbstractProblem();
    AbstractProblem(const AbstractProblem&);
    
    //! Private to prohibit copying
    AbstractProblem& operator=(const AbstractProblem&);

    //! Local utility function to construct unique string from Nodeset name and dof name
    std::string constructDBCName(const std::string ns, const std::string dof) const;
  };

}

#endif // ALBANY_ABSTRACTPROBLEM_HPP
