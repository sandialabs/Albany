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


#ifndef THERMO_MECHANICAL_PROBLEM_HPP
#define THERMO_MECHANICAL_PROBLEM_HPP

#include "Teuchos_RCP.hpp"
#include "Teuchos_ParameterList.hpp"

#include "Albany_AbstractProblem.hpp"

#include "Phalanx.hpp"
#include "PHAL_Workset.hpp"
#include "PHAL_Dimension.hpp"
#include "Albany_ProblemUtils.hpp"
#include "Albany_ResponseUtilities.hpp"
#include "Albany_EvaluatorUtils.hpp"
#include "PHAL_AlbanyTraits.hpp"

namespace Albany {

  /*!
   * \brief ThermoMechanical Coupling Problem
   */
  class ThermoMechanicalProblem : public Albany::AbstractProblem {
  public:
  
    //! Default constructor
    ThermoMechanicalProblem( const Teuchos::RCP<Teuchos::ParameterList>& params,
			     const Teuchos::RCP<ParamLib>& paramLib,
			     const int numEq );

    //! Destructor
    virtual ~ThermoMechanicalProblem();

    //! Build the PDE instantiations, boundary conditions, and initial solution
    virtual void 
    buildProblem( Teuchos::ArrayRCP<Teuchos::RCP< Albany::MeshSpecsStruct > >  meshSpecs,
		  StateManager& stateMgr,
		  Teuchos::ArrayRCP< Teuchos::RCP< Albany::AbstractResponseFunction > >& responses );

    //! Each problem must generate it's list of valid parameters
    Teuchos::RCP<const Teuchos::ParameterList> getValidProblemParameters() const;

    void getAllocatedStates( Teuchos::ArrayRCP<Teuchos::ArrayRCP<Teuchos::RCP<Intrepid::FieldContainer<RealType> > > > oldState_,
			     Teuchos::ArrayRCP<Teuchos::ArrayRCP<Teuchos::RCP<Intrepid::FieldContainer<RealType> > > > newState_ ) const;

  private:

    //! Private to prohibit copying
    ThermoMechanicalProblem( const ThermoMechanicalProblem& );
    
    //! Private to prohibit copying
    ThermoMechanicalProblem& operator = ( const ThermoMechanicalProblem& );

    //    template <typename EvalT>
    //    void constructEvaluators<EvalT>(const Albany::MeshSpecsStruct& meshSpecs,
    //                             Albany::StateManager& stateMgr,
    //                             Teuchos::ArrayRCP< Teuchos::RCP<Albany::AbstractResponseFunction> >& responses);

    void constructDirichletEvaluators( const Albany::MeshSpecsStruct& meshSpecs );

  protected:

    //! Boundary conditions on source term
    bool haveSource;
    int T_offset;  //Position of T unknown in nodal DOFs
    int X_offset;  //Position of X unknown in nodal DOFs, followed by Y,Z
    int numDim;    //Number of spatial dimensions and displacement variable 

    Teuchos::ArrayRCP< Teuchos::ArrayRCP< Teuchos::RCP< Intrepid::FieldContainer< RealType > > > > oldState;
    Teuchos::ArrayRCP< Teuchos::ArrayRCP< Teuchos::RCP< Intrepid::FieldContainer< RealType > > > > newState;

  private:
    template < typename EvalT >
    void constructEvaluators( PHX::FieldManager< PHAL::AlbanyTraits >& fm0,
			      const Albany::MeshSpecsStruct& meshSpecs,
			      Albany::StateManager& stateMgr,
			      Teuchos::ArrayRCP< Teuchos::RCP< Albany::AbstractResponseFunction > >& responses,
			      bool constructResponses = false);
  };

}

#endif // ALBANY_ELASTICITYPROBLEM_HPP
