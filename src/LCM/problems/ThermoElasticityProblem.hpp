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


#ifndef THERMOELASTICITYPROBLEM_HPP
#define THERMOELASTICITYPROBLEM_HPP

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
  class ThermoElasticityProblem : public Albany::AbstractProblem {
  public:
  
    //! Default constructor
    ThermoElasticityProblem(
			    const Teuchos::RCP<Teuchos::ParameterList>& params,
			    const Teuchos::RCP<ParamLib>& paramLib,
			    const int numEq);

    //! Destructor
    virtual ~ThermoElasticityProblem();

    //! Build the PDE instantiations, boundary conditions, and initial solution
    virtual void 
    buildProblem(
       const Albany::MeshSpecsStruct& meshSpecs,
       StateManager& stateMgr,
       std::vector< Teuchos::RCP<Albany::AbstractResponseFunction> >& responses);

    //! Each problem must generate it's list of valid parameters
    Teuchos::RCP<const Teuchos::ParameterList> getValidProblemParameters() const;

    void getAllocatedStates(Teuchos::ArrayRCP<Teuchos::ArrayRCP<Teuchos::RCP<Intrepid::FieldContainer<RealType> > > > oldState_,
			    Teuchos::ArrayRCP<Teuchos::ArrayRCP<Teuchos::RCP<Intrepid::FieldContainer<RealType> > > > newState_
			    ) const;

  private:

    //! Private to prohibit copying
    ThermoElasticityProblem(const ThermoElasticityProblem&);
    
    //! Private to prohibit copying
    ThermoElasticityProblem& operator=(const ThermoElasticityProblem&);

    void constructEvaluators(const Albany::MeshSpecsStruct& meshSpecs,
                             Albany::StateManager& stateMgr,
                             std::vector< Teuchos::RCP<Albany::AbstractResponseFunction> >& responses);
    void constructDirichletEvaluators(const Albany::MeshSpecsStruct& meshSpecs);
  protected:

    //! Boundary conditions on source term
    bool haveSource;
    int T_offset;  //Position of T unknown in nodal DOFs
    int X_offset;  //Position of X unknown in nodal DOFs, followed by Y,Z
    int numDim;    //Number of spatial dimensions and displacement variable 

    Teuchos::ArrayRCP<Teuchos::ArrayRCP<Teuchos::RCP<Intrepid::FieldContainer<RealType> > > > oldState;
    Teuchos::ArrayRCP<Teuchos::ArrayRCP<Teuchos::RCP<Intrepid::FieldContainer<RealType> > > > newState;
  };

}

#endif // ALBANY_ELASTICITYPROBLEM_HPP
