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

#include "Albany_StateManager.hpp"
#include "Albany_StateInfoStruct.hpp" // contains MeshSpecsStuct

#include "Phalanx.hpp"
#include "Albany_DataTypes.hpp"
#include "Albany_BCUtils.hpp"

#include "PHAL_AlbanyTraits.hpp"
#include "PHAL_Workset.hpp"
#include "PHAL_Dimension.hpp"

#include "Teuchos_VerboseObject.hpp"
#include <Intrepid_FieldContainer.hpp>

#include "Intrepid_HGRAD_LINE_C1_FEM.hpp"
#include "Intrepid_HGRAD_LINE_Cn_FEM.hpp"
#include "Intrepid_HGRAD_QUAD_C1_FEM.hpp"
#include "Intrepid_HGRAD_QUAD_C2_FEM.hpp"
#include "Intrepid_HGRAD_TRI_C1_FEM.hpp"
#include "Intrepid_HGRAD_TRI_C2_FEM.hpp"
#include "Intrepid_HGRAD_HEX_C1_FEM.hpp"
#include "Intrepid_HGRAD_HEX_C2_FEM.hpp"
#include "Intrepid_HGRAD_TET_C1_FEM.hpp"
#include "Intrepid_HGRAD_TET_C2_FEM.hpp"
#include "Intrepid_HGRAD_TET_COMP12_FEM.hpp"
#include "Intrepid_FieldContainer.hpp"
#include "Intrepid_DefaultCubatureFactory.hpp"
#include "Shards_CellTopology.hpp"
#include "PHAL_FactoryTraits.hpp"


namespace Albany {

  class Application;

  enum FieldManagerChoice {BUILD_RESID_FM, BUILD_RESPONSE_FM, BUILD_STATE_FM};

  /*!
   * \brief Abstract interface for representing a 1-D finite element
   * problem.
   */
  class AbstractProblem {
  public:
  
    //! Only constructor
    AbstractProblem( const Teuchos::RCP<Teuchos::ParameterList>& params_,
                     const Teuchos::RCP<ParamLib>& paramLib_,
                     const int neq_ = 0);

    //! Destructor
    virtual ~AbstractProblem() {};

     virtual void getRBMInfoForML(
       int& numPDEs,
       int& numElasticityDim,
       int& numScalar,
       int& nullSpaceDim);

    //! Get the number of equations
    unsigned int numEquations() const;
    void setNumEquations(const int neq_);
    unsigned int numStates() const;

    //! Get spatial dimension
    virtual int spatialDimension() const = 0;

    //! Build the PDE instantiations, boundary conditions, and initial solution
    //! And construct the evaluators and field managers
    virtual void buildProblem(
      Teuchos::ArrayRCP<Teuchos::RCP<Albany::MeshSpecsStruct> >  meshSpecs,
      StateManager& stateMgr) = 0;

    // Build evaluators
    virtual Teuchos::Array< Teuchos::RCP<const PHX::FieldTag> > 
    buildEvaluators(
      PHX::FieldManager<PHAL::AlbanyTraits>& fm0,
      const Albany::MeshSpecsStruct& meshSpecs,
      Albany::StateManager& stateMgr,
      Albany::FieldManagerChoice fmchoice,
      const Teuchos::RCP<Teuchos::ParameterList>& responseList) = 0;

    Teuchos::ArrayRCP<Teuchos::RCP<PHX::FieldManager<PHAL::AlbanyTraits> > > getFieldManager();
    Teuchos::RCP<PHX::FieldManager<PHAL::AlbanyTraits> > getDirichletFieldManager() ;
    Teuchos::RCP<PHX::FieldManager<PHAL::AlbanyTraits> > getNeumannFieldManager() ;

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

    //! Number of equations per node being solved
    unsigned int neq;

    //! Problem parameters
    Teuchos::RCP<Teuchos::ParameterList> params;

    //! Parameter library
    Teuchos::RCP<ParamLib> paramLib;

    //! Field manager for Volumettric Fill
    Teuchos::ArrayRCP<Teuchos::RCP<PHX::FieldManager<PHAL::AlbanyTraits> > > fm;

    //! Field manager for Dirchlet Conditions Fill
    Teuchos::RCP<PHX::FieldManager<PHAL::AlbanyTraits> > dfm;

    //! Field manager for Neumann Conditions Fill
    Teuchos::RCP<PHX::FieldManager<PHAL::AlbanyTraits> > nfm;

  private:

    //! Private to prohibit default or copy constructor
    AbstractProblem();
    AbstractProblem(const AbstractProblem&);
    
    //! Private to prohibit copying
    AbstractProblem& operator=(const AbstractProblem&);
  };

  template <typename ProblemType>
  struct ConstructEvaluatorsOp {
    ProblemType& prob;
    PHX::FieldManager<PHAL::AlbanyTraits>& fm;
    const Albany::MeshSpecsStruct& meshSpecs;
    Albany::StateManager& stateMgr;
    Albany::FieldManagerChoice fmchoice;
    Teuchos::RCP<Teuchos::ParameterList> responseList;
    Teuchos::RCP< Teuchos::Array< Teuchos::RCP<const PHX::FieldTag> > > tags;
    ConstructEvaluatorsOp(
      ProblemType& prob_,
      PHX::FieldManager<PHAL::AlbanyTraits>& fm_,
      const Albany::MeshSpecsStruct& meshSpecs_,
      Albany::StateManager& stateMgr_,
      Albany::FieldManagerChoice fmchoice_ = BUILD_RESID_FM,
      const Teuchos::RCP<Teuchos::ParameterList>& responseList_ = Teuchos::null)
      : prob(prob_), fm(fm_), meshSpecs(meshSpecs_), 
	stateMgr(stateMgr_), fmchoice(fmchoice_), responseList(responseList_) {
      tags = 
	Teuchos::rcp(new Teuchos::Array< Teuchos::RCP<const PHX::FieldTag> >);
    }
    template <typename T> void operator() (T x) {
      tags->push_back(
	prob.template constructEvaluators<T>(fm, meshSpecs, stateMgr, 
					     fmchoice, responseList));
    }
  };

}

#endif // ALBANY_ABSTRACTPROBLEM_HPP
