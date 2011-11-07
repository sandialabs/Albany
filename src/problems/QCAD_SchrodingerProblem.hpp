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


#ifndef QCAD_SCHRODINGERPROBLEM_HPP
#define QCAD_SCHRODINGERPROBLEM_HPP

#include "Teuchos_RCP.hpp"
#include "Teuchos_ParameterList.hpp"

#include "Albany_AbstractProblem.hpp"
#include "Albany_ProblemUtils.hpp"
#include "Albany_ResponseUtilities.hpp"

#include "Phalanx.hpp"
#include "PHAL_Workset.hpp"
#include "PHAL_Dimension.hpp"

namespace QCAD {

  /*!
   * \brief Abstract interface for representing a 1-D finite element
   * problem.
   */
  class SchrodingerProblem : public Albany::AbstractProblem {
  public:
  
    //! Default constructor
    SchrodingerProblem(
                         const Teuchos::RCP<Teuchos::ParameterList>& params,
                         const Teuchos::RCP<ParamLib>& paramLib,
                         const int numDim_,
			 const Teuchos::RCP<const Epetra_Comm>& comm_);

    //! Destructor
    ~SchrodingerProblem();

    //! Build the PDE instantiations, boundary conditions, and initial solution
    void buildProblem(
       Teuchos::ArrayRCP<Teuchos::RCP<Albany::MeshSpecsStruct> >  meshSpecs,
       Albany::StateManager& stateMgr,
       Teuchos::ArrayRCP< Teuchos::RCP<Albany::AbstractResponseFunction> >& responses);

    //! Each problem must generate it's list of valide parameters
    Teuchos::RCP<const Teuchos::ParameterList> getValidProblemParameters() const;

  private:

    //! Private to prohibit copying
    SchrodingerProblem(const SchrodingerProblem&);
    
    //! Private to prohibit copying
    SchrodingerProblem& operator=(const SchrodingerProblem&);

    //! Main problem setup routine. Not directly called, but indirectly by following functions
    template <typename EvalT>
    void constructEvaluators(
            PHX::FieldManager<PHAL::AlbanyTraits>& fm0,
            const Albany::MeshSpecsStruct& meshSpecs,
            Albany::StateManager& stateMgr,
            Albany::FieldManagerChoice fmchoice,
            Teuchos::ArrayRCP< Teuchos::RCP<Albany::AbstractResponseFunction> >& responses);

    //! Interface for Residual (PDE) field manager
    template <typename EvalT>
    void constructResidEvaluators(
            PHX::FieldManager<PHAL::AlbanyTraits>& fm0,
            const Albany::MeshSpecsStruct& meshSpecs,
            Albany::StateManager& stateMgr)
    {
      Teuchos::ArrayRCP< Teuchos::RCP<Albany::AbstractResponseFunction> > junk;
      constructEvaluators<EvalT>(fm0, meshSpecs, stateMgr, Albany::BUILD_RESID_FM, junk);
    }

    //! Interface for Response field manager, except for residual type
    template <typename EvalT>
    void constructResponseEvaluators(
            PHX::FieldManager<PHAL::AlbanyTraits>& fm0,
            const Albany::MeshSpecsStruct& meshSpecs,
            Albany::StateManager& stateMgr)
    {
      Teuchos::ArrayRCP< Teuchos::RCP<Albany::AbstractResponseFunction> > junk;
      constructEvaluators<EvalT>(fm0, meshSpecs, stateMgr, Albany::BUILD_RESPONSE_FM, junk);
    }

    //! Interface for Response field manager, Residual type.
    // This version loads the responses variable, that needs to be constructed just once
    template <typename EvalT>
    void constructResponseEvaluators(
            PHX::FieldManager<PHAL::AlbanyTraits>& fm0,
            const Albany::MeshSpecsStruct& meshSpecs,
            Albany::StateManager& stateMgr,
            Teuchos::ArrayRCP< Teuchos::RCP<Albany::AbstractResponseFunction> >& responses)
    {
      constructEvaluators<EvalT>(fm0, meshSpecs, stateMgr, Albany::BUILD_RESPONSE_FM, responses);
    }

    void constructDirichletEvaluators(const Albany::MeshSpecsStruct& meshSpecs);

    template <typename EvalT>
    void
    constructResponses(PHX::FieldManager<PHAL::AlbanyTraits>& fm0,
                       Teuchos::ArrayRCP< Teuchos::RCP<Albany::AbstractResponseFunction> >& responses,
		       Teuchos::ParameterList& responseList, 
		       Albany::StateManager& stateMgr,
                       Albany::ResponseUtilities<EvalT, PHAL::AlbanyTraits>& respUtils);

  protected:
    Teuchos::RCP<const Epetra_Comm> comm;
    bool havePotential;
    bool haveMaterial;
    double energy_unit_in_eV, length_unit_in_m;
    std::string potentialStateName;
    std::string mtrlDbFilename;

    int numDim;
    int nEigenvectorsToOuputAsStates;
    bool bOnlySolveInQuantumBlocks;
  };

}

#include "QCAD_MaterialDatabase.hpp"

#include "Intrepid_FieldContainer.hpp"
#include "Intrepid_DefaultCubatureFactory.hpp"
#include "Shards_CellTopology.hpp"

#include "Albany_Utils.hpp"
#include "Albany_EvaluatorUtils.hpp"

#include "QCAD_SchrodingerPotential.hpp"
#include "QCAD_SchrodingerResid.hpp"
#include "QCAD_ResponseSaddleValue.hpp"


template <typename EvalT>
void QCAD::SchrodingerProblem::constructEvaluators(
        PHX::FieldManager<PHAL::AlbanyTraits>& fm0,
        const Albany::MeshSpecsStruct& meshSpecs,
        Albany::StateManager& stateMgr,
        Albany::FieldManagerChoice fieldManagerChoice,
        Teuchos::ArrayRCP< Teuchos::RCP<Albany::AbstractResponseFunction> >& responses)
{
   using Teuchos::RCP;
   using Teuchos::rcp;
   using Teuchos::ParameterList;
   using PHX::DataLayout;
   using PHX::MDALayout;
   using std::vector;
   using PHAL::AlbanyTraits;

   RCP<shards::CellTopology> cellType = rcp(new shards::CellTopology (&meshSpecs.ctd));
   RCP<Intrepid::Basis<RealType, Intrepid::FieldContainer<RealType> > >
     intrepidBasis = Albany::getIntrepidBasis(meshSpecs.ctd);

   const int numNodes = intrepidBasis->getCardinality();
   const int worksetSize = meshSpecs.worksetSize;

   Intrepid::DefaultCubatureFactory<RealType> cubFactory;
   RCP <Intrepid::Cubature<RealType> > cubature = cubFactory.create(*cellType, meshSpecs.cubatureDegree);

   const int numQPts = cubature->getNumPoints();
   const int numVertices = cellType->getNodeCount();

   *out << "Field Dimensions: Workset=" << worksetSize 
        << ", Vertices= " << numVertices
        << ", Nodes= " << numNodes
        << ", QuadPts= " << numQPts
        << ", Dim= " << numDim << endl;

   RCP<Albany::Layouts> dl = rcp(new Albany::Layouts(worksetSize,numVertices,numNodes,numQPts,numDim));
   Albany::EvaluatorUtils<EvalT, PHAL::AlbanyTraits> evalUtils(dl);
   bool supportsTransient=true;

   // Temporary variable used numerous times below
   Teuchos::RCP<PHX::Evaluator<AlbanyTraits> > ev;

   // Define Field Names

   Teuchos::ArrayRCP<string> dof_names(neq);
     dof_names[0] = "psi";

   Teuchos::ArrayRCP<string> dof_names_dot(neq);
   if (supportsTransient) {
     for (int i=0; i<neq; i++) dof_names_dot[i] = dof_names[i]+"_dot";
   }

   Teuchos::ArrayRCP<string> resid_names(neq);
     for (int i=0; i<neq; i++) resid_names[i] = dof_names[i]+" Residual";

   if (supportsTransient) fm0.template registerEvaluator<EvalT>
       (evalUtils.constructGatherSolutionEvaluator(false, dof_names, dof_names_dot));
   else fm0.template registerEvaluator<EvalT>
       (evalUtils.constructGatherSolutionEvaluator_noTransient(false, dof_names));

   fm0.template registerEvaluator<EvalT>
     (evalUtils.constructScatterResidualEvaluator(false, resid_names));

   fm0.template registerEvaluator<EvalT>
     (evalUtils.constructGatherCoordinateVectorEvaluator());

   fm0.template registerEvaluator<EvalT>
     (evalUtils.constructMapToPhysicalFrameEvaluator(cellType, cubature));

   fm0.template registerEvaluator<EvalT>
     (evalUtils.constructComputeBasisFunctionsEvaluator(cellType, intrepidBasis, cubature));

   for (int i=0; i<neq; i++) {
     fm0.template registerEvaluator<EvalT>
       (evalUtils.constructDOFInterpolationEvaluator(dof_names[i]));

     if (supportsTransient)
     fm0.template registerEvaluator<EvalT>
         (evalUtils.constructDOFInterpolationEvaluator(dof_names_dot[i]));

     fm0.template registerEvaluator<EvalT>
       (evalUtils.constructDOFGradInterpolationEvaluator(dof_names[i]));
  }

   // Create Material Database
   RCP<QCAD::MaterialDatabase> materialDB = rcp(new QCAD::MaterialDatabase(mtrlDbFilename, comm));

  if (havePotential) { // Potential energy
    RCP<ParameterList> p = rcp(new ParameterList);

    p->set<string>("QP Variable Name", "psi");
    p->set<string>("QP Potential Name", potentialStateName);
    p->set<string>("QP Coordinate Vector Name", "Coord Vec");
    p->set< RCP<DataLayout> >("Node Data Layout", dl->node_scalar);
    p->set< RCP<DataLayout> >("QP Scalar Data Layout", dl->qp_scalar);
    p->set< RCP<DataLayout> >("QP Vector Data Layout", dl->qp_vector);

    p->set<RCP<ParamLib> >("Parameter Library", paramLib);
    Teuchos::ParameterList& paramList = params->sublist("Potential");
    p->set<Teuchos::ParameterList*>("Parameter List", &paramList);

    //Global Problem Parameters
    p->set<double>("Energy unit in eV", energy_unit_in_eV);
    p->set<double>("Length unit in m", length_unit_in_m);

    ev = rcp(new QCAD::SchrodingerPotential<EvalT,AlbanyTraits>(*p));
    fm0.template registerEvaluator<EvalT>(ev);
  }

  { // Wavefunction (psi) Resid
    RCP<ParameterList> p = rcp(new ParameterList("Wavefunction Resid"));

    //Input
    p->set<string>("Weighted BF Name", "wBF");
    p->set< RCP<DataLayout> >("Node QP Scalar Data Layout", dl->node_qp_scalar);
    p->set<string>("QP Variable Name", "psi");
    p->set<string>("QP Time Derivative Variable Name", "psi_dot");
    p->set<string>("QP Coordinate Vector Name", "Coord Vec");

    p->set<bool>("Have Potential", havePotential);
    p->set<bool>("Have Material", haveMaterial);
    p->set<string>("Potential Name", potentialStateName); // was "V"
    p->set< RCP<DataLayout> >("QP Scalar Data Layout", dl->qp_scalar);

    p->set<string>("Gradient QP Variable Name", "psi Gradient");
    p->set< RCP<DataLayout> >("QP Vector Data Layout", dl->qp_vector);

    p->set<string>("Weighted Gradient BF Name", "wGrad BF");
    p->set< RCP<DataLayout> >("Node QP Vector Data Layout", dl->node_qp_vector);

    p->set< RCP<DataLayout> >("QP Tensor Data Layout", dl->qp_tensor);
  
    //Output
    p->set<string>("Residual Name", "psi Residual");
    p->set< RCP<DataLayout> >("Node Scalar Data Layout", dl->node_scalar);

    if(haveMaterial) {
      Teuchos::ParameterList& paramList = params->sublist("Material");
      p->set<Teuchos::ParameterList*>("Material Parameter List", &paramList);
    }

    //Global Problem Parameters
    p->set<double>("Energy unit in eV", energy_unit_in_eV);
    p->set<double>("Length unit in m", length_unit_in_m);
    p->set<bool>("Only solve in quantum blocks", bOnlySolveInQuantumBlocks);
    p->set< RCP<QCAD::MaterialDatabase> >("MaterialDB", materialDB);

    //Pass the Potential parameter list to test Finite Wall with different effective mass
    p->set<RCP<ParamLib> >("Parameter Library", paramLib);
    Teuchos::ParameterList& paramList = params->sublist("Potential");
    p->set<Teuchos::ParameterList*>("Parameter List", &paramList);

    ev = rcp(new QCAD::SchrodingerResid<EvalT,AlbanyTraits>(*p));
    fm0.template registerEvaluator<EvalT>(ev);
  }


  if (fieldManagerChoice == Albany::BUILD_RESID_FM)  {
    PHX::Tag<typename EvalT::ScalarT> res_tag("Scatter", dl->dummy);
    fm0.requireField<EvalT>(res_tag);
  }

  else {
    Teuchos::ParameterList& responseList = params->sublist("Response Functions");
    Albany::ResponseUtilities<EvalT, PHAL::AlbanyTraits> respUtils(dl);
    //respUtils.constructResponses(fm0, responses, responseList, stateMgr);
    this->constructResponses(fm0, responses, responseList, stateMgr, respUtils);
  }
}

template<typename EvalT>
void
QCAD::SchrodingerProblem::constructResponses(
  PHX::FieldManager<PHAL::AlbanyTraits>& fm0,
  Teuchos::ArrayRCP< Teuchos::RCP<Albany::AbstractResponseFunction> >& responses,
  Teuchos::ParameterList& responseList, 
  Albany::StateManager& stateMgr,
  Albany::ResponseUtilities<EvalT, PHAL::AlbanyTraits>& respUtils)
{
  using Teuchos::RCP;
  using Teuchos::rcp;
  using Teuchos::ParameterList;
  using PHX::DataLayout;
  using std::string;
  using PHAL::AlbanyTraits;
  Albany::Layouts& dl = *respUtils.get_dl();

   // Parameters for Response Evaluators
   //  Iterate through list of responses (from input xml file).  For each, create a response
   //  function and possibly a parameter list to construct a response evaluator.
   int num_responses = responseList.get("Number", 0);
   responses.resize(num_responses);

   std::vector<string> responseIDs_to_require;

   // First, add in response targets for PHAL_SaveStateField evaluators created
   //  during problem setup (these evaluators only act for residual type)
   if(typeid(EvalT) == typeid(PHAL::AlbanyTraits::Residual))
     responseIDs_to_require = stateMgr.getResidResponseIDsToRequire();

   for (int i=0; i<num_responses; i++) 
   {
     std::string responseID = Albany::strint("Response",i);
     std::string name = responseList.get(responseID, "??");

     Teuchos::RCP< PHX::Evaluator<PHAL::AlbanyTraits> > ev;
     if( respUtils.getStdResponseFn(name, i, responseList, responses, stateMgr, ev) ) {
       if(ev != Teuchos::null) {
         fm0.template registerEvaluator<EvalT>(ev);
	 responseIDs_to_require.push_back(responseID);
       }
     }

     else if (name == "Saddle Value")
     { 
       std::string responseParamsID = Albany::strint("ResponseParams",i);              
       ParameterList& responseParams = responseList.sublist(responseParamsID);
       RCP<ParameterList> p = rcp(new ParameterList);
       
       RCP<QCAD::SaddleValueResponseFunction> 
	 svResponse = rcp(new QCAD::SaddleValueResponseFunction(
					     numDim, responseParams)); 
       responses[i] = svResponse;
       
       p->set<string>("Response ID", responseID);
       p->set<int>   ("Response Index", i);
       p->set< Teuchos::RCP<QCAD::SaddleValueResponseFunction> >
	 ("Response Function", svResponse);
       p->set<Teuchos::ParameterList*>("Parameter List", &responseParams);
       p->set< RCP<DataLayout> >("Dummy Data Layout", dl.dummy);
       
       p->set<string>("Coordinate Vector Name", "Coord Vec");
       p->set<string>("Weights Name",   "Weights");
       p->set< RCP<DataLayout> >("QP Scalar Data Layout", dl.qp_scalar);
       p->set< RCP<DataLayout> >("QP Vector Data Layout", dl.qp_vector);

       ev = rcp(new QCAD::ResponseSaddleValue<EvalT,AlbanyTraits>(*p));
       fm0.template registerEvaluator<EvalT>(ev);

       responseIDs_to_require.push_back(responseID);
     }

     else {
       TEUCHOS_TEST_FOR_EXCEPTION(true, Teuchos::Exceptions::InvalidParameter,
				  std::endl << "Error!  Unknown response function " << name <<
				  "!" << std::endl << "Supplied parameter list is " <<
				  std::endl << responseList);
     }
   } // end of loop over responses

   //! Create field manager for responses
   respUtils.createResponseFieldManager(fm0, responseIDs_to_require);
}
#endif
