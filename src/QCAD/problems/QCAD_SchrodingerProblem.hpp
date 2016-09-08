//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

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
    SchrodingerProblem(const Teuchos::RCP<Teuchos::ParameterList>& params,
		       const Teuchos::RCP<ParamLib>& paramLib,
		       const int numDim_,
                       Teuchos::RCP<const Teuchos::Comm<int> >& commT_); 

    //! Destructor
    ~SchrodingerProblem();

    //! Return number of spatial dimensions
    virtual int spatialDimension() const { return numDim; }

    //! Build the PDE instantiations, boundary conditions, and initial solution
    virtual void buildProblem(
      Teuchos::ArrayRCP<Teuchos::RCP<Albany::MeshSpecsStruct> >  meshSpecs,
      Albany::StateManager& stateMgr);

    // Build evaluators
    virtual Teuchos::Array< Teuchos::RCP<const PHX::FieldTag> >
    buildEvaluators(
      PHX::FieldManager<PHAL::AlbanyTraits>& fm0,
      const Albany::MeshSpecsStruct& meshSpecs,
      Albany::StateManager& stateMgr,
      Albany::FieldManagerChoice fmchoice,
      const Teuchos::RCP<Teuchos::ParameterList>& responseList);

    //! Each problem must generate it's list of valide parameters
    Teuchos::RCP<const Teuchos::ParameterList> getValidProblemParameters() const;

  private:

    //! Private to prohibit copying
    SchrodingerProblem(const SchrodingerProblem&);
    
    //! Private to prohibit copying
    SchrodingerProblem& operator=(const SchrodingerProblem&);

  public:

    //! Main problem setup routine. Not directly called, but indirectly by following functions
    template <typename EvalT>
    Teuchos::RCP<const PHX::FieldTag>
    constructEvaluators(
      PHX::FieldManager<PHAL::AlbanyTraits>& fm0,
      const Albany::MeshSpecsStruct& meshSpecs,
      Albany::StateManager& stateMgr,
      Albany::FieldManagerChoice fmchoice,
      const Teuchos::RCP<Teuchos::ParameterList>& responseList);

    void constructDirichletEvaluators(const Albany::MeshSpecsStruct& meshSpecs);

  protected:
    Teuchos::RCP<const Teuchos::Comm<int> > commT; 
    bool havePotential;
    double energy_unit_in_eV, length_unit_in_m;
    std::string potentialFieldName;
    int potentialAuxIndex;
    std::string mtrlDbFilename;

    int numDim;
    int nEigenvectorsToOuputAsStates;
    bool bOnlySolveInQuantumBlocks;
  };

}

#include "QCAD_MaterialDatabase.hpp"

#include "Intrepid2_FieldContainer.hpp"
#include "Intrepid2_DefaultCubatureFactory.hpp"
#include "Shards_CellTopology.hpp"

#include "Albany_Utils.hpp"
#include "Albany_EvaluatorUtils.hpp"

#include "PHAL_DOFInterpolation.hpp"

#include "QCAD_SchrodingerPotential.hpp"
#include "QCAD_SchrodingerResid.hpp"
#if defined(ALBANY_EPETRA)
#include "QCAD_ResponseSaddleValue.hpp"
#endif


template <typename EvalT>
Teuchos::RCP<const PHX::FieldTag> 
QCAD::SchrodingerProblem::constructEvaluators(
  PHX::FieldManager<PHAL::AlbanyTraits>& fm0,
  const Albany::MeshSpecsStruct& meshSpecs,
  Albany::StateManager& stateMgr,
  Albany::FieldManagerChoice fieldManagerChoice,
  const Teuchos::RCP<Teuchos::ParameterList>& responseList)
{
   using Teuchos::RCP;
   using Teuchos::rcp;
   using Teuchos::ParameterList;
   using PHX::DataLayout;
   using std::vector;
   using std::string;
   using PHAL::AlbanyTraits;

   RCP<shards::CellTopology> cellType = rcp(new shards::CellTopology (&meshSpecs.ctd));
   RCP<Intrepid2::Basis<RealType, Intrepid2::FieldContainer_Kokkos<RealType, PHX::Layout, PHX::Device> > >
     intrepidBasis = Albany::getIntrepid2Basis(meshSpecs.ctd);

   const int numNodes = intrepidBasis->getCardinality();
   const int worksetSize = meshSpecs.worksetSize;

   Intrepid2::DefaultCubatureFactory<RealType, Intrepid2::FieldContainer_Kokkos<RealType, PHX::Layout, PHX::Device> > cubFactory;
   RCP <Intrepid2::Cubature<RealType, Intrepid2::FieldContainer_Kokkos<RealType, PHX::Layout,PHX::Device> > > cubature = cubFactory.create(*cellType, meshSpecs.cubatureDegree);

   const int numQPts = cubature->getNumPoints();
   const int numVertices = cellType->getNodeCount();

   *out << "Field Dimensions: Workset=" << worksetSize 
        << ", Vertices= " << numVertices
        << ", Nodes= " << numNodes
        << ", QuadPts= " << numQPts
        << ", Dim= " << numDim << std::endl;

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
     for (unsigned int i=0; i<neq; i++) dof_names_dot[i] = dof_names[i]+"_dot";
   }

   Teuchos::ArrayRCP<string> resid_names(neq);
     for (unsigned int i=0; i<neq; i++) resid_names[i] = dof_names[i]+" Residual";

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

   for (unsigned int i=0; i<neq; i++) {
     fm0.template registerEvaluator<EvalT>
       (evalUtils.constructDOFInterpolationEvaluator(dof_names[i]));

     if (supportsTransient)
     fm0.template registerEvaluator<EvalT>
         (evalUtils.constructDOFInterpolationEvaluator(dof_names_dot[i]));

     fm0.template registerEvaluator<EvalT>
       (evalUtils.constructDOFGradInterpolationEvaluator(dof_names[i]));
  }

   // Create Material Database
   RCP<QCAD::MaterialDatabase> materialDB = rcp(new QCAD::MaterialDatabase(mtrlDbFilename, commT));

  if (havePotential) { // If a "Potential" sublist is specified in the input, add a potential energy term

    if(potentialAuxIndex < 0) { 

      // Case when potential is given using the "Potential" parameter sublist

      RCP<ParameterList> p = rcp(new ParameterList);

      p->set<string>("QP Variable Name", "psi");
      p->set<string>("QP Potential Name", potentialFieldName);
      p->set<string>("QP Coordinate Vector Name", "Coord Vec");

      p->set<RCP<ParamLib> >("Parameter Library", paramLib);
      Teuchos::ParameterList& paramList = params->sublist("Potential");
      p->set<Teuchos::ParameterList*>("Parameter List", &paramList);

      //Global Problem Parameters
      p->set<double>("Energy unit in eV", energy_unit_in_eV);
      p->set<double>("Length unit in m", length_unit_in_m);

      ev = rcp(new QCAD::SchrodingerPotential<EvalT,AlbanyTraits>(*p,dl));
      fm0.template registerEvaluator<EvalT>(ev);
    }
    else {
     
#if defined(ALBANY_EPETRA) 
      //Case when we load the potential from an aux data vector (on the nodes)
      // to the potential field (on the quad points).  Note this requires the
      // "Type" parameter of the "Potential" input file sublist to be "From Aux Data Vector"
      // directives found in the "Potential" input file sublist.  This is used
      // when coupling the Poisson and Schrodinger problems (see QCAD::CoupledPoissonSchrodinger)

      Teuchos::ParameterList& paramList = params->sublist("Potential");
      std::string potentialType = paramList.get("Type", "not-given");
      TEUCHOS_TEST_FOR_EXCEPTION (potentialType != "From Aux Data Vector", Teuchos::Exceptions::InvalidParameter, std::endl 
	  << "Error! Schrodinger potential type must be \"From Aux Data Vector\" when an aux vector index is specified!" << std::endl);

      // Gather the aux data vector into potentialFieldName
      RCP<ParameterList> p = rcp(new ParameterList);
      p->set<string>("Field Name", potentialFieldName);
      p->set<int>("Aux Data Vector Index", potentialAuxIndex);

      ev = rcp(new PHAL::GatherAuxData<EvalT,AlbanyTraits>(*p,dl));
      fm0.template registerEvaluator<EvalT>(ev);

      // Interpolate potential to quad points (use DOFInterpolation)
      p = rcp(new ParameterList("Interpolate potential to quad points"));
      p->set<string>("Variable Name", potentialFieldName); // assumes same Name for output as for input 
      p->set<string>("BF Name", "BF");
      p->set<int>("Offset of First DOF", 0);

      ev = rcp(new PHAL::DOFInterpolation<EvalT,AlbanyTraits>(*p,dl));
      fm0.template registerEvaluator<EvalT>(ev);
#endif
    }
  }

  { // Wavefunction (psi) Resid
    RCP<ParameterList> p = rcp(new ParameterList("Wavefunction Resid"));

    //Input
    p->set<string>("Weighted BF Name", "wBF");
    p->set<string>("QP Variable Name", "psi");
    p->set<string>("QP Time Derivative Variable Name", "psi_dot");
    p->set<string>("QP Coordinate Vector Name", "Coord Vec");

    p->set<bool>("Have Potential", havePotential);
    p->set<string>("Potential Name", potentialFieldName); // was "V"

    p->set<string>("Gradient QP Variable Name", "psi Gradient");

    p->set<string>("Weighted Gradient BF Name", "wGrad BF");

    //Output
    p->set<string>("Residual Name", "psi Residual");

    //Global Problem Parameters
    p->set<double>("Energy unit in eV", energy_unit_in_eV);
    p->set<double>("Length unit in m", length_unit_in_m);
    p->set<bool>("Only solve in quantum blocks", bOnlySolveInQuantumBlocks);
    p->set< RCP<QCAD::MaterialDatabase> >("MaterialDB", materialDB);

    //Pass the Potential parameter list to test Finite Wall with different effective mass
    p->set<RCP<ParamLib> >("Parameter Library", paramLib);
    Teuchos::ParameterList& paramList = params->sublist("Potential");
    p->set<Teuchos::ParameterList*>("Parameter List", &paramList);

    ev = rcp(new QCAD::SchrodingerResid<EvalT,AlbanyTraits>(*p,dl));
    fm0.template registerEvaluator<EvalT>(ev);
  }


  if (fieldManagerChoice == Albany::BUILD_RESID_FM)  {
    PHX::Tag<typename EvalT::ScalarT> res_tag("Scatter", dl->dummy);
    fm0.requireField<EvalT>(res_tag);
    return res_tag.clone();
  }

  else if (fieldManagerChoice == Albany::BUILD_RESPONSE_FM) {

    // Parameters to be sent to all response constructors (whether they use them or not).
    RCP<ParameterList> pFromProb = rcp(new ParameterList("Response Parameters from Problem"));
    pFromProb->set<double>("Length unit in m", length_unit_in_m);
    pFromProb->set< RCP<QCAD::MaterialDatabase> >("MaterialDB", materialDB);

    Albany::ResponseUtilities<EvalT, PHAL::AlbanyTraits> respUtils(dl);
    return respUtils.constructResponses(fm0, *responseList, pFromProb, stateMgr);
  }

  return Teuchos::null;
}
#endif
