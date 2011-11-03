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


#ifndef ALBANY_MULTIHEATPROBLEM_HPP
#define ALBANY_MULTIHEATPROBLEM_HPP

#include "Teuchos_RCP.hpp"
#include "Teuchos_ParameterList.hpp"

#include "Albany_AbstractProblem.hpp"

#include "Phalanx.hpp"
#include "PHAL_Workset.hpp"
#include "PHAL_Dimension.hpp"

#include "QCAD_MaterialDatabase.hpp"

namespace Albany {

  /*!
   * \brief Abstract interface for representing a 1-D finite element
   * problem.
   */
  class MultiHeatProblem : public AbstractProblem {
  public:
  
    //! Default constructor
    MultiHeatProblem(
                         const Teuchos::RCP<Teuchos::ParameterList>& params,
                         const Teuchos::RCP<ParamLib>& paramLib,
                         const int numDim_, 
                         const Teuchos::RCP<const Epetra_Comm>& comm_);

    //! Destructor
    ~MultiHeatProblem();

    //! Build the PDE instantiations, boundary conditions, and initial solution
    void buildProblem(
       Teuchos::ArrayRCP<Teuchos::RCP<Albany::MeshSpecsStruct> >  meshSpecs,
       StateManager& stateMgr,
       Teuchos::ArrayRCP< Teuchos::RCP<Albany::AbstractResponseFunction> >& responses);

    //! Each problem must generate it's list of valide parameters
    Teuchos::RCP<const Teuchos::ParameterList> getValidProblemParameters() const;

  private:

    //! Private to prohibit copying
    MultiHeatProblem(const MultiHeatProblem&);
    
    //! Private to prohibit copying
    MultiHeatProblem& operator=(const MultiHeatProblem&);

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
      constructEvaluators<EvalT>(fm0, meshSpecs, stateMgr, BUILD_RESID_FM, junk);
    }

    //! Interface for Response field manager, except for residual type
    template <typename EvalT>
    void constructResponseEvaluators(
            PHX::FieldManager<PHAL::AlbanyTraits>& fm0,
            const Albany::MeshSpecsStruct& meshSpecs,
            Albany::StateManager& stateMgr)
    {
      Teuchos::ArrayRCP< Teuchos::RCP<Albany::AbstractResponseFunction> > junk;
      constructEvaluators<EvalT>(fm0, meshSpecs, stateMgr, BUILD_RESPONSE_FM, junk);
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
      constructEvaluators<EvalT>(fm0, meshSpecs, stateMgr, BUILD_RESPONSE_FM, responses);
    }

    void constructDirichletEvaluators(const Albany::MeshSpecsStruct& meshSpecs);

  protected:

    //! Boundary conditions on source term
    bool periodic;
    bool haveSource;
    bool haveAbsorption;
	bool haveMatDB;
    int numDim;

    std::string mtrlDbFilename;
    Teuchos::RCP<QCAD::MaterialDatabase> materialDB;
    Teuchos::RCP<const Epetra_Comm> comm;

  };

}

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

#include "PHAL_ThermalConductivity.hpp"
#include "PHAL_Absorption.hpp"
#include "PHAL_Source.hpp"
#include "PHAL_HeatEqResid.hpp"


template <typename EvalT>
void Albany::MultiHeatProblem::constructEvaluators(
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

   RCP<Intrepid::Basis<RealType, Intrepid::FieldContainer<RealType> > >
     intrepidBasis = Albany::getIntrepidBasis(meshSpecs.ctd);
   RCP<shards::CellTopology> cellType = rcp(new shards::CellTopology (&meshSpecs.ctd));

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

   // Parser will build parameter list that determines the field

   RCP<Albany::Layouts> dl = rcp(new Albany::Layouts(worksetSize,numVertices,numNodes,numQPts,numDim));
   Albany::EvaluatorUtils<EvalT, PHAL::AlbanyTraits> evalUtils(dl);

   Teuchos::ArrayRCP<string> dof_names(neq);
     dof_names[0] = "Temperature";
   Teuchos::ArrayRCP<string> dof_names_dot(neq);
     dof_names_dot[0] = "Temperature_dot";
   Teuchos::ArrayRCP<string> resid_names(neq);
     resid_names[0] = "Temperature Residual";

   fm0.template registerEvaluator<EvalT>
     (evalUtils.constructGatherSolutionEvaluator(false, dof_names, dof_names_dot));

   fm0.template registerEvaluator<EvalT>
     (evalUtils.constructScatterResidualEvaluator(false, resid_names));

   fm0.template registerEvaluator<EvalT>
     (evalUtils.constructGatherCoordinateVectorEvaluator());

   fm0.template registerEvaluator<EvalT>
     (evalUtils.constructMapToPhysicalFrameEvaluator( cellType, cubature));

   fm0.template registerEvaluator<EvalT>
     (evalUtils.constructComputeBasisFunctionsEvaluator(cellType, intrepidBasis, cubature));

  for (unsigned int i=0; i<neq; i++) {
    fm0.template registerEvaluator<EvalT>
      (evalUtils.constructDOFInterpolationEvaluator(dof_names[i]));

    fm0.template registerEvaluator<EvalT>
      (evalUtils.constructDOFInterpolationEvaluator(dof_names_dot[i]));

    fm0.template registerEvaluator<EvalT>
      (evalUtils.constructDOFGradInterpolationEvaluator(dof_names[i]));
  }

  // Temporary variable used numerous times below
  Teuchos::RCP<PHX::Evaluator<AlbanyTraits> > ev;

  { // Thermal conductivity
    RCP<ParameterList> p = rcp(new ParameterList);

    p->set<string>("QP Variable Name", "Thermal Conductivity");
    p->set<string>("QP Coordinate Vector Name", "Coord Vec");
    p->set< RCP<DataLayout> >("Node Data Layout", dl->node_scalar);
    p->set< RCP<DataLayout> >("QP Scalar Data Layout", dl->qp_scalar);
    p->set< RCP<DataLayout> >("QP Vector Data Layout", dl->qp_vector);

    p->set<RCP<ParamLib> >("Parameter Library", paramLib);
    Teuchos::ParameterList& paramList = params->sublist("Thermal Conductivity");
    p->set<Teuchos::ParameterList*>("Parameter List", &paramList);

    p->set<bool>("Have MatDB", haveMatDB);
    // Here we assume that the instance of this problem applies on a single element block
    p->set<string>("Element Block Name", meshSpecs.ebName);

    if(haveMatDB)
      p->set< RCP<QCAD::MaterialDatabase> >("MaterialDB", materialDB);

    ev = rcp(new PHAL::ThermalConductivity<EvalT,AlbanyTraits>(*p));
    fm0.template registerEvaluator<EvalT>(ev);
  }

  if (haveAbsorption) { // Absorption
    RCP<ParameterList> p = rcp(new ParameterList);

    p->set<string>("QP Variable Name", "Absorption");
    p->set<string>("QP Coordinate Vector Name", "Coord Vec");
    p->set< RCP<DataLayout> >("Node Data Layout", dl->node_scalar);
    p->set< RCP<DataLayout> >("QP Scalar Data Layout", dl->qp_scalar);
    p->set< RCP<DataLayout> >("QP Vector Data Layout", dl->qp_vector);

    p->set<RCP<ParamLib> >("Parameter Library", paramLib);
    Teuchos::ParameterList& paramList = params->sublist("Absorption");
    p->set<Teuchos::ParameterList*>("Parameter List", &paramList);

    ev = rcp(new PHAL::Absorption<EvalT,AlbanyTraits>(*p));
    fm0.template registerEvaluator<EvalT>(ev);
  }

  if (haveSource) { // Source
    RCP<ParameterList> p = rcp(new ParameterList);

    p->set<string>("Source Name", "Source");
    p->set<string>("Variable Name", "Temperature");
    p->set< RCP<DataLayout> >("QP Scalar Data Layout", dl->qp_scalar);

    p->set<RCP<ParamLib> >("Parameter Library", paramLib);
    Teuchos::ParameterList& paramList = params->sublist("Source Functions");
    p->set<Teuchos::ParameterList*>("Parameter List", &paramList);

    ev = rcp(new PHAL::Source<EvalT,AlbanyTraits>(*p));
    fm0.template registerEvaluator<EvalT>(ev);
  }

  { // Temperature Resid
    RCP<ParameterList> p = rcp(new ParameterList("Temperature Resid"));

    //Input
    p->set<string>("Weighted BF Name", "wBF");
    p->set< RCP<DataLayout> >("Node QP Scalar Data Layout", dl->node_qp_scalar);
    p->set<string>("QP Variable Name", "Temperature");

    p->set<string>("QP Time Derivative Variable Name", "Temperature_dot");

    p->set<bool>("Have Source", haveSource);
    p->set<bool>("Have Absorption", haveAbsorption);
    p->set<string>("Source Name", "Source");

    p->set<string>("Thermal Conductivity Name", "Thermal Conductivity");
    p->set< RCP<DataLayout> >("QP Scalar Data Layout", dl->qp_scalar);

    p->set<string>("Absorption Name", "Thermal Conductivity");
    p->set< RCP<DataLayout> >("QP Scalar Data Layout", dl->qp_scalar);
    
    p->set<string>("Gradient QP Variable Name", "Temperature Gradient");
    p->set< RCP<DataLayout> >("QP Vector Data Layout", dl->qp_vector);

    p->set<string>("Weighted Gradient BF Name", "wGrad BF");
    p->set< RCP<DataLayout> >("Node QP Vector Data Layout", dl->node_qp_vector);
    if (params->isType<string>("Convection Velocity"))
    	p->set<string>("Convection Velocity",
                       params->get<string>("Convection Velocity"));
    if (params->isType<bool>("Have Rho Cp"))
    	p->set<bool>("Have Rho Cp", params->get<bool>("Have Rho Cp"));

    //Output
    p->set<string>("Residual Name", "Temperature Residual");
    p->set< RCP<DataLayout> >("Node Scalar Data Layout", dl->node_scalar);

    ev = rcp(new PHAL::HeatEqResid<EvalT,AlbanyTraits>(*p));
    fm0.template registerEvaluator<EvalT>(ev);
  }

  if (fieldManagerChoice == Albany::BUILD_RESID_FM)  {
     PHX::Tag<typename EvalT::ScalarT> res_tag("Scatter", dl->dummy);
     fm0.requireField<EvalT>(res_tag);
   }
   else {
     Teuchos::ParameterList& responseList = params->sublist("Response Functions");
     Albany::ResponseUtilities<EvalT, PHAL::AlbanyTraits> respUtils(dl);
     respUtils.constructResponses(fm0, responses, responseList, stateMgr);
   }
}
#endif // ALBANY_HEATNONLINEARSOURCEPROBLEM_HPP
