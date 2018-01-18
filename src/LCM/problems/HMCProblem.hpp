//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#ifndef HMCPROBLEM_HPP
#define HMCPROBLEM_HPP

#include "Teuchos_RCP.hpp"
#include "Teuchos_ParameterList.hpp"

#include "Albany_AbstractProblem.hpp"
#include "ConstitutiveModelInterface.hpp"
#include "Albany_MaterialDatabase.hpp"

#ifdef ALBANY_ATO
#include "ATO_OptimizationProblem.hpp"
#include "ATO_Utils.hpp"
#endif

#include "PHAL_Workset.hpp"
#include "PHAL_Dimension.hpp"
#include "PHAL_AlbanyTraits.hpp"

// To do:
// --  Add multiblock support (See mechanics example problem)
// --  Add density as input.  Currently hardwired to implicit value of 1.0.
// --  Add Currant limit.  Newmark integrator only seems to work for beta=0.25.
// --  Add artificial viscosity.
// --  Add hourglass stabilization for single point integration.
// --  Fix output.  Currently it's dumping everything!


/*\begin{text}
This source has been annotated with latex comments.  Use the eqcc script to compile into a summary pdf.  The source is best viewed using folding in vim (i.e., \begin{verbatim} :g/\\begin{text}/foldc \end{verbatim})
\end{text}*/


namespace Albany {

  /*!
   * \brief Abstract interface for representing a 2-D finite element
   * problem.
   */
  class HMCProblem :
#ifdef ALBANY_ATO
    public ATO::OptimizationProblem,
#endif
    public virtual Albany::AbstractProblem {
  public:

    //! Default constructor
    HMCProblem(
		      const Teuchos::RCP<Teuchos::ParameterList>& params_,
		      const Teuchos::RCP<ParamLib>& paramLib_,
		      const int numDim_,
                      Teuchos::RCP<const Teuchos::Comm<int>>& commT);


    //! Destructor
    virtual ~HMCProblem();

    //! Return number of spatial dimensions
    virtual int spatialDimension() const { return numDim; }

    //! Get boolean telling code if SDBCs are utilized
    virtual bool useSDBCs() const {return use_sdbcs_; }

    //! Build the PDE instantiations, boundary conditions, and initial solution
    virtual void buildProblem(
      Teuchos::ArrayRCP<Teuchos::RCP<Albany::MeshSpecsStruct>>  meshSpecs,
      StateManager& stateMgr);

    // Build evaluators
    virtual Teuchos::Array< Teuchos::RCP<const PHX::FieldTag>>
    buildEvaluators(
      PHX::FieldManager<PHAL::AlbanyTraits>& fm0,
      const Albany::MeshSpecsStruct& meshSpecs,
      Albany::StateManager& stateMgr,
      Albany::FieldManagerChoice fmchoice,
      const Teuchos::RCP<Teuchos::ParameterList>& responseList);

    //! Each problem must generate it's list of valid parameters
    Teuchos::RCP<const Teuchos::ParameterList> getValidProblemParameters() const;

    void getAllocatedStates(Teuchos::ArrayRCP<Teuchos::ArrayRCP<Teuchos::RCP<Kokkos::DynRankView<RealType, PHX::Device>>>> oldState_,
			    Teuchos::ArrayRCP<Teuchos::ArrayRCP<Teuchos::RCP<Kokkos::DynRankView<RealType, PHX::Device>>>> newState_
			    ) const;

  private:

    //! Private to prohibit copying
    HMCProblem(const HMCProblem&);

    //! Private to prohibit copying
    HMCProblem& operator=(const HMCProblem&);


    void parseMaterialModel(Teuchos::RCP<Teuchos::ParameterList>& p,
                       const Teuchos::RCP<Teuchos::ParameterList>& params) const;

    Teuchos::RCP<Albany::MaterialDatabase> material_db_;


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
    void constructNeumannEvaluators(const Teuchos::RCP<Albany::MeshSpecsStruct>& meshSpecs);


  protected:

    ///
    ///Boolean marking whether SDBCs are used
    bool use_sdbcs_;

    //! Boundary conditions on source term
    bool haveSource;
    int numDim;
    int numMicroScales;

    std::string matModel;
    Teuchos::RCP<Albany::Layouts> dl;

    Teuchos::ArrayRCP<Teuchos::ArrayRCP<Teuchos::RCP<Kokkos::DynRankView<RealType, PHX::Device>>>> oldState;
    Teuchos::ArrayRCP<Teuchos::ArrayRCP<Teuchos::RCP<Kokkos::DynRankView<RealType, PHX::Device>>>> newState;

    template <typename EvalT>
    void registerStateVariables(
       Teuchos::RCP<LCM::ConstitutiveModelInterface<EvalT, PHAL::AlbanyTraits>> cmiEv,
       PHX::FieldManager<PHAL::AlbanyTraits>& fm0,
       Albany::StateManager& stateMgr, std::string eb_name, int numDim);
  };

}

#include "Albany_SolutionAverageResponseFunction.hpp"
#include "Albany_SolutionTwoNormResponseFunction.hpp"
#include "Albany_SolutionMaxValueResponseFunction.hpp"
#include "Albany_Utils.hpp"
#include "Albany_ProblemUtils.hpp"
#include "Albany_ResponseUtilities.hpp"
#include "Albany_EvaluatorUtils.hpp"
#include "HMC_StrainDifference.hpp"
#include "HMC_TotalStress.hpp"
#include "FieldNameMap.hpp"

#include "Strain.hpp"
#include "DefGrad.hpp"
#include "PHAL_SaveStateField.hpp"
#include "PHAL_SaveCellStateField.hpp"
#include "UpdateField.hpp"
#include "ElasticityResid.hpp"
#include "HMC_MicroResidual.hpp"
#ifdef ALBANY_ATO
#include "ATO_TopologyFieldWeighting.hpp"
#include "ATO_TopologyWeighting.hpp"
#endif

#include "Time.hpp"
#include "ConstitutiveModelParameters.hpp"


#include <sstream>

template <typename EvalT>
Teuchos::RCP<const PHX::FieldTag>
Albany::HMCProblem::constructEvaluators(
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
   using PHX::MDALayout;
   using std::vector;
   using PHAL::AlbanyTraits;

  // get the name of the current element block
   std::string eb_name = meshSpecs.ebName;

  // get the name of the material model to be used (and make sure there is one)
  std::string material_model_name =
      material_db_->
          getElementBlockSublist(eb_name, "Material Model").get<std::string>(
          "Model Name");
  TEUCHOS_TEST_FOR_EXCEPTION(material_model_name.length() == 0, std::logic_error,
      "A material model must be defined for block: "
          + eb_name);

#ifdef ALBANY_VERBOSE
  *out << "In MechanicsProblem::constructEvaluators" << std::endl;
  *out << "element block name: " << eb_name << std::endl;
  *out << "material model name: " << material_model_name << std::endl;
#endif

   RCP<shards::CellTopology> cellType = rcp(new shards::CellTopology (&meshSpecs.ctd));
   RCP<Intrepid2::Basis<PHX::Device, RealType, RealType>>
     intrepidBasis = Albany::getIntrepid2Basis(meshSpecs.ctd);

   const int numNodes = intrepidBasis->getCardinality();
   const int worksetSize = meshSpecs.worksetSize;

   Intrepid2::DefaultCubatureFactory cubFactory;
   RCP <Intrepid2::Cubature<PHX::Device>  > cubature = cubFactory.create<PHX::Device, RealType, RealType>(*cellType, meshSpecs.cubatureDegree);

   const int numDim = cubature->getDimension();
   const int numQPts = cubature->getNumPoints();
   const int numVertices = cellType->getNodeCount();

   *out << "Field Dimensions: Workset=" << worksetSize
        << ", Vertices= " << numVertices
        << ", Nodes= " << numNodes
        << ", QuadPts= " << numQPts
        << ", Dim= " << numDim << std::endl;


   // volume average pressure
   bool volume_average_pressure(false);
   if (material_db_->isElementBlockParam(eb_name, "Volume Average Pressure"))
     volume_average_pressure =
       material_db_->getElementBlockParam<bool>(eb_name,"Volume Average Pressure");


   // Construct standard FEM evaluators with standard field names
   dl = rcp(new Albany::Layouts(worksetSize,numVertices,numNodes,numQPts,numDim));
   TEUCHOS_TEST_FOR_EXCEPTION(dl->vectorAndGradientLayoutsAreEquivalent==false, std::logic_error,
                              "Data Layout Usage in Mechanics problems assume vecDim = numDim");

   Albany::EvaluatorUtils<EvalT, PHAL::AlbanyTraits> evalUtils(dl);
#ifdef ALBANY_ATO
   ATO::Utils<EvalT, PHAL::AlbanyTraits> atoUtils(dl, numDim);
#endif

   // independent variables
   std::string strDisplacement("Displacement");
   std::string strMicrostrain("Microstrain");

   // Define Field Names:
   /*\begin{text}
   The 'Displacement' and 'Microstrain_n' variables defined below are increments, i.e. $\Delta u_{Ii}$, etc.
   \end{text}*/
   // displacement increment
   //   macro
   const int numMacroScales = 1;
   Teuchos::ArrayRCP<std::string> macro_dof_names(numMacroScales);
   macro_dof_names[0] = std::string("Delta") + strDisplacement;
   Teuchos::ArrayRCP<std::string> macro_resid_names(numMacroScales);
   macro_resid_names[0] = macro_dof_names[0] + " Residual";

   //   micro
   Teuchos::ArrayRCP< Teuchos::ArrayRCP<std::string>> micro_dof_names(numMicroScales);
   Teuchos::ArrayRCP< Teuchos::ArrayRCP<std::string>> micro_resid_names(numMicroScales);
   Teuchos::ArrayRCP< Teuchos::ArrayRCP<std::string>> micro_scatter_names(numMicroScales);
   for(int i=0;i<numMicroScales;i++){
      micro_dof_names[i].resize(1);
      micro_resid_names[i].resize(1);
      micro_scatter_names[i].resize(1);
      micro_dof_names[i][0] = std::string("Delta") + Albany::strint(strMicrostrain,i);
      micro_resid_names[i][0] = micro_dof_names[i][0] + " Residual";
      micro_scatter_names[i][0] = micro_dof_names[i][0] + " Scatter";
    }

   // acceleration
   //   macro
   Teuchos::ArrayRCP<std::string> macro_dof_names_dotdot(numMacroScales);
   Teuchos::ArrayRCP<std::string> macro_resid_names_dotdot(numMacroScales);
   Teuchos::ArrayRCP< Teuchos::ArrayRCP<std::string>> micro_dof_names_dotdot(numMicroScales);
   Teuchos::ArrayRCP< Teuchos::ArrayRCP<std::string>> micro_resid_names_dotdot(numMicroScales);
   Teuchos::ArrayRCP< Teuchos::ArrayRCP<std::string>> micro_scatter_names_dotdot(numMicroScales);
   macro_dof_names_dotdot[0] = macro_dof_names[0]+"_dotdot";
   macro_resid_names_dotdot[0] = macro_resid_names[0]+" Residual";
   //   micro
   for(int i=0;i<numMicroScales;i++){
     micro_dof_names_dotdot[i].resize(1);
     micro_resid_names_dotdot[i].resize(1);
     micro_scatter_names_dotdot[i].resize(1);
     micro_dof_names_dotdot[i][0] = micro_dof_names[i][0]+"_dotdot";
     micro_resid_names_dotdot[i][0] = micro_resid_names_dotdot[i][0]+" Residual";
     micro_scatter_names_dotdot[i][0] = micro_scatter_names_dotdot[i][0]+" Scatter";
   }

  // Temporary variable used numerous times below
  Teuchos::RCP<PHX::Evaluator<AlbanyTraits>> ev;


// Create displacement state variable.
/*\begin{text}
   The total displacement variable is registered as two state (i.e., registerOldState=true),
   so Displacement is at state N+1 and Displacement_old is at N. \\
   \textbf{DEPENDENT FIELDS:} \\
     None. \\
  \textbf{EVALUATED FIELDS:} \\
  \begin{tabular}{l l l l}
     $u_{Ii}^N$     & Displacement at state N   & "Displacement_current"  & dims(cell,I=nNodes,i=vecDim) \\
     $u_{Ii}^{N+1}$ & Displacement at state N+1 & "Displacement_updated"  & dims(cell,I=nNodes,i=vecDim)
  \end{tabular}
\end{text}*/
  std::string strDisplacement_Updated = strDisplacement;        // updated state, i.e., state at N+1
  std::string strDisplacement_Current = strDisplacement+"_old"; // current state, i.e., state at N
  {
    RCP<ParameterList> p = rcp(new ParameterList);
    p = stateMgr.registerStateVariable(strDisplacement_Updated,dl->node_vector, dl->dummy, eb_name,
               /* init type = */       "scalar",
               /* init value = */       0.0,
               /* registerOldState = */ true,
               /* write as output = */  true);
    ev = rcp(new PHAL::SaveStateField<EvalT,AlbanyTraits>(*p));
    fm0.template registerEvaluator<EvalT>(ev);
  }

// Compute updated displacement
/*\begin{text}
   $ u_{Ii}^{N+1} = u_{Ii}^N + \Delta u_{Ii} $ \\
   \textbf{DEPENDENT FIELDS:} \\
     $u_{Ii}^N$      & Displacement at state N   & "Displacement_old"   & dims(cell,I=nNodes,i=vecDim) \\
     $\Delta u_{Ii}$ & Displacement Increment    & "DeltaDisplacement"  & dims(cell,I=nNodes,i=vecDim) \\
  \textbf{EVALUATED FIELDS:} \\
  \begin{tabular}{l l l l}
     $u_{Ii}^{N+1}$  & Displacement at state N+1 & "Displacement"       & dims(cell,I=nNodes,i=vecDim) \\
  \end{tabular}
\end{text}*/
  {
    RCP<ParameterList> p = rcp(new ParameterList("Update Displacement"));
    p->set< RCP<DataLayout>>("Field Layout", dl->node_vector);
    // Input
    p->set<std::string>("Current State Name", strDisplacement_Current);
    p->set<std::string>("Increment Name", macro_dof_names[0]);
    // Output
    p->set<std::string>("Updated Field Name", strDisplacement_Updated);
    ev = rcp(new LCM::UpdateField<EvalT,AlbanyTraits>(*p));
    fm0.template registerEvaluator<EvalT>(ev);
  }

// Create microstrain_n state variables.
/*\begin{text}
   The total microstrain variables are registered as two state (i.e., registerOldState=true),
   so totalMicrostrain_n is at state N+1 and totalMicrostrain_n_old is at N. \\
   \textbf{DEPENDENT FIELDS:} \\
     None. \\
  \textbf{EVALUATED FIELDS:} \\
  \begin{tabular}{l l l l}
     $\epsilon_{Iij}^{n\ N}$   & Total Microstrain n at state N   & "totalMicrostrain_n"     & dims(cell,I=nNodes,i=vecDim) \\
     $\epsilon_{Iij}^{n\ N+1}$ & Total Microstrain n at state N+1 & "totalMicrostrain_n_old" & dims(cell,I=nNodes,i=vecDim)
  \end{tabular}
\end{text}*/
  vector<std::string> strMicrostrains(numMicroScales);
  vector<std::string> strMicrostrains_Updated(numMicroScales);
  vector<std::string> strMicrostrains_Current(numMicroScales);
  for(int i=0; i<numMicroScales; i++){

    strMicrostrains[i] = Albany::strint(strMicrostrain,i);

    RCP<ParameterList> p = rcp(new ParameterList);

    // save cell average for output
    std::string cn[3] = {"x","y","z"};
    for(int j=0; j< numDim; j++)
      for(int k=0; k< numDim; k++){
        std::string varname(strMicrostrains[i]);
        varname += " ";
        varname += cn[j];
        varname += cn[k];
        varname += "_ave ";
        p = stateMgr.registerStateVariable(varname,
            dl->cell_scalar, dl->dummy, eb_name, "scalar",
            0.0, /* save state = */ false, /* write output = */ true);
        p->set("Field Layout", dl->qp_tensor);
        p->set("Field Name", strMicrostrains[i]);
        p->set("Weights Layout", dl->qp_scalar);
        p->set("Weights Name", "Weights");
        p->set("component i", j);
        p->set("component j", k);
        ev = Teuchos::rcp(new PHAL::SaveCellStateField<EvalT, PHAL::AlbanyTraits>(*p));
        fm0.template registerEvaluator<EvalT>(ev);
      }

    strMicrostrains_Updated[i] = strMicrostrains[i];          // updated state, i.e., state at N+1
    strMicrostrains_Current[i] = strMicrostrains[i]+"_old";   // current state, i.e., state at N
    p = stateMgr.registerStateVariable(strMicrostrains[i],dl->qp_tensor, dl->dummy, eb_name,
               /* init type = */       "scalar",
               /* init value = */       0.0,
               /* registerOldState = */ true,
               /* write as output = */  false);
    ev = rcp(new PHAL::SaveStateField<EvalT,AlbanyTraits>(*p));
    fm0.template registerEvaluator<EvalT>(ev);
  }

// Compute updated microstrain from delta
/*\begin{text}
   $ \epsilon_{Iij}^{N+1} = \epsilon_{Iij}^N + \Delta \epsilon_{Iij} $ \\
   \textbf{DEPENDENT FIELDS:} \\
     $\epsilon_{Iij}^N$      & Microstrain n at state N   & "Microstrain_n_old"   & dims(cell,I=nNodes,i=vecDim,j=vecDim) \\
     $\Delta \epsilon_{Iij}$ & Microstrain n Increment    & "DeltaMicrostrain_n"  & dims(cell,I=nNodes,i=vecDim,j=vecDim) \\
  \textbf{EVALUATED FIELDS:} \\
  \begin{tabular}{l l l l}
     $\epsilon_{Iij}^{N+1}$  & Microstrain n at state N+1 & "Microstrain_n"       & dims(cell,I=nNodes,i=vecDim,j=vecDim) \\
  \end{tabular}
\end{text}*/
  for(int i=0; i<numMicroScales; i++){
    RCP<ParameterList> p = rcp(new ParameterList("Updated Microstrain"));
    p->set< RCP<DataLayout>>("Field Layout", dl->qp_tensor);
    // Input
    p->set<std::string>("Current State Name", strMicrostrains_Current[i]);
    p->set<std::string>("Increment Name", micro_dof_names[i][0]);
    // Output
    p->set<std::string>("Updated Field Name", strMicrostrains_Updated[i]);
    ev = rcp(new LCM::UpdateField<EvalT,AlbanyTraits>(*p));
    fm0.template registerEvaluator<EvalT>(ev);
  }

// Gather Solution (displacement and acceleration)
/*\begin{text}
  Gather solution data from solver data structures to grid based structures.  \\
   \textbf{DEPENDENT FIELDS:} \\
     None. \\
  \textbf{EVALUATED FIELDS:} \\
  \begin{tabular}{l l l l}
     $u_{Ii}$ & Nodal displacements & "Displacement"         & dims(cell,I=nNodes,i=vecDim) \\
     $a_{Ii}$ & Nodal accelerations & "Displacement\_dotdot" & dims(cell,I=nNodes,i=vecDim)
  \end{tabular}
\end{text}*/

   int vectorRank = 1;
   fm0.template registerEvaluator<EvalT>
     (evalUtils.constructGatherSolutionEvaluator_withAcceleration(vectorRank, macro_dof_names, Teuchos::null, macro_dof_names_dotdot));

// Gather Solution (microstrains and micro accelerations)
/*\begin{text}
  Gather solution data from solver data structures to grid based structures.  \\
   \textbf{DEPENDENT FIELDS:} \\
     None. \\
  \textbf{EVALUATED FIELDS:} \\
  \begin{tabular}{l l l l}
     $\epsilon^n_{Iij}$        & Nodal microstrains at scale 'n'        & "Microstrain\_n"         & dims(cell,I=nNodes,i=vecDim,j=vecDim) \\
     $\ddot{\epsilon}^n_{Iij}$ & Nodal micro accelerations at scale 'n' & "Microstrain\_n\_dotdot" & dims(cell,I=nNodes,i=vecDim,j=vecDim)
  \end{tabular} \\
\end{text}*/
   int dof_offset = numDim; // dof layout is {x, y, ..., xx, xy, xz, yx, ...}
   int dof_stride = numDim*numDim;
   int tensorRank = 2;
   for(int i=0;i<numMicroScales;i++){
     fm0.template registerEvaluator<EvalT>
       (evalUtils.constructGatherSolutionEvaluator_withAcceleration(
          tensorRank, micro_dof_names[i], Teuchos::null, micro_dof_names_dotdot[i], dof_offset+i*dof_stride) );
   }

// Gather Coordinates
/*\begin{text}
  Gather coordinate data from solver data structures to grid based structures.  The single
  argument to constructGatherCoordinateVectorEvaluator causes the created evaluator to add
  the variable named strDisplacement_Current to be added to the coordinates. \\
   \textbf{DEPENDENT FIELDS:} \\
     None. \\
  \textbf{EVALUATED FIELDS:} \\
   \begin{tabular}{l l l l}
     $x_{Ii}$ & Nodal coordinates & "Coord Vec" & dims(cell,I=nNodes,i=vecDim) \\
   \end{tabular} \\
\end{text}*/
   fm0.template registerEvaluator<EvalT>
     (evalUtils.constructGatherCoordinateVectorEvaluator(strDisplacement_Current));


// Compute gradient matrix and weighted basis function values in current coordinates
/*\begin{text}
    Register new evaluator. \\
    \textbf{DEPENDENT FIELDS:}\\
    \begin{tabular}{l l l l}
    $x_{Ii}$ & Nodal coordinates & "Coord Vec" & dims(cell,I=nNodes,i=vecDim) \\
    \end{tabular} \\
    \textbf{EVALUATED FIELDS:} \\
    \begin{tabular}{l l l l}
    $det\left(\frac{\partial x_{ip}}{\partial \xi_j}\right) \omega_p$ &
    Weighted measure & "Weights" & dims(cell,p=nQPs)\\
    %
    $det\left(\frac{\partial x_{ip}}{\partial \xi_j}\right)$ &
    Jacobian determinant & Jacobian Det" & dims(cell,p=nQPs) \\
    %
    $N_I(\mathbf{x}_p)$ &
    Basis function values & "BF" & dims(cell,I=nNodes,p=nQPs)\\
    %
    $N_I(\mathbf{x}_p)\ det\left(\frac{\partial x_{ip}}{\partial \xi_j}\right) \omega_p$ &
    Weighted basis function values & "wBF" & dims(cell,I=nNode,p=nQPs)\\
    %
    $\frac{\partial N_I (\mathbf{x}_p)}{\partial \xi_k} J^{-1}_{kj}$ &
    Gradient matrix wrt physical frame & "Grad BF" & dims(cell,I=nNodes,p=nQPs,j=spcDim)\\
    %
    $\frac{\partial N_I (\mathbf{x}_p)}{\partial \xi_k} J^{-1}_{kj} det\left(\frac{\partial x_{ip}}{\partial \xi_j}\right) \omega_p$ &
    Weighted gradient matrix wrt current config & "wGrad BF" & dims(cell,I=nNodes,p=nQPs,j=spcDim)\\
    \end{tabular} \\
 \end{text}*/
   fm0.template registerEvaluator<EvalT>
     (evalUtils.constructComputeBasisFunctionsEvaluator(cellType, intrepidBasis, cubature));

// Project displacements to Gauss points
/*\begin{text}
   Register new evaluator: \\
  \begin{align*}
     u_i(\xi_p)&=N_I(\xi_p) u_{Ii}\\
     (c,p,i)&=(c,I,p)*(c,I,i)
  \end{align*}
   \textbf{DEPENDENT FIELDS:} \\
   \begin{tabular}{l l l l}
     $u_{Ii}$      & Nodal Displacements  & "Displacements"  & dims(cell,I=nNodes,i=vecDim) \\
     $N_I(\xi_p)$  & Basis Functions      & "BF"             & dims(cell,I=nNodes,p=nQPs)   \\
   \end{tabular} \\
  \textbf{EVALUATED FIELDS:} \\
  \begin{tabular}{l l l l}
     $u_i(\xi_p)$ & Displacements at quadrature points & "Displacements" & dims(cell,p=nQPs,i=vecDim)
  \end{tabular} \\
\end{text}*/
   fm0.template registerEvaluator<EvalT>
     (evalUtils.constructDOFVecInterpolationEvaluator(macro_dof_names[0]));

// Project microstrains to Gauss points
/*\begin{text}
   Register new evaluator: \\
  \begin{align*}
     \epsilon^n_{ij}(\xi_p)&=N_I(\xi_p) \epsilon^n_{ijI}\\
     (c,p,i,j)&=(c,I,p)*(c,I,i,j)
  \end{align*}
   \textbf{DEPENDENT FIELDS:} \\
   \begin{tabular}{l l l l}
     $\epsilon^n_{Iij}$  & Nodal microstrains at scale 'n'  & "Microstrain\_n"  & dims(cell,I=nNodes,i=vecDim,j=vecDim) \\
     $N_I(\xi_p)$        & Basis Functions                  & "BF"              & dims(cell,I=nNodes,p=nQPs)            \\
   \end{tabular} \\
  \textbf{EVALUATED FIELDS:} \\
  \begin{tabular}{l l l l}
     $\epsilon^n_{ij}(\xi_p)$ & Microstrains at scale 'n' at quadrature points  & "Microstrain\_n"  & dims(cell,p=nQPs,i=vecDim,j=spcDim)
  \end{tabular} \\
\end{text}*/
   for(int i=0;i<numMicroScales;i++)
     fm0.template registerEvaluator<EvalT>
       (evalUtils.constructDOFTensorInterpolationEvaluator(micro_dof_names[i][0],
        dof_offset+i*dof_stride));

// Project accelerations to Gauss points
/*\begin{text}
   Register new evaluator: \\
  \begin{align*}
     a_i(\xi_p)&=N_I(\xi_p) a_{Ii}\\
     (c,p,i)&=(c,I,p)*(c,I,i)
  \end{align*}
   \textbf{DEPENDENT FIELDS:} \\
   \begin{tabular}{l l l l}
     $a_{Ii}$      & Nodal Acceleration  & "Displacement\_dotdot"  & dims(cell,I=nNodes,i=vecDim) \\
     $N_I(\xi_p)$  & Basis Functions     & "BF"                    & dims(cell,I=nNodes,p=nQPs)   \\
   \end{tabular} \\
  \textbf{EVALUATED FIELDS:} \\
  \begin{tabular}{l l l l}
     $a_i(\xi_p)$ & Acceleration at quadrature points  & "Displacement\_dotdot"  & dims(cell,p=nQPs,i=vecDim)
  \end{tabular} \\
\end{text}*/
   fm0.template registerEvaluator<EvalT>
     (evalUtils.constructDOFVecInterpolationEvaluator(macro_dof_names_dotdot[0]));

// Project micro accelerations to Gauss points
/*\begin{text}
   Register new evaluator:
  \begin{align*}
     \ddot{\epsilon}^n_{ij}(\xi_p)&=N_I(\xi_p) \ddot{\epsilon}^n_{Iij}\\
     (c,p,i,j)&=(c,I,p)*(c,I,i,j)
  \end{align*}
   \textbf{DEPENDENT FIELDS:} \\
   \begin{tabular}{l l l l}
     $\ddot{\epsilon}^n_{Iij}$  & Nodal micro acceleration at scale 'n'  & "Microstrain\_n\_dotdot"  & dims(cell,I=nNodes,i=vecDim,j=vecDim)\\
     $N_I(\xi_p)$               & Basis Functions                        & "BF"                      & dims(cell,I=nNodes,p=nQPs) \\
   \end{tabular} \\
  \textbf{EVALUATED FIELDS:} \\
  \begin{tabular}{l l l l}
     $\ddot{\epsilon}^n_{ij}(\xi_p)$  & Micro acceleration at scale 'n' at quadrature points &  "Microstrain\_n\_dotdot"  & dims(cell,p=nQPs,i=vecDim,j=vecDim)
  \end{tabular} \\
\end{text}*/
   for(int i=0;i<numMicroScales;i++)
     fm0.template registerEvaluator<EvalT>
       (evalUtils.constructDOFTensorInterpolationEvaluator(micro_dof_names_dotdot[i][0],
        dof_offset+i*dof_stride));

// Project nodal coordinates to Gauss points
/*\begin{text}
   Register new evaluator: Compute Gauss point locations from nodal locations.
  \begin{align*}
     x_{pi} &= N_{I}(\xi_p) x_{Ii}\\
   (c,p,i) &= (c,I,p)*(c,I,i)
  \end{align*}
   \textbf{DEPENDENT FIELDS:} \\
   \begin{tabular}{l l l l}
     $x_{Ii}$  & Nodal coordinates        & "Coord Vec"  & dims(cell,I=nNodes,i=vecDim) \\
   \end{tabular} \\
  \textbf{EVALUATED FIELDS:} \\
   \begin{tabular}{l l l l}
     $x_{pi}$  & Gauss point coordinates  & "Coord Vec"  & dims(cell,p=nQPs,i=vecDim) \\
   \end{tabular} \\
\end{text}*/
   fm0.template registerEvaluator<EvalT>
     (evalUtils.constructMapToPhysicalFrameEvaluator(cellType, cubature));

// Compute displacement gradient
/*\begin{text}
   New evaluator:
  \begin{align*}
     \left.\frac{\partial u_i}{\partial x_j}\right|_{\xi_p} &= \partial_j N_{I}(\xi_p) u_{iI}\\
   (c,p,i,j) &= (c,I,p,j)*(c,I,i)
  \end{align*}
   \textbf{DEPENDENT FIELDS:} \\
   \begin{tabular}{l l l l}
     $u_{Ii}$      & Nodal Displacement           & "Displacement"  & dims(cell,I=nNodes,i=vecDim)        \\
     $B_I(\xi_p)$  & Gradient of Basis Functions  & "Grad BF"       & dims(cell,I=nNodes,p=nQPs,i=vecDim) \\
   \end{tabular} \\
  \textbf{EVALUATED FIELDS:} \\
  \begin{tabular}{l l l l}
     $\left.\frac{\partial u_i}{\partial x_j}\right|_{\xi_p}$  \\
     & Gradient of node vector  & "Displacement Gradient"  & dims(cell,p=nQPs,i=vecDim,j=spcDim)
  \end{tabular} \\
\end{text}*/
   fm0.template registerEvaluator<EvalT>
     (evalUtils.constructDOFVecGradInterpolationEvaluator(macro_dof_names[0]));

// Compute increment of the microstrain gradient
/*\begin{text}
   Register new evaluator:
  \begin{align*}
     \left.\frac{\partial \epsilon^n_{ij}}{\partial x_k}\right|_{\xi_p} &= \partial_k N_{I}(\xi_p) \epsilon^n_{Iij}\\
   (c,p,i,j,k) &= (c,I,p,k)*(c,I,i,j)
  \end{align*}
   \textbf{DEPENDENT FIELDS:} \\
   \begin{tabular}{l l l l}
     $\epsilon^n_{Iij}$  & Nodal microstrain at scale 'n'  & "Microstrain\_n"  & dims(cell,I=nNodes,i=vecDim,j=vecDim) \\
     $B_I(\xi_p)$        & Gradient of Basis Functions     & "Grad BF"         & dims(cell,I=nNodes,p=nQPs,i=vecDim)   \\
   \end{tabular} \\
  \textbf{EVALUATED FIELDS:} \\
  \begin{tabular}{l l l l}
     $\left.\frac{\partial \epsilon^n_{ij}}{\partial x_k}\right|_{\xi_p}$
     & Microstrain gradient at scale 'n'  & "Microstrain\_n Gradient"  & dims(cell,p=nQPs,i=vecDim,j=vecDim,k=spcDim)
  \end{tabular} \\
\end{text}*/
   for(int i=0;i<numMicroScales;i++){
     fm0.template registerEvaluator<EvalT>
       (evalUtils.constructDOFTensorGradInterpolationEvaluator(micro_dof_names[i][0],dof_offset+i*dof_stride));


     std::string strMSGrad_Inc = micro_dof_names[i][0] + " Gradient";
     std::string msGrad = Albany::strint(strMicrostrain,i) + " Gradient";
     {
       RCP<ParameterList> p = rcp(new ParameterList);
       p = stateMgr.registerStateVariable(msGrad, dl->qp_tensor3, dl->dummy, eb_name,
                  /* init type = */       "scalar",
                  /* init value = */       0.0,
                  /* registerOldState = */ true,
                  /* write as output = */  false);
       ev = rcp(new PHAL::SaveStateField<EvalT,AlbanyTraits>(*p));
       fm0.template registerEvaluator<EvalT>(ev);
     }
     std::string strMSGrad_Updated = msGrad;          // updated state, i.e., state at N+1
     std::string strMSGrad_Current = msGrad+"_old";   // current state, i.e., state at N
     {
       RCP<ParameterList> p = rcp(new ParameterList("Microstrain Gradient"));
       p->set< RCP<DataLayout>>("Field Layout", dl->qp_tensor3);
       // Input
       p->set<std::string>("Current State Name", strMSGrad_Current);
       p->set<std::string>("Increment Name", strMSGrad_Inc);
       // Output
       p->set<std::string>("Updated Field Name", strMSGrad_Updated);
       ev = rcp(new LCM::UpdateField<EvalT,AlbanyTraits>(*p));
       fm0.template registerEvaluator<EvalT>(ev);
     }
  }

// Compute strain increment and add to current strain to get updated strain
/*\begin{text}
   New evaluator:\\
  \begin{align*}
     \epsilon^p_{ij} &=
               \frac{1}{2}\left(\left.\frac{\partial u_i}{\partial x_j}\right|_{\xi_p}
                               +\left.\frac{\partial u_j}{\partial x_i}\right|_{\xi_p} \right)\\
   (c,p,i,j) &= ((c,p,i,j)+(c,p,j,i)/2.0)
  \end{align*}
   \textbf{DEPENDENT FIELDS:} \\
   \begin{tabular}{l l l l}
     $\left.\frac{\partial u_i}{\partial x_j}\right|_{\xi_p}$
     & Gradient of displacement  & "Displacement Gradient"  & dims(cell, p=nQPs, i=vecDim, j=spcDim)
   \end{tabular} \\
  \textbf{EVALUATED FIELDS:} \\
  \begin{tabular}{l l l l}
     $\epsilon^p_{ij}$ & Infinitesimal strain  & "Strain" & dims(cell, p=nQPs, i=vecDim, j=spcDim) \\
  \end{tabular} \\
\end{text}*/
  {
    RCP<ParameterList> p = rcp(new ParameterList("Strain Increment"));

    //Input
    p->set<std::string>("Gradient QP Variable Name", macro_dof_names[0] + " Gradient");

    //Output
    p->set<std::string>("Strain Name", "Strain Increment");

    ev = rcp(new LCM::Strain<EvalT,AlbanyTraits>(*p,dl));
    fm0.template registerEvaluator<EvalT>(ev);
  }
  std::string strStrain("Strain");
  {
    RCP<ParameterList> p = rcp(new ParameterList);

    // save cell average for output
    std::string cn[3] = {"x","y","z"};
    for(int j=0; j< numDim; j++)
      for(int k=0; k< numDim; k++){
        std::string varname(strStrain);
        varname += " ";
        varname += cn[j];
        varname += cn[k];
        varname += "_ave ";
        p = stateMgr.registerStateVariable(varname,
            dl->cell_scalar, dl->dummy, eb_name, "scalar",
            0.0, /* save state = */ false, /* write output = */ true);
        p->set("Field Layout", dl->qp_tensor);
        p->set("Field Name", strStrain);
        p->set("Weights Layout", dl->qp_scalar);
        p->set("Weights Name", "Weights");
        p->set("component i", j);
        p->set("component j", k);
        ev = Teuchos::rcp(new PHAL::SaveCellStateField<EvalT, PHAL::AlbanyTraits>(*p));
        fm0.template registerEvaluator<EvalT>(ev);
      }

    p = stateMgr.registerStateVariable(strStrain, dl->qp_tensor, dl->dummy, eb_name,
               /* init type = */       "scalar",
               /* init value = */       0.0,
               /* registerOldState = */ true,
               /* write as output = */  false);
    ev = rcp(new PHAL::SaveStateField<EvalT,AlbanyTraits>(*p));
    fm0.template registerEvaluator<EvalT>(ev);
  }
  std::string strStrain_Updated = strStrain;          // updated state, i.e., state at N+1
  std::string strStrain_Current = strStrain+"_old";   // current state, i.e., state at N
  {
    RCP<ParameterList> p = rcp(new ParameterList("Strain"));
    p->set< RCP<DataLayout>>("Field Layout", dl->qp_tensor);
    // Input
    p->set<std::string>("Current State Name", strStrain_Current);
    p->set<std::string>("Increment Name", "Strain Increment");
    // Output
    p->set<std::string>("Updated Field Name", strStrain_Updated);
    ev = rcp(new LCM::UpdateField<EvalT,AlbanyTraits>(*p));
    fm0.template registerEvaluator<EvalT>(ev);
  }

// Compute microstrain difference
/*\begin{text}
   Register new evaluator:
  \begin{align*}
     \Delta\epsilon^{np}_{ij} = \epsilon^p_{ij} - \epsilon^{np}_{ij}\\
  \end{align*}
   \textbf{DEPENDENT FIELDS:} \\
   \begin{tabular}{l l l l}
     $\epsilon^p_{ij}$   & Macro strain                     & "Strain"          & dims(cell, p=nQPs, i=vecDim, j=spcDim) \\
     $\epsilon^n_{ijI}$  & Nodal microstrains at scale 'n'  & "Microstrain\_1"  & dims(cell, I=nNodes, i=vecDim, j=vecDim) \\
   \end{tabular} \\
  \textbf{EVALUATED FIELDS:} \\
  \begin{tabular}{l l l l}
     $\Delta\epsilon^{np}_{ij}$ & Strain Difference at scale 'n'  & "Strain Difference n"  & dims(cell, p=nQPs, i=vecDim, j=spcDim) \\
  \end{tabular} \\
\end{text}*/
  for(int i=0;i<numMicroScales;i++){
    RCP<ParameterList> p = rcp(new ParameterList("Strain Difference"));

    //Input
    p->set<std::string>("Micro Strain Name", strMicrostrains_Updated[i]);
    p->set<std::string>("Macro Strain Name", "Strain");

    //Output
    p->set<std::string>("Strain Difference Name", Albany::strint("Strain Difference",i));

    ev = rcp(new HMC::StrainDifference<EvalT,AlbanyTraits>(*p,dl));
    fm0.template registerEvaluator<EvalT>(ev);
  }
  for(int i=0;i<numMicroScales;i++){
    RCP<ParameterList> p = rcp(new ParameterList("Increment of Strain Difference"));

    //Input
    p->set<std::string>("Micro Strain Name", micro_dof_names[i][0]);
    p->set<std::string>("Macro Strain Name", "Strain Increment");

    //Output
    p->set<std::string>("Strain Difference Name", Albany::strint("Strain Difference",i)+" Increment");

    ev = rcp(new HMC::StrainDifference<EvalT,AlbanyTraits>(*p,dl));
    fm0.template registerEvaluator<EvalT>(ev);
  }


  { // Constitutive Model Parameters
    RCP<ParameterList> p = rcp(
        new ParameterList("Constitutive Model Parameters"));
    std::string matName = material_db_->getElementBlockParam<std::string>(
        eb_name, "material");
    Teuchos::ParameterList& param_list =
        material_db_->getElementBlockSublist(eb_name, matName);

    // pass through material properties
    p->set<Teuchos::ParameterList*>("Material Parameters", &param_list);

    RCP<LCM::ConstitutiveModelParameters<EvalT, AlbanyTraits>> cmpEv =
        rcp(new LCM::ConstitutiveModelParameters<EvalT, AlbanyTraits>(*p, dl));
    fm0.template registerEvaluator<EvalT>(cmpEv);
  }

// Compute stresses
/*\begin{text}
   Register new evaluator:
  \begin{align*}
   \{\sigma^p_{ij}, \bar{\beta}^{np}_{ij}, \bar{\bar{\beta}}^{np}_{ijk}\}
      = f(\{\epsilon^p_{ij}, \Delta\epsilon^{np}_{ij}, \epsilon^{np}_{ij,k}\})
  \end{align*}
   \textbf{DEPENDENT FIELDS:} \\
   \begin{tabular}{l l l l}
     $\epsilon^p_{ij}$          & Macro strain             & "Strain"                   & dims(cell, p=nQPs, i=vecDim, j=spcDim) \\
     $\Delta\epsilon^{np}_{ij}$ & Strain Difference 'n'    & "Strain Difference 1"      & dims(cell, p=nQPs, i=vecDim, j=spcDim) \\
     $\epsilon^{np}_{ij,k}$     & Microstrain gradient 'n' & "Microstrain\_n Gradient"  & dims(cell, p=nQPs, i=vecDim, j=vecDim, k=spcDim)
   \end{tabular} \\
  \textbf{EVALUATED FIELDS:} \\
  \begin{tabular}{l l l l}
     $\sigma^p_{ij}$               & Macro Stress       & "Stress"           & dims(cell, p=nQPs, i=vecDim, j=spcDim) \\
     $\bar{beta}^{np}_{ij}$        & Micro stress 'n'   & "Micro Stress n"   & dims(cell, p=nQPs, i=vecDim, j=spcDim) \\
     $\bar{\bar{beta}}^{np}_{ijk}$ & Double stress 'n'  & "Double Stress n"  & dims(cell, p=nQPs, i=vecDim, j=spcDim, k=spcDim) \\
  \end{tabular} \\
\end{text}*/
  {
    RCP<ParameterList> p = rcp(new ParameterList("Constitutive Model Interface"));
    std::string matName = material_db_->getElementBlockParam<std::string>(eb_name, "material");
    Teuchos::ParameterList& param_list = material_db_->getElementBlockSublist(eb_name, matName);

    // construct field name map
    // required
    LCM::FieldNameMap
    field_name_map(false);
    RCP<std::map<std::string, std::string>>
    fnm = field_name_map.getMap();
    param_list.set<RCP<std::map<std::string, std::string>>>("Name Map", fnm);
    p->set<Teuchos::ParameterList*>("Material Parameters", &param_list);
    // end required


    p->set<bool>("Volume Average Pressure", volume_average_pressure);
    if (volume_average_pressure) {
      p->set<std::string>("Weights Name", "Weights");
    }

    RCP<LCM::ConstitutiveModelInterface<EvalT, AlbanyTraits>> cmiEv =
        rcp(new LCM::ConstitutiveModelInterface<EvalT, AlbanyTraits>(*p, dl));
    fm0.template registerEvaluator<EvalT>(cmiEv);

    registerStateVariables(cmiEv, fm0, stateMgr, eb_name, numDim);

  }

// Compute total stress
/*\begin{text}
   Register new evaluator:
  \begin{align*}
    \sigma^{tp}_{ij} = \sigma^p_{ij} + \sum_n \bar{\beta}^{np}_{ij}
  \end{align*}
   \textbf{DEPENDENT FIELDS:} \\
   \begin{tabular}{l l l l}
     $\sigma^{tp}_{ij}$            & Total stress       & "Total Stress"     & dims(cell, p=nQPs, i=vecDim, j=spcDim) \\
   \end{tabular} \\
  \textbf{EVALUATED FIELDS:} \\
  \begin{tabular}{l l l l}
     $\sigma^p_{ij}$               & Macro Stress       & "Stress"           & dims(cell, p=nQPs, i=vecDim, j=spcDim) \\
     $\bar{beta}^{np}_{ij}$        & Micro stress 'n'   & "Micro Stress n"   & dims(cell, p=nQPs, i=vecDim, j=spcDim) \\
  \end{tabular} \\
\end{text}*/
  std::string macroKinematicFieldName = "Total Stress";
  {
    RCP<ParameterList> p = rcp(new ParameterList(macroKinematicFieldName));

    p->set<int>("Additional Scales", numMicroScales);

    //Input
    p->set<std::string>("Macro Stress Name", "Stress");
    p->set< RCP<DataLayout>>("QP 2Tensor Data Layout", dl->qp_tensor);
    for(int i=0;i<numMicroScales;i++){
      std::string ms = Albany::strint("Micro Stress",i);
      std::string msname(ms); msname += " Name";
      p->set<std::string>(msname, ms);
    }
    //Output
    p->set<std::string>("Total Stress Name", macroKinematicFieldName);

    ev = rcp(new HMC::TotalStress<EvalT,AlbanyTraits>(*p,dl));
    fm0.template registerEvaluator<EvalT>(ev);
  }

#ifdef ALBANY_ATO
  /******************************************************************************/
  /** Begin topology weighting **************************************************/
  /******************************************************************************/
  if( params->isType<Teuchos::RCP<ATO::Topology>>("Topology") )
  {
    Teuchos::RCP<ATO::Topology> topology = params->get<Teuchos::RCP<ATO::Topology>>("Topology");

    if( topology->getEntityType() == "Distributed Parameter" ){
      RCP<ParameterList> p = rcp(new ParameterList("Distributed Parameter"));
      p->set<std::string>("Parameter Name", topology->getName());
      ev = rcp(new PHAL::GatherScalarNodalParameter<EvalT,AlbanyTraits>(*p, dl) );
      fm0.template registerEvaluator<EvalT>(ev);
    }

    // macro
    RCP<ParameterList> p = rcp(new ParameterList("TopologyWeighting"));

    p->set<Teuchos::RCP<ATO::Topology>>("Topology",topology);

    Teuchos::ParameterList& wfParams = params->sublist("Apply Topology Weight Functions");

    p->set<std::string>("BF Name", "BF");
    p->set<std::string>("Unweighted Variable Name", macroKinematicFieldName);
    p->set<std::string>("Weighted Variable Name", macroKinematicFieldName+"_Weighted");
    p->set<std::string>("Variable Layout", "QP Tensor");
    p->set<int>("Function Index", wfParams.get<int>(macroKinematicFieldName));

    if( topology->getEntityType() == "Distributed Parameter" )
      ev = rcp(new ATO::TopologyFieldWeighting<EvalT,AlbanyTraits>(*p,dl));
    else
      ev = rcp(new ATO::TopologyWeighting<EvalT,AlbanyTraits>(*p,dl));

    fm0.template registerEvaluator<EvalT>(ev);

    //if(some input stuff)
    atoUtils.SaveCellStateField(fm0, stateMgr, macroKinematicFieldName+"_Weighted", eb_name, dl->qp_tensor);

    // microscales
    for(int i=0;i<numMicroScales;i++){
      RCP<ParameterList> p = rcp(new ParameterList("TopologyWeighting"));

      Teuchos::RCP<ATO::Topology> topology = params->get<Teuchos::RCP<ATO::Topology>>("Topology");
      p->set<Teuchos::RCP<ATO::Topology>>("Topology",topology);

      Teuchos::ParameterList& wfParams = params->sublist("Apply Topology Weight Functions");

      p->set<std::string>("BF Name", "BF");
      std::string ms = Albany::strint("Micro Stress",i);
      p->set<std::string>("Unweighted Variable Name", ms);
      p->set<std::string>("Weighted Variable Name", ms+"_Weighted");
      p->set<std::string>("Variable Layout", "QP Tensor");
      p->set<int>("Function Index", wfParams.get<int>("Micro Stress"));

      if( topology->getEntityType() == "Distributed Parameter" )
        ev = rcp(new ATO::TopologyFieldWeighting<EvalT,AlbanyTraits>(*p,dl));
      else
        ev = rcp(new ATO::TopologyWeighting<EvalT,AlbanyTraits>(*p,dl));

      fm0.template registerEvaluator<EvalT>(ev);

      //if(some input stuff)
      atoUtils.SaveCellStateField(fm0, stateMgr, ms+"_Weighted", eb_name, dl->qp_tensor);
    }
    for(int i=0;i<numMicroScales;i++){
      RCP<ParameterList> p = rcp(new ParameterList("TopologyWeighting"));

      Teuchos::RCP<ATO::Topology> topology = params->get<Teuchos::RCP<ATO::Topology>>("Topology");
      p->set<Teuchos::RCP<ATO::Topology>>("Topology",topology);

      Teuchos::ParameterList& wfParams = params->sublist("Apply Topology Weight Functions");

      p->set<std::string>("BF Name", "BF");
      std::string ds = Albany::strint("Double Stress",i);
      p->set<std::string>("Unweighted Variable Name", ds);
      p->set<std::string>("Weighted Variable Name", ds+"_Weighted");
      p->set<std::string>("Variable Layout", "QP Tensor3");
      p->set<int>("Function Index", wfParams.get<int>("Double Stress"));

      if( topology->getEntityType() == "Distributed Parameter" )
        ev = rcp(new ATO::TopologyFieldWeighting<EvalT,AlbanyTraits>(*p,dl));
      else
        ev = rcp(new ATO::TopologyWeighting<EvalT,AlbanyTraits>(*p,dl));

      fm0.template registerEvaluator<EvalT>(ev);

      //if(some input stuff)
      atoUtils.SaveCellStateField(fm0, stateMgr, ds+"_Weighted", eb_name, dl->qp_tensor3);
    }
  }
  /******************************************************************************/
  /** End topology weighting ****************************************************/
  /******************************************************************************/
#endif


// Compute macro residual
/*\begin{text}
   Register new evaluator:
  \begin{align*}
    f_{Ii} = \sum_p
    \frac{\partial N_I (\mathbf{x}_p)}{\partial \xi_k} J^{-1}_{kj} det\left(\frac{\partial x_{ip}}{\partial \xi_j}\right) \omega_p \sigma^{tp}_{ij}
   + \sum_p N_I(\mathbf{x}_p)\ det\left(\frac{\partial x_{ip}}{\partial \xi_j}\right) \omega_p a^p_i
  \end{align*}
   \textbf{DEPENDENT FIELDS:} \\
   \begin{tabular}{l l l l}
     $\sigma^{tp}_{ij}$ & Total stress &  "Total Stress" & dims(cell, p=nQPs, i=vecDim, j=spcDim) \\
     $\frac{\partial N_I (\mathbf{x}_p)}{\partial \xi_k} J^{-1}_{kj} det\left(\frac{\partial x_{ip}}{\partial \xi_j}\right) \omega_p$ &
     Weighted Gradient matrix wrt current config & "wGrad BF" & dims(cell, I=nNodes, p=nQPs, i=spcDim)\\
     $a^p_i$ & Acceleration at quadrature points & "Displacement\_dotdot" & dims(cell, p=nQPs, i=vecDim)\\
     $N_I(\mathbf{x}_p)\ det\left(\frac{\partial x_{ip}}{\partial \xi_j}\right) \omega_p$ &
     Weighted BF &  "wBF"  & dims(cell, I=nNodes, p=nQPs)\\
   \end{tabular} \\
  \textbf{EVALUATED FIELDS:} \\
  \begin{tabular}{l l l l}
     $f_{Ii}$ & Macroscale Residual  & "Displacement Residual"  & dims(cell, I=nNodes, i=spcDim)\\
  \end{tabular} \\
\end{text}*/
  {
    RCP<ParameterList> p = rcp(new ParameterList("Displacement Resid"));

    //Input
#ifdef ALBANY_ATO
    if( params->isType<Teuchos::RCP<ATO::Topology>>("Topology") )
      p->set<std::string>("Stress Name", macroKinematicFieldName+"_Weighted");
    else
#endif
      p->set<std::string>("Stress Name", macroKinematicFieldName);

    p->set< RCP<DataLayout>>("QP Tensor Data Layout", dl->qp_tensor);

    p->set<std::string>("Weighted Gradient BF Name", "wGrad BF");
    p->set< RCP<DataLayout>>("Node QP Vector Data Layout", dl->node_qp_vector);

    // extra input for time dependent term
    p->set<std::string>("Weighted BF Name", "wBF");
    p->set< RCP<DataLayout>>("Node QP Scalar Data Layout", dl->node_qp_scalar);
    p->set<std::string>("Time Dependent Variable Name", macro_dof_names_dotdot[0]);
    p->set< RCP<DataLayout>>("QP Vector Data Layout", dl->qp_vector);

    //Output
    p->set<std::string>("Residual Name", macro_resid_names[0]);
    p->set< RCP<DataLayout>>("Node Vector Data Layout", dl->node_vector);

    ev = rcp(new LCM::ElasticityResid<EvalT,AlbanyTraits>(*p));
    fm0.template registerEvaluator<EvalT>(ev);
  }
// Compute micro residuals
/*\begin{text}
   Register new evaluator:
  \begin{align*}
    f^n_{Iij} = \sum_p
    \frac{\partial N_I (\mathbf{x}_p)}{\partial \xi_l} J^{-1}_{lk} \bar{\bar{\beta}}^{np}_{ijk}\}
       det\left(\frac{\partial x_{ip}}{\partial \xi_j}\right) \omega_p
   + \sum_p N_I(\mathbf{x}_p) \bar{\beta}^{np}_{ij}
      \det\left(\frac{\partial x_{ip}}{\partial \xi_j}\right) \omega_p
   + \sum_p N_I(\mathbf{x}_p) \ddot{\epsilon}^{np}_{ij}
      \det\left(\frac{\partial x_{ip}}{\partial \xi_j}\right) \omega_p
  \end{align*}
   \textbf{DEPENDENT FIELDS:} \\
   \begin{tabular}{l l l l}
     $\bar{beta}^{np}_{ij}$         & Micro stress 'n'         & "Micro Stress n"         & dims(cell, p=nQPs, i=vecDim, j=spcDim) \\
     $\bar{\bar{beta}}^{np}_{ijk}$  & Double stress 'n'        & "Double Stress n"        & dims(cell, p=nQPs, i=vecDim, j=spcDim, k=spcDim) \\
     $\ddot{\epsilon}^n_{Iij}$      & micro accel at scale 'n' & "Microstrain\_n\_dotdot" & dims(cell,I=nNodes,i=vecDim,j=vecDim) \\
     $\frac{\partial N_I (\mathbf{x}_p)}{\partial \xi_l} J^{-1}_{lk} det\left(\frac{\partial x_{ip}}{\partial \xi_j}\right) \omega_p$ &
     Weighted Gradient matrix wrt current config & "wGrad BF" & dims(cell, I=nNodes, p=nQPs, i=spcDim)\\
     $N_I(\mathbf{x}_p)\ det\left(\frac{\partial x_{ip}}{\partial \xi_j}\right) \omega_p$ &
     Weighted BF &  "wBF"  & dims(cell, I=nNodes, p=nQPs)\\
   \end{tabular} \\
  \textbf{EVALUATED FIELDS:} \\
  \begin{tabular}{l l l l}
     $f^n_{Iij}$  & Residual at scale 'n'  & "Microstrain\_n Residual"  & dims(cell, I=nNodes, i=vecDim, j=vecDim)\\
  \end{tabular} \\
\end{text}*/
  for(int i=0;i<numMicroScales;i++){
    RCP<ParameterList> p = rcp(new ParameterList("Microstrain Resid"));

    //Input: Micro stresses
    std::string ms = Albany::strint("Micro Stress",i);

#ifdef ALBANY_ATO
    if( params->isType<Teuchos::RCP<ATO::Topology>>("Topology") )
      p->set<std::string>("Micro Stress Name", ms+"_Weighted");
    else
#endif
      p->set<std::string>("Micro Stress Name", ms);
    p->set< RCP<DataLayout>>("QP Tensor Data Layout", dl->qp_tensor);


    std::string ds = Albany::strint("Double Stress",i);
#ifdef ALBANY_ATO
    if( params->isType<Teuchos::RCP<ATO::Topology>>("Topology") )
      p->set<std::string>("Double Stress Name", ds+"_Weighted");
    else
#endif
      p->set<std::string>("Double Stress Name", ds);
    p->set< RCP<DataLayout>>("QP 3Tensor Data Layout", dl->qp_tensor3);

    p->set<std::string>("Weighted Gradient BF Name", "wGrad BF");
    p->set< RCP<DataLayout>>("Node QP Vector Data Layout", dl->node_qp_vector);

    p->set<std::string>("Weighted BF Name", "wBF");
    p->set< RCP<DataLayout>>("Node QP Scalar Data Layout", dl->node_qp_scalar);

    // extra input for time dependent term
    p->set<std::string>("Time Dependent Variable Name", micro_dof_names[i][0]+"_dotdot");
    p->set< RCP<DataLayout>>("QP Vector Data Layout", dl->qp_vector);

    //Output
    p->set<std::string>("Residual Name", micro_resid_names[i][0]);
    p->set< RCP<DataLayout>>("Node Tensor Data Layout", dl->node_tensor);

    ev = rcp(new HMC::MicroResidual<EvalT,AlbanyTraits>(*p));
    fm0.template registerEvaluator<EvalT>(ev);
  }

// Scatter macroscale forces
/*\begin{text}
   Register new evaluator: Scatter the nodal forces from grid based structures to solver data structures.\\
   \textbf{DEPENDENT FIELDS:} \\
   \begin{tabular}{l l l l}
     $f_{Ii}$ & Macroscale Residual  & "Displacement Residual"  & dims(cell, I=nNodes, i=vecDim) \\
   \end{tabular} \\
  \textbf{EVALUATED FIELDS:} \\
     None. \\
\end{text}*/
   fm0.template registerEvaluator<EvalT>
     (evalUtils.constructScatterResidualEvaluator(vectorRank, macro_resid_names));

// Scatter microscale forces
/*\begin{text}
   Register new evaluator: Scatter the nodal forces from grid based structures to solver data structures.\\
   \textbf{DEPENDENT FIELDS:} \\
   \begin{tabular}{l l l l}
     $f^n_{Iij}$ & Microscale Residual  & "Microstrain\_n Residual"  & dims(cell, I=nNodes, i=vecDim, j=vecDim) \\
   \end{tabular} \\
  \textbf{EVALUATED FIELDS:} \\
     None. \\
\end{text}*/
  int numTensorFields = numDim*numDim;
  int dofOffset = numDim;
  for(int i=0;i<numMicroScales;i++){ // Micro forces
    fm0.template registerEvaluator<EvalT>
      (evalUtils.constructScatterResidualEvaluator(tensorRank, micro_resid_names[i], dofOffset, micro_scatter_names[i][0]));
    dofOffset += numTensorFields;
  }

  { // Time
    RCP<ParameterList> p = rcp(new ParameterList);

    p->set<std::string>("Time Name", "Time");
    p->set<std::string>("Delta Time Name", "Delta Time");
    p->set< RCP<DataLayout>>("Workset Scalar Data Layout", dl->workset_scalar);
    p->set<RCP<ParamLib>>("Parameter Library", paramLib);
    p->set<bool>("Disable Transient", true);

    ev = rcp(new LCM::Time<EvalT,AlbanyTraits>(*p));
    fm0.template registerEvaluator<EvalT>(ev);
    p = stateMgr.registerStateVariable("Time",dl->workset_scalar, dl->dummy, eb_name, "scalar", 0.0, true);
    ev = rcp(new PHAL::SaveStateField<EvalT,AlbanyTraits>(*p));
    fm0.template registerEvaluator<EvalT>(ev);
  }


  if (fieldManagerChoice == Albany::BUILD_RESID_FM)  {
    PHX::Tag<typename EvalT::ScalarT> res_tag("Scatter", dl->dummy);
    fm0.requireField<EvalT>(res_tag);
    for(int i=0;i<numMicroScales;i++){ // Micro forces
      PHX::Tag<typename EvalT::ScalarT> res_tag(micro_scatter_names[i][0], dl->dummy);
      fm0.requireField<EvalT>(res_tag);
    }
    return res_tag.clone();
  }
  else if (fieldManagerChoice == Albany::BUILD_RESPONSE_FM) {
    Albany::ResponseUtilities<EvalT, PHAL::AlbanyTraits> respUtils(dl);
    return respUtils.constructResponses(fm0, *responseList, params, stateMgr);
  }

  return Teuchos::null;
}


/******************************************************************************/
template <typename EvalT>
void Albany::HMCProblem::registerStateVariables(
  Teuchos::RCP<LCM::ConstitutiveModelInterface<EvalT, PHAL::AlbanyTraits>> cmiEv,
  PHX::FieldManager<PHAL::AlbanyTraits>& fm0,
  Albany::StateManager& stateMgr, std::string eb_name, int numDim)
/******************************************************************************/
{
  Teuchos::RCP<Teuchos::ParameterList> p;
  Teuchos::RCP<PHX::Evaluator<PHAL::AlbanyTraits>> ev;

  // register state variables
  for (int sv(0); sv < cmiEv->getNumStateVars(); ++sv) {
    cmiEv->fillStateVariableStruct(sv);

    //
    // QUAD POINT SCALARS
    if( (cmiEv->getLayout() == dl->qp_scalar) &&
        (cmiEv->getOutputFlag() == true) ){

      // save cell average for output
      p = stateMgr.registerStateVariable(cmiEv->getName()+"_ave",
          dl->cell_scalar, dl->dummy, eb_name, cmiEv->getInitType(),
          cmiEv->getInitValue(), /* save state = */ false, /* write output = */ true);
      p->set("Field Layout", dl->qp_scalar);
      p->set("Field Name", cmiEv->getName());
      p->set("Weights Layout", dl->qp_scalar);
      p->set("Weights Name", "Weights");
      ev = Teuchos::rcp(new PHAL::SaveCellStateField<EvalT, PHAL::AlbanyTraits>(*p));
      fm0.template registerEvaluator<EvalT>(ev);

      // save state w/o output
      p = stateMgr.registerStateVariable(cmiEv->getName(),
          cmiEv->getLayout(), dl->dummy, eb_name, cmiEv->getInitType(),
          cmiEv->getInitValue(), cmiEv->getStateFlag(), /* write output = */ false);
      ev = Teuchos::rcp(new PHAL::SaveStateField<EvalT, PHAL::AlbanyTraits>(*p));
      fm0.template registerEvaluator<EvalT>(ev);

    } else

    //
    // QUAD POINT TENSORS
    if( (cmiEv->getLayout() == dl->qp_tensor) &&
        (cmiEv->getOutputFlag() == true) ){

      std::string cn[3] = {"x","y","z"};

      // save cell average for output
      for(int i=0; i< numDim; i++)
        for(int j=0; j< numDim; j++){
          std::string varname(cmiEv->getName());
          varname += " ";
          varname += cn[i];
          varname += cn[j];
          varname += "_ave ";
          p = stateMgr.registerStateVariable(varname,
              dl->cell_scalar, dl->dummy, eb_name, cmiEv->getInitType(),
              cmiEv->getInitValue(), /* save state = */ false, /* write output = */ true);
          p->set("Field Layout", dl->qp_tensor);
          p->set("Field Name", cmiEv->getName());
          p->set("Weights Layout", dl->qp_scalar);
          p->set("Weights Name", "Weights");
          p->set("component i", i);
          p->set("component j", j);
          ev = Teuchos::rcp(new PHAL::SaveCellStateField<EvalT, PHAL::AlbanyTraits>(*p));
          fm0.template registerEvaluator<EvalT>(ev);
        }

      // save state w/o output
      p = stateMgr.registerStateVariable(cmiEv->getName(),
          cmiEv->getLayout(), dl->dummy, eb_name, cmiEv->getInitType(),
          cmiEv->getInitValue(), cmiEv->getStateFlag(), /* write output = */ false);
      ev = Teuchos::rcp(new PHAL::SaveStateField<EvalT, PHAL::AlbanyTraits>(*p));
      fm0.template registerEvaluator<EvalT>(ev);

    } else

    //
    // QUAD POINT THIRD RANK TENSORS
    if( (cmiEv->getLayout() == dl->qp_tensor3) &&
        (cmiEv->getOutputFlag() == true) ){

      std::string cn[3] = {"x","y","z"};

      // save cell average for output
      for(int i=0; i< numDim; i++)
        for(int j=0; j< numDim; j++)
          for(int k=0; k< numDim; k++){
            std::string varname(cmiEv->getName());
            varname += " ";
            varname += cn[i];
            varname += cn[j];
            varname += cn[k];
            varname += "_ave ";
            p = stateMgr.registerStateVariable(varname,
                dl->cell_scalar, dl->dummy, eb_name, cmiEv->getInitType(),
                cmiEv->getInitValue(), /* save state = */ false, /* write output = */ true);
            p->set("Field Layout", dl->qp_tensor3);
            p->set("Field Name", cmiEv->getName());
            p->set("Weights Layout", dl->qp_scalar);
            p->set("Weights Name", "Weights");
            p->set("component i", i);
            p->set("component j", j);
            p->set("component k", k);
            ev = Teuchos::rcp(new PHAL::SaveCellStateField<EvalT, PHAL::AlbanyTraits>(*p));
            fm0.template registerEvaluator<EvalT>(ev);
          }

      // save state w/o output
      p = stateMgr.registerStateVariable(cmiEv->getName(),
          cmiEv->getLayout(), dl->dummy, eb_name, cmiEv->getInitType(),
          cmiEv->getInitValue(), cmiEv->getStateFlag(), /* write output = */ false);
      ev = Teuchos::rcp(new PHAL::SaveStateField<EvalT, PHAL::AlbanyTraits>(*p));
      fm0.template registerEvaluator<EvalT>(ev);

    } else {
      p = stateMgr.registerStateVariable(cmiEv->getName(),
          cmiEv->getLayout(),
          dl->dummy,
          eb_name,
          cmiEv->getInitType(),
          cmiEv->getInitValue(),
          cmiEv->getStateFlag(),
          cmiEv->getOutputFlag());
      ev = Teuchos::rcp(new PHAL::SaveStateField<EvalT, PHAL::AlbanyTraits>(*p));
      fm0.template registerEvaluator<EvalT>(ev);
    }
  }
}
#endif // ALBANY_ELASTICITYPROBLEM_HPP
