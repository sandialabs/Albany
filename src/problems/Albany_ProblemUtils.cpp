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

#include "Albany_ProblemUtils.hpp"
#include "Albany_DataTypes.hpp"

#include "PHAL_FactoryTraits.hpp"
#ifdef ALBANY_LCM       
#include "LCM_FactoryTraits.hpp"
#endif

#include "Intrepid_HGRAD_LINE_Cn_FEM.hpp"


Albany::Layouts::Layouts (int worksetSize, int  numVertices,
                          int numNodes, int numQPts, int numDim)
{
  using Teuchos::rcp;
  using PHX::MDALayout;
  
  // Solution Fields
  node_scalar = rcp(new MDALayout<Cell,Node>(worksetSize,numNodes));
  qp_scalar   = rcp(new MDALayout<Cell,QuadPoint>(worksetSize,numQPts));
  cell_scalar = rcp(new MDALayout<Cell,QuadPoint>(worksetSize,1));

  node_vector = rcp(new MDALayout<Cell,Node,Dim>(worksetSize,numNodes,numDim));
  qp_vector   = rcp(new MDALayout<Cell,QuadPoint,Dim>(worksetSize,numQPts,numDim));

  node_tensor = rcp(new MDALayout<Cell,Node,Dim,Dim>(worksetSize,numNodes,numDim,numDim));
  qp_tensor   = rcp(new MDALayout<Cell,QuadPoint,Dim,Dim>(worksetSize,numQPts,numDim,numDim));

  // Coordinates
  vertices_vector = rcp(new MDALayout<Cell,Vertex, Dim>(worksetSize,numVertices,numDim));

  // Basis Functions
  node_qp_scalar = rcp(new MDALayout<Cell,Node,QuadPoint>(worksetSize,numNodes, numQPts));
  node_qp_vector = rcp(new MDALayout<Cell,Node,QuadPoint,Dim>(worksetSize,numNodes, numQPts,numDim));

  dummy = rcp(new MDALayout<Dummy>(0));
}

/*************************************************************************/

Albany::ProblemUtils::ProblemUtils(
     Teuchos::RCP<Albany::Layouts> dl_, std::string facTraits_) :
     dl(dl_), facTraits(facTraits_)
{
   TEST_FOR_EXCEPTION(facTraits!="PHAL" && facTraits!="LCM", std::logic_error,
       "ProblemUtils constructor: unrecognized  facTraits flag: "<< facTraits);
}


Teuchos::RCP<Teuchos::ParameterList>
Albany::ProblemUtils::constructGatherSolutionEvaluator(
       bool isVectorField,
       Teuchos::ArrayRCP<std::string> dof_names,
       Teuchos::ArrayRCP<std::string> dof_names_dot, 
       int offsetToFirstDOF)
{
    using Teuchos::RCP;
    using PHX::DataLayout;
    using Teuchos::ParameterList;
    if (facTraits=="PHAL")     type = PHAL::FactoryTraits<PHAL::AlbanyTraits>::id_gather_solution;
#ifdef ALBANY_LCM       
    else if (facTraits=="LCM") type =  LCM::FactoryTraits<PHAL::AlbanyTraits>::id_gather_solution;
#endif

    RCP<ParameterList> p = rcp(new ParameterList("Gather Solution"));
    p->set<int>("Type", type);
    p->set< Teuchos::ArrayRCP<std::string> >("Solution Names", dof_names);

    p->set<bool>("Vector Field", isVectorField);
    if (isVectorField) p->set< RCP<DataLayout> >("Data Layout", dl->node_vector);
    else               p->set< RCP<DataLayout> >("Data Layout", dl->node_scalar);

    p->set<int>("Offset of First DOF", offsetToFirstDOF);

    p->set< Teuchos::ArrayRCP<std::string> >("Time Dependent Solution Names", dof_names_dot);
    return p;
}

Teuchos::RCP<Teuchos::ParameterList>
Albany::ProblemUtils::constructGatherSolutionEvaluator_noTransient(
       bool isVectorField,
       Teuchos::ArrayRCP<std::string> dof_names,
       int offsetToFirstDOF)
{
    using Teuchos::RCP;
    using PHX::DataLayout;
    using Teuchos::ParameterList;
    if (facTraits=="PHAL")     type = PHAL::FactoryTraits<PHAL::AlbanyTraits>::id_gather_solution;
#ifdef ALBANY_LCM       
    else if (facTraits=="LCM") type =  LCM::FactoryTraits<PHAL::AlbanyTraits>::id_gather_solution;
#endif

    RCP<ParameterList> p = rcp(new ParameterList("Gather Solution"));
    p->set<int>("Type", type);
    p->set< Teuchos::ArrayRCP<std::string> >("Solution Names", dof_names);

    p->set<bool>("Vector Field", isVectorField);
    if (isVectorField) p->set< RCP<DataLayout> >("Data Layout", dl->node_vector);
    else               p->set< RCP<DataLayout> >("Data Layout", dl->node_scalar);

    p->set<int>("Offset of First DOF", offsetToFirstDOF);
    p->set<bool>("Disable Transient", true);

    return p;
}


Teuchos::RCP<Teuchos::ParameterList>
Albany::ProblemUtils::constructScatterResidualEvaluator(
       bool isVectorField,
       Teuchos::ArrayRCP<std::string> resid_names,
       int offsetToFirstDOF, std::string scatterName)
{
    using Teuchos::RCP;
    using PHX::DataLayout;
    using Teuchos::ParameterList;
    if (facTraits=="PHAL")     type = PHAL::FactoryTraits<PHAL::AlbanyTraits>::id_scatter_residual;
#ifdef ALBANY_LCM       
    else if (facTraits=="LCM") type =  LCM::FactoryTraits<PHAL::AlbanyTraits>::id_scatter_residual;
#endif

    RCP<ParameterList> p = rcp(new ParameterList("Scatter Residual"));
    p->set<int>("Type", type);
    p->set< Teuchos::ArrayRCP<std::string> >("Residual Names", resid_names);

    p->set<bool>("Vector Field", isVectorField);
    if (isVectorField) p->set< RCP<DataLayout> >("Data Layout", dl->node_vector);
    else               p->set< RCP<DataLayout> >("Data Layout", dl->node_scalar);

    p->set< RCP<DataLayout> >("Dummy Data Layout", dl->dummy);
    p->set<int>("Offset of First DOF", offsetToFirstDOF);
    p->set<string>("Scatter Field Name", scatterName);

    return p;
}

Teuchos::RCP<Teuchos::ParameterList>
Albany::ProblemUtils::constructGatherCoordinateVectorEvaluator()
{
    using Teuchos::RCP;
    using PHX::DataLayout;
    using Teuchos::ParameterList;
    if (facTraits=="PHAL")     type = PHAL::FactoryTraits<PHAL::AlbanyTraits>::id_gather_coordinate_vector;
#ifdef ALBANY_LCM       
    else if (facTraits=="LCM") type =  LCM::FactoryTraits<PHAL::AlbanyTraits>::id_gather_coordinate_vector;
#endif

    RCP<ParameterList> p = rcp(new ParameterList("Gather Coordinate Vector"));
    p->set<int> ("Type", type);

    // Input: Periodic BC flag
    p->set<bool>("Periodic BC", false);
 
    // Output:: Coordindate Vector at vertices
    p->set< RCP<DataLayout> >  ("Coordinate Data Layout",  dl->vertices_vector);
    p->set< string >("Coordinate Vector Name", "Coord Vec");
 
    return p;
}

Teuchos::RCP<Teuchos::ParameterList>
Albany::ProblemUtils::constructMapToPhysicalFrameEvaluator(
    const Teuchos::RCP<shards::CellTopology>& cellType,
    const Teuchos::RCP<Intrepid::Cubature<RealType> > cubature)
{
    using Teuchos::RCP;
    using PHX::DataLayout;
    using Teuchos::ParameterList;
    if (facTraits=="PHAL")     type = PHAL::FactoryTraits<PHAL::AlbanyTraits>::id_map_to_physical_frame;
#ifdef ALBANY_LCM       
    else if (facTraits=="LCM") type =  LCM::FactoryTraits<PHAL::AlbanyTraits>::id_map_to_physical_frame;
#endif

    RCP<ParameterList> p = rcp(new ParameterList("Map To Physical Frame"));
    p->set<int>   ("Type", type);
 
    // Input: X, Y at vertices
    p->set< string >("Coordinate Vector Name", "Coord Vec");
    p->set< RCP<DataLayout> >("Coordinate Data Layout", dl->vertices_vector);
 
    p->set<RCP <Intrepid::Cubature<RealType> > >("Cubature", cubature);
    p->set<RCP<shards::CellTopology> >("Cell Type", cellType);
 
    // Output: X, Y at Quad Points (same name as input)
    p->set< RCP<DataLayout> >("QP Vector Data Layout", dl->qp_vector);
 
    return p;
}

Teuchos::RCP<Teuchos::ParameterList>
Albany::ProblemUtils::constructComputeBasisFunctionsEvaluator(
    const Teuchos::RCP<shards::CellTopology>& cellType,
    const Teuchos::RCP<Intrepid::Basis<RealType, Intrepid::FieldContainer<RealType> > > intrepidBasis,
    const Teuchos::RCP<Intrepid::Cubature<RealType> > cubature)
{
    using Teuchos::RCP;
    using PHX::DataLayout;
    using Teuchos::ParameterList;
    if (facTraits=="PHAL")     type = PHAL::FactoryTraits<PHAL::AlbanyTraits>::id_compute_basis_functions;
#ifdef ALBANY_LCM       
    else if (facTraits=="LCM") type =  LCM::FactoryTraits<PHAL::AlbanyTraits>::id_compute_basis_functions;
#endif

    RCP<ParameterList> p = rcp(new ParameterList("Compute Basis Functions"));
    p->set<int>   ("Type", type);

    // Inputs: X, Y at nodes, Cubature, and Basis
    p->set<string>("Coordinate Vector Name","Coord Vec");
    p->set< RCP<DataLayout> >("Coordinate Data Layout", dl->vertices_vector);
    p->set< RCP<Intrepid::Cubature<RealType> > >("Cubature", cubature);
 
    p->set< RCP<Intrepid::Basis<RealType, Intrepid::FieldContainer<RealType> > > >
        ("Intrepid Basis", intrepidBasis);
 
    p->set<RCP<shards::CellTopology> >("Cell Type", cellType);
    // Outputs: BF, weightBF, Grad BF, weighted-Grad BF, all in physical space
     p->set<string>("Weights Name",          "Weights");
    p->set< RCP<DataLayout> >("QP Scalar Data Layout", dl->qp_scalar);
    p->set<string>("BF Name",          "BF");
    p->set<string>("Weighted BF Name", "wBF");
    p->set< RCP<DataLayout> >("Node QP Scalar Data Layout", dl->node_qp_scalar);
 
    p->set<string>("Gradient BF Name",          "Grad BF");
    p->set<string>("Weighted Gradient BF Name", "wGrad BF");
    p->set< RCP<DataLayout> >("Node QP Vector Data Layout", dl->node_qp_vector);

    return p;
}

Teuchos::RCP<Teuchos::ParameterList>
Albany::ProblemUtils::constructDOFInterpolationEvaluator(
       std::string& dof_name)
{
    using Teuchos::RCP;
    using PHX::DataLayout;
    using Teuchos::ParameterList;
    if (facTraits=="PHAL")     type = PHAL::FactoryTraits<PHAL::AlbanyTraits>::id_dof_interpolation;
#ifdef ALBANY_LCM       
    else if (facTraits=="LCM") type =  LCM::FactoryTraits<PHAL::AlbanyTraits>::id_dof_interpolation;
#endif

    RCP<ParameterList> p = rcp(new ParameterList("DOF Interpolation "+dof_name));
    p->set<int>   ("Type", type);
    // Input
    p->set<string>("Variable Name", dof_name);
    p->set< RCP<DataLayout> >("Node Data Layout", dl->node_scalar);

    p->set<string>("BF Name", "BF");
    p->set< RCP<DataLayout> >("Node QP Scalar Data Layout", dl->node_qp_scalar);

    // Output (assumes same Name as input)
    p->set< RCP<DataLayout> >("QP Scalar Data Layout", dl->qp_scalar);

    return p;
}

Teuchos::RCP<Teuchos::ParameterList>
Albany::ProblemUtils::constructDOFGradInterpolationEvaluator(
       std::string& dof_name)
{
    using Teuchos::RCP;
    using PHX::DataLayout;
    using Teuchos::ParameterList;
    if (facTraits=="PHAL")     type = PHAL::FactoryTraits<PHAL::AlbanyTraits>::id_dof_grad_interpolation;
#ifdef ALBANY_LCM       
    else if (facTraits=="LCM") type =  LCM::FactoryTraits<PHAL::AlbanyTraits>::id_dof_grad_interpolation;
#endif

    RCP<ParameterList> p = rcp(new ParameterList("DOF Interpolation "+dof_name));
    p->set<int>   ("Type", type);
    // Input
    p->set<string>("Variable Name", dof_name);
    p->set< RCP<DataLayout> >("Node Data Layout", dl->node_scalar);

    p->set<string>("Gradient BF Name", "Grad BF");
    p->set< RCP<DataLayout> >("Node QP Vector Data Layout", dl->node_qp_vector);

    // Output (assumes same Name as input)
    p->set<string>("Gradient Variable Name", dof_name+" Gradient");
    p->set< RCP<DataLayout> >("QP Vector Data Layout", dl->qp_vector);

    return p;
}

Teuchos::RCP<Teuchos::ParameterList>
Albany::ProblemUtils::constructDOFVecInterpolationEvaluator(
       std::string& dof_name)
{
    using Teuchos::RCP;
    using PHX::DataLayout;
    using Teuchos::ParameterList;
    if (facTraits=="PHAL")     type = PHAL::FactoryTraits<PHAL::AlbanyTraits>::id_dofvec_interpolation;
#ifdef ALBANY_LCM       
    else if (facTraits=="LCM") type =  LCM::FactoryTraits<PHAL::AlbanyTraits>::id_dofvec_interpolation;
#endif

    RCP<ParameterList> p = rcp(new ParameterList("DOFVec Interpolation "+dof_name));
    p->set<int>   ("Type", type);
    // Input
    p->set<string>("Variable Name", dof_name);
    p->set< RCP<DataLayout> >("Node Vector Data Layout", dl->node_vector);

    p->set<string>("BF Name", "BF");
    p->set< RCP<DataLayout> >("Node QP Scalar Data Layout", dl->node_qp_scalar);

    // Output (assumes same Name as input)
    p->set< RCP<DataLayout> >("QP Vector Data Layout", dl->qp_vector);

    return p;
}

Teuchos::RCP<Teuchos::ParameterList>
Albany::ProblemUtils::constructDOFVecGradInterpolationEvaluator(
       std::string& dof_name)
{
    using Teuchos::RCP;
    using PHX::DataLayout;
    using Teuchos::ParameterList;
    if (facTraits=="PHAL")     type = PHAL::FactoryTraits<PHAL::AlbanyTraits>::id_dofvec_grad_interpolation;
#ifdef ALBANY_LCM       
    else if (facTraits=="LCM") type =  LCM::FactoryTraits<PHAL::AlbanyTraits>::id_dofvec_grad_interpolation;
#endif

    RCP<ParameterList> p = rcp(new ParameterList("DOFVecGrad Interpolation "+dof_name));
    p->set<int>   ("Type", type);
    // Input
    p->set<string>("Variable Name", dof_name);
    p->set< RCP<DataLayout> >("Node Vector Data Layout", dl->node_vector);

    p->set<string>("Gradient BF Name", "Grad BF");
    p->set< RCP<DataLayout> >("Node QP Vector Data Layout", dl->node_qp_vector);

    // Output (assumes same Name as input)
    p->set<string>("Gradient Variable Name", dof_name+" Gradient");
    p->set< RCP<DataLayout> >("QP Tensor Data Layout", dl->qp_tensor);

    return p;
}

Teuchos::RCP<PHX::FieldManager<PHAL::AlbanyTraits> > 
Albany::ProblemUtils::constructDirichletEvaluators(
      const std::vector<std::string>& nodeSetIDs,
      const std::vector<std::string>& dirichletNames,
      Teuchos::RCP<Teuchos::ParameterList> params,
      Teuchos::RCP<ParamLib> paramLib)
{
   using Teuchos::RCP;
   using Teuchos::rcp;
   using Teuchos::ParameterList;
   using PHX::DataLayout;
   using PHX::MDALayout;
   using std::vector;
   using std::map;
   using std::string;

   using PHAL::AlbanyTraits;

   type = PHAL::DirichletFactoryTraits<AlbanyTraits>::id_dirichlet;
   
   Teuchos::ParameterList DBCparams = params->sublist("Dirichlet BCs");
   DBCparams.validateParameters(*(getValidDirichletBCParameters(nodeSetIDs,dirichletNames)),0);

   map<string, RCP<ParameterList> > evaluators_to_build;
   RCP<DataLayout> dummy = rcp(new MDALayout<Dummy>(0));
   vector<string> dbcs;

   // Check for all possible standard BCs (every dof on every nodeset) to see which is set
   for (std::size_t i=0; i<nodeSetIDs.size(); i++) {
     for (std::size_t j=0; j<dirichletNames.size(); j++) {
       std::string ss = constructDBCName(nodeSetIDs[i],dirichletNames[j]);

       if (DBCparams.isParameter(ss)) {
         RCP<ParameterList> p = rcp(new ParameterList);
         p->set<int>("Type", type);

         p->set< RCP<DataLayout> >("Data Layout", dummy);
         p->set< string >  ("Dirichlet Name", ss);
         p->set< RealType >("Dirichlet Value", DBCparams.get<double>(ss));
         p->set< string >  ("Node Set ID", nodeSetIDs[i]);
        // p->set< int >     ("Number of Equations", dirichletNames.size());
	 p->set< int >     ("Equation Offset", j);

         p->set<RCP<ParamLib> >("Parameter Library", paramLib);

         std::stringstream ess; ess << "Evaluator for " << ss;
         evaluators_to_build[ess.str()] = p;

         dbcs.push_back(ss);
       }
     }
   }

   for (std::size_t i=0; i<nodeSetIDs.size(); i++) 
   {
     std::string ss = constructDBCName(nodeSetIDs[i],"K");
     
     if (DBCparams.isSublist(ss)) 
     {
       // grab the sublist
       ParameterList& sub_list = DBCparams.sublist(ss);

       // Only for LCM problems, but no harm in leaving off ifdefs
       if (sub_list.get<string>("BC Function") == "Kfield" )
       {
	 RCP<ParameterList> p = rcp(new ParameterList);
	 type = PHAL::DirichletFactoryTraits<AlbanyTraits>::id_kfield_bc;
	 p->set<int>("Type", type);

	 // This BC needs a shear modulus and poissons ratio defined
	 TEST_FOR_EXCEPTION(!params->isSublist("Shear Modulus"), 
			    Teuchos::Exceptions::InvalidParameter, 
			    "This BC needs a Shear Modulus");
	 ParameterList& shmd_list = params->sublist("Shear Modulus");
	 TEST_FOR_EXCEPTION(!(shmd_list.get("Shear Modulus Type","") == "Constant"), 
			    Teuchos::Exceptions::InvalidParameter,
			    "Invalid Shear Modulus type");
	 p->set< RealType >("Shear Modulus", shmd_list.get("Value", 1.0));

	 TEST_FOR_EXCEPTION(!params->isSublist("Poissons Ratio"), 
			    Teuchos::Exceptions::InvalidParameter, 
			    "This BC needs a Poissons Ratio");
	 ParameterList& pr_list = params->sublist("Poissons Ratio");
	 TEST_FOR_EXCEPTION(!(pr_list.get("Poissons Ratio Type","") == "Constant"), 
			    Teuchos::Exceptions::InvalidParameter,
			    "Invalid Poissons Ratio type");
	 p->set< RealType >("Poissons Ratio", pr_list.get("Value", 1.0));

	 // Extract BC parameters
	 p->set< string >("Kfield KI Name", "Kfield KI");
	 p->set< string >("Kfield KII Name", "Kfield KII");
	 p->set< RealType >("KI Value", sub_list.get<double>("Kfield KI"));
	 p->set< RealType >("KII Value", sub_list.get<double>("Kfield KII"));

	 // Fill up ParameterList with things DirichletBase wants
	 p->set< RCP<DataLayout> >("Data Layout", dummy);
	 p->set< string >  ("Dirichlet Name", ss);
         p->set< RealType >("Dirichlet Value", 0.0);
	 p->set< string >  ("Node Set ID", nodeSetIDs[i]);
         //p->set< int >     ("Number of Equations", dirichletNames.size());
	 p->set< int >     ("Equation Offset", 0);
	 
	 p->set<RCP<ParamLib> >("Parameter Library", paramLib);
	 std::stringstream ess; ess << "Evaluator for " << ss;
	 evaluators_to_build[ess.str()] = p;

	 dbcs.push_back(ss);
       }
     }
   }

   string allDBC="Evaluator for all Dirichlet BCs";
   {
      RCP<ParameterList> p = rcp(new ParameterList);
      type = PHAL::DirichletFactoryTraits<AlbanyTraits>::id_dirichlet_aggregator;
      p->set<int>("Type", type);

      p->set<vector<string>* >("DBC Names", &dbcs);
      p->set< RCP<DataLayout> >("Data Layout", dummy);
      p->set<string>("DBC Aggregator Name", allDBC);
      evaluators_to_build[allDBC] = p;
   }

   // Build Field Evaluators for each evaluation type
   PHX::EvaluatorFactory<AlbanyTraits,PHAL::DirichletFactoryTraits<AlbanyTraits> > factory;
   RCP< vector< RCP<PHX::Evaluator_TemplateManager<AlbanyTraits> > > > evaluators;
   evaluators = factory.buildEvaluators(evaluators_to_build);

   // Create a DirichletFieldManager
   Teuchos::RCP<PHX::FieldManager<PHAL::AlbanyTraits> > dfm
     = Teuchos::rcp(new PHX::FieldManager<AlbanyTraits>);

   // Register all Evaluators
   PHX::registerEvaluators(evaluators, *dfm);

   PHX::Tag<AlbanyTraits::Residual::ScalarT> res_tag0(allDBC, dummy);
   dfm->requireField<AlbanyTraits::Residual>(res_tag0);

   PHX::Tag<AlbanyTraits::Jacobian::ScalarT> jac_tag0(allDBC, dummy);
   dfm->requireField<AlbanyTraits::Jacobian>(jac_tag0);

   PHX::Tag<AlbanyTraits::Tangent::ScalarT> tan_tag0(allDBC, dummy);
   dfm->requireField<AlbanyTraits::Tangent>(tan_tag0);

   PHX::Tag<AlbanyTraits::SGResidual::ScalarT> sgres_tag0(allDBC, dummy);
   dfm->requireField<AlbanyTraits::SGResidual>(sgres_tag0);

   PHX::Tag<AlbanyTraits::SGJacobian::ScalarT> sgjac_tag0(allDBC, dummy);
   dfm->requireField<AlbanyTraits::SGJacobian>(sgjac_tag0);

   PHX::Tag<AlbanyTraits::MPResidual::ScalarT> mpres_tag0(allDBC, dummy);
   dfm->requireField<AlbanyTraits::MPResidual>(mpres_tag0);

   PHX::Tag<AlbanyTraits::MPJacobian::ScalarT> mpjac_tag0(allDBC, dummy);
   dfm->requireField<AlbanyTraits::MPJacobian>(mpjac_tag0);

   return dfm;
}

Teuchos::RCP<const Teuchos::ParameterList>
Albany::ProblemUtils::getValidDirichletBCParameters(
  const std::vector<std::string>& nodeSetIDs,
  const std::vector<std::string>& dirichletNames) const
{
  Teuchos::RCP<Teuchos::ParameterList> validPL =
     Teuchos::rcp(new Teuchos::ParameterList("Valid Dirichlet BC List"));;

  for (std::size_t i=0; i<nodeSetIDs.size(); i++) {
    for (std::size_t j=0; j<dirichletNames.size(); j++) {
      std::string ss = constructDBCName(nodeSetIDs[i],dirichletNames[j]);
      validPL->set<double>(ss, 0.0, "Value of BC corresponding to nodeSetID and dofName");
    }
  }
  
  for (std::size_t i=0; i<nodeSetIDs.size(); i++) 
  {
    std::string ss = constructDBCName(nodeSetIDs[i],"K");
    validPL->sublist(ss, false, "");
  }

  return validPL;
}

std::string
Albany::ProblemUtils::constructDBCName(const std::string ns,
                                       const std::string dof) const
{
  std::stringstream ss; ss << "DBC on NS " << ns << " for DOF " << dof;
  return ss.str();
}

Teuchos::RCP<Intrepid::Basis<RealType, Intrepid::FieldContainer<RealType> > >
Albany::getIntrepidBasis(const CellTopologyData& ctd, bool compositeTet)
{
   using Teuchos::rcp;
   using Intrepid::FieldContainer;
   Teuchos::RCP<Intrepid::Basis<RealType, FieldContainer<RealType> > > intrepidBasis;
   const int& numNodes = ctd.node_count;
   const int& numDim = ctd.dimension;
   std::string name = ctd.name;

   cout << "CellTopology is " << name << " with nodes " << numNodes << "  dim " << numDim << endl;

   if (name == "Line_2" )
       intrepidBasis = rcp(new Intrepid::Basis_HGRAD_LINE_C1_FEM<RealType, FieldContainer<RealType> >() );
// No HGRAD_LINE_C2 in Intrepid
   else if (name == "Line_3" )
       intrepidBasis = rcp(new Intrepid::Basis_HGRAD_LINE_Cn_FEM<RealType, FieldContainer<RealType> >(2, Intrepid::POINTTYPE_EQUISPACED) );
   else if (name == "Triangle_3" )
         intrepidBasis = rcp(new Intrepid::Basis_HGRAD_TRI_C1_FEM<RealType, FieldContainer<RealType> >() );
   else if (name == "Triangle_6" )
         intrepidBasis = rcp(new Intrepid::Basis_HGRAD_TRI_C2_FEM<RealType, FieldContainer<RealType> >() );
   else if (name == "Quadrilateral_4" )
         intrepidBasis = rcp(new Intrepid::Basis_HGRAD_QUAD_C1_FEM<RealType, FieldContainer<RealType> >() );
   else if (name == "Quadrilateral_9" )
         intrepidBasis = rcp(new Intrepid::Basis_HGRAD_QUAD_C2_FEM<RealType, FieldContainer<RealType> >() );
   else if (name == "Hexahedron_8" )
         intrepidBasis = rcp(new Intrepid::Basis_HGRAD_HEX_C1_FEM<RealType, FieldContainer<RealType> >() );
   else if (name == "Hexahedron_27" )
         intrepidBasis = rcp(new Intrepid::Basis_HGRAD_HEX_C2_FEM<RealType, FieldContainer<RealType> >() );
   else if (name == "Tetrahedron_4" )
           intrepidBasis = rcp(new Intrepid::Basis_HGRAD_TET_C1_FEM<RealType, FieldContainer<RealType> >() );
   else if (name == "Tetrahedron_10" && !compositeTet )
           intrepidBasis = rcp(new Intrepid::Basis_HGRAD_TET_C2_FEM<RealType, FieldContainer<RealType> >() );
   else if (name == "Tetrahedron_10" && compositeTet )
           intrepidBasis = rcp(new Intrepid::Basis_HGRAD_TET_COMP12_FEM<RealType, FieldContainer<RealType> >() );
   else
     TEST_FOR_EXCEPTION( //JTO compiler doesn't like this --> ctd.name != "Recognized Element Name", 
			true,
			Teuchos::Exceptions::InvalidParameter,
			"Albany::ProblemUtils::getIntrepidBasis did not recognize element name: "
			<< ctd.name);

   return intrepidBasis;
}
