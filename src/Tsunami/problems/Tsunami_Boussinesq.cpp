//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#include "Tsunami_Boussinesq.hpp"

#include "Intrepid2_DefaultCubatureFactory.hpp"
#include "Shards_CellTopology.hpp"
#include "PHAL_FactoryTraits.hpp"
#include "Albany_Utils.hpp"
#include "Albany_BCUtils.hpp"
#include "Albany_ProblemUtils.hpp"



Tsunami::Boussinesq::
Boussinesq( const Teuchos::RCP<Teuchos::ParameterList>& params_,
             const Teuchos::RCP<ParamLib>& paramLib_,
             const int numDim_) : 
  Albany::AbstractProblem(params_, paramLib_),
  haveSource(false),
  numDim(numDim_),
  use_sdbcs_(false), 
  use_params_on_mesh(false), 
  h(1.0), 
  neq(5) 
{

  if (params->isSublist("Tsunami Parameters")) {
    h = params->sublist("Tsunami Parameters").get<double>("Water Depth",1.0);
    if (h <= 0) {
      TEUCHOS_TEST_FOR_EXCEPTION(true, std::logic_error,
                              "Invalid value of Water Depth in Tsunami Problem = "
                               << h <<"!  Wader depth must be >0.");
    }
    use_params_on_mesh = params->sublist("Tsunami Parameters").get<bool>("Use Parameters on Mesh", false);
  }

  haveSource = true;

  this->setNumEquations(neq);

  // Need to allocate a fields in mesh database
  if (params->isParameter("Required Fields"))
  {
    // Need to allocate a fields in mesh database
    Teuchos::Array<std::string> req = params->get<Teuchos::Array<std::string> > ("Required Fields");
    for (int i(0); i<req.size(); ++i)
      this->requirements.push_back(req[i]);
  }

  // Print out a summary of the problem
  *out << "Navier Stokes problem:" << std::endl
       << "\tSpatial dimension:      " << numDim << std::endl
       << "\tNumber of Equations:    " << neq << std::endl
       << "\tUse Parameters on Mesh: " << use_params_on_mesh << std::endl;
}

Tsunami::Boussinesq::
~Boussinesq()
{
}

void
Tsunami::Boussinesq::
buildProblem(
  Teuchos::ArrayRCP<Teuchos::RCP<Albany::MeshSpecsStruct> >  meshSpecs,
  Albany::StateManager& stateMgr)
{
  using Teuchos::rcp;

 /* Construct All Phalanx Evaluators */
  TEUCHOS_TEST_FOR_EXCEPTION(meshSpecs.size()!=1,std::logic_error,"Problem supports one Material Block");
  fm.resize(1);
  fm[0]  = rcp(new PHX::FieldManager<PHAL::AlbanyTraits>);
  elementBlockName = meshSpecs[0]->ebName;
  buildEvaluators(*fm[0], *meshSpecs[0], stateMgr, Albany::BUILD_RESID_FM,
      Teuchos::null);
  constructDirichletEvaluators(*meshSpecs[0]);
  //construct Neumann evaluators
  constructNeumannEvaluators(meshSpecs[0]);
}

Teuchos::Array< Teuchos::RCP<const PHX::FieldTag> >
Tsunami::Boussinesq::
buildEvaluators(
  PHX::FieldManager<PHAL::AlbanyTraits>& fm0,
  const Albany::MeshSpecsStruct& meshSpecs,
  Albany::StateManager& stateMgr,
  Albany::FieldManagerChoice fmchoice,
  const Teuchos::RCP<Teuchos::ParameterList>& responseList)
{
  // Call constructeEvaluators<EvalT>(*rfm[0], *meshSpecs[0], stateMgr);
  // for each EvalT in PHAL::AlbanyTraits::BEvalTypes
  Albany::ConstructEvaluatorsOp<Boussinesq> op(
    *this, fm0, meshSpecs, stateMgr, fmchoice, responseList);
  Sacado::mpl::for_each<PHAL::AlbanyTraits::BEvalTypes> fe(op);
  return *op.tags;
}

void
Tsunami::Boussinesq::constructDirichletEvaluators(
        const Albany::MeshSpecsStruct& meshSpecs)
{
   // Construct Dirichlet evaluators for all nodesets and names
   std::vector<std::string> dirichletNames(neq);
   int index = 0;
   dirichletNames[index++] = "eta";
   dirichletNames[index++] = "ualpha";
   dirichletNames[index++] = "valpha";
   dirichletNames[index++] = "E1";
   dirichletNames[index++] = "E2";
   Albany::BCUtils<Albany::DirichletTraits> dirUtils;
   dfm = dirUtils.constructBCEvaluators(meshSpecs.nsNames, dirichletNames,
                                          this->params, this->paramLib);
   use_sdbcs_ = dirUtils.useSDBCs(); 
   offsets_ = dirUtils.getOffsets();
   nodeSetIDs_ = dirUtils.getNodeSetIDs();
}

//Neumann BCs
void
Tsunami::Boussinesq::constructNeumannEvaluators(const Teuchos::RCP<Albany::MeshSpecsStruct>& meshSpecs)
{

   // Note: we only enter this function if sidesets are defined in the mesh file
   // i.e. meshSpecs.ssNames.size() > 0

    Albany::BCUtils<Albany::NeumannTraits> nbcUtils;

   // Check to make sure that Neumann BCs are given in the input file

   if(!nbcUtils.haveBCSpecified(this->params)) {
      return;
   }

   // Construct BC evaluators for all side sets and names
   // Note that the string index sets up the equation offset, so ordering is important
   //
   // Currently we aren't exactly doing this right.  I think to do this
   // correctly we need different neumann evaluators for each DOF (velocity,
   // pressure, temperature, flux) since velocity is a vector and the
   // others are scalars.  The dof_names stuff is only used
   // for robin conditions, so at this point, as long as we don't enable
   // robin conditions, this should work.

   std::vector<std::string> nbcNames;
   Teuchos::RCP< Teuchos::Array<std::string> > dof_names =
     Teuchos::rcp(new Teuchos::Array<std::string>);
   Teuchos::Array<Teuchos::Array<int> > offsets;
   int idx = 0;
   nbcNames.push_back("eta");
   offsets.push_back(Teuchos::Array<int>(1,idx++));
   nbcNames.push_back("ualpha");
   offsets.push_back(Teuchos::Array<int>(1,idx++));
   nbcNames.push_back("valpha");
   offsets.push_back(Teuchos::Array<int>(1,idx++));
   nbcNames.push_back("E1");
   offsets.push_back(Teuchos::Array<int>(1,idx++));
   nbcNames.push_back("E2");
   offsets.push_back(Teuchos::Array<int>(1,idx++));

   // Construct BC evaluators for all possible names of conditions
   // Should only specify flux vector components (dudx, dudy, dudz), or dudn, not both
   std::vector<std::string> condNames(3); //dudx, dudy, dudz, dudn, basal

   // Note that sidesets are only supported for two and 3D currently
   //
   //IKT, FIXME: the following needs to be changed for Tsunami problem! 
   if(numDim == 2)
    condNames[0] = "(dudx, dudy)";
   else
    TEUCHOS_TEST_FOR_EXCEPTION(true, Teuchos::Exceptions::InvalidParameter,
       std::endl << "Error: Sidesets only supported in 2 and 3D." << std::endl);

   condNames[1] = "dudn";

   condNames[2] = "basal";

   nfm.resize(1);


   nfm[0] = nbcUtils.constructBCEvaluators(meshSpecs, nbcNames,
                                           Teuchos::arcp(dof_names),
                                           true, 0, condNames, offsets, dl,
                                           this->params, this->paramLib);
}



Teuchos::RCP<const Teuchos::ParameterList>
Tsunami::Boussinesq::getValidProblemParameters() const
{
  Teuchos::RCP<Teuchos::ParameterList> validPL =
    this->getGenericProblemParams("ValidStokesParams");

  validPL->sublist("Body Force", false, "");
  validPL->sublist("Flow", false, "");
  validPL->sublist("Tsunami Parameters", false, "");

  return validPL;
}

