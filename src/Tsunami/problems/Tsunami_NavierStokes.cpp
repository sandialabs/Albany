//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#include "Tsunami_NavierStokes.hpp"

#include "Intrepid2_DefaultCubatureFactory.hpp"
#include "Shards_CellTopology.hpp"
#include "PHAL_FactoryTraits.hpp"
#include "Albany_Utils.hpp"
#include "Albany_BCUtils.hpp"
#include "Albany_ProblemUtils.hpp"


Tsunami::NavierStokes::
NavierStokes( const Teuchos::RCP<Teuchos::ParameterList>& params_,
             const Teuchos::RCP<ParamLib>& paramLib_,
             const int numDim_, 
             const bool haveAdvection_, 
             const bool haveUnsteady_) :
  Albany::AbstractProblem(params_, paramLib_),
  haveSource(false),
  havePSPG(true),
  haveSUPG(false),
  numDim(numDim_),
  haveAdvection(haveAdvection_),
  haveUnsteady(haveUnsteady_),
  stabType("Shakib-Hughes"),
  use_sdbcs_(false), 
  use_params_on_mesh(false), 
  mu(1.0), 
  rho(1.0) 
{

  if (params->isSublist("Tsunami Parameters")) {
    mu = params->sublist("Tsunami Parameters").get<double>("Viscosity",1.0);
    if (mu <= 0) {
      TEUCHOS_TEST_FOR_EXCEPTION(true, std::logic_error,
                              "Invalid value of Viscosity in Tsunami Problem = "
                               << mu <<"!  Viscosity must be >0.");
    }
    rho = params->sublist("Tsunami Parameters").get<double>("Density",1.0);
    if (rho <= 0) {
      TEUCHOS_TEST_FOR_EXCEPTION(true, std::logic_error,
                              "Invalid value of Density in Tsunami Problem = "
                               << rho <<"!  Density must be >0.");
    }
    haveSUPG = params->sublist("Tsunami Parameters").get<bool>("Have SUPG Stabilization", true);
    use_params_on_mesh = params->sublist("Tsunami Parameters").get<bool>("Use Parameters on Mesh", false);
    stabType = params->sublist("Tsunami Parameters").get<std::string>("Stabilization Type", "Shakib-Hughes");
    if ((stabType != "Shakib-Hughes") && (stabType != "Tsunami")) {
      TEUCHOS_TEST_FOR_EXCEPTION(true, std::logic_error,
                              "Invalid Stabilizaton Type = "
                               << stabType <<"!  Valid types are 'Shakib-Hughes' and 'Tsunami'.");
    }
  }

  if (haveAdvection == false) haveSUPG = false; 
  haveSource = true;

  // Compute number of equations
  int num_eq = 0;
  num_eq += numDim+1;
  this->setNumEquations(num_eq);

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
       << "\tHave Advection:         " << haveAdvection << std::endl
       << "\tHave Unsteady:          " << haveUnsteady << std::endl
       << "\tPSPG stabilization:     " << havePSPG << std::endl
       << "\tSUPG stabilization:     " << haveSUPG << std::endl
       << "\tStabilization type:     " << stabType << std::endl
       << "\tUse Parameters on Mesh: " << use_params_on_mesh << std::endl;
}

Tsunami::NavierStokes::
~NavierStokes()
{
}

void
Tsunami::NavierStokes::
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
Tsunami::NavierStokes::
buildEvaluators(
  PHX::FieldManager<PHAL::AlbanyTraits>& fm0,
  const Albany::MeshSpecsStruct& meshSpecs,
  Albany::StateManager& stateMgr,
  Albany::FieldManagerChoice fmchoice,
  const Teuchos::RCP<Teuchos::ParameterList>& responseList)
{
  // Call constructeEvaluators<EvalT>(*rfm[0], *meshSpecs[0], stateMgr);
  // for each EvalT in PHAL::AlbanyTraits::BEvalTypes
  Albany::ConstructEvaluatorsOp<NavierStokes> op(
    *this, fm0, meshSpecs, stateMgr, fmchoice, responseList);
  Sacado::mpl::for_each<PHAL::AlbanyTraits::BEvalTypes> fe(op);
  return *op.tags;
}

void
Tsunami::NavierStokes::constructDirichletEvaluators(
        const Albany::MeshSpecsStruct& meshSpecs)
{
   // Construct Dirichlet evaluators for all nodesets and names
   std::vector<std::string> dirichletNames(neq);
   int index = 0;
   dirichletNames[index++] = "ux";
   if (numDim>=2) dirichletNames[index++] = "uy";
   if (numDim==3) dirichletNames[index++] = "uz";
   dirichletNames[index++] = "p";
   Albany::BCUtils<Albany::DirichletTraits> dirUtils;
   dfm = dirUtils.constructBCEvaluators(meshSpecs.nsNames, dirichletNames,
                                          this->params, this->paramLib);
   use_sdbcs_ = dirUtils.useSDBCs(); 
   offsets_ = dirUtils.getOffsets();
   nodeSetIDs_ = dirUtils.getNodeSetIDs();
}

//Neumann BCs
void
Tsunami::NavierStokes::constructNeumannEvaluators(const Teuchos::RCP<Albany::MeshSpecsStruct>& meshSpecs)
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
   nbcNames.push_back("ux");
   offsets.push_back(Teuchos::Array<int>(1,idx++));
   if (numDim>=2) {
     nbcNames.push_back("uy");
     offsets.push_back(Teuchos::Array<int>(1,idx++));
   }
   if (numDim==3) {
     nbcNames.push_back("uz");
     offsets.push_back(Teuchos::Array<int>(1,idx++));
   }
   nbcNames.push_back("p");
   offsets.push_back(Teuchos::Array<int>(1,idx++));
   dof_names->push_back("Velocity");
   dof_names->push_back("Pressure");

   // Construct BC evaluators for all possible names of conditions
   // Should only specify flux vector components (dudx, dudy, dudz), or dudn, not both
   std::vector<std::string> condNames(3); //dudx, dudy, dudz, dudn, basal

   // Note that sidesets are only supported for two and 3D currently
   //
   if(numDim == 2)
    condNames[0] = "(dudx, dudy)";
   else if(numDim == 3)
    condNames[0] = "(dudx, dudy, dudz)";
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
Tsunami::NavierStokes::getValidProblemParameters() const
{
  Teuchos::RCP<Teuchos::ParameterList> validPL =
    this->getGenericProblemParams("ValidStokesParams");

  validPL->sublist("Body Force", false, "");
  validPL->sublist("Flow", false, "");
  validPL->sublist("Tsunami Parameters", false, "");

  return validPL;
}

