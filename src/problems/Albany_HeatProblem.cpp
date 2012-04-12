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


#include "Albany_HeatProblem.hpp"
#include "Albany_InitialCondition.hpp"

#include "Intrepid_FieldContainer.hpp"
#include "Intrepid_DefaultCubatureFactory.hpp"
#include "Shards_CellTopology.hpp"
#include "PHAL_FactoryTraits.hpp"
#include "Albany_Utils.hpp"


Albany::HeatProblem::
HeatProblem( const Teuchos::RCP<Teuchos::ParameterList>& params_,
             const Teuchos::RCP<ParamLib>& paramLib_,
             const int numDim_,
             const Teuchos::RCP<const Epetra_Comm>& comm_) :
  Albany::AbstractProblem(params_, paramLib_),
  haveSource(false),
  haveAbsorption(false),
  numDim(numDim_),
  comm(comm_)
{
  this->setNumEquations(1);

  if (numDim==1) periodic = params->get("Periodic BC", false);
  else           periodic = false;
  if (periodic) *out <<" Periodic Boundary Conditions being used." <<std::endl;

  haveSource =  params->isSublist("Source Functions");
  haveAbsorption =  params->isSublist("Absorption");

  if(params->isType<string>("MaterialDB Filename")){

    std::string mtrlDbFilename = params->get<string>("MaterialDB Filename");
 // Create Material Database
    materialDB = Teuchos::rcp(new QCAD::MaterialDatabase(mtrlDbFilename, comm));

  }


}

Albany::HeatProblem::
~HeatProblem()
{
}

void
Albany::HeatProblem::
buildProblem(
  Teuchos::ArrayRCP<Teuchos::RCP<Albany::MeshSpecsStruct> >  meshSpecs,
  Albany::StateManager& stateMgr)
{
  /* Construct All Phalanx Evaluators */
  TEUCHOS_TEST_FOR_EXCEPTION(meshSpecs.size()!=1,std::logic_error,"Problem supports one Material Block");

  fm.resize(1);
  fm[0]  = Teuchos::rcp(new PHX::FieldManager<PHAL::AlbanyTraits>);
  buildEvaluators(*fm[0], *meshSpecs[0], stateMgr, BUILD_RESID_FM, 
		  Teuchos::null);

  if(meshSpecs[0]->nsNames.size() > 0) // Build a nodeset evaluator if nodesets are present

    constructDirichletEvaluators(meshSpecs[0]->nsNames);

  if(meshSpecs[0]->ssNames.size() > 0) // Build a sideset evaluator if sidesets are present

    constructNeumannEvaluators(meshSpecs[0]);

}

Teuchos::Array<Teuchos::RCP<const PHX::FieldTag> >
Albany::HeatProblem::
buildEvaluators(
  PHX::FieldManager<PHAL::AlbanyTraits>& fm0,
  const Albany::MeshSpecsStruct& meshSpecs,
  Albany::StateManager& stateMgr,
  Albany::FieldManagerChoice fmchoice,
  const Teuchos::RCP<Teuchos::ParameterList>& responseList)
{
  // Call constructEvaluators<EvalT>(*rfm[0], *meshSpecs[0], stateMgr);
  // for each EvalT in PHAL::AlbanyTraits::BEvalTypes
  ConstructEvaluatorsOp<HeatProblem> op(
    *this, fm0, meshSpecs, stateMgr, fmchoice, responseList);
  boost::mpl::for_each<PHAL::AlbanyTraits::BEvalTypes>(op);
  return *op.tags;
}

// Dirichlet BCs
void
Albany::HeatProblem::constructDirichletEvaluators(const std::vector<std::string>& nodeSetIDs)
{
   // Construct BC evaluators for all node sets and names
   std::vector<string> bcNames(neq);
   bcNames[0] = "T";
   Albany::BCUtils<Albany::DirichletTraits> bcUtils;
   dfm = bcUtils.constructBCEvaluators(nodeSetIDs, bcNames,
                                          this->params, this->paramLib);
}

// Neumann BCs
void
Albany::HeatProblem::constructNeumannEvaluators(const Teuchos::RCP<Albany::MeshSpecsStruct>& meshSpecs)
{
   // Note: we only enter this function if sidesets are defined in the mesh file
   // i.e. meshSpecs.ssNames.size() > 0

   Albany::BCUtils<Albany::NeumannTraits> bcUtils;

   // Check to make sure that Neumann BCs are given in the input file

   if(!bcUtils.haveNeumann(this->params))

      return;

   // Construct BC evaluators for all side sets and names
   // Note that the string index sets up the equation offset, so ordering is important
   std::vector<string> bcNames(neq);
   Teuchos::Array<Teuchos::Array<int> > offsets;
   offsets.resize(neq);

   bcNames[0] = "T";
   offsets[0].resize(1);
   offsets[0][0] = 0;


   // Construct BC evaluators for all possible names of conditions
   // Should only specify flux vector components (dudx, dudy, dudz), or dudn, not both
   std::vector<string> condNames(3); //dudx, dudy, dudz, dudn or scaled jump (internal surface)

   // Note that sidesets are only supported for two and 3D currently
   if(numDim == 2)
    condNames[0] = "(dudx, dudy)";
   else if(numDim == 3)
    condNames[0] = "(dudx, dudy, dudz)";
   else
    TEUCHOS_TEST_FOR_EXCEPTION(true, Teuchos::Exceptions::InvalidParameter,
       std::endl << "Error: Sidesets only supported in 2 and 3D." << std::endl);

   condNames[1] = "dudn";

   condNames[2] = "scaled jump";

   nfm.resize(1); // Heat problem only has one element block
   nfm[0] = bcUtils.constructBCEvaluators(meshSpecs, bcNames, condNames, offsets, dl, 
                                          this->params, this->paramLib, materialDB);

}

Teuchos::RCP<const Teuchos::ParameterList>
Albany::HeatProblem::getValidProblemParameters() const
{
  Teuchos::RCP<Teuchos::ParameterList> validPL =
    this->getGenericProblemParams("ValidHeatProblemParams");

  if (numDim==1)
    validPL->set<bool>("Periodic BC", false, "Flag to indicate periodic BC for 1D problems");
  validPL->sublist("Thermal Conductivity", false, "");
  validPL->set("Convection Velocity", "{0,0,0}", "");
  validPL->set<bool>("Have Rho Cp", false, "Flag to indicate if rhoCp is used");
  validPL->set<string>("MaterialDB Filename","materials.xml","Filename of material database xml file");

  return validPL;
}

