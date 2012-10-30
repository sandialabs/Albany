//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#include "FELIX_StokesFO.hpp"
#include "Albany_InitialCondition.hpp"

#include "Intrepid_FieldContainer.hpp"
#include "Intrepid_DefaultCubatureFactory.hpp"
#include "Shards_CellTopology.hpp"
#include "PHAL_FactoryTraits.hpp"
#include "Albany_Utils.hpp"
#include "Albany_ProblemUtils.hpp"
#include <string>


FELIX::StokesFO::
StokesFO( const Teuchos::RCP<Teuchos::ParameterList>& params_,
             const Teuchos::RCP<ParamLib>& paramLib_,
             const int numDim_) :
  Albany::AbstractProblem(params_, paramLib_),
  numDim(numDim_)
{
  // Get number of species equations from Problem specifications
  neq = params_->get("Number of Species", numDim);
}

FELIX::StokesFO::
~StokesFO()
{
}

void
FELIX::StokesFO::
buildProblem(
  Teuchos::ArrayRCP<Teuchos::RCP<Albany::MeshSpecsStruct> >  meshSpecs,
  Albany::StateManager& stateMgr)
{
  using Teuchos::rcp;

  cout << "In StokesFO Problem!" << endl; 

 /* Construct All Phalanx Evaluators */
  TEUCHOS_TEST_FOR_EXCEPTION(meshSpecs.size()!=1,std::logic_error,"Problem supports one Material Block");
  fm.resize(1);
  fm[0]  = rcp(new PHX::FieldManager<PHAL::AlbanyTraits>);
  buildEvaluators(*fm[0], *meshSpecs[0], stateMgr, Albany::BUILD_RESID_FM, 
		  Teuchos::null);
  constructDirichletEvaluators(*meshSpecs[0]);
  
  if(meshSpecs[0]->ssNames.size() > 0) // Build a sideset evaluator if sidesets are present
     constructNeumannEvaluators(meshSpecs[0]);
}

Teuchos::Array< Teuchos::RCP<const PHX::FieldTag> >
FELIX::StokesFO::
buildEvaluators(
  PHX::FieldManager<PHAL::AlbanyTraits>& fm0,
  const Albany::MeshSpecsStruct& meshSpecs,
  Albany::StateManager& stateMgr,
  Albany::FieldManagerChoice fmchoice,
  const Teuchos::RCP<Teuchos::ParameterList>& responseList)
{
  // Call constructeEvaluators<EvalT>(*rfm[0], *meshSpecs[0], stateMgr);
  // for each EvalT in PHAL::AlbanyTraits::BEvalTypes
  Albany::ConstructEvaluatorsOp<StokesFO> op(
    *this, fm0, meshSpecs, stateMgr, fmchoice, responseList);
  boost::mpl::for_each<PHAL::AlbanyTraits::BEvalTypes>(op);
  return *op.tags;
}

void
FELIX::StokesFO::constructDirichletEvaluators(
        const Albany::MeshSpecsStruct& meshSpecs)
{
   // Construct Dirichlet evaluators for all nodesets and names
   std::vector<string> dirichletNames(neq);
   for (int i=0; i<neq; i++) {
     std::stringstream s; s << "C" << i;
     dirichletNames[i] = s.str();
   }
   Albany::BCUtils<Albany::DirichletTraits> dirUtils;
   dfm = dirUtils.constructBCEvaluators(meshSpecs.nsNames, dirichletNames,
                                          this->params, this->paramLib);
}

// Neumann BCs
void
FELIX::StokesFO::constructNeumannEvaluators(const Teuchos::RCP<Albany::MeshSpecsStruct>& meshSpecs)
{

  cout << "setting Neumann evaluators" << endl; 
   // Note: we only enter this function if sidesets are defined in the mesh file
   // i.e. meshSpecs.ssNames.size() > 0

   Albany::BCUtils<Albany::NeumannTraits> nbcUtils;

   // Check to make sure that Neumann BCs are given in the input file

   if(!nbcUtils.haveBCSpecified(this->params)) {
      return;
   }


   // Construct BC evaluators for all side sets and names
   // Note that the string index sets up the equation offset, so ordering is important

   // Currently we aren't exactly doing this right.  I think to do this
   // correctly we need different neumann evaluators for each DOF (velocity,
   // pressure, temperature, flux) since velocity is a vector and the 
   // others are scalars.  The dof_names stuff is only used
   // for robin conditions, so at this point, as long as we don't enable
   // robin conditions, this should work.

   std::vector<string> nbcNames;
   Teuchos::RCP< Teuchos::Array<string> > dof_names = 
     Teuchos::rcp(new Teuchos::Array<string>);
   Teuchos::Array<Teuchos::Array<int> > offsets;
   int idx = 0;
     nbcNames.push_back("C0");
     offsets.push_back(Teuchos::Array<int>(1,idx++));
     if (numDim>=2) {
       nbcNames.push_back("C1");
       offsets.push_back(Teuchos::Array<int>(1,idx++));
     }
     offsets.push_back(Teuchos::Array<int>(1,idx++));
     dof_names->push_back("Velocity");
   // Construct BC evaluators for all possible names of conditions
   // Should only specify flux vector components (dudx, dudy, dudz), or dudn, not both
   std::vector<string> condNames(3); //dCdx, dCdy, dCdz, dCdn, basal

   // Note that sidesets are only supported for two and 3D currently
   if(numDim == 2)
    condNames[0] = "(dCdx, dCdy)";
   else if(numDim == 3)
    condNames[0] = "(dCdx, dCdy, dCdz)";
   else
    TEUCHOS_TEST_FOR_EXCEPTION(true, Teuchos::Exceptions::InvalidParameter,
       std::endl << "Error: Sidesets only supported in 2 and 3D." << std::endl);

   condNames[1] = "dCdn";
   
   condNames[2] = "basal";

   nfm.resize(1);

   nfm[0] = nbcUtils.constructBCEvaluators(meshSpecs, nbcNames,
                                           Teuchos::arcp(dof_names),
                                           false, 0, condNames, offsets, dl,
                                           this->params, this->paramLib);

}

Teuchos::RCP<const Teuchos::ParameterList>
FELIX::StokesFO::getValidProblemParameters() const
{
  Teuchos::RCP<Teuchos::ParameterList> validPL =
    this->getGenericProblemParams("ValidStokesFOProblemParams");

  validPL->set("Number of Species", 1, "Number of species eqs in GPAM equation set");
  validPL->sublist("FELIX Viscosity", false, "");
  validPL->sublist("Body Force", false, "");
  return validPL;
}

