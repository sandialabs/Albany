//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#include "Aeras_ShallowWaterProblem.hpp"

#include "Intrepid_FieldContainer.hpp"
#include "Shards_CellTopology.hpp"
#include "PHAL_FactoryTraits.hpp"
#include "Albany_Utils.hpp"
#include "Albany_ProblemUtils.hpp"
#include <string>


Aeras::ShallowWaterProblem::
ShallowWaterProblem( const Teuchos::RCP<Teuchos::ParameterList>& params_,
             const Teuchos::RCP<ParamLib>& paramLib_,
             const int spatialDim_) :
  Albany::AbstractProblem(params_, paramLib_),
  spatialDim(spatialDim_)
{
  TEUCHOS_TEST_FOR_EXCEPTION(spatialDim!=2 && spatialDim!=3,std::logic_error,"Shallow water problem is only written for 2 or 3D.");
  // Set number of scalar equation per node, neq,  based on spatialDim
  if      (spatialDim==2) { modelDim=2; neq=3; } // Planar 2D problem
  else if (spatialDim==3) { modelDim=2; neq=3; } // 2D shells embedded in 3D

  // Set the num PDEs for the null space object to pass to ML
  this->rigidBodyModes->setNumPDEs(neq);
}

Aeras::ShallowWaterProblem::
~ShallowWaterProblem()
{
}

void
Aeras::ShallowWaterProblem::
buildProblem(
  Teuchos::ArrayRCP<Teuchos::RCP<Albany::MeshSpecsStruct> >  meshSpecs,
  Albany::StateManager& stateMgr)
{
  using Teuchos::rcp;

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
Aeras::ShallowWaterProblem::
buildEvaluators(
  PHX::FieldManager<PHAL::AlbanyTraits>& fm0,
  const Albany::MeshSpecsStruct& meshSpecs,
  Albany::StateManager& stateMgr,
  Albany::FieldManagerChoice fmchoice,
  const Teuchos::RCP<Teuchos::ParameterList>& responseList)
{
  // Call constructeEvaluators<EvalT>(*rfm[0], *meshSpecs[0], stateMgr);
  // for each EvalT in PHAL::AlbanyTraits::BEvalTypes
  Albany::ConstructEvaluatorsOp<ShallowWaterProblem> op(
    *this, fm0, meshSpecs, stateMgr, fmchoice, responseList);
  boost::mpl::for_each<PHAL::AlbanyTraits::BEvalTypes>(op);
  return *op.tags;
}

void
Aeras::ShallowWaterProblem::constructDirichletEvaluators(
        const Albany::MeshSpecsStruct& meshSpecs)
{
   // Construct Dirichlet evaluators for all nodesets and names
   std::vector<std::string> dirichletNames(neq);
   dirichletNames[0] = "Depth";
   dirichletNames[1] = "Vx";
   dirichletNames[2] = "Vy";
   Albany::BCUtils<Albany::DirichletTraits> dirUtils;
   dfm = dirUtils.constructBCEvaluators(meshSpecs.nsNames, dirichletNames,
                                          this->params, this->paramLib);
}

// Neumann BCs
void
Aeras::ShallowWaterProblem::constructNeumannEvaluators(const Teuchos::RCP<Albany::MeshSpecsStruct>& meshSpecs)
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

   std::vector<std::string> neumannNames(neq + 1);
   Teuchos::Array<Teuchos::Array<int> > offsets;
   offsets.resize(neq + 1);

   neumannNames[0] = "Depth";
   offsets[0].resize(1);
   offsets[0][0] = 0;
   offsets[neq].resize(neq);
   offsets[neq][0] = 0;

   if (neq>1){
      neumannNames[1] = "Vx";
      offsets[1].resize(1);
      offsets[1][0] = 1;
      offsets[neq][1] = 1;
   }

   if (neq>2){
     neumannNames[2] = "Vy";
      offsets[2].resize(1);
      offsets[2][0] = 2;
      offsets[neq][2] = 2;
   }

   neumannNames[neq] = "all";

   // Construct BC evaluators for all possible names of conditions
   // Should only specify flux vector components (dUdx, dUdy, dUdz)
   std::vector<std::string>       condNames(1); //(dUdx, dUdy, dUdz)
   Teuchos::ArrayRCP<std::string> dof_names(1);
     dof_names[0] = "Velocity";

//   condNames[1] = "dFluxdn";
//   condNames[2] = "basal";
//   condNames[3] = "P";
//   condNames[4] = "lateral";

   nfm.resize(1); // Aeras problem only has one element block

   nfm[0] = nbcUtils.constructBCEvaluators(meshSpecs, neumannNames, dof_names, true, 0,
                                          condNames, offsets, dl,
                                          this->params, this->paramLib);


}

Teuchos::RCP<const Teuchos::ParameterList>
Aeras::ShallowWaterProblem::getValidProblemParameters() const
{
  Teuchos::RCP<Teuchos::ParameterList> validPL =
    this->getGenericProblemParams("ValidShallowWaterProblemParams");

  validPL->sublist("Shallow Water Problem", false, "");
  validPL->sublist("Aeras Surface Height", false, "");
  validPL->sublist("Aeras Shallow Water Source", false, "");
  return validPL;
}

