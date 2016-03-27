//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#include "HydMorphProblem.hpp"

#include "Intrepid2_FieldContainer.hpp"
#include "Intrepid2_DefaultCubatureFactory.hpp"
#include "Shards_CellTopology.hpp"
#include "PHAL_FactoryTraits.hpp"
#include "Albany_Utils.hpp"


Albany::HydMorphProblem::
HydMorphProblem( const Teuchos::RCP<Teuchos::ParameterList>& params_,
             const Teuchos::RCP<ParamLib>& paramLib_,
             const int numDim_,
             Teuchos::RCP<const Teuchos::Comm<int> >& commT_):  
  Albany::AbstractProblem(params_, paramLib_),
  numDim(numDim_),
  commT(commT_)
{

  this->setNumEquations(2);

  if(params->isType<std::string>("MaterialDB Filename")){

    std::string mtrlDbFilename = params->get<std::string>("MaterialDB Filename");
 // Create Material Database
    materialDB = Teuchos::rcp(new QCAD::MaterialDatabase(mtrlDbFilename, commT));

  }
}

Albany::HydMorphProblem::
~HydMorphProblem()
{
}

void
Albany::HydMorphProblem::
buildProblem(
  Teuchos::ArrayRCP<Teuchos::RCP<Albany::MeshSpecsStruct> >  meshSpecs,
  Albany::StateManager& stateMgr)
{

  /* Construct All Phalanx Evaluators */
  int physSets = meshSpecs.size();
  std::cout << "HydMorphology Problem Num MeshSpecs: " << physSets << std::endl;
  fm.resize(physSets);

  for (int ps=0; ps<physSets; ps++) {
    fm[ps]  = Teuchos::rcp(new PHX::FieldManager<PHAL::AlbanyTraits>);
    buildEvaluators(*fm[ps], *meshSpecs[ps], stateMgr, BUILD_RESID_FM,
                    Teuchos::null);
  }



  if(meshSpecs[0]->nsNames.size() > 0) // Build a nodeset evaluator if nodesets are present

    constructDirichletEvaluators(meshSpecs[0]->nsNames);

  if(meshSpecs[0]->ssNames.size() > 0) // Build a sideset evaluator if sidesets are present

    constructNeumannEvaluators(meshSpecs[0]);


}

Teuchos::Array<Teuchos::RCP<const PHX::FieldTag> >
Albany::HydMorphProblem::
buildEvaluators(
  PHX::FieldManager<PHAL::AlbanyTraits>& fm0,
  const Albany::MeshSpecsStruct& meshSpecs,
  Albany::StateManager& stateMgr,
  Albany::FieldManagerChoice fmchoice,
  const Teuchos::RCP<Teuchos::ParameterList>& responseList)
{
  // Call constructEvaluators<EvalT>(*rfm[0], *meshSpecs[0], stateMgr);
  // for each EvalT in PHAL::AlbanyTraits::BEvalTypes
  ConstructEvaluatorsOp<HydMorphProblem> op(
    *this, fm0, meshSpecs, stateMgr, fmchoice, responseList);
  Sacado::mpl::for_each<PHAL::AlbanyTraits::BEvalTypes> fe(op);
  return *op.tags;
}

// Dirichlet BCs
void
Albany::HydMorphProblem::constructDirichletEvaluators(const std::vector<std::string>& nodeSetIDs)
{
   // Construct BC evaluators for all node sets and names
   std::vector<std::string> bcNames(neq);
   bcNames[0] = "T";
   bcNames[1] = "Ch";

   Albany::BCUtils<Albany::DirichletTraits> bcUtils;
   dfm = bcUtils.constructBCEvaluators(nodeSetIDs, bcNames,
                                          this->params, this->paramLib);
   offsets_ = bcUtils.getOffsets(); 
}

// Neumann BCs
void
Albany::HydMorphProblem::constructNeumannEvaluators(
        const Teuchos::RCP<Albany::MeshSpecsStruct>& meshSpecs)
{
   // Note: we only enter this function if sidesets are defined in the mesh file
   // i.e. meshSpecs.ssNames.size() > 0

   Albany::BCUtils<Albany::NeumannTraits> neuUtils;

   // Check to make sure that Neumann BCs are given in the input file

   if(!neuUtils.haveBCSpecified(this->params))

      return;

   // Construct BC evaluators for all side sets and names
   // Note that the string index sets up the equation offset, so ordering is important
   std::vector<std::string> neumannNames(neq);
   Teuchos::Array<Teuchos::Array<int> > offsets;
   offsets.resize(neq+1);

   neumannNames[0] = "T";
   neumannNames[1] = "Ch";
   offsets[0].resize(1);
   offsets[0][0] = 0;
   offsets[1].resize(2);
   offsets[1][0] = 0;
   offsets[1][1] = 1;

   if (numDim>1){
      neumannNames[1] = "Ty";
      offsets[1].resize(1);
      offsets[1][0] = 1;
      offsets[neq][1] = 1;
   }

   if (numDim>2){
     neumannNames[2] = "Tz";
      offsets[2].resize(1);
      offsets[2][0] = 2;
      offsets[neq][2] = 2;
   }

   neumannNames[numDim] = "cFlux";
   offsets[numDim].resize(1);
   offsets[numDim][0] = numDim;
   offsets[neq][numDim] = numDim;

   neumannNames[numDim + 1] = "wFlux";
   offsets[numDim+1].resize(1);
   offsets[numDim+1][0] = numDim+1;
   offsets[neq][numDim+1] = numDim+1;

   neumannNames[neq] = "all";

   // Construct BC evaluators for all possible names of conditions
   // Should only specify flux vector components (dudx, dudy, dudz), or dudn, not both
   std::vector<std::string> condNames(3); //dudx, dudy, dudz, dudn,
   Teuchos::ArrayRCP<std::string> dof_names(1);
     dof_names[0] = "Displacement";

   // Note that sidesets are only supported for two and 3D currently
   if(numDim == 2)
    condNames[0] = "(dudx, dudy)";
   else if(numDim == 3)
    condNames[0] = "(dudx, dudy, dudz)";
   else
    TEUCHOS_TEST_FOR_EXCEPTION(true, Teuchos::Exceptions::InvalidParameter,
       std::endl << "Error: Sidesets only supported in 2 and 3D." << std::endl);

   condNames[1] = "dudn";
   condNames[2] = "P";

   nfm.resize(1); // Elasticity problem only has one element block

   nfm[0] = neuUtils.constructBCEvaluators(meshSpecs, neumannNames, dof_names, true, 0,
                                          condNames, offsets, dl,
                                          this->params, this->paramLib);

}

Teuchos::RCP<const Teuchos::ParameterList>
Albany::HydMorphProblem::getValidProblemParameters() const
{
  Teuchos::RCP<Teuchos::ParameterList> validPL =
    this->getGenericProblemParams("ValidHydMorphProblemParams");

  Teuchos::Array<int> defaultPeriod;
  validPL->sublist("Thermal Conductivity", false, "");
  validPL->sublist("Hydrogen Conductivity", false, "");
  validPL->set<bool>("Have Rho Cp", false, "Flag to indicate if rhoCp is used");
  validPL->set<std::string>("MaterialDB Filename","materials.xml","Filename of material database xml file");


  return validPL;
}

