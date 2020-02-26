//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#include "TDManufacturing.hpp"

#include "Intrepid2_DefaultCubatureFactory.hpp"
#include "Shards_CellTopology.hpp"
#include "PHAL_FactoryTraits.hpp"
#include "Albany_Utils.hpp"

//Constructor for 3DM
Albany::TDManufacturing::
TDManufacturing(const Teuchos::RCP<Teuchos::ParameterList>& params_,
		       const Teuchos::RCP<ParamLib>& param_lib,
		       const int num_dims,
		       Teuchos::RCP<const Teuchos::Comm<int> >& commT) :
  Albany::AbstractProblem(params_, param_lib), 
  num_dims_(num_dims), Subtractive_(true)
{
  // Read the "MaterialDB Filename" parameter from the input deck and create the MaterialDatabase
  std::string filename = params->get<std::string>("MaterialDB Filename");
  material_db_ = Teuchos::rcp(new Albany::MaterialDatabase(filename, commT));
  
  // get subtractive flag from material input deck. If not specified then assign true.
  Subtractive_ = material_db_->getParam<bool>("Subtractive",true);
  
  // Tell user if subtractive is on or off
  Teuchos::RCP<Teuchos::FancyOStream> out = Teuchos::VerboseObjectBase::getDefaultOStream();
  
  if ( Subtractive_ != true ) {
      *out << "*******************************" << std::endl;
      *out << "WARNING: Subtractive is OFF" << std::endl;
      *out << "*******************************" << std::endl;
    sim_type = params->get<std::string>("Simulation Type");

  /*
  powder_layer_thickness = params->get("Powder Layer Thickness", 50e-6);
  initial_porosity = params->get("Powder Layer Initial Porosity", 0.652);
  powder_diameter = params->get("Powder Diameter", 30e-6); 
  laser_beam_radius = params->get("Laser Beam Radius", 1);
  laser_path_filename = params->get<std::string>("Laser Path Input Filename");
  
  //Make sure sim_type is valid. For now, only "SLM Additive" will be valid
  if (sim_type != "SLM Additive"){
      TEUCHOS_TEST_FOR_EXCEPTION(true,std::runtime_error,"*** Unsupported Simulation Type. Currently supported sim types: \"SLM Additive\" ***\n")
  */ 

  }
  
  this->setNumEquations(1);
}

Albany::TDManufacturing::
~TDManufacturing()
{ }

// This function return true if compute subtractive was specified in the
// input deck. By default subtractive is on.

bool
Albany::TDManufacturing::Subtractive() const
{
  return Subtractive_;
}

void Albany::TDManufacturing::
buildProblem(
	     Teuchos::ArrayRCP<Teuchos::RCP<Albany::MeshSpecsStruct> >  meshSpecs,
	     Albany::StateManager& stateMgr)
{
  //phys_sets = number of element blocks (for current state of the problem)
  int phys_sets = meshSpecs.size();
  *out << "Num MeshSpecs: " << phys_sets << std::endl;
  fm.resize(phys_sets);
  
  for (int ps=0; ps<phys_sets; ps++) {
    fm[ps]  = Teuchos::rcp(new PHX::FieldManager<PHAL::AlbanyTraits>);
    buildEvaluators(*fm[ps], *meshSpecs[ps], stateMgr, BUILD_RESID_FM,
		    Teuchos::null);
  }
	
  //Dirichlet and Neumann conditions applied only to first meshspec
  if(meshSpecs[0]->nsNames.size() > 0)
    constructDirichletEvaluators(meshSpecs[0]->nsNames);

  if(meshSpecs[0]->ssNames.size() > 0)
    constructNeumannEvaluators(meshSpecs[0]);

}

Teuchos::Array<Teuchos::RCP<const PHX::FieldTag> >
Albany::TDManufacturing::
buildEvaluators(
		PHX::FieldManager<PHAL::AlbanyTraits>& fm0,
		const Albany::MeshSpecsStruct& meshSpecs,
		Albany::StateManager& stateMgr,
		Albany::FieldManagerChoice fmchoice,
		const Teuchos::RCP<Teuchos::ParameterList>& responseList)
{
  ConstructEvaluatorsOp<TDManufacturing> op(
		*this, fm0, meshSpecs, stateMgr, fmchoice, responseList);
  Sacado::mpl::for_each<PHAL::AlbanyTraits::BEvalTypes> fe(op);
  return *op.tags;
}

// Dirichlet BCs
void Albany::TDManufacturing::constructDirichletEvaluators(
								  const std::vector<std::string>& nodeSetIDs)
{
  std::vector<std::string> bcNames(neq);
  bcNames[0] = "Temperature";
  //If boundary conditions are entered for consolidation, they will be added as shown below
  // reference: ThermoElasticityProblem.cpp
  //bcNames[1] = "Z";   //may need to switch Temperature and Z depending on ordering in the problem
  Albany::BCUtils<Albany::DirichletTraits> bcUtils;
  dfm = bcUtils.constructBCEvaluators(nodeSetIDs, bcNames,
				      this->params, this->paramLib);
  offsets_ = bcUtils.getOffsets(); 
  //removed use_sdbcs_ and nodeSetIDs_ variables set from bcUtils
}

// Neumann BCs
void Albany::TDManufacturing::constructNeumannEvaluators(
								const Teuchos::RCP<Albany::MeshSpecsStruct>& meshSpecs)
{
  Albany::BCUtils<Albany::NeumannTraits> bcUtils;

  if(!bcUtils.haveBCSpecified(this->params))
    return;

  std::vector<std::string> bcNames(neq);
  Teuchos::ArrayRCP<std::string> dof_names(neq);
  Teuchos::Array<Teuchos::Array<int> > offsets;
  offsets.resize(neq);

  bcNames[0] = "U";
  dof_names[0] = "u";
  offsets[0].resize(1);
  offsets[0][0] = 0;

  // Construct BC evaluators for all possible names of conditions
  // Should only specify flux vector components (dudx, dudy, dudz), or dudn, not both
  std::vector<std::string> condNames(5); 
  //dudx, dudy, dudz, dudn, scaled jump (internal surface), or robin (like DBC plus scaled jump)

  // Note that sidesets are only supported for two and 3D currently
  if(num_dims_ == 2)
    condNames[0] = "(dudx, dudy)";
  else if(num_dims_ == 3)
    condNames[0] = "(dudx, dudy, dudz)";
  else
    TEUCHOS_TEST_FOR_EXCEPTION(true, Teuchos::Exceptions::InvalidParameter,
			       std::endl << "Error: Sidesets only supported in 2 and 3D." << std::endl);

  condNames[1] = "dudn";
  condNames[2] = "scaled jump";
  condNames[3] = "robin";
  condNames[4] = "radiate";

  //This may change when consolidation is added
  nfm.resize(1); // Heat problem only has one physics set   
  nfm[0] = bcUtils.constructBCEvaluators(meshSpecs, bcNames, dof_names, false, 0,
					 condNames, offsets, dl_, this->params, this->paramLib, material_db_);

}

Teuchos::RCP<const Teuchos::ParameterList>
Albany::TDManufacturing::getValidProblemParameters() const
{
  Teuchos::RCP<Teuchos::ParameterList> validPL =
    this->getGenericProblemParams("ValidTDManufacturingParams");

  validPL->set<std::string>("MaterialDB Filename",
                            "materials.xml",
                            "Filename of material database xml file");
  validPL->set<std::string>("Simulation Type","SLM Additive");
  validPL->set<std::string>("Laser Path Input Filename","LaserCenter.txt");
  // this is for use in CTM project
  validPL->set<bool>("Transient",
		     true,
		     "Specify if you want a transient analysis or not");

  return validPL;
}
