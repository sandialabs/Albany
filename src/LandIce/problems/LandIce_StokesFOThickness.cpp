//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#include <string>

#include "PHAL_FactoryTraits.hpp"
#include "Albany_BCUtils.hpp"
#include <string.hpp> // For util::upper_case (do not confuse this with <string>! string.hpp is an Albany file)

#include "LandIce_StokesFOThickness.hpp"

// Uncomment for some setup output
#define OUTPUT_TO_SCREEN

namespace LandIce {

StokesFOThickness::StokesFOThickness(
            const Teuchos::RCP<Teuchos::ParameterList>& params_,
            const Teuchos::RCP<Teuchos::ParameterList>& discParams_,
            const Teuchos::RCP<ParamLib>& paramLib_,
            const int numDim_) :
  StokesFOBase(params_, discParams_, paramLib_, numDim_)
{
  //Set # of PDEs per node.
  std::string eqnSet = params_->sublist("Equation Set").get<std::string>("Type", "LandIce");
  neq = 3; //LandIce FO Stokes system is a system of 2 PDEs; add one for the thickness.

  // Set the num PDEs for the null space object to pass to ML
  // this->rigidBodyModes->setNumPDEs(neq);

  TEUCHOS_TEST_FOR_EXCEPTION (surfaceSideName=="__INVALID__", std::runtime_error,
    "Error! StokesFOThickness requires a valid surfaceSideName, since the thickness equation is solved on the surface.\n");
  // Defining the thickness equation only in 2D (basal side)
  sideSetEquations[2].push_back(surfaceSideName);

  dof_names.resize(2);
  dof_names[1] = "ice_thickness Increment";

  resid_names.resize(2);
  resid_names[1] = dof_names[1] + " Residual";

  scatter_names.resize(2);
  scatter_names[1] = "Scatter " + resid_names[1];

  dof_offsets.resize(2);
  dof_offsets[1] = vecDimFO;

  // We have two values for ice_thickness: the initial one, and the updated one.
  initial_ice_thickness_name = ice_thickness_name;
  ice_thickness_name += "_computed";
}

Teuchos::Array< Teuchos::RCP<const PHX::FieldTag> >
StokesFOThickness::buildEvaluators(
  PHX::FieldManager<PHAL::AlbanyTraits>& fm0,
  const Albany::MeshSpecsStruct& meshSpecs,
  Albany::StateManager& stateMgr,
  Albany::FieldManagerChoice fmchoice,
  const Teuchos::RCP<Teuchos::ParameterList>& responseList)
{
  // Call constructeEvaluators<EvalT>(*rfm[0], *meshSpecs[0], stateMgr);
  // for each EvalT in PHAL::AlbanyTraits::BEvalTypes

  Albany::ConstructEvaluatorsOp<StokesFOThickness> op(
    *this, fm0, meshSpecs, stateMgr, fmchoice, responseList);

  Sacado::mpl::for_each<PHAL::AlbanyTraits::BEvalTypes> fe(op);

  return *op.tags;
}

void StokesFOThickness::constructDirichletEvaluators(const Albany::MeshSpecsStruct& meshSpecs)
{
   // Construct Dirichlet evaluators for all nodesets and names
   std::vector<std::string> dirichletNames(neq);
   for (size_t i=0; i<neq-1; i++) {
     std::stringstream s; s << "U" << i;
     dirichletNames[i] = s.str();
   }
   dirichletNames[neq-1] = "H";
   Albany::BCUtils<Albany::DirichletTraits> dirUtils;
   dfm = dirUtils.constructBCEvaluators(meshSpecs.nsNames, dirichletNames,
                                          this->params, this->paramLib);
   use_sdbcs_ = dirUtils.useSDBCs();
   offsets_ = dirUtils.getOffsets();
   nodeSetIDs_ = dirUtils.getNodeSetIDs();
}

// Neumann BCs
void
LandIce::StokesFOThickness::constructNeumannEvaluators(const Teuchos::RCP<Albany::MeshSpecsStruct>& meshSpecs)
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

   neumannNames[0] = "U0";
   offsets[0].resize(1);
   offsets[0][0] = 0;
   offsets[neq].resize(neq-1);
   offsets[neq][0] = 0;

   if (neq-1>1){
      neumannNames[1] = "U1";
      offsets[1].resize(1);
      offsets[1][0] = 1;
      offsets[neq][1] = 1;
   }

   if (neq-1>2){
     neumannNames[2] = "U2";
      offsets[2].resize(1);
      offsets[2][0] = 2;
      offsets[neq][2] = 2;
   }

   neumannNames[neq-1] = "H";
   offsets[neq-1].resize(1);
   offsets[neq-1][0] = neq-1;

   neumannNames[neq] = "all";

   // Construct BC evaluators for all possible names of conditions
   // Should only specify flux vector components (dCdx, dCdy, dCdz), or dCdn, not both
   std::vector<std::string> condNames(3); //(dCdx, dCdy, dCdz), dCdn, P

   // Note that sidesets are only supported for two and 3D currently
   if(numDim == 2)
    condNames[0] = "(dFluxdx, dFluxdy)";
   else if(numDim == 3)
    condNames[0] = "(dFluxdx, dFluxdy, dFluxdz)";
   else
    TEUCHOS_TEST_FOR_EXCEPTION(true, Teuchos::Exceptions::InvalidParameter,
       std::endl << "Error: Sidesets only supported in 2 and 3D." << std::endl);

   condNames[1] = "dFluxdn";
   condNames[2] = "P";

   nfm.resize(1); // LandIce problem only has one element block

   nfm[0] = nbcUtils.constructBCEvaluators(meshSpecs, neumannNames, dof_names, true, 0,
                                          condNames, offsets, dl,
                                          this->params, this->paramLib);
}

Teuchos::RCP<const Teuchos::ParameterList>
StokesFOThickness::getValidProblemParameters() const
{
  Teuchos::RCP<Teuchos::ParameterList> validPL = StokesFOBase::getStokesFOBaseProblemParameters();

  validPL->sublist("Equation Set", false, "");
  validPL->sublist("Body Force", false, "");
  validPL->set<double>("Time Step", 1.0, "Time step for divergence flux ");
  validPL->set<Teuchos::RCP<double> >("Time Step Ptr", Teuchos::null, "Time step ptr for divergence flux ");

  return validPL;
}

void StokesFOThickness::setFieldsProperties () {
  StokesFOBase::setFieldsProperties();

  // Fix the scalar type of ice_thickness_name, since in StokesFOThickness it depends on the solution.
  field_scalar_type[ice_thickness_name] = FST::Scalar;

  // Mark the thickness increment and initial_thickness+increment as computed
  is_computed_field[ice_thickness_name] = true;
  if (Albany::mesh_depends_on_solution()) {
    // With a moving mesh, the surface height is an output variable
    is_computed_field[surface_height_name] = true;
  }

  field_rank[ice_thickness_name]  = FRT::Scalar;
  field_rank[surface_height_name] = FRT::Scalar;
}

} // namespace LandIce
