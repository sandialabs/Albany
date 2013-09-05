//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//
#include "ConcurrentMultiscaleProblem.hpp"
#include "Albany_Utils.hpp"
#include "Albany_ProblemUtils.hpp"
#include "PHAL_AlbanyTraits.hpp"

//------------------------------------------------------------------------------
Albany::ConcurrentMultiscaleProblem::
ConcurrentMultiscaleProblem(const Teuchos::RCP<Teuchos::ParameterList>& params,
                 const Teuchos::RCP<ParamLib>& param_lib,
                 const int num_dims,
                 const Teuchos::RCP<const Epetra_Comm>& comm) :
  Albany::AbstractProblem(params, param_lib),
  have_source_(false),
  num_dims_(num_dims)
{

  std::string& method = params->get("Name", "Mechanics ");
  *out << "Problem Name = " << method << std::endl;

  bool I_Do_Not_Have_A_Valid_Material_DB(true);
  if(params->isType<std::string>("MaterialDB Filename")){
    I_Do_Not_Have_A_Valid_Material_DB = false;
    std::string filename = params->get<std::string>("MaterialDB Filename");
    material_db_ = Teuchos::rcp(new QCAD::MaterialDatabase(filename, comm));
  }
  TEUCHOS_TEST_FOR_EXCEPTION(I_Do_Not_Have_A_Valid_Material_DB, 
                             std::logic_error,
                             "ConcurrentMultiscale Problem Requires a Material Database");


  // Compute number of equations
  int num_eq = num_dims_;
  this->setNumEquations(num_eq);

  //the following function returns the problem information required for
  //setting the rigid body modes (RBMs)
  int num_PDEs = neq;
  int num_elasticity_dim = num_dims_;
  int num_scalar = neq - num_elasticity_dim;
  int null_space_dim;
  if (num_dims_ == 1) {null_space_dim = 1; }
  if (num_dims_ == 2) {null_space_dim = 3; }
  if (num_dims_ == 3) {null_space_dim = 6; }

  rigidBodyModes->setParameters(num_PDEs, num_elasticity_dim, num_scalar, null_space_dim);
  
}
//------------------------------------------------------------------------------
Albany::ConcurrentMultiscaleProblem::
~ConcurrentMultiscaleProblem()
{
}
//------------------------------------------------------------------------------
void
Albany::ConcurrentMultiscaleProblem::
buildProblem(Teuchos::ArrayRCP<Teuchos::RCP<Albany::MeshSpecsStruct> >  meshSpecs,
             Albany::StateManager& stateMgr)
{
  // Construct All Phalanx Evaluators
  int physSets = meshSpecs.size();
  std::cout << "Num MeshSpecs: " << physSets << std::endl;
  fm.resize(physSets);

  std::cout << "Calling ConcurrentMultiscaleProblem::buildEvaluators" << std::endl;
  for (int ps=0; ps < physSets; ++ps) {

    std::string eb_name = meshSpecs[ps]->ebName;

    if ( material_db_->isElementBlockParam(eb_name,"Lagrange Multiplier Block") ) {
      lm_overlap_map_.insert(std::make_pair(eb_name, 
         material_db_->getElementBlockParam<bool>(eb_name,"Lagrange Multiplier Block") ) );
    }

    if ( material_db_->isElementBlockParam(eb_name,"Coarse Overlap Block") ) {
      coarse_overlap_map_.insert(std::make_pair(eb_name, 
         material_db_->getElementBlockParam<bool>(eb_name,"Coarse Overlap Block") ) );
    }

    if ( material_db_->isElementBlockParam(eb_name,"Fine Overlap Block") ) {
      fine_overlap_map_.insert(std::make_pair(eb_name, 
         material_db_->getElementBlockParam<bool>(eb_name,"Fine Overlap Block") ) );
    }
    
    fm[ps]  = Teuchos::rcp(new PHX::FieldManager<PHAL::AlbanyTraits>);
    buildEvaluators(*fm[ps], *meshSpecs[ps], stateMgr, BUILD_RESID_FM,
                    Teuchos::null);
  }
  constructDirichletEvaluators(*meshSpecs[0]);
}
//------------------------------------------------------------------------------
Teuchos::Array<Teuchos::RCP<const PHX::FieldTag> >
Albany::ConcurrentMultiscaleProblem::
buildEvaluators(PHX::FieldManager<PHAL::AlbanyTraits>& fm0,
                const Albany::MeshSpecsStruct& meshSpecs,
                Albany::StateManager& stateMgr,
                Albany::FieldManagerChoice fmchoice,
                const Teuchos::RCP<Teuchos::ParameterList>& responseList)
{
  // Call constructeEvaluators<EvalT>(*rfm[0], *meshSpecs[0], stateMgr);
  // for each EvalT in PHAL::AlbanyTraits::BEvalTypes
  ConstructEvaluatorsOp<ConcurrentMultiscaleProblem> 
    op( *this,
        fm0,
        meshSpecs,
        stateMgr,
        fmchoice,
        responseList );
  boost::mpl::for_each<PHAL::AlbanyTraits::BEvalTypes>(op);
  return *op.tags;
}
//------------------------------------------------------------------------------
void
Albany::ConcurrentMultiscaleProblem::
constructDirichletEvaluators(const Albany::MeshSpecsStruct& meshSpecs)
{

  // Construct Dirichlet evaluators for all nodesets and names
  std::vector<std::string> dirichletNames(neq);
  int index = 0;
  dirichletNames[index++] = "X";
  if (neq>1) dirichletNames[index++] = "Y";
  if (neq>2) dirichletNames[index++] = "Z";

  Albany::BCUtils<Albany::DirichletTraits> dirUtils;
  dfm = dirUtils.constructBCEvaluators(meshSpecs.nsNames, dirichletNames,
                                       this->params, this->paramLib);
}
//------------------------------------------------------------------------------
Teuchos::RCP<const Teuchos::ParameterList>
Albany::ConcurrentMultiscaleProblem::
getValidProblemParameters() const
{
  Teuchos::RCP<Teuchos::ParameterList> validPL =
    this->getGenericProblemParams("ValidConcurrentMultiscaleProblemParams");

  validPL->set<std::string>("MaterialDB Filename",
                            "materials.xml",
                            "Filename of material database xml file");

  return validPL;
}

//------------------------------------------------------------------------------
void
Albany::ConcurrentMultiscaleProblem::
getAllocatedStates(
   Teuchos::ArrayRCP<Teuchos::ArrayRCP<Teuchos::RCP<Intrepid::FieldContainer<RealType> > > > old_state,
   Teuchos::ArrayRCP<Teuchos::ArrayRCP<Teuchos::RCP<Intrepid::FieldContainer<RealType> > > > new_state
                   ) const
{
  old_state = old_state_;
  new_state = new_state_;
}
