//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//
#include "SchwarzMultiscale.hpp"
#include "Albany_SolverFactory.hpp" 
#include "Albany_ModelFactory.hpp" 

LCM::
SchwarzMultiscale::
SchwarzMultiscale(const Teuchos::RCP<Teuchos::ParameterList>& appParams,
                  const Teuchos::RCP<const Teuchos::Comm<int> >& commT,  
                  const Teuchos::RCP<const Tpetra_Vector>& initial_guessT)
{
  std::cout << "Initializing Schwarz Multiscale constructor!" << std::endl;

  // Get "Coupled Schwarz" parameter sublist
  Teuchos::ParameterList& coupledSystemParams = appParams->sublist("Coupled System");
  // Get names of individual model xml input files from problem parameterlist
  Teuchos::Array<std::string> model_filenames =
      coupledSystemParams.get<Teuchos::Array<std::string> >("Model XML Files");
  //number of models
  int num_models = model_filenames.size(); 
  std::cout << "DEBUG: num_models: " << num_models << std::endl;
  Teuchos::Array< Teuchos::RCP<Albany::Application> > apps(num_models);
  Teuchos::Array< Teuchos::RCP<Thyra::ModelEvaluator<ST> > > models(num_models);
  Teuchos::Array< Teuchos::RCP<Teuchos::ParameterList> > modelAppParams(num_models);
  Teuchos::Array< Teuchos::RCP<Teuchos::ParameterList> > modelProblemParams(num_models);
  
  //Create a dummy solverFactory for validating application parameter lists 
  //(see QCAD::CoupledPoissonSchorodinger)
  //FIXME: look into how this is used, uncomment if necessary 
  /*Albany::SolverFactory validFactory( Teuchos::createParameterList("Empty dummy for Validation"), commT );
  Teuchos::RCP<const Teuchos::ParameterList> validAppParams = validFactory.getValidAppParameters();
  Teuchos::RCP<const Teuchos::ParameterList> validParameterParams = validFactory.getValidParameterParameters();
  Teuchos::RCP<const Teuchos::ParameterList> validResponseParams = validFactory.getValidResponseParameters();
  */

  //Set up each application and model object in Teuchos::Array
  //(similar logic to that in Albany::SolverFactory::createAlbanyAppAndModelT) 
  for (int m=0; m<num_models; m++) {
    //get parameterlist from mth model *.xml file 
    Albany::SolverFactory slvrfctry(model_filenames[m], commT);
    Teuchos::ParameterList& appParams_m = slvrfctry.getParameters(); 
    modelAppParams[m] = Teuchos::rcp(&(appParams_m)); 
    Teuchos::RCP<Teuchos::ParameterList> problemParams_m = Teuchos::sublist(modelAppParams[m], "Problem"); 
    modelProblemParams[m] = problemParams_m;
    std::string &problem_name = problemParams_m->get("Name", "");
    //FIXME: should we throw an error if the problem names in the input files don't match??
    std::cout << "DEBUG: name of problem #" << m <<": " << problem_name << std::endl; 
    //create application for mth model 
    //FIXME: initial_guessT needs to be made the right one for the mth model!  Or can it be null?  
    apps[m] = Teuchos::rcp(new Albany::Application(commT, modelAppParams[m], initial_guessT));
    //Validate parameter lists
    //FIXME: add relevant things to validate to getValid* functions below
    //Uncomment
    //problemParams_m->validateParameters(*getValidProblemParameters(),0); 
    //problemParams_m->sublist("Parameters").validateParameters(*validParameterParams, 0);
    //problemParams_m->sublist("Response Functions").validateParameters(*validResponseParams, 0);
    //Create model evaluator
    Albany::ModelFactory modelFactory(modelAppParams[m], apps[m]); 
    models[m] = modelFactory.createT();     
   }
   std::cout <<"Finished creating Albany apps and models!" << std::endl; 
  
}


LCM::SchwarzMultiscale::~SchwarzMultiscale()
{
}

// Overridden from Thyra::ModelEvaluator<ST>
Teuchos::RCP<const Thyra::VectorSpaceBase<ST> >
LCM::SchwarzMultiscale::get_x_space() const
{
  //to fill in!
  //Teuchos::RCP<const Thyra::VectorSpaceBase<ST> > x_space = Thyra::createVectorSpace<ST>(map);
  //return x_space;
}


Teuchos::RCP<const Thyra::VectorSpaceBase<ST> >
LCM::SchwarzMultiscale::get_f_space() const
{
  /*Teuchos::RCP<const Thyra::VectorSpaceBase<ST> > f_space = Thyra::createVectorSpace<ST>(map);
  return f_space;*/
}


Teuchos::RCP<const Thyra::VectorSpaceBase<ST> >
LCM::SchwarzMultiscale::get_p_space(int l) const
{
  /*TEUCHOS_TEST_FOR_EXCEPTION(
    l >= num_param_vecs + num_dist_param_vecs || l < 0,
    Teuchos::Exceptions::InvalidParameter,
    std::endl <<
    "Error!  LCM::SchwarzMultiscale::get_p_space():  " <<
    "Invalid parameter index l = " << l << std::endl);
  Teuchos::RCP<const Tpetra_Map> map; 
  if (l < num_param_vecs)
    map = tpetra_param_map[l];  
  //IK, 7/1/14: commenting this out for now
  //map = distParamLib->get(dist_param_names[l-num_param_vecs])->map(); 
  Teuchos::RCP<const Thyra::VectorSpaceBase<ST> > tpetra_param_space = Thyra::createVectorSpace<ST>(map);
  return tpetra_param_space;*/
}


Teuchos::RCP<const Thyra::VectorSpaceBase<ST> >
LCM::SchwarzMultiscale::get_g_space(int l) const
{
  /*TEUCHOS_TEST_FOR_EXCEPTION(
      l >= app->getNumResponses() || l < 0,
      Teuchos::Exceptions::InvalidParameter,
      std::endl <<
      "Error!  LCM::SchwarzMultiscale::get_g_space():  " <<
      "Invalid response index l = " << l << std::endl);

  Teuchos::RCP<const Tpetra_Map> mapT = app->getResponse(l)->responseMapT();
  Teuchos::RCP<const Thyra::VectorSpaceBase<ST> > gT_space = Thyra::createVectorSpace<ST>(mapT);
  return gT_space;*/
}


Teuchos::RCP<const Teuchos::Array<std::string> >
LCM::SchwarzMultiscale::get_p_names(int l) const
{
/*
  TEUCHOS_TEST_FOR_EXCEPTION(
    l >= num_param_vecs + num_dist_param_vecs || l < 0,
    Teuchos::Exceptions::InvalidParameter,
    std::endl <<
    "Error!  LCM::SchwarzMultiscale::get_p_names():  " <<
    "Invalid parameter index l = " << l << std::endl);

  if (l < num_param_vecs)
    return param_names[l];
  return Teuchos::rcp(new Teuchos::Array<std::string>(1, dist_param_names[l-num_param_vecs]));
*/
}


Thyra::ModelEvaluatorBase::InArgs<ST>
LCM::SchwarzMultiscale::getNominalValues() const
{
  //return nominalValues;
}


Thyra::ModelEvaluatorBase::InArgs<ST>
LCM::SchwarzMultiscale::getLowerBounds() const
{
  return Thyra::ModelEvaluatorBase::InArgs<ST>(); // Default value
}


Thyra::ModelEvaluatorBase::InArgs<ST>
LCM::SchwarzMultiscale::getUpperBounds() const
{
  return Thyra::ModelEvaluatorBase::InArgs<ST>(); // Default value
}


Teuchos::RCP<Thyra::LinearOpBase<ST> >
LCM::SchwarzMultiscale::create_W_op() const
{
  /*const Teuchos::RCP<Tpetra_Operator> W =
    Teuchos::rcp(new Tpetra_CrsMatrix(app->getJacobianGraphT()));
  return Thyra::createLinearOp(W);
  */
}

Teuchos::RCP<Thyra::PreconditionerBase<ST> >
LCM::SchwarzMultiscale::create_W_prec() const
{
  // TODO: Analog of EpetraExt::ModelEvaluator::Preconditioner does not exist in Thyra yet!
  const bool W_prec_not_supported = true;
  TEUCHOS_TEST_FOR_EXCEPT(W_prec_not_supported);
  return Teuchos::null;
}

Teuchos::RCP<Thyra::LinearOpBase<ST> >
LCM::SchwarzMultiscale::create_DfDp_op_impl(int j) const
{
  /*TEUCHOS_TEST_FOR_EXCEPTION(
    j >= num_param_vecs+num_dist_param_vecs || j < num_param_vecs,
    Teuchos::Exceptions::InvalidParameter,
    std::endl <<
    "Error!  LCM::SchwarzMultiscale::create_DfDp_op_impl():  " <<
    "Invalid parameter index j = " << j << std::endl);

  const Teuchos::RCP<Tpetra_Operator> DfDp = Teuchos::rcp(new DistributedParameterDerivativeOpT(
                      app, dist_param_names[j-num_param_vecs]));

  return Thyra::createLinearOp(DfDp); */
}


Teuchos::RCP<const Thyra::LinearOpWithSolveFactoryBase<ST> >
LCM::SchwarzMultiscale::get_W_factory() const
{
  return Teuchos::null;
}


Thyra::ModelEvaluatorBase::InArgs<ST>
LCM::SchwarzMultiscale::createInArgs() const
{

  //return this->createInArgsImpl();

}


void
LCM::SchwarzMultiscale::reportFinalPoint(
    const Thyra::ModelEvaluatorBase::InArgs<ST>& finalPoint,
    const bool wasSolved)
{
//fill in
}

void
LCM::SchwarzMultiscale::
allocateVectors()
{
//fill in
}


/// Create operator form of dg/dx for distributed responses
Teuchos::RCP<Thyra::LinearOpBase<ST> >
LCM::SchwarzMultiscale::
create_DgDx_op_impl(int j) const
{
//fill in
}

/// Create operator form of dg/dx_dot for distributed responses
Teuchos::RCP<Thyra::LinearOpBase<ST> >
LCM::SchwarzMultiscale::
create_DgDx_dot_op_impl(int j) const
{
//fill in
}

/// Create OutArgs
Thyra::ModelEvaluatorBase::OutArgs<ST>
LCM::SchwarzMultiscale::
createOutArgsImpl() const
{
//fill in
}

/// Evaluate model on InArgs
void 
LCM::SchwarzMultiscale::
evalModelImpl(Thyra::ModelEvaluatorBase::InArgs<ST> const & in_args,
      Thyra::ModelEvaluatorBase::OutArgs<ST> const & out_args) const 
{
//fill in
}

//Copied from Albany::SolverFactory -- used to validate applicaton parameters of applications not created via a SolverFactory
Teuchos::RCP<const Teuchos::ParameterList>
LCM::SchwarzMultiscale::
getValidAppParameters() const
{  
  Teuchos::RCP<Teuchos::ParameterList> validPL = Teuchos::rcp(new Teuchos::ParameterList("ValidAppParams"));;
  validPL->sublist("Problem",            false, "Problem sublist");
  validPL->sublist("Debug Output",       false, "Debug Output sublist");
  validPL->sublist("Discretization",     false, "Discretization sublist");
  validPL->sublist("Quadrature",         false, "Quadrature sublist");
  validPL->sublist("Regression Results", false, "Regression Results sublist");
  validPL->sublist("VTK",                false, "DEPRECATED  VTK sublist");
  validPL->sublist("Piro",               false, "Piro sublist");
  validPL->sublist("Coupled System",     false, "Coupled system sublist");

  return validPL;
}

Teuchos::RCP<const Teuchos::ParameterList> 
LCM::SchwarzMultiscale::
getValidProblemParameters() const
{
  Teuchos::RCP<Teuchos::ParameterList> validPL = Teuchos::createParameterList("ValidCoupledSchwarzProblemParams");

  validPL->set<std::string>("Name", "", "String to designate Problem Class");
  validPL->set<int>("Phalanx Graph Visualization Detail", 0, "Flag to select output of Phalanx Graph and level of detail");
  //FIXME: anything else to validate? 
  validPL->set<std::string>("Solution Method", "Steady", "Flag for Steady, Transient, or Continuation");
  
  return validPL;
}
