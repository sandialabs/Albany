//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//
#include "SchwarzMultiscale.hpp"


LCM::
SchwarzMultiscale::
SchwarzMultiscale(
  const Teuchos::Array<Teuchos::RCP<Thyra::ModelEvaluator<ST> > >& models,
  const Teuchos::Array<Teuchos::RCP<Teuchos::ParameterList> >& params,
  const Teuchos::RCP<const Teuchos::Comm<int> >& commT): 
  models_(models),
  params_(params),
  commT_(commT)
{
  // Setup VerboseObject
  //Teuchos::readVerboseObjectSublist(params.get(), this);

  n_models_ = models.size();
  solvers_.resize(n_models_);

  /*
  // Create solvers for models - rewrite using Albany solver factory
    Piro::Epetra::SolverFactory solverFactory;
    for (int i=0; i<n_models; i++)
      solvers[i] = solverFactory.createSolver(params[i], models[i]);
   */

  // Get number of parameter and response vectors
  solver_inargs_.resize(n_models_); 
  solver_outargs_.resize(n_models_);
  num_params_.resize(n_models_);
  num_responses_.resize(n_models_);
  num_params_total_ = 0;
  num_responses_total_ = 0;
  for (int i=0; i<n_models_; i++) {
    solver_inargs_[i] = solvers_[i]->createInArgs();
    solver_outargs_[i] = solvers_[i]->createOutArgs();
    num_params_[i] = solver_inargs_[i].Np();
    num_responses_[i] = solver_outargs_[i].Ng();
    num_params_total_ += num_params_[i];
    num_responses_total_ += num_responses_[i];
  }
 
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

//fill in rest of functions 
