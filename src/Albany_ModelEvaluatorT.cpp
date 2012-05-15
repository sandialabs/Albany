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


#include "Albany_ModelEvaluatorT.hpp"
#include "Teuchos_ScalarTraits.hpp"
#include "Teuchos_TestForException.hpp"
#include "Stokhos_EpetraVectorOrthogPoly.hpp"
#include "Stokhos_EpetraMultiVectorOrthogPoly.hpp"
#include "Stokhos_EpetraOperatorOrthogPoly.hpp"
#include "Thyra_TpetraThyraWrappers.hpp"
#include "Tpetra_ConfigDefs.hpp"
#include "Petra_Converters.hpp"

using Teuchos::rcp_dynamic_cast; 

typedef Thyra::TpetraOperatorVectorExtraction<ST, int> ConverterT; 

Albany::ModelEvaluatorT::ModelEvaluatorT(
  const Teuchos::RCP<Albany::Application>& app_,
  const Teuchos::RCP<Teuchos::ParameterList>& appParams) 
  : app(app_),
    supplies_prec(app_->suppliesPreconditioner())
{
  Teuchos::RCP<Teuchos::FancyOStream> out = 
    Teuchos::VerboseObjectBase::getDefaultOStream();

  // Parameters (e.g., for sensitivities, SG expansions, ...)
  Teuchos::ParameterList& problemParams = appParams->sublist("Problem");
  Teuchos::ParameterList& parameterParams = 
    problemParams.sublist("Parameters");
  int num_param_vecs = 
    parameterParams.get("Number of Parameter Vectors", 0);
  bool using_old_parameter_list = false;
  if (parameterParams.isType<int>("Number")) {
    int numParameters = parameterParams.get<int>("Number");
    if (numParameters > 0) {
      num_param_vecs = 1;
      using_old_parameter_list = true;
    }
  }
  param_names.resize(num_param_vecs);
  *out << "Number of parameters vectors  = " << num_param_vecs << endl;
  for (int i=0; i<num_param_vecs; i++) {
    Teuchos::ParameterList* pList;
    if (using_old_parameter_list)
      pList = &parameterParams;
    else
      pList = &(parameterParams.sublist(Albany::strint("Parameter Vector",i)));
    int numParameters = pList->get<int>("Number");
    TEUCHOS_TEST_FOR_EXCEPTION(
      numParameters == 0, 
      Teuchos::Exceptions::InvalidParameter,
      std::endl << "Error!  FEApp::ModelEvaluator::ModelEvaluator():  " <<
      "Parameter vector " << i << " has zero parameters!" << std::endl);
    param_names[i] = 
      Teuchos::rcp(new Teuchos::Array<std::string>(numParameters));
    for (int j=0; j<numParameters; j++) {
      (*param_names[i])[j] = 
	pList->get<std::string>(Albany::strint("Parameter",j));
    }
    *out << "Number of parameters in parameter vector " << i << " = " 
	 << numParameters << endl;
  }

  // Setup sacado and tpetra storage for parameters  
  sacado_param_vec.resize(num_param_vecs);
  //Tpetra analog of above
  tpetra_param_map.resize(num_param_vecs);
  p_sg_vals.resize(num_param_vecs);
  p_mp_vals.resize(num_param_vecs);
  //`const Epetra_Comm& comm = app->getMap()->Comm();
  const Epetra_Comm& comm = *app->getComm();
  Teuchos::RCP<const Teuchos::Comm<int> > commT = Albany::createTeuchosCommFromMpiComm(Albany::getMpiCommFromEpetraComm(comm));
  Teuchos::ParameterList kokkosNodeParams;
  Teuchos::RCP<KokkosNode> nodeT = Teuchos::rcp(new KokkosNode (kokkosNodeParams));


  for (int i=0; i<num_param_vecs; i++) {

    // Initialize Sacado parameter vector
    app->getParamLib()->fillVector<PHAL::AlbanyTraits::Residual>(
      *(param_names[i]), sacado_param_vec[i]);

 
    // Create Tpetra map for parameter vector
    Tpetra::LocalGlobal lg = Tpetra::LocallyReplicated; 
    tpetra_param_map[i] = 
      Teuchos::rcp(new Tpetra_Map(sacado_param_vec[i].size(), 0, commT, lg));

    // Create Tpetra vector for parameters
    tpetra_param_vec[i] = 
      Teuchos::rcp(new Tpetra_Vector(tpetra_param_map[i]));
    for (unsigned int j=0; j<sacado_param_vec[i].size(); j++) {
      Teuchos::ArrayRCP<ST> tpetra_param_vec_nonConstView = tpetra_param_vec[i]->get1dViewNonConst();
      tpetra_param_vec_nonConstView[j] = sacado_param_vec[i][j].baseValue;
    }

    p_sg_vals[i].resize(sacado_param_vec[i].size());
    p_mp_vals[i].resize(sacado_param_vec[i].size());
  }  

  //Create the structure of the problem (Tpetra/Thyra)
  Thyra::ModelEvaluatorBase::InArgsSetup<ST> inArgsT;
  inArgsT.setModelEvalDescription(this->description());

  inArgsT.setSupports(Thyra::ModelEvaluatorBase::IN_ARG_t,true);
  inArgsT.setSupports(Thyra::ModelEvaluatorBase::IN_ARG_x,true);
  inArgsT.setSupports(Thyra::ModelEvaluatorBase::IN_ARG_x_dot,true);
  //Are these defined here?
  inArgsT.setSupports(Thyra::ModelEvaluatorBase::IN_ARG_alpha,true);
  inArgsT.setSupports(Thyra::ModelEvaluatorBase::IN_ARG_beta,true);
  inArgsT.set_Np(param_names.size());
  //Will need to add stochastic Galerkin and multi-point stuff
  prototypeInArgsT = inArgsT;  

  Thyra::ModelEvaluatorBase::OutArgsSetup<ST> outArgsT;
  outArgsT.setModelEvalDescription(this->description());
  int n_g = app->getNumResponses();
  // Deterministic
  outArgsT.setSupports(Thyra::ModelEvaluatorBase::OUT_ARG_f,true);
  outArgsT.setSupports(Thyra::ModelEvaluatorBase::OUT_ARG_W,true);
  outArgsT.set_W_properties(
    Thyra::ModelEvaluatorBase::DerivativeProperties(Thyra::ModelEvaluatorBase::DERIV_LINEARITY_UNKNOWN, Thyra::ModelEvaluatorBase::DERIV_RANK_FULL, true));
  //I believe the following is not implemented in Thyra::ModelEvaluatorBase
  //if (supplies_prec) outArgsT.setSupports(Thyra::ModelEvaluatorBase::OUT_ARG_WPrec, true);
  outArgsT.set_Np_Ng(param_names.size(), n_g);
  
  for (int i=0; i<param_names.size(); i++)
    outArgsT.setSupports(Thyra::ModelEvaluatorBase::OUT_ARG_DfDp, i, Thyra::ModelEvaluatorBase::DerivativeSupport(Thyra::ModelEvaluatorBase::DERIV_MV_BY_COL));
  for (int i=0; i<n_g; i++) {
    if (app->getResponse(i)->isScalarResponse()) {
      outArgsT.setSupports(Thyra::ModelEvaluatorBase::OUT_ARG_DgDx, i, 
			  Thyra::ModelEvaluatorBase::DerivativeSupport(Thyra::ModelEvaluatorBase::DERIV_TRANS_MV_BY_ROW));
      outArgsT.setSupports(Thyra::ModelEvaluatorBase::OUT_ARG_DgDx_dot, i, 
			  Thyra::ModelEvaluatorBase::DerivativeSupport(Thyra::ModelEvaluatorBase::DERIV_TRANS_MV_BY_ROW));
    }
    else {
       outArgsT.setSupports(Thyra::ModelEvaluatorBase::OUT_ARG_DgDx, i, 
			   Thyra::ModelEvaluatorBase::DerivativeSupport(Thyra::ModelEvaluatorBase::DERIV_LINEAR_OP));
       outArgsT.setSupports(Thyra::ModelEvaluatorBase::OUT_ARG_DgDx_dot, i, 
			   Thyra::ModelEvaluatorBase::DerivativeSupport(Thyra::ModelEvaluatorBase::DERIV_LINEAR_OP));
    }
    for (int j=0; j<param_names.size(); j++)
      outArgsT.setSupports(Thyra::ModelEvaluatorBase::OUT_ARG_DgDp, i, j, 
			  Thyra::ModelEvaluatorBase::DerivativeSupport(Thyra::ModelEvaluatorBase::DERIV_MV_BY_COL));
  }
  //will need to add stochastic Galerkin and multi-point stuff
  prototypeOutArgsT = outArgsT; 

  nominalValues = inArgsT;
  //Create Tpetra objects to be wrapped in Thyra
  Teuchos::RCP<const Tpetra_Vector> xT_init = app->getInitialSolutionT(); 
  Teuchos::RCP<const Tpetra_Vector> x_dotT_init = app->getInitialSolutionDotT(); 
  Teuchos::RCP<const Tpetra_Map> map = app->getMapT();
  Teuchos::RCP<const Thyra::VectorSpaceBase<ST> > xT_space = Thyra::createVectorSpace<ST>(map);
  //create non-const versions of xT_init and x_dotT_init 
  Teuchos::RCP<Tpetra_Vector> xT_init_nonconst = Teuchos::rcp(new Tpetra_Vector(*xT_init)); 
  Teuchos::RCP<Tpetra_Vector> x_dotT_init_nonconst = Teuchos::rcp(new Tpetra_Vector(*x_dotT_init)); 
  nominalValues.set_x(Thyra::createVector(xT_init_nonconst, xT_space)); 
  nominalValues.set_x_dot(Thyra::createVector(x_dotT_init_nonconst, xT_space)); //x_init and x_dot_init have same space?
  //would this be correct way to set p_init?
  for (int l =0; l<num_param_vecs; l++) {
     Teuchos::RCP<const Tpetra_Map> map = tpetra_param_map[l];
     Teuchos::RCP<const Thyra::VectorSpaceBase<ST> > tpetra_param_space = Thyra::createVectorSpace<ST>(map);
     nominalValues.set_p(l, Thyra::createVector(tpetra_param_vec[l], tpetra_param_space));  
  }

  timer = Teuchos::TimeMonitor::getNewTimer("Albany: **Total Fill Time**");
}

// Overridden from Thyra::ModelEvaluatorBase


Teuchos::RCP<const Thyra::VectorSpaceBase<ST> >
Albany::ModelEvaluatorT::get_x_space() const
{
  Teuchos::RCP<const Tpetra_Map> map = app->getMapT();
  Teuchos::RCP<const Thyra::VectorSpaceBase<ST> > x_space = Thyra::createVectorSpace<ST>(map); 
  return x_space;   
}


Teuchos::RCP<const Thyra::VectorSpaceBase<ST> >
Albany::ModelEvaluatorT::get_f_space() const
{
  Teuchos::RCP<const Tpetra_Map> map = app->getMapT();
  Teuchos::RCP<const Thyra::VectorSpaceBase<ST> > f_space = Thyra::createVectorSpace<ST>(map); 
  return f_space;   
}


Teuchos::RCP<const Thyra::VectorSpaceBase<ST> >
Albany::ModelEvaluatorT::get_p_space(int l) const
{
   TEUCHOS_TEST_FOR_EXCEPTION(
    l >= static_cast<int>(tpetra_param_map.size()) || l < 0,
    Teuchos::Exceptions::InvalidParameter,
    std::endl <<
    "Error!  Albany::ModelEvaluatorT::get_p_space():  " <<
    "Invalid parameter index l = " << l << std::endl);
  Teuchos::RCP<const Tpetra_Map> map = tpetra_param_map[l]; 
  Teuchos::RCP<const Thyra::VectorSpaceBase<ST> > tpetra_param_space = Thyra::createVectorSpace<ST>(map); 
  return tpetra_param_space;
}


Teuchos::RCP<const Thyra::VectorSpaceBase<ST> >
Albany::ModelEvaluatorT::get_g_space(int l) const
{

  //Ultimately will need to rewrite responseMap() function in responses to return a Tpetra Map!
   TEUCHOS_TEST_FOR_EXCEPTION(
    l >= app->getNumResponses() || l < 0,
    Teuchos::Exceptions::InvalidParameter,
    std::endl <<
    "Error!  Albany::ModelEvaluatorT::get_g_space():  " <<
    "Invalid response index l = " << l << std::endl);

  Teuchos::RCP<const Epetra_Map> map = app->getResponse(l)->responseMap();
  const Epetra_Comm& comm = *app->getComm();
  Teuchos::RCP<const Teuchos::Comm<int> > commT = Albany::createTeuchosCommFromMpiComm(Albany::getMpiCommFromEpetraComm(comm));
  Teuchos::ParameterList kokkosNodeParams;
  Teuchos::RCP<KokkosNode> nodeT = Teuchos::rcp(new KokkosNode (kokkosNodeParams));
  Teuchos::RCP<const Tpetra_Map> mapT = Petra::EpetraMap_To_TpetraMap(map, commT, nodeT); 
  Teuchos::RCP<const Thyra::VectorSpaceBase<ST> >gT_space = Thyra::createVectorSpace<ST>(mapT); 
  return gT_space; 
}

Teuchos::RCP<const Teuchos::Array<std::string> >
Albany::ModelEvaluatorT::get_p_names(int l) const
{
  TEUCHOS_TEST_FOR_EXCEPTION(l >= static_cast<int>(param_names.size()) || l < 0, 
		     Teuchos::Exceptions::InvalidParameter,
                     std::endl << 
                     "Error!  Albany::ModelEvaluatorT::get_p_names():  " <<
                     "Invalid parameter index l = " << l << std::endl);

  return param_names[l];
}


Thyra::ModelEvaluatorBase::InArgs<ST> 
Albany::ModelEvaluatorT::getNominalValues() const
{
  return nominalValues; 
}


Teuchos::RCP<LinearOpBase<ST> >
Albany::ModelEvaluatorT::create_W_op() const
{
   //Create Tpetra_CrsMatrix from Jacobian Graph
   Teuchos::RCP<Tpetra_Operator> W = Teuchos::rcp(new Tpetra_CrsMatrix(app->getJacobianGraphT())); 
   //convert Tpetra_CrsMatrix to Thyra LinearOpBase object
   return Thyra::createLinearOp(W); 
}


//Analog of EpetraExt::ModelEvaluator::Preconditioner does not exist in thyra yet!
/*Teuchos::RCP<EpetraExt::ModelEvaluator::Preconditioner>
Albany::ModelEvaluator::create_WPrec() const
{
  Teuchos::RCP<Epetra_Operator> precOp = app->getPreconditioner();

  // Teko prec needs space for Jacobian as well
  Extra_W_crs = Teuchos::rcp_dynamic_cast<Epetra_CrsMatrix>(create_W(), true);

  // bool is answer to: "Prec is already inverted?"
  return Teuchos::rcp(new EpetraExt::ModelEvaluator::Preconditioner(precOp,true));
}
*/

Teuchos::RCP<LinearOpBase<ST> > 
Albany::ModelEvaluatorT::create_DgDx_op(int j) const
{
    TEUCHOS_TEST_FOR_EXCEPTION(
    j >= app->getNumResponses() || j < 0,
    Teuchos::Exceptions::InvalidParameter,
    std::endl <<
    "Error!  Albany::ModelEvaluatorT::create_DgDx_op():  " <<
    "Invalid response index j = " << j << std::endl);

  return Thyra::createLinearOp(app->getResponse(j)->createGradientOpT());
}


Teuchos::RCP<LinearOpBase<ST> > 
Albany::ModelEvaluatorT::create_DgDx_dot_op(int j) const
{
   TEUCHOS_TEST_FOR_EXCEPTION(
    j >= app->getNumResponses() || j < 0,
    Teuchos::Exceptions::InvalidParameter,
    std::endl <<
    "Error!  Albany::ModelEvaluatorT::create_DgDx_dot_op():  " <<
    "Invalid response index j = " << j << std::endl);

  return Thyra::createLinearOp(app->getResponse(j)->createGradientOpT());
}


Thyra::ModelEvaluatorBase::InArgs<ST> 
Albany::ModelEvaluatorT::createInArgs() const
{
  //The following are not implemented in Thyra::ModelEvaluatorBase yet -- SG and MP (?)
 /*inArgsT.setSupports(Thyra::ModelEvaluatorBase::IN_ARG_x_sg,true);
  inArgsT.setSupports(Thyra::ModelEvaluatorBase::IN_ARG_x_dot_sg,true);
  for (int i=0; i<param_names.size(); i++)
    inArgsT.setSupports(Thyra::ModelEvaluatorBase::IN_ARG_p_sg, i, true);
  inArgsT.setSupports(Thyra::ModelEvaluatorBase::IN_ARG_sg_basis,true);
  inArgsT.setSupports(Thyra::ModelEvaluatorBase::IN_ARG_sg_quadrature,true);
  inArgsT.setSupports(Thyra::ModelEvaluatorBase::IN_ARG_sg_expansion,true);
  
  inArgsT.setSupports(Thyra::ModelEvaluatorBase::IN_ARG_x_mp,true);
  inArgsT.setSupports(Thyra::ModelEvaluatorBase::IN_ARG_x_dot_mp,true);
  for (int i=0; i<param_names.size(); i++)
    inArgsT.setSupports(Thyra::ModelEvaluatorBase::IN_ARG_p_mp, i, true); 
  */
  return prototypeInArgsT;
}

Thyra::ModelEvaluatorBase::OutArgs<ST> 
Albany::ModelEvaluatorT::createOutArgs() const
{
  //The following I believe are not yet implemented in Thyra::ModelEvaluatorBase 
  // Stochastic
  /*outArgs.setSupports(OUT_ARG_f_sg,true);
  outArgs.setSupports(OUT_ARG_W_sg,true);
  for (int i=0; i<param_names.size(); i++)
    outArgs.setSupports(OUT_ARG_DfDp_sg, i, DerivativeSupport(DERIV_MV_BY_COL));
  for (int i=0; i<n_g; i++) {
    outArgs.setSupports(OUT_ARG_g_sg, i, true);
    if (app->getResponse(i)->isScalarResponse()) {
      outArgs.setSupports(OUT_ARG_DgDx_sg, i, 
			  DerivativeSupport(DERIV_TRANS_MV_BY_ROW));
      outArgs.setSupports(OUT_ARG_DgDx_dot_sg, i, 
			  DerivativeSupport(DERIV_TRANS_MV_BY_ROW));
    }
    else {
       outArgs.setSupports(OUT_ARG_DgDx_sg, i, 
			   DerivativeSupport(DERIV_LINEAR_OP));
       outArgs.setSupports(OUT_ARG_DgDx_dot_sg, i, 
			   DerivativeSupport(DERIV_LINEAR_OP));
    }
    for (int j=0; j<param_names.size(); j++)
      outArgs.setSupports(OUT_ARG_DgDp_sg, i, j, 
			  DerivativeSupport(DERIV_MV_BY_COL));
  }
      
  // Multi-point
  outArgs.setSupports(OUT_ARG_f_mp,true);
  outArgs.setSupports(OUT_ARG_W_mp,true);
  for (int i=0; i<param_names.size(); i++)
    outArgs.setSupports(OUT_ARG_DfDp_mp, i, DerivativeSupport(DERIV_MV_BY_COL));
  for (int i=0; i<n_g; i++) {
    outArgs.setSupports(OUT_ARG_g_mp, i, true);
    if (app->getResponse(i)->isScalarResponse()) {
      outArgs.setSupports(OUT_ARG_DgDx_mp, i, 
			  DerivativeSupport(DERIV_TRANS_MV_BY_ROW));
      outArgs.setSupports(OUT_ARG_DgDx_dot_mp, i, 
			  DerivativeSupport(DERIV_TRANS_MV_BY_ROW));
    }
    else {
       outArgs.setSupports(OUT_ARG_DgDx_mp, i, 
			   DerivativeSupport(DERIV_LINEAR_OP));
       outArgs.setSupports(OUT_ARG_DgDx_dot_mp, i, 
			   DerivativeSupport(DERIV_LINEAR_OP));
    }
    for (int j=0; j<param_names.size(); j++)
      outArgs.setSupports(OUT_ARG_DgDp_mp, i, j, 
			  DerivativeSupport(DERIV_MV_BY_COL));
  }
*/
  return prototypeOutArgsT;
}


void 
Albany::ModelEvaluatorT::evalModel(const Thyra::ModelEvaluatorBase::InArgs<ST>& inArgsT, 
				   const Thyra::ModelEvaluatorBase::OutArgs<ST>& outArgsT) const
{
  Teuchos::TimeMonitor Timer(*timer); //start timer
  //
  // Get the input arguments
  //
  Teuchos::RCP<const Tpetra_Vector> xT = ConverterT::getConstTpetraVector(inArgsT.get_x());
  Teuchos::RCP<const Tpetra_Vector> x_dotT;
  double alpha     = 0.0;
  double beta      = 1.0;
  double curr_time = 0.0;
  x_dotT = ConverterT::getConstTpetraVector(inArgsT.get_x_dot());
  if (x_dotT != Teuchos::null) {
    alpha = inArgsT.get_alpha();
    beta = inArgsT.get_beta();
    curr_time  = inArgsT.get_t();
  }
  
  for (int i=0; i<inArgsT.Np(); i++) {
    Teuchos::RCP<const Tpetra_Vector> pT = ConverterT::getConstTpetraVector(inArgsT.get_p(i));
    if (pT != Teuchos::null) {
      Teuchos::ArrayRCP<const ST> pT_constView = pT->get1dView();
      for (unsigned int j=0; j<sacado_param_vec[i].size(); j++)
	sacado_param_vec[i][j].baseValue = pT_constView[j];
    }
  }

  //
  // Get the output arguments
  //
  //CAN WE JUST MAKE fT_out an RCP to a Tpetra_VECTOR???
  //EpetraExt::ModelEvaluator::Evaluation<Epetra_Vector> f_out = outArgs.get_f();
  Teuchos::RCP<Tpetra_Vector> fT_out = ConverterT::getTpetraVector(outArgsT.get_f()); 
  Teuchos::RCP<Tpetra_Operator> WT_out =ConverterT::getTpetraOperator(outArgsT.get_W());

  // Cast W to a CrsMatrix, throw an exception if this fails
  Teuchos::RCP<Tpetra_CrsMatrix> W_out_crsT;
  if (WT_out != Teuchos::null)
    W_out_crsT = Teuchos::rcp_dynamic_cast<Tpetra_CrsMatrix>(WT_out, true);


  //I believe OUT_ARG_WPrec is not implemented yet in Thyra::ModelEvaluatorBase  
  // Get preconditioner operator, if requested
  //Teuchos::RCP<Epetra_Operator> WPrec_out;
  //if (outArgs.supports(OUT_ARG_WPrec)) WPrec_out = outArgs.get_WPrec();

  //
  // Compute the functions
  //
  bool f_already_computed = false;

  // W matrix
  if (WT_out != Teuchos::null) {
    app->computeGlobalJacobianT(alpha, beta, curr_time, x_dotT.get(), *xT, 
			       sacado_param_vec, fT_out.get(), *W_out_crsT); 
    /*cout << "W_out_crs" << endl; 
    cout << *W_out_crs << endl; 
    cout << "x_dot" << endl; 
    cout << x_dot << endl; 
    cout << "f" << endl; 
    cout << f_out << endl; 
    cout << "x" << endl; 
    cout << *x << endl; 
    cout << "curr_time" << endl; 
    cout << curr_time << endl; 
    */
    f_already_computed=true;
  }
 
  //Stuff having to do with WPrec_out is not implemented in Thyra
  /*if (WPrec_out != Teuchos::null) {
    app->computeGlobalJacobian(alpha, beta, curr_time, x_dot.get(), *x, 
			       sacado_param_vec, f_out.get(), *Extra_W_crs);
    f_already_computed=true;
    //convert computeGlobalPreconditioner to Tpetra??
    app->computeGlobalPreconditioner(Extra_W_crs, WPrec_out);
  }
  */

  // df/dp
  for (int i=0; i<outArgsT.Np(); i++) {
    Teuchos::RCP<Tpetra_MultiVector> dfdp_outT = 
      ConverterT::getTpetraMultiVector(outArgsT.get_DfDp(i).getMultiVector());
    if (dfdp_outT != Teuchos::null) {
      //Thyra::ModelEvaluatorBase::DerivativeMultiVector does not have method getParamIndexes()!
      //Teuchos::Array<int> p_indexes = 
	//outArgsT.get_DfDp(i).getDerivativeMultiVector().getParamIndexes();
      Teuchos::RCP<ParamVec> p_vec;
      //if (p_indexes.size() == 0)
	//p_vec = Teuchos::rcp(&sacado_param_vec[i],false);
      //else {
	//p_vec = Teuchos::rcp(new ParamVec);
	//for (int j=0; j<p_indexes.size(); j++)
	  //p_vec->addParam(sacado_param_vec[i][p_indexes[j]].family, 
	//		  sacado_param_vec[i][p_indexes[j]].baseValue);
      //}
  
      app->computeGlobalTangentT(curr_time, 0.0, 0.0, false, x_dotT.get(), *xT, 
				sacado_param_vec, p_vec.get(),
				NULL, NULL, NULL, fT_out.get(), NULL, 
				dfdp_outT.get());
      
      f_already_computed=true;
    }
  }
 
  // f
  if (app->is_adjoint) {
    Thyra::ModelEvaluatorBase::Derivative<ST> f_derivT(outArgsT.get_f(), Thyra::ModelEvaluatorBase::DERIV_TRANS_MV_BY_ROW);
    int response_index = 0; // need to add capability for sending this in
    app->evaluateResponseDerivativeT(response_index, curr_time, x_dotT.get(), *xT, 
				    sacado_param_vec, NULL, 
				    NULL, f_derivT, Thyra::ModelEvaluatorBase::Derivative<ST>::Derivative(), Thyra::ModelEvaluatorBase::Derivative<ST>::Derivative());
  }
  else {
    if (fT_out != Teuchos::null && !f_already_computed) {
      app->computeGlobalResidualT(curr_time, x_dotT.get(), *xT, 
  			       sacado_param_vec, *fT_out);
    }
  }
  // Response functions
  for (int i=0; i<outArgsT.Ng(); i++) {
    Teuchos::RCP<Tpetra_Vector> gT_out = ConverterT::getTpetraVector(outArgsT.get_g(i));
    Thyra::ModelEvaluatorBase::Derivative<ST> dgdxT_out = outArgsT.get_DgDx(i);
    Thyra::ModelEvaluatorBase::Derivative<ST> dgdxdotT_out = outArgsT.get_DgDx_dot(i);
    bool g_computed = false;

    // dg/dx, dg/dxdot
    if (!dgdxT_out.isEmpty() || !dgdxdotT_out.isEmpty()) {
      app->evaluateResponseDerivativeT(i, curr_time, x_dotT.get(), *xT, 
				      sacado_param_vec, NULL,
				      gT_out.get(), dgdxT_out, 
				      dgdxdotT_out, Thyra::ModelEvaluatorBase::Derivative<ST>::Derivative());
      g_computed = true;
    }
    
    // dg/dp
    for (int j=0; j<outArgsT.Np(); j++) {
      Teuchos::RCP<Tpetra_MultiVector> dgdpT_out = 
	ConverterT::getTpetraMultiVector(outArgsT.get_DgDp(i,j).getMultiVector());
      if (dgdpT_out != Teuchos::null) {
        //Thyra::ModelEvaluatorBase::Derivative does not appear to have method getParamIndexes
	//Teuchos::Array<int> p_indexes = 
	//  outArgs.get_DgDp(i,j).getDerivativeMultiVector().getParamIndexes();
	Teuchos::RCP<ParamVec> p_vec;
	//if (p_indexes.size() == 0)
	//  p_vec = Teuchos::rcp(&sacado_param_vec[j],false);
	//else {
	//  p_vec = Teuchos::rcp(new ParamVec);
	//  for (int k=0; k<p_indexes.size(); k++)
	//    p_vec->addParam(sacado_param_vec[j][p_indexes[k]].family, 
	//		    sacado_param_vec[j][p_indexes[k]].baseValue);
	//}
	app->evaluateResponseTangentT(i, alpha, beta, curr_time, false,
				     x_dotT.get(), *xT, 
				     sacado_param_vec, p_vec.get(),
				     NULL, NULL, NULL, gT_out.get(), NULL,
				     dgdpT_out.get());
	g_computed = true;
      }
    }

    
    if (gT_out != Teuchos::null && !g_computed) {
      app->evaluateResponseT(i, curr_time, x_dotT.get(), *xT, sacado_param_vec, 
			    *gT_out);
    }
    
  }
  
  /*
  //
  // Stochastic Galerkin
  //
  InArgs::sg_const_vector_t x_sg = inArgs.get_x_sg();
  if (x_sg != Teuchos::null) {
    app->init_sg(inArgs.get_sg_basis(), 
		 inArgs.get_sg_quadrature(), 
		 inArgs.get_sg_expansion(), 
		 x_sg->productComm());
    InArgs::sg_const_vector_t x_dot_sg  = inArgs.get_x_dot_sg();
    InArgs::sg_const_vector_t epetra_p_sg = inArgs.get_p_sg(0);
    Teuchos::Array<int> p_sg_index;
    for (int i=0; i<inArgs.Np(); i++) {
      InArgs::sg_const_vector_t p_sg = inArgs.get_p_sg(i);
      if (p_sg != Teuchos::null) {
	p_sg_index.push_back(i);
	for (int j=0; j<p_sg_vals[i].size(); j++) {
	  int num_sg_blocks = p_sg->size();
	  p_sg_vals[i][j].reset(app->getStochasticExpansion(), num_sg_blocks);
	  p_sg_vals[i][j].copyForWrite();
	  for (int l=0; l<num_sg_blocks; l++) {
	    p_sg_vals[i][j].fastAccessCoeff(l) = (*p_sg)[l][j];
	  }
	}
      }
    }

    OutArgs::sg_vector_t f_sg = outArgs.get_f_sg();
    OutArgs::sg_operator_t W_sg = outArgs.get_W_sg();
    bool f_sg_computed = false;

    // W_sg
    if (W_sg != Teuchos::null) {
      Stokhos::VectorOrthogPoly<Epetra_CrsMatrix> W_sg_crs(W_sg->basis(), 
							   W_sg->map());
      for (int i=0; i<W_sg->size(); i++)
	W_sg_crs.setCoeffPtr(
	  i,
	  Teuchos::rcp_dynamic_cast<Epetra_CrsMatrix>(W_sg->getCoeffPtr(i)));
      app->computeGlobalSGJacobian(alpha, beta, curr_time, 
				   x_dot_sg.get(), *x_sg, 
				   sacado_param_vec, p_sg_index, p_sg_vals,
				   f_sg.get(), W_sg_crs);
      f_sg_computed = true;
    }

    // df/dp_sg
    for (int i=0; i<outArgs.Np(); i++) {
      Teuchos::RCP< Stokhos::EpetraMultiVectorOrthogPoly > dfdp_sg 
	= outArgs.get_DfDp_sg(i).getMultiVector();
      if (dfdp_sg != Teuchos::null) {
	Teuchos::Array<int> p_indexes = 
	  outArgs.get_DfDp_sg(i).getDerivativeMultiVector().getParamIndexes();
	Teuchos::RCP<ParamVec> p_vec;
	if (p_indexes.size() == 0)
	  p_vec = Teuchos::rcp(&sacado_param_vec[i],false);
	else {
	  p_vec = Teuchos::rcp(new ParamVec);
	  for (int j=0; j<p_indexes.size(); j++)
	    p_vec->addParam(sacado_param_vec[i][p_indexes[j]].family, 
			    sacado_param_vec[i][p_indexes[j]].baseValue);
	}
	
	app->computeGlobalSGTangent(0.0, 0.0, curr_time, false, 
				    x_dot_sg.get(), *x_sg, 
				    sacado_param_vec, p_sg_index, p_sg_vals, 
				    p_vec.get(), NULL, NULL, NULL,  
				    f_sg.get(), NULL, dfdp_sg.get());
	
	f_sg_computed = true;
      }
    }

    if (f_sg != Teuchos::null && !f_sg_computed)
      app->computeGlobalSGResidual(curr_time, x_dot_sg.get(), *x_sg, 
				   sacado_param_vec, p_sg_index, p_sg_vals,
				   *f_sg);

    // Response functions
    for (int i=0; i<outArgs.Ng(); i++) {
      OutArgs::sg_vector_t g_sg = outArgs.get_g_sg(i);
      SGDerivative dgdx_sg = outArgs.get_DgDx_sg(i);
      SGDerivative dgdxdot_sg = outArgs.get_DgDx_dot_sg(i);
      bool g_sg_computed = false;

      // dg/dx, dg/dxdot
      if (!dgdx_sg.isEmpty() || !dgdxdot_sg.isEmpty()) {
	app->evaluateSGResponseDerivative(
	  i, curr_time, x_dot_sg.get(), *x_sg, 
	  sacado_param_vec, p_sg_index, p_sg_vals,
	  NULL, g_sg.get(), dgdx_sg, 
	  dgdxdot_sg, SGDerivative());
	g_sg_computed = true;
      }
    
      // dg/dp
      for (int j=0; j<outArgs.Np(); j++) {
	Teuchos::RCP< Stokhos::EpetraMultiVectorOrthogPoly > dgdp_sg = 
	  outArgs.get_DgDp_sg(i,j).getMultiVector();
	if (dgdp_sg != Teuchos::null) {
	  Teuchos::Array<int> p_indexes = 
	    outArgs.get_DgDp_sg(i,j).getDerivativeMultiVector().getParamIndexes();
	  Teuchos::RCP<ParamVec> p_vec;
	  if (p_indexes.size() == 0)
	    p_vec = Teuchos::rcp(&sacado_param_vec[j],false);
	  else {
	    p_vec = Teuchos::rcp(new ParamVec);
	    for (int k=0; k<p_indexes.size(); k++)
	      p_vec->addParam(sacado_param_vec[j][p_indexes[k]].family, 
			      sacado_param_vec[j][p_indexes[k]].baseValue);
	  }
	  app->evaluateSGResponseTangent(i, alpha, beta, curr_time, false,
					 x_dot_sg.get(), *x_sg, 
					 sacado_param_vec, p_sg_index, 
					 p_sg_vals, p_vec.get(), 
					 NULL, NULL, NULL, g_sg.get(), 
					 NULL, dgdp_sg.get());
	  g_sg_computed = true;
					 
	}
      }
      
      if (g_sg != Teuchos::null && !g_sg_computed)
	app->evaluateSGResponse(i, curr_time, x_dot_sg.get(), *x_sg, 
				sacado_param_vec, p_sg_index, p_sg_vals, 
				*g_sg);
    }
  }

  //
  // Multi-point evaluation
  //
  mp_const_vector_t x_mp = inArgs.get_x_mp();
  if (x_mp != Teuchos::null) {
    mp_const_vector_t x_dot_mp  = inArgs.get_x_dot_mp();
    Teuchos::Array<int> p_mp_index;
    for (int i=0; i<inArgs.Np(); i++) {
      mp_const_vector_t p_mp = inArgs.get_p_mp(i);
      if (p_mp != Teuchos::null) {
	p_mp_index.push_back(i);
	for (int j=0; j<p_mp_vals[i].size(); j++) {
	  int num_mp_blocks = p_mp->size();
	  p_mp_vals[i][j].reset(num_mp_blocks);
	  p_mp_vals[i][j].copyForWrite();
	  for (int l=0; l<num_mp_blocks; l++) {
	    p_mp_vals[i][j].fastAccessCoeff(l) = (*p_mp)[l][j];
	  }
	}
      }
    }
    
    mp_vector_t f_mp = outArgs.get_f_mp();
    mp_operator_t W_mp = outArgs.get_W_mp();
    bool f_mp_computed = false;

    // W_mp
    if (W_mp != Teuchos::null) {
      Stokhos::ProductContainer<Epetra_CrsMatrix> W_mp_crs(W_mp->map());
      for (int i=0; i<W_mp->size(); i++)
	W_mp_crs.setCoeffPtr(
	  i,
	  Teuchos::rcp_dynamic_cast<Epetra_CrsMatrix>(W_mp->getCoeffPtr(i)));
      app->computeGlobalMPJacobian(alpha, beta, curr_time, 
				   x_dot_mp.get(), *x_mp, 
				   sacado_param_vec, p_mp_index, p_mp_vals,
				   f_mp.get(), W_mp_crs);
      f_mp_computed = true;
    }

    // df/dp_mp
    for (int i=0; i<outArgs.Np(); i++) {
      Teuchos::RCP< Stokhos::ProductEpetraMultiVector > dfdp_mp 
	= outArgs.get_DfDp_mp(i).getMultiVector();
      if (dfdp_mp != Teuchos::null) {
	Teuchos::Array<int> p_indexes = 
	  outArgs.get_DfDp_mp(i).getDerivativeMultiVector().getParamIndexes();
	Teuchos::RCP<ParamVec> p_vec;
	if (p_indexes.size() == 0)
	  p_vec = Teuchos::rcp(&sacado_param_vec[i],false);
	else {
	  p_vec = Teuchos::rcp(new ParamVec);
	  for (int j=0; j<p_indexes.size(); j++)
	    p_vec->addParam(sacado_param_vec[i][p_indexes[j]].family, 
			    sacado_param_vec[i][p_indexes[j]].baseValue);
	}
	    
	app->computeGlobalMPTangent(0.0, 0.0, curr_time, false, 
				    x_dot_mp.get(), *x_mp, 
				    sacado_param_vec, p_mp_index, p_mp_vals, 
				    p_vec.get(), NULL, NULL, NULL,
				    f_mp.get(), NULL, dfdp_mp.get());
	
	f_mp_computed = true;
      }
    }

    if (f_mp != Teuchos::null && !f_mp_computed)
      app->computeGlobalMPResidual(curr_time, x_dot_mp.get(), *x_mp, 
				   sacado_param_vec, p_mp_index, p_mp_vals,
				   *f_mp);

    // Response functions
    for (int i=0; i<outArgs.Ng(); i++) {
      mp_vector_t g_mp = outArgs.get_g_mp(i);
      MPDerivative dgdx_mp = outArgs.get_DgDx_mp(i);
      MPDerivative dgdxdot_mp = outArgs.get_DgDx_dot_mp(i);
      bool g_mp_computed = false;

      // dg/dx, dg/dxdot
      if (!dgdx_mp.isEmpty() || !dgdxdot_mp.isEmpty()) {
	app->evaluateMPResponseDerivative(
	  i, curr_time, x_dot_mp.get(), *x_mp, 
	  sacado_param_vec, p_mp_index, p_mp_vals,
	  NULL, g_mp.get(), dgdx_mp, 
	  dgdxdot_mp, MPDerivative());
	g_mp_computed = true;
      }
      
      // dg/dp
      for (int j=0; j<outArgs.Np(); j++) {
	Teuchos::RCP< Stokhos::ProductEpetraMultiVector > dgdp_mp = 
	  outArgs.get_DgDp_mp(i,j).getMultiVector();
	if (dgdp_mp != Teuchos::null) {
	  Teuchos::Array<int> p_indexes = 
	    outArgs.get_DgDp_mp(i,j).getDerivativeMultiVector().getParamIndexes();
	  Teuchos::RCP<ParamVec> p_vec;
	  if (p_indexes.size() == 0)
	    p_vec = Teuchos::rcp(&sacado_param_vec[j],false);
	  else {
	    p_vec = Teuchos::rcp(new ParamVec);
	    for (int k=0; k<p_indexes.size(); k++)
	      p_vec->addParam(sacado_param_vec[j][p_indexes[k]].family, 
			      sacado_param_vec[j][p_indexes[k]].baseValue);
	  }
	  app->evaluateMPResponseTangent(i, alpha, beta, curr_time, false,
					 x_dot_mp.get(), *x_mp, 
					 sacado_param_vec, p_mp_index, 
					 p_mp_vals, p_vec.get(), 
					 NULL, NULL, NULL, g_mp.get(), 
					 NULL, dgdp_mp.get());
	  g_mp_computed = true;
	}
      }
          
      if (g_mp != Teuchos::null && !g_mp_computed)
	app->evaluateMPResponse(i, curr_time, x_dot_mp.get(), *x_mp, 
				sacado_param_vec, p_mp_index, p_mp_vals, 
				*g_mp);
    }
  }
 */ 
}
