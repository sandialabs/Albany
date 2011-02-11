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


#include "Albany_ModelEvaluator.hpp"
#include "Teuchos_ScalarTraits.hpp"
#include "Teuchos_TestForException.hpp"
#include "Stokhos_EpetraVectorOrthogPoly.hpp"

Albany::ModelEvaluator::ModelEvaluator(
  const Teuchos::RCP<Albany::Application>& app_,
  const Teuchos::RCP< Teuchos::Array<std::string> >& free_param_names,
  const Teuchos::RCP< Teuchos::Array<std::string> >& sg_param_names) 
  : app(app_),
    supports_p(false),
    supports_g(false),
    supports_sg(false),
    supplies_prec(app_->suppliesPreconditioner())
{
  // Compute number of parameter vectors
  int num_param_vecs = 0;
  if (free_param_names != Teuchos::null) {
    supports_p = true;
    num_param_vecs = 1;
    if (sg_param_names != Teuchos::null) {
      supports_sg = true;
      num_param_vecs = 2;
    }
  }
  else {
    TEST_FOR_EXCEPTION(sg_param_names != Teuchos::null, Teuchos::Exceptions::InvalidParameter,
                       std::endl << "Error in Albany::ModelEvaluator " <<
                       "logic for sg_params requires having free_params "<< std::endl);
  }

  if (num_param_vecs > 0) {
    param_names.resize(num_param_vecs);
    sacado_param_vec.resize(num_param_vecs);
    epetra_param_map.resize(num_param_vecs);
    epetra_param_vec.resize(num_param_vecs);

    // Set parameter names
    param_names[0] = free_param_names;
    if (num_param_vecs == 2)
      param_names[1] = sg_param_names;

    // Initialize each parameter vector
    const Epetra_Comm& comm = app->getMap()->Comm();
    for (int i=0; i<num_param_vecs; i++) {

      // Initialize Sacado parameter vector
      sacado_param_vec[i] = Teuchos::rcp(new ParamVec);
      if (param_names[i] != Teuchos::null)
	app->getParamLib()->fillVector<PHAL::AlbanyTraits::Residual>(*(param_names[i]), *(sacado_param_vec[i]));

      // Create Epetra map for parameter vector
      epetra_param_map[i] = 
	Teuchos::rcp(new Epetra_LocalMap(sacado_param_vec[i]->size(), 0, comm));

      // Create Epetra vector for parameters
      epetra_param_vec[i] = 
	Teuchos::rcp(new Epetra_Vector(*(epetra_param_map[i])));
  
      // Set parameters
      for (unsigned int j=0; j<sacado_param_vec[i]->size(); j++)
	(*(epetra_param_vec[i]))[j] = (*(sacado_param_vec[i]))[j].baseValue;

    }

    // Create storage for SG parameter values
    if (supports_sg)
      p_sg_vals.resize(sg_param_names->size());
  }

  supports_g = (app->getResponseMap() != Teuchos::null);

  timer = Teuchos::TimeMonitor::getNewTimer("Albany: **Total Fill Time**");
}

// Overridden from EpetraExt::ModelEvaluator

Teuchos::RCP<const Epetra_Map>
Albany::ModelEvaluator::get_x_map() const
{
  return app->getMap();
}

Teuchos::RCP<const Epetra_Map>
Albany::ModelEvaluator::get_f_map() const
{
  return app->getMap();
}

Teuchos::RCP<const Epetra_Map>
Albany::ModelEvaluator::get_p_map(int l) const
{
  TEST_FOR_EXCEPTION(supports_p == false, 
                     Teuchos::Exceptions::InvalidParameter,
                     std::endl << 
                     "Error!  Albany::ModelEvaluator::get_p_map():  " <<
                     "No parameters have been supplied.  " <<
                     "Supplied index l = " << l << std::endl);
  TEST_FOR_EXCEPTION(l >= static_cast<int>(epetra_param_map.size()) || l < 0, 
		     Teuchos::Exceptions::InvalidParameter,
                     std::endl << 
                     "Error!  Albany::ModelEvaluator::get_p_map():  " <<
                     "Invalid parameter index l = " << l << std::endl);

  return epetra_param_map[l];
}

Teuchos::RCP<const Epetra_Map>
Albany::ModelEvaluator::get_p_sg_map(int l) const
{
  TEST_FOR_EXCEPTION(supports_sg == false, 
                     Teuchos::Exceptions::InvalidParameter,
                     std::endl << 
                     "Error!  Albany::ModelEvaluator::get_p_sg_map():  " <<
                     "SG is not enabled.");
  TEST_FOR_EXCEPTION(l != 0, 
		     Teuchos::Exceptions::InvalidParameter,
                     std::endl << 
                     "Error!  Albany::ModelEvaluator::get_p_sg_map():  " <<
                     "Invalid parameter index l = " << l << std::endl);

  return epetra_param_map[1];
}

Teuchos::RCP<const Epetra_Map>
Albany::ModelEvaluator::get_g_map(int l) const
{
  TEST_FOR_EXCEPTION(supports_g == false, 
                     Teuchos::Exceptions::InvalidParameter,
                     std::endl << 
                     "Error!  Albany::ModelEvaluator::get_g_map():  " <<
                     "No response functions have been supplied.  " <<
                     "Supplied index l = " << l << std::endl);
  TEST_FOR_EXCEPTION(l != 0, Teuchos::Exceptions::InvalidParameter,
                     std::endl << 
                     "Error!  Albany::ModelEvaluator::get_g_map() only " <<
                     " supports 1 response vector.  Supplied index l = " << 
                     l << std::endl);

  return app->getResponseMap();
}

Teuchos::RCP<const Epetra_Map>
Albany::ModelEvaluator::get_g_sg_map(int l) const
{
  TEST_FOR_EXCEPTION(supports_sg == false, 
                     Teuchos::Exceptions::InvalidParameter,
                     std::endl << 
                     "Error!  Albany::ModelEvaluator::get_g_sg_map():  " <<
                     "SG is not enabled.");
  TEST_FOR_EXCEPTION(l != 0, Teuchos::Exceptions::InvalidParameter,
                     std::endl << 
                     "Error!  Albany::ModelEvaluator::get_g_sg_map() only " <<
                     " supports 1 response vector.  Supplied index l = " << 
                     l << std::endl);

  return app->getResponseMap();
}

Teuchos::RCP<const Teuchos::Array<std::string> >
Albany::ModelEvaluator::get_p_names(int l) const
{
  TEST_FOR_EXCEPTION(supports_p == false, 
                     Teuchos::Exceptions::InvalidParameter,
                     std::endl << 
                     "Error!  Albany::ModelEvaluator::get_p_names():  " <<
                     "No parameters have been supplied.  " <<
                     "Supplied index l = " << l << std::endl);
  TEST_FOR_EXCEPTION(l >= static_cast<int>(param_names.size()) || l < 0, 
		     Teuchos::Exceptions::InvalidParameter,
                     std::endl << 
                     "Error!  Albany::ModelEvaluator::get_p_names():  " <<
                     "Invalid parameter index l = " << l << std::endl);

  return param_names[l];
}

Teuchos::RCP<const Teuchos::Array<std::string> >
Albany::ModelEvaluator::get_p_sg_names(int l) const
{
  TEST_FOR_EXCEPTION(supports_sg == false, 
                     Teuchos::Exceptions::InvalidParameter,
                     std::endl << 
                     "Error!  Albany::ModelEvaluator::get_p_names():  " <<
                     "SG is not enabled.");
  TEST_FOR_EXCEPTION(l != 0, 
		     Teuchos::Exceptions::InvalidParameter,
                     std::endl << 
                     "Error!  Albany::ModelEvaluator::get_p_sg_names():  " <<
                     "Invalid parameter index l = " << l << std::endl);

  return param_names[1];
}

Teuchos::RCP<const Epetra_Vector>
Albany::ModelEvaluator::get_x_init() const
{
  return app->getInitialSolution();
}

Teuchos::RCP<const Epetra_Vector>
Albany::ModelEvaluator::get_x_dot_init() const
{
  return app->getInitialSolutionDot();
}

Teuchos::RCP<const Epetra_Vector>
Albany::ModelEvaluator::get_p_init(int l) const
{
  TEST_FOR_EXCEPTION(supports_p == false, 
                     Teuchos::Exceptions::InvalidParameter,
                     std::endl << 
                     "Error!  Albany::ModelEvaluator::get_p_init():  " <<
                     "No parameters have been supplied.  " <<
                     "Supplied index l = " << l << std::endl);
  TEST_FOR_EXCEPTION(l >= static_cast<int>(param_names.size()) || l < 0, 
		     Teuchos::Exceptions::InvalidParameter,
                     std::endl << 
                     "Error!  Albany::ModelEvaluator::get_p_init():  " <<
                     "Invalid parameter index l = " << l << std::endl);
  
  return epetra_param_vec[l];
}

Teuchos::RCP<Epetra_Operator>
Albany::ModelEvaluator::create_W() const
{
  return 
    Teuchos::rcp(new Epetra_CrsMatrix(::Copy, *(app->getJacobianGraph())));
}

Teuchos::RCP<EpetraExt::ModelEvaluator::Preconditioner>
Albany::ModelEvaluator::create_WPrec() const
{
  Teuchos::RCP<Epetra_Operator> precOp = app->getPreconditioner();

  // Teko prec needs space for Jacobian as well
  Extra_W_crs = Teuchos::rcp_dynamic_cast<Epetra_CrsMatrix>(create_W(), true);

  // bool is answer to: "Prec is already inverted?"
  return Teuchos::rcp(new EpetraExt::ModelEvaluator::Preconditioner(precOp,true));
}

EpetraExt::ModelEvaluator::InArgs
Albany::ModelEvaluator::createInArgs() const
{
  InArgsSetup inArgs;
  inArgs.setModelEvalDescription(this->description());
  inArgs.setSupports(IN_ARG_x,true);
  inArgs.set_Np(param_names.size());
  if (supports_sg) {
    inArgs.setSupports(IN_ARG_x_sg,true);
    inArgs.set_Np_sg(1); // 1 SG parameter vector
    inArgs.setSupports(IN_ARG_sg_expansion,true);
  }
  else
    inArgs.set_Np_sg(0);

  inArgs.setSupports(IN_ARG_t,true);
  inArgs.setSupports(IN_ARG_x_dot,true);
  inArgs.setSupports(IN_ARG_alpha,true);
  inArgs.setSupports(IN_ARG_beta,true);
  if (supports_sg)
    inArgs.setSupports(IN_ARG_x_dot_sg,true);

  return inArgs;
}

EpetraExt::ModelEvaluator::OutArgs
Albany::ModelEvaluator::createOutArgs() const
{
  OutArgsSetup outArgs;
  outArgs.setModelEvalDescription(this->description());
  
  if (supports_p && supports_g) {
    outArgs.set_Np_Ng(param_names.size(), 1);
    for (unsigned int i=0; i<param_names.size(); i++)
      outArgs.setSupports(OUT_ARG_DgDp, 0, i, 
			  DerivativeSupport(DERIV_MV_BY_COL));
  }
  else if (supports_p)
    outArgs.set_Np_Ng(1, 0);
  else if (supports_g)
    outArgs.set_Np_Ng(0, 1);
  else
    outArgs.set_Np_Ng(0, 0);

  if (supports_g) {
    outArgs.setSupports(OUT_ARG_DgDx, 0, DerivativeSupport(DERIV_MV_BY_COL));
    outArgs.setSupports(OUT_ARG_DgDx_dot, 0, DerivativeSupport(DERIV_MV_BY_COL));
  }
  if (supports_p)
    for (unsigned int i=0; i<param_names.size(); i++)
      outArgs.setSupports(OUT_ARG_DfDp, i, DerivativeSupport(DERIV_MV_BY_COL));

  outArgs.setSupports(OUT_ARG_f,true);
  outArgs.setSupports(OUT_ARG_W,true);
  outArgs.set_W_properties(
    DerivativeProperties(
      DERIV_LINEARITY_UNKNOWN ,DERIV_RANK_FULL ,true)
    );
  
  if (supplies_prec) outArgs.setSupports(OUT_ARG_WPrec, true);

  if (supports_sg) {
    outArgs.setSupports(OUT_ARG_f_sg,true);
    outArgs.setSupports(OUT_ARG_W_sg,true);
    if (supports_p && supports_g)
      outArgs.set_Np_Ng_sg(param_names.size(), 1);
    else if (supports_p)
      outArgs.set_Np_Ng_sg(1, 0);
    else if (supports_g)
      outArgs.set_Np_Ng_sg(0, 1);
    else
      outArgs.set_Np_Ng_sg(0, 0);
  }

  return outArgs;
}

void 
Albany::ModelEvaluator::evalModel(const InArgs& inArgs, 
				 const OutArgs& outArgs) const
{
  Teuchos::TimeMonitor Timer(*timer); //start timer
  //
  // Get the input arguments
  //
  Teuchos::RCP<const Epetra_Vector> x = inArgs.get_x();
  Teuchos::RCP<const Epetra_Vector> x_dot;
  double alpha     = 0.0;
  double beta      = 1.0;
  double curr_time = 0.0;
  x_dot = inArgs.get_x_dot();
  if (x_dot != Teuchos::null) {
    alpha = inArgs.get_alpha();
    beta = inArgs.get_beta();
    curr_time  = inArgs.get_t();
  }
  for (int i=0; i<inArgs.Np(); i++) {
    Teuchos::RCP<const Epetra_Vector> p = inArgs.get_p(i);
    if (p != Teuchos::null) {
      for (unsigned int j=0; j<sacado_param_vec[i]->size(); j++)
	(*(sacado_param_vec[i]))[j].baseValue = (*p)[j];
    }
  }

  //
  // Get the output arguments
  //
  EpetraExt::ModelEvaluator::Evaluation<Epetra_Vector> f_out = outArgs.get_f();
  Teuchos::RCP<Epetra_Operator> W_out = outArgs.get_W();
  Teuchos::RCP<Epetra_MultiVector> dfdp_out;
  if (outArgs.Np() > 0)
    dfdp_out = outArgs.get_DfDp(0).getMultiVector();

  // Cast W to a CrsMatrix, throw an exception if this fails
  Teuchos::RCP<Epetra_CrsMatrix> W_out_crs = 
    Teuchos::rcp_dynamic_cast<Epetra_CrsMatrix>(W_out, true);
  
  // Get preconditioner operator, if requested
  Teuchos::RCP<Epetra_Operator> WPrec_out;
  if (outArgs.supports(OUT_ARG_WPrec)) WPrec_out = outArgs.get_WPrec();

  //
  // Compute the functions
  //
  bool f_already_computed=false;

  // W matrix
  if (W_out != Teuchos::null) {
    app->computeGlobalJacobian(alpha, beta, curr_time, x_dot.get(), *x, 
			       sacado_param_vec, f_out.get(), *W_out_crs);
    f_already_computed=true;
  }

  if (WPrec_out != Teuchos::null) {
    app->computeGlobalJacobian(alpha, beta, curr_time, x_dot.get(), *x, 
			       sacado_param_vec, f_out.get(), *Extra_W_crs);
    f_already_computed=true;

    app->computeGlobalPreconditioner(Extra_W_crs, WPrec_out);
  }

  // df/dp
  if (supports_p) {
    for (int i=0; i<outArgs.Np(); i++) {
      Teuchos::RCP<Epetra_MultiVector> dfdp_out = 
	outArgs.get_DfDp(i).getMultiVector();
      if (dfdp_out != Teuchos::null) {
	Teuchos::Array<int> p_indexes = 
	  outArgs.get_DfDp(i).getDerivativeMultiVector().getParamIndexes();
	unsigned int n_params = p_indexes.size();
	Teuchos::RCP<ParamVec> p_vec;
	if (n_params > 0) {
	  Teuchos::Array<std::string> p_names(n_params);
	  for (unsigned int j=0; j<n_params; j++)
	    p_names[j] = (*(param_names[i]))[p_indexes[j]];
	  p_vec = Teuchos::rcp(new ParamVec);
	  app->getParamLib()->fillVector<PHAL::AlbanyTraits::Residual>(p_names, 
								       *p_vec);
	for (unsigned int j=0; j<p_vec->size(); j++)
	  (*p_vec)[j].baseValue = 
	    (*(sacado_param_vec[i]))[p_indexes[j]].baseValue;
	}
	else
	  p_vec = sacado_param_vec[i];
  
	app->computeGlobalTangent(0.0, 0.0, curr_time, false, 
				  x_dot.get(), *x, sacado_param_vec, 
				  p_vec.get(),
				  NULL, NULL, NULL, f_out.get(), NULL, 
				  dfdp_out.get());

	f_already_computed=true;
      }
    }
  }

  // f
  if (f_out != Teuchos::null && !f_already_computed) {
    app->computeGlobalResidual(curr_time, x_dot.get(), *x, 
			       sacado_param_vec, *f_out);
  }

  // Response functions
  if (outArgs.Ng() > 0 && supports_g) {
    Teuchos::RCP<Epetra_Vector> g_out = outArgs.get_g(0);
    Teuchos::RCP<Epetra_MultiVector> dgdx_out = 
      outArgs.get_DgDx(0).getMultiVector();
    Teuchos::RCP<Epetra_MultiVector> dgdxdot_out;
      dgdxdot_out = outArgs.get_DgDx_dot(0).getMultiVector();
    
    Teuchos::Array< Teuchos::RCP<ParamVec> > p_vec(outArgs.Np());
    Teuchos::Array< Teuchos::RCP<Epetra_MultiVector> > dgdp_out(outArgs.Np());
    bool have_dgdp = false;
    for (int i=0; i<outArgs.Np(); i++) {
      dgdp_out[i] = outArgs.get_DgDp(0,i).getMultiVector();
      if (dgdp_out[i] != Teuchos::null)
	have_dgdp = true;
      Teuchos::Array<int> p_indexes = 
	outArgs.get_DgDp(0,i).getDerivativeMultiVector().getParamIndexes();
      unsigned int n_params = p_indexes.size();
      if (n_params > 0) {
	Teuchos::Array<std::string> p_names(n_params);
	for (unsigned int j=0; j<n_params; j++)
	  p_names[i] = (*(param_names[i]))[p_indexes[j]];
	p_vec[i] = Teuchos::rcp(new ParamVec);
	app->getParamLib()->fillVector<PHAL::AlbanyTraits::Residual>(p_names, *(p_vec[i]));
	for (unsigned int j=0; j<p_vec[i]->size(); j++)
	  (*(p_vec[i]))[j].baseValue = (*(sacado_param_vec[i]))[p_indexes[j]].baseValue;
      }
      else
	p_vec[i] = sacado_param_vec[i];
    }

    if (have_dgdp ||dgdx_out != Teuchos::null || dgdxdot_out != Teuchos::null) {
      app->evaluateResponseGradients(x_dot.get(), *x, sacado_param_vec, p_vec, 
                                     g_out.get(), dgdx_out.get(), 
                                     dgdxdot_out.get(), dgdp_out);
    }
    else if (g_out != Teuchos::null)
      app->evaluateResponses(x_dot.get(), *x, sacado_param_vec, *g_out);
  }

  // Stochastic Galerkin
  if (supports_sg) {
    InArgs::sg_const_vector_t x_sg = inArgs.get_x_sg();
    if (x_sg != Teuchos::null) {
      app->init_sg(inArgs.get_sg_expansion());
      InArgs::sg_const_vector_t x_dot_sg = inArgs.get_x_dot_sg();
      InArgs::sg_const_vector_t epetra_p_sg = inArgs.get_p_sg(0);
      Teuchos::Array<SGType> *p_sg_ptr = NULL;
      if (epetra_p_sg != Teuchos::null) {
	for (unsigned int i=0; i<p_sg_vals.size(); i++) {
	  int num_sg_blocks = epetra_p_sg->size();
	  p_sg_vals[i].reset(app->getStochasticExpansion(), num_sg_blocks);
	  p_sg_vals[i].copyForWrite();
	  for (int j=0; j<num_sg_blocks; j++) {
	    p_sg_vals[i].fastAccessCoeff(j) = (*epetra_p_sg)[j][i];
	  }
	}
	p_sg_ptr = &p_sg_vals;
      }
      OutArgs::sg_vector_t f_sg = outArgs.get_f_sg();
      OutArgs::sg_operator_t W_sg = outArgs.get_W_sg();
      if (W_sg != Teuchos::null) {
	Stokhos::VectorOrthogPoly<Epetra_CrsMatrix> W_sg_crs(W_sg->basis(), 
							     W_sg->size());
	for (int i=0; i<W_sg->size(); i++)
	  W_sg_crs.setCoeffPtr(
	    i,
	    Teuchos::rcp_dynamic_cast<Epetra_CrsMatrix>(W_sg->getCoeffPtr(i)));
	app->computeGlobalSGJacobian(alpha, beta, curr_time, 
				     x_dot_sg.get(), *x_sg, 
				     sacado_param_vec[0].get(), 
				     sacado_param_vec[1].get(), p_sg_ptr,
				     f_sg.get(), W_sg_crs);
      }
      else if (f_sg != Teuchos::null)
	app->computeGlobalSGResidual(curr_time, x_dot_sg.get(), *x_sg, 
				     sacado_param_vec[0].get(), 
				     sacado_param_vec[1].get(), p_sg_ptr,
				     *f_sg);

      // Response functions
      if (outArgs.Ng() > 0 && supports_g) {
	OutArgs::sg_vector_t g_sg = outArgs.get_g_sg(0);
	if (g_sg != Teuchos::null)
	  app->evaluateSGResponses(x_dot_sg.get(), *x_sg, 
				   sacado_param_vec[0].get(), 
				   sacado_param_vec[1].get(), p_sg_ptr, 
				   *g_sg);
      }
    }
  }
}
