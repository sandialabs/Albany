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


#include "ENAT_SGNOXSolver.hpp"
#include "Piro_Epetra_NOXSolver.hpp"
#include "Stokhos.hpp"
#include "Stokhos_Epetra.hpp"
#include "Sacado_PCE_OrthogPoly.hpp"
#include "Teuchos_XMLParameterListHelpers.hpp"

#include "NOX_Epetra_LinearSystem_Stratimikos.H"
#include "NOX_Epetra_LinearSystem_MPBD.hpp"

ENAT::SGNOXSolver::
SGNOXSolver(const Teuchos::RCP<Teuchos::ParameterList>& appParams,
	    const Teuchos::RCP<EpetraExt::ModelEvaluator>& model,
	    const Teuchos::RCP<const Epetra_Comm>& comm,
            Teuchos::RCP<NOX::Epetra::Observer> noxObserver)
{
  Teuchos::ParameterList& problemParams = appParams->sublist("Problem");
  Teuchos::ParameterList& sgParams =
    problemParams.sublist("Stochastic Galerkin");
  sgParams.validateParameters(*getValidSGParameters(),0);
  Teuchos::RCP<Teuchos::ParameterList> sgSolverParams = 
    Teuchos::rcp(&(sgParams.sublist("SG Solver Parameters")),false);

  // Get SG expansion type
  std::string sg_type = sgParams.get("SG Method", "AD");
  SG_METHOD sg_method;
  if (sg_type == "AD")
    sg_method = SG_AD;
  else if (sg_type == "Global")
    sg_method = SG_GLOBAL;
  else if (sg_type == "Non-intrusive")
    sg_method = SG_NI;
  else if (sg_type == "Multi-point Non-intrusive")
    sg_method = SG_MPNI;
  else
    TEST_FOR_EXCEPTION(true, Teuchos::Exceptions::InvalidParameter,
		       std::endl << "Error!  ENAT_SGNOXSolver():  " <<
		       "Invalid SG Method  " << sg_type << std::endl);

  // Create SG basis
  Teuchos::ParameterList& sg_parameterParams = 
    sgParams.sublist("SG Parameters");
  Teuchos::ParameterList& sg_basisParams = sgParams.sublist("Basis");
  int numParameters = sg_parameterParams.get("Number", 0);
  int dim = sg_basisParams.get("Dimension", numParameters);
  TEST_FOR_EXCEPTION(dim != numParameters, std::logic_error,
		     std::endl << "Error!  ENAT_SGNOXSolver():  " <<
		     "Basis dimension (" << dim << ") does not match number " 
		     << " of SG parameters (" << numParameters << ")!" << 
		     std::endl);
  basis = Stokhos::BasisFactory<int,double>::create(sgParams);
  if (comm->MyPID()==0) 
    std::cout << "Basis size = " << basis->size() << std::endl;

  // SG Quadrature
  Teuchos::ParameterList& expParams = sgParams.sublist("Expansion");
  std::string exp_type = expParams.get("Type", "Quadrature");
  if (exp_type == "Quadrature" || 
      sg_method == SG_GLOBAL ||
      sg_method == SG_NI ||
      sg_method == SG_MPNI) {
    quad = Stokhos::QuadratureFactory<int,double>::create(sgParams);
    if (comm->MyPID()==0) 
      std::cout << "Quadrature size = " << quad->size() << std::endl;
  }

  // Create expansion
  Teuchos::RCP<Stokhos::OrthogPolyExpansion<int,double> > expansion;
  Teuchos::RCP<const Stokhos::Sparse3Tensor<int,double> > Cijk;
  if (sg_method != SG_NI && sg_method != SG_MPNI) {
    expansion = 
      Stokhos::ExpansionFactory<int,double>::create(sgParams);
    Cijk = 
      sgParams.get< Teuchos::RCP<const Stokhos::Sparse3Tensor<int,double> > >("Triple Product Tensor");
  }

   // Create stochastic parallel distribution
  Teuchos::RCP<const EpetraExt::MultiComm> sg_comm = 
    Teuchos::rcp_dynamic_cast<const EpetraExt::MultiComm>(comm, true);
  Teuchos::RCP<Stokhos::ParallelData> sg_parallel_data =
    Teuchos::rcp(new Stokhos::ParallelData(basis, Cijk, sg_comm,
					   sgParams));
    
  // Set up stochastic Galerkin model
  Teuchos::RCP<EpetraExt::ModelEvaluator> sg_model;
  if (sg_method == SG_AD) {
    sg_model = model;
  }
  else if (sg_method == SG_MPNI) {
    int num_mp = quad->size();
    Teuchos::RCP<const Epetra_Comm> mp_comm = 
      Stokhos::getStochasticComm(sg_comm);
    Teuchos::RCP<const Epetra_Map> mp_block_map = 
      Teuchos::rcp(new Epetra_Map(num_mp, 0, *mp_comm));
    Teuchos::RCP<EpetraExt::ModelEvaluator> mp_model = model;

    // Turn mp_model into an MP-nonlinear problem
    Teuchos::RCP<Teuchos::ParameterList> mpParams = 
    Teuchos::rcp(&(sgParams.sublist("MP Solver Parameters")),false);
    Teuchos::RCP<Stokhos::MPModelEvaluator> mp_nonlinear_model =
      Teuchos::rcp(new Stokhos::MPModelEvaluator(mp_model, sg_comm,
						 mp_block_map, mpParams));

    bool use_mpbd_solver = mpParams->get("Use MPBD Solver", false);
    Teuchos::RCP<NOX::Epetra::LinearSystem> linsys;
    Teuchos::RCP<NOX::Epetra::ModelEvaluatorInterface> nox_interface;
    if (use_mpbd_solver) {
      nox_interface = 
	Teuchos::rcp(new NOX::Epetra::ModelEvaluatorInterface(mp_nonlinear_model));
      Teuchos::RCP<Epetra_Operator> A = 
	mp_nonlinear_model->create_W();
      Teuchos::RCP<Epetra_Operator> M = 
	mp_nonlinear_model->create_WPrec()->PrecOp;
      Teuchos::RCP<NOX::Epetra::Interface::Required> iReq = 
	nox_interface;
      Teuchos::RCP<NOX::Epetra::Interface::Jacobian> iJac = 
	nox_interface;
      Teuchos::RCP<NOX::Epetra::Interface::Preconditioner> iPrec = 
	nox_interface;

      Teuchos::ParameterList& noxParams = appParams->sublist("NOX");
      Teuchos::ParameterList& printParams = noxParams.sublist("Printing");
      Teuchos::ParameterList& newtonParams = 
	noxParams.sublist("Direction").sublist("Newton");
      Teuchos::ParameterList& noxstratlsParams = 
	newtonParams.sublist("Stratimikos Linear Solver");

      Teuchos::RCP<const Teuchos::ParameterList> ortho_params = 
	Teuchos::rcp(new Teuchos::ParameterList);
      noxstratlsParams.sublist("Stratimikos").sublist("Linear Solver Types").sublist("Belos").sublist("Solver Types").sublist("GCRODR").set("Orthogonalization Parameters", ortho_params);


      Teuchos::ParameterList& mpbdParams = 
	mpParams->sublist("MPBD Linear Solver");
      mpbdParams.sublist("Deterministic Solver Parameters") = 
	noxstratlsParams;
      Teuchos::RCP<Epetra_Operator> inner_A = model->create_W();
      Teuchos::RCP<NOX::Epetra::ModelEvaluatorInterface> inner_nox_interface = 
	Teuchos::rcp(new NOX::Epetra::ModelEvaluatorInterface(model));
      Teuchos::RCP<NOX::Epetra::Interface::Required> inner_iReq = 
	inner_nox_interface;
      Teuchos::RCP<NOX::Epetra::Interface::Jacobian> inner_iJac = 
	inner_nox_interface;
      Teuchos::RCP<const Epetra_Vector> inner_u = model->get_x_init();
      Teuchos::RCP<NOX::Epetra::LinearSystem> inner_linsys = 
	Teuchos::rcp(new NOX::Epetra::LinearSystemStratimikos(
		       printParams, 
		       noxstratlsParams,
		       inner_iJac, inner_A, *inner_u));
      linsys = 
	Teuchos::rcp(new NOX::Epetra::LinearSystemMPBD(printParams, 
						       mpbdParams,
						       inner_linsys,
						       iReq, iJac, A,
						       model->get_x_map()));
    }

    // Create solver to map p -> g
    Teuchos::RCP<Piro::Epetra::NOXSolver> mp_solver =
      Teuchos::rcp(new Piro::Epetra::NOXSolver(appParams, mp_nonlinear_model,
					       Teuchos::null, nox_interface,
					       linsys));

    // Create MP inverse model evaluator to map p_mp -> g_mp
    Teuchos::Array<int> non_mp_inverse_p_index = 
      mp_nonlinear_model->get_non_p_mp_indices();
    Teuchos::Array<int> mp_inverse_p_index = 
      mp_nonlinear_model->get_p_mp_indices();
    Teuchos::Array<int> non_mp_inverse_g_index = 
      mp_nonlinear_model->get_non_g_mp_indices();
    Teuchos::Array<int> mp_inverse_g_index = 
      mp_nonlinear_model->get_g_mp_indices();
    Teuchos::Array< Teuchos::RCP<const Epetra_Map> > base_p_maps = 
      mp_nonlinear_model->get_p_mp_base_maps();
    Teuchos::Array< Teuchos::RCP<const Epetra_Map> > base_g_maps = 
      mp_nonlinear_model->get_g_mp_base_maps();
    Teuchos::RCP<EpetraExt::ModelEvaluator> mp_inverse_solver =
      Teuchos::rcp(new Stokhos::MPInverseModelEvaluator(mp_solver,
							mp_inverse_p_index, 
							non_mp_inverse_p_index,
							mp_inverse_g_index, 
							non_mp_inverse_g_index,
							base_p_maps, 
							base_g_maps));

    // Create MP-based SG Quadrature model evaluator to calculate g_sg
    sg_model =
      Teuchos::rcp(new Stokhos::SGQuadMPModelEvaluator(mp_inverse_solver, 
						       sg_comm, 
						       mp_block_map));
  }
  else {
    Teuchos::RCP<EpetraExt::ModelEvaluator> underlying_model;
    if (sg_method == SG_GLOBAL)
      underlying_model = model;
    else 
      underlying_model =
	Teuchos::rcp(new Piro::Epetra::NOXSolver(appParams, model));
    sg_model =
      Teuchos::rcp(new Stokhos::SGQuadModelEvaluator(underlying_model));
  }

  // Set up SG nonlinear model
  sg_nonlin_model =
    Teuchos::rcp(new Stokhos::SGModelEvaluator(sg_model, basis, quad, expansion,
					       sg_parallel_data, 
					       sgSolverParams));

  // Set up stochastic parameters
  Epetra_LocalMap p_sg_map(numParameters, 0, *comm);
  int sg_p_index;
  if (sg_method == SG_AD || sg_method == SG_MPNI) {
    sg_p_index = 0;
  }
  else {
    // When SGQuadModelEvaluator is used, there are 2 SG parameter vectors
    sg_p_index = 1;
  }
  Teuchos::RCP<Stokhos::EpetraVectorOrthogPoly> sg_p =
    sg_nonlin_model->create_p_sg(sg_p_index);
  Teuchos::ParameterList& basisParams = sgParams.sublist("Basis");
  for (int i=0; i<numParameters; i++) {
    std::ostringstream ss;
    ss << "Basis " << i;
    Teuchos::ParameterList& bp = basisParams.sublist(ss.str());
    Teuchos::Array<double> initial_p_vals;
    initial_p_vals = bp.get("Initial Expansion Coefficients",initial_p_vals);
    if (initial_p_vals.size() == 0) {
      sg_p->term(i,0)[i] = 0.0;
      sg_p->term(i,1)[i] = 1.0;  // Set order 1 coeff to 1 for this RV
    }
    else
      for (Teuchos::Array<double>::size_type j = 0; j<initial_p_vals.size(); j++)
	(*sg_p)[j][i] = initial_p_vals[j];
  }
  sg_nonlin_model->set_p_sg_init(sg_p_index, *sg_p);

  // Set other sg parameter vector when using quadrature
  if (sg_method != SG_AD && sg_method != SG_MPNI) {
    Teuchos::RCP<Stokhos::EpetraVectorOrthogPoly> sg_p0 =
    sg_nonlin_model->create_p_sg(0);
    (*sg_p0)[0] = *(model->get_p_init(0));
    sg_nonlin_model->set_p_sg_init(0, *sg_p0);
  }

  // Setup stochastic initial guess
  if (sg_method != SG_NI && sg_method != SG_MPNI) {
    Teuchos::RCP<Stokhos::EpetraVectorOrthogPoly> sg_x = 
      sg_nonlin_model->create_x_sg();
    sg_x->init(0.0);
    if (sg_x->myGID(0))
      (*sg_x)[0] = *(model->get_x_init());
    sg_nonlin_model->set_x_sg_init(*sg_x);
  }

  // Set up Observer to call noxObserver for each vector block
  Teuchos::RCP<NOX::Epetra::Observer> sgnoxObserver;
  if (noxObserver != Teuchos::null && sg_method != SG_NI && sg_method != SG_MPNI) {
    int save_moments = sgParams.get("Save Moments",-1);
    sgnoxObserver = 
      Teuchos::rcp(new Piro::Epetra::StokhosNOXObserver(
        noxObserver, basis, 
        sg_nonlin_model->get_overlap_stochastic_map(),
	model->get_x_map(), 
        sg_nonlin_model->get_x_sg_overlap_map(),
        sg_comm, sg_nonlin_model->get_x_sg_importer(), save_moments));
  }

  // Create SG NOX solver
  Teuchos::RCP<EpetraExt::ModelEvaluator> sg_block_solver;
  if (sg_method != SG_NI && sg_method != SG_MPNI) {
    // Will find preconditioner for Matrix-Free method
    sg_block_solver = Teuchos::rcp(new Piro::Epetra::NOXSolver(appParams, 
							       sg_nonlin_model, 
							       sgnoxObserver));
  }
  else 
    sg_block_solver = sg_nonlin_model;

  // Create SG Inverse model evaluator
  Teuchos::Array<int> non_sg_inverse_p_index = 
    sg_nonlin_model->get_non_p_sg_indices();
  Teuchos::Array<int> sg_inverse_p_index = sg_nonlin_model->get_p_sg_indices();
  Teuchos::Array<int> non_sg_inverse_g_index = 
    sg_nonlin_model->get_non_g_sg_indices();
  Teuchos::Array<int> sg_inverse_g_index = sg_nonlin_model->get_g_sg_indices();
  Teuchos::Array< Teuchos::RCP<const Epetra_Map> > base_p_maps = 
    sg_nonlin_model->get_p_sg_base_maps();
  Teuchos::Array< Teuchos::RCP<const Epetra_Map> > base_g_maps = 
    sg_nonlin_model->get_g_sg_base_maps();
  // Add sg_u response function supplied by Piro::Epetra::NOXSolver
  if (sg_method != SG_NI && sg_method != SG_MPNI) {
    sg_inverse_g_index.push_back(sg_inverse_g_index[sg_inverse_g_index.size()-1]+1);
    base_g_maps.push_back(model->get_x_map());
  }
  sg_solver = 
    Teuchos::rcp(new Stokhos::SGInverseModelEvaluator(sg_block_solver, 
						      sg_inverse_p_index, 
						      non_sg_inverse_p_index, 
						      sg_inverse_g_index, 
						      non_sg_inverse_g_index, 
						      base_p_maps, 
						      base_g_maps));
}

ENAT::SGNOXSolver::~SGNOXSolver()
{
}

Teuchos::RCP<const Epetra_Map> 
ENAT::SGNOXSolver::get_x_map() const
{
  return sg_solver->get_x_map();
}

Teuchos::RCP<const Epetra_Map> 
ENAT::SGNOXSolver::get_f_map() const
{
  return sg_solver->get_f_map();
}

Teuchos::RCP<const Epetra_Map> 
ENAT::SGNOXSolver::get_p_map(int l) const
{
  return sg_solver->get_p_map(l);
}

Teuchos::RCP<const Epetra_Map> 
ENAT::SGNOXSolver::get_p_sg_map(int l) const
{
  return sg_solver->get_p_sg_map(l);
}

Teuchos::RCP<const Epetra_Map> 
ENAT::SGNOXSolver::get_g_map(int j) const
{
  return sg_solver->get_g_map(j);
}

Teuchos::RCP<const Epetra_Map> 
ENAT::SGNOXSolver::get_g_sg_map(int j) const
{
  return sg_solver->get_g_sg_map(j);
}

Teuchos::RCP<const Epetra_Vector> 
ENAT::SGNOXSolver::get_x_init() const
{
  return sg_solver->get_x_init();
}

Teuchos::RCP<const Epetra_Vector> 
ENAT::SGNOXSolver::get_p_init(int l) const
{
  return sg_solver->get_p_init(l);
}

Teuchos::RCP<const Stokhos::EpetraVectorOrthogPoly>
ENAT::SGNOXSolver::get_p_sg_init(int l) const
{
  return sg_nonlin_model->get_p_sg_init(l);
}

EpetraExt::ModelEvaluator::InArgs 
ENAT::SGNOXSolver::createInArgs() const
{
  return sg_solver->createInArgs();
}

EpetraExt::ModelEvaluator::OutArgs 
ENAT::SGNOXSolver::createOutArgs() const
{
  return sg_solver->createOutArgs();
}

void 
ENAT::SGNOXSolver::evalModel(const InArgs& inArgs,
			     const OutArgs& outArgs ) const
{
  sg_solver->evalModel(inArgs, outArgs);
}

Teuchos::RCP<const Teuchos::ParameterList>
ENAT::SGNOXSolver::getValidSGParameters() const
{
  Teuchos::RCP<Teuchos::ParameterList> validPL =
     rcp(new Teuchos::ParameterList("ValidSGParams"));;
  validPL->sublist("SG Parameters", false, "");
  validPL->sublist("SG Solver Parameters", false, "");
  validPL->sublist("MP Solver Parameters", false, "");
  validPL->sublist("Basis", false, "");
  validPL->sublist("Expansion", false, "");
  validPL->sublist("Quadrature", false, "");
  validPL->set<std::string>("SG Method", "","");
  validPL->set<std::string>("Triple Product Size", "","");
  validPL->set<bool>("Rebalance Stochastic Graph", false, "");
  validPL->set<int>("Save Moments", -1, "Set to 2 for Mean and Variance. Default writes Coeffs");
  validPL->sublist("Isorropia", false, "");
  return validPL;
}

