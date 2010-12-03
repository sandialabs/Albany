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
  else
    TEST_FOR_EXCEPTION(true, Teuchos::Exceptions::InvalidParameter,
		       std::endl << "Error!  ENAT_SGNOXSolver():  " <<
		       "Invalid SG Method  " << sg_type << std::endl);

  // Create SG basis
  basis = Stokhos::BasisFactory<int,double>::create(sgParams);
  int numParameters = basis->dimension();
  if (comm->MyPID()==0) std::cout << "Basis size = " << basis->size() << std::endl;

  // Set up stochastic parameters
  Epetra_LocalMap p_sg_map(numParameters, 0, *comm);
  Teuchos::Array< Teuchos::RCP<Stokhos::EpetraVectorOrthogPoly> > sg_p;
  int sg_p_index;
  if (sg_method == SG_AD) {
    sg_p.resize(1);
    sg_p_index = 0;
  }
  else {
    // When SGQuadModelEvaluator is used, there are 2 SG parameter vectors
    sg_p.resize(2);
    sg_p_index = 1;
  }
  sg_p[sg_p_index] = 
    Teuchos::rcp(new Stokhos::EpetraVectorOrthogPoly(basis, p_sg_map));
  Teuchos::ParameterList& basisParams = sgParams.sublist("Basis");
  for (int i=0; i<numParameters; i++) {
    std::ostringstream ss;
    ss << "Basis " << i;
    Teuchos::ParameterList& bp = basisParams.sublist(ss.str());
    Teuchos::Array<double> initial_p_vals = 
      Teuchos::getArrayFromStringParameter<double>(
	bp, std::string("Initial Expansion Coefficients"), -1, false);
    if (initial_p_vals.size() == 0) {
      sg_p[sg_p_index]->term(i,0)[i] = 0.0;
      sg_p[sg_p_index]->term(i,1)[i] = 1.0;  // Set order 1 coeff to 1 for this RV
    }
    else
      for (unsigned int j = 0; j<initial_p_vals.size(); j++)
	(*sg_p[sg_p_index])[j][i] = initial_p_vals[j];
  }

  // Setup stochastic initial guess
  Teuchos::RCP<Stokhos::EpetraVectorOrthogPoly> sg_x = 
    Teuchos::rcp(new Stokhos::EpetraVectorOrthogPoly(basis, 
						     *(model->get_x_map())));

  // SG Quadrature
  std::string exp_type = sgParams.get("AD Expansion Type", "Quadrature");
  if (exp_type == "Quadrature" || 
      sg_method == SG_GLOBAL ||
      sg_method == SG_NI) {
    quad = Stokhos::QuadratureFactory<int,double>::create(sgParams);
  }

  // Create expansion
  Teuchos::RCP<Stokhos::OrthogPolyExpansion<int,double> > expansion = 
    Stokhos::ExpansionFactory<int,double>::create(sgParams);
  Teuchos::RCP<const Stokhos::Sparse3Tensor<int,double> > Cijk = 
    sgParams.get< Teuchos::RCP<const Stokhos::Sparse3Tensor<int,double> > >("Triple Product Tensor");
    
  // Set up stochastic Galerkin model
  Teuchos::RCP<EpetraExt::ModelEvaluator> sg_model;
  if (sg_method == SG_AD) {
    sg_model = model;
  }
  else {
    Teuchos::RCP<EpetraExt::ModelEvaluator> underlying_model;
    if (sg_method == SG_GLOBAL)
      underlying_model = model;
    else 
      underlying_model =
	Teuchos::rcp(new Piro::Epetra::NOXSolver(appParams, model));
    sg_model =
      Teuchos::rcp(new Stokhos::SGQuadModelEvaluator(underlying_model, 
						     basis));
  }

  // Create stochastic parallel distribution
  Teuchos::RCP<const EpetraExt::MultiComm> sg_comm = 
    Teuchos::rcp_dynamic_cast<const EpetraExt::MultiComm>(comm, true);
  Teuchos::ParameterList parallelParams;
  Teuchos::RCP<Stokhos::ParallelData> sg_parallel_data =
    Teuchos::rcp(new Stokhos::ParallelData(basis, Cijk, sg_comm,
					   parallelParams));

  // Set up SG nonlinear model
  Teuchos::RCP<Stokhos::SGModelEvaluator> sg_nonlin_model =
    Teuchos::rcp(new Stokhos::SGModelEvaluator(sg_model, basis, quad, expansion,
					       sg_parallel_data, 
					       sgSolverParams,
					       Teuchos::null, sg_p));

  // Set up Observer to call noxObserver for each vector block
  Teuchos::RCP<NOX::Epetra::Observer> sgnoxObserver;
  if (noxObserver != Teuchos::null)
    sgnoxObserver = 
      Teuchos::rcp(new Piro::Epetra::StokhosNOXObserver(noxObserver,
							*(model->get_x_map()),
							basis->size()));

  // Create SG NOX solver
  Teuchos::RCP<EpetraExt::ModelEvaluator> sg_block_solver;
  if (sg_method != SG_NI) {
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
  if (sg_method != SG_NI) {
    sg_inverse_g_index.push_back(sg_inverse_g_index[sg_inverse_g_index.size()-1]+1);
    base_g_maps.push_back(model->get_x_map());
  }
  sg_solver = 
    Teuchos::rcp(new Stokhos::SGInverseModelEvaluator(sg_block_solver, 
						      basis,
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
  return sg_solver->get_p_sg_init(l);
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
  validPL->sublist("Basis", false, "");
  validPL->sublist("Expansion", false, "");
  validPL->sublist("Quadrature", false, "");
  validPL->set<std::string>("SG Method", "","");
  validPL->set<std::string>("Triple Product Size", "","");
  return validPL;
}

