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
  Teuchos::ParameterList& sg_parameterParams =
    sgParams.sublist("SG Parameters");
  int numParameters = sg_parameterParams.get("Number", 0);
  Teuchos::ParameterList& basisParams = sgParams.sublist("Basis");
  Teuchos::Array< Teuchos::RCP<const Stokhos::OneDOrthogPolyBasis<int,double> > > bases(numParameters);
  for (int i=0; i<numParameters; i++) {
    std::ostringstream ss;
    ss << "Basis " << i;
    Teuchos::ParameterList& bp = basisParams.sublist(ss.str());
    std::string type = bp.get("Type","Legendre");
    int order = bp.get("Order", 3);
    if (type == "Legendre")
      bases[i] = Teuchos::rcp(new Stokhos::LegendreBasis<int,double>(order));
    else if (type == "Clenshaw-Curtis")
      bases[i] = Teuchos::rcp(new Stokhos::ClenshawCurtisLegendreBasis<int,double>(order));
    else if (type == "Hermite")
      bases[i] = Teuchos::rcp(new Stokhos::HermiteBasis<int,double>(order));
    else if (type == "Rys") {
      double cut = bp.get("Weight Cut", 1.0);
      bool normalize = bp.get("Normalize", false);
      bases[i] = Teuchos::rcp(new Stokhos::RysBasis<int,double>(order, cut,
								normalize));
    }
    else
      TEST_FOR_EXCEPTION(true, Teuchos::Exceptions::InvalidParameter,
			 std::endl << "Error!  ENAT_SGNOXSolver():  " <<
			 "Invalid basis type  " << type << std::endl);

    
  }
  basis = 
    Teuchos::rcp(new Stokhos::CompletePolynomialBasis<int,double>(bases));
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
    std::string quad_type = sgParams.get("Quadrature Type", "Tensor Product");
    if (quad_type == "Tensor Product")
      quad = 
	Teuchos::rcp(new Stokhos::TensorProductQuadrature<int,double>(basis));
    else if (quad_type == "Sparse Grid") {
#ifdef HAVE_STOKHOS_DAKOTA
      if (sgParams.isType<int>("Sparse Grid Level")) {
	int level = sgParams.get<int>("Sparse Grid Level");
	quad = 
	  Teuchos::rcp(new Stokhos::SparseGridQuadrature<int,double>(basis,
								   level));
      }
      else
	quad = 
	  Teuchos::rcp(new Stokhos::SparseGridQuadrature<int,double>(basis));
#else
      TEST_FOR_EXCEPTION(true, Teuchos::Exceptions::InvalidParameter,
			 std::endl << "Error!  ENAT_SGNOXSolver():  " <<
			 "SparseGrid quadrature requires Dakota" << std::endl);
#endif
    }
    else
      TEST_FOR_EXCEPTION(true, Teuchos::Exceptions::InvalidParameter,
			 std::endl << "Error!  ENAT_SGNOXSolver():  " <<
			 "Invalid quadrature type  " << quad_type << std::endl);
  }

  // Triple product tensor
  std::string Cijk_type = sgSolverParams->get("Triple Product Size", "Full");
  int Cijk_sz = basis->size();
  if (Cijk_type == "Linear")
    Cijk_sz = basis->dimension()+1;
  Teuchos::RCP<Stokhos::Sparse3Tensor<int,double> > Cijk =
    basis->computeTripleProductTensor(Cijk_sz);

  // SG AD Expansion
  Teuchos::RCP<Stokhos::OrthogPolyExpansion<int,double> > expansion;
  if (exp_type == "Quadrature")
    expansion = 
      Teuchos::rcp(new Stokhos::QuadOrthogPolyExpansion<int,double>(basis,
								    Cijk,
								    quad));
  else if (exp_type == "Algebraic")
    expansion = 
      Teuchos::rcp(new Stokhos::AlgebraicOrthogPolyExpansion<int,double>(
		     basis, Cijk));
#ifdef HAVE_STOKHOS_FORUQTK
  else if (exp_type == "For UQTK") {
    if (sgParams.isType<double>("Taylor Expansion Tolerance")) {
      double rtol = sgParams.get<double>("Taylor Expansion Tolerance");
      expansion = 
	Teuchos::rcp(new Stokhos::ForUQTKOrthogPolyExpansion<int,double>(
		       basis, Cijk,
		       Stokhos::ForUQTKOrthogPolyExpansion<int,double>::TAYLOR,
		       rtol));
    }
    else
      expansion = 
	Teuchos::rcp(new Stokhos::ForUQTKOrthogPolyExpansion<int,double>(
		       basis, Cijk));
  }
#endif
  else if (exp_type == "Derivative") {
    Teuchos::RCP<Teuchos::SerialDenseMatrix<int,double> > Bij = 
      basis->computeDerivDoubleProductTensor();
    Teuchos::RCP<Stokhos::Dense3Tensor<int,double> > Dijk = 
      basis->computeDerivTripleProductTensor(Bij, Cijk);
    expansion = 
      Teuchos::rcp(new Stokhos::DerivOrthogPolyExpansion<int,double>(
		     basis, Bij, Cijk, Dijk));
  }
  else
    TEST_FOR_EXCEPTION(true, Teuchos::Exceptions::InvalidParameter,
		       std::endl << "Error!  ENAT_SGNOXSolver():  " <<
		       "Invalid expansion type  " << exp_type << std::endl);

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

  // Set up SG nonlinear model
  Teuchos::RCP<Stokhos::SGModelEvaluator> sg_nonlin_model =
    Teuchos::rcp(new Stokhos::SGModelEvaluator(sg_model, basis, quad, expansion,
					       Cijk, sgSolverParams, comm,
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
  validPL->set<std::string>("SG Method", "","");
  validPL->set<std::string>("SG Method", "","");
  validPL->set<std::string>("AD Expansion Type", "","");
  validPL->set<std::string>("Quadrature Type", "","");
  validPL->set<int>("Sparse Grid Level", 0);

  return validPL;
}

