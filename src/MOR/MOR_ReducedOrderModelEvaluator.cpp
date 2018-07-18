//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//
#include "MOR_ReducedOrderModelEvaluator.hpp"

#include "MOR_ReducedSpace.hpp"
#include "MOR_ReducedOperatorFactory.hpp"

#include "Epetra_Vector.h"
#include "Epetra_CrsMatrix.h"

#include "Teuchos_Tuple.hpp"
#include "Teuchos_TestForException.hpp"

#include "MOR_ContainerUtils.hpp"
#include "EpetraExt_MultiVectorOut.h"
#include "EpetraExt_BlockMapOut.h"

// includes to make the DBC fix possible
#include <Tpetra_MultiVectorFiller.hpp>
#include "Petra_Converters.hpp"
#include "EpetraExt_RowMatrixOut.h"
#include "MatrixMarket_Tpetra.hpp"

// for mkdir
#include <sys/stat.h>

// Set precLBonly to true if you want to ONLY apply a preconditioner to the left basis (Psi)
// NOTE: if we're using the implementation where the preconditioner is just applied twice to
//   the left basis Psi, this behavior can be replicated by simply calling the singular
//   version of the perconditioner call.
// This is useful as a sanity check of the LSPG method...
//   if you only apply J_inverse to the left basis via. preconditioner types
//   "InverseJacobian" (which requires an additional define) or "Ifpack_Amesos",
//   then LSPG should be equivalent to Galerkin.
#define precLBonly false

// Set invJacPrec to true only if you REALLY want to enable preconditioning
//   with the inverse Jacobian.  It's a memory hog and causes issues for large
//   problems (i.e. PCAP), so it's commented out for now.
#define invJacPrec false // ALSO MOR_GaussNewtonOperatorFactory.cpp

int count_jacr_pl;

namespace MOR {

using Teuchos::RCP;
using Teuchos::rcp;
using Teuchos::rcp_dynamic_cast;
using Teuchos::null;
using Teuchos::is_null;
using Teuchos::nonnull;
using Teuchos::Tuple;
using Teuchos::tuple;

static void _mkdir(const char *dir)
{
	char tmp[256];
	char *p = NULL;
	size_t len;

	snprintf(tmp, sizeof(tmp),"%s",dir);
	len = strlen(tmp);
	if(tmp[len - 1] == '/')
	tmp[len - 1] = 0;
	for(p = tmp + 1; *p; p++)
	if(*p == '/')
	{
		*p = 0;
		mkdir(tmp, S_IRWXU);
		*p = '/';
	}
	mkdir(tmp, S_IRWXU);
}

void ReducedOrderModelEvaluator::parOut(std::string text) const
{
	if (app_->getEpetraComm()->MyPID() == 0)
		std::cout << text << std::endl;
}

void ReducedOrderModelEvaluator::parOut(std::string text, double val) const
{
	if (app_->getEpetraComm()->MyPID() == 0)
		std::cout << text << val << std::endl;
}

void ReducedOrderModelEvaluator::parOut(std::string text, int val) const
{
	if (app_->getEpetraComm()->MyPID() == 0)
		std::cout << text << val << std::endl;
}

ReducedOrderModelEvaluator::ReducedOrderModelEvaluator(const RCP<EpetraExt::ModelEvaluator> &fullOrderModel,
		const RCP<const ReducedSpace> &solutionSpace,
		const RCP<ReducedOperatorFactory> &reducedOpFactory,
		const bool* outputFlags,
		const std::string preconditionerType) :
		fullOrderModel_(fullOrderModel),
		solutionSpace_(solutionSpace),
		reducedOpFactory_(reducedOpFactory),
		x_init_(null),
		x_dot_init_(null),
		prev_time_(-1.0),
		step_(0),
		iter_(0)
{
	reset_x_and_x_dot_init();

	app_ = dynamic_cast<Albany::ModelEvaluator &>(*fullOrderModel_).get_app();
	// get data we'll need later on
	morParams_ = Teuchos::sublist(Teuchos::sublist(app_->getProblemPL(), "Model Order Reduction", true), "Reduced-Order Model",true);
	isThermoMech_ = morParams_->get<bool>("Use thermo-mechanical first step fix", false); // depreciated behavior meant as a work-around for some changes to the thermo-mechanical residual code by Coleman Alleman where the Jacobain would be singular.  However, there were several other concerns with his approach and so it was decided that we should just restart beyond the first step to avoid refactoring code later, so this work-around is no longer needed.  If you really want to use the ROM at the initialization step, mechanical problems should be fine.
	apply_bcs_ = morParams_->get<bool>("Apply BCs", true);
	run_nan_check_ = morParams_->get<bool>("Run nan Check", true);
	run_singular_check_ = morParams_->get<bool>("Run singular Check", true);
	prec_full_jac_ = morParams_->get<bool>("Precondition Full Jacobian", false);
	num_dbc_modes_ = morParams_->get("Number of DBC Modes", 0);
	extract_DBC_data(Teuchos::sublist(app_->getProblemPL(), "Dirichlet BCs"));
	outdir_ = morParams_->get("Output Directory",".") + "/" ;
	_mkdir(outdir_.c_str());

	outputTrace_ = outputFlags[0];
	writeJacobian_reduced_ = outputFlags[1];
	writeResidual_reduced_ = outputFlags[2];
	writeSolution_reduced_ = outputFlags[3];
	writePreconditioner_ = outputFlags[4];
	count_jacr_pl = 0;

	if (preconditionerType.compare("None") == 0)
	{
		// nothing to do
	}
	else if (preconditionerType.compare("Identity") == 0)
	{
		PrecondType=identity;
		parOut("Preconditioning: Identity");
#if !invJacPrec
		TEUCHOS_TEST_FOR_EXCEPTION(true, std::runtime_error, "Preconditioning using the identity matrix is currently disabled.  This is because it piggy-backs on some of the inverse Jacobian calls, and the memory declaration for that preconditioner segfaults for large problems.  If you really want to use this preconditioner for a smaller problem, change \"invJacPrec\" to true and rebuild.\n");
#endif
	}
	else if (preconditionerType.compare("DiagonalScaling") == 0)
	{
		PrecondType=scaling;
		parOut("Preconditioning: Diagonal Scaling");
	}
	else if (preconditionerType.compare("InverseJacobian") == 0)
	{
		PrecondType=invJac;
		parOut("Preconditioning: Inverse Jacobian");
#if !invJacPrec
		TEUCHOS_TEST_FOR_EXCEPTION(true, std::runtime_error, "Preconditioning using the inverse Jacobian is currently disabled.  The memory declaration for the preconditioner segfaults for large problems.  If you really want to use this preconditioner for a smaller problem, change \"invJacPrec\" to true and rebuild.\n");
#endif
	}
	else if (preconditionerType.compare("ProjectedSolution") == 0)
	{
		PrecondType=projSoln;
		parOut("Preconditioning: Projected Solution");
	}
	else if (preconditionerType.compare("Ifpack_Jacobi") == 0)
	{
		PrecondType=ifPack;
		ifpackType_ = "Jacobi";
		parOut("Preconditioning: Ifpack - Type: " + ifpackType_);
	}
	else if (preconditionerType.compare("Ifpack_GaussSeidel") == 0)
	{
		PrecondType=ifPack;
		ifpackType_ = "GaussSeidel";
		parOut("Preconditioning: Ifpack - Type: " + ifpackType_);
	}
	else if (preconditionerType.compare("Ifpack_SymmetricGaussSeidel") == 0)
	{
		PrecondType=ifPack;
		ifpackType_ = "SymmetricGaussSeidel";
		parOut("Preconditioning: Ifpack - Type: " + ifpackType_);
	}
	else if (preconditionerType.compare("Ifpack_ILU0") == 0)
	{
		PrecondType=ifPack;
		ifpackType_ = "ILU0";
		parOut("Preconditioning: Ifpack - Type: " + ifpackType_);
	}
	else if (preconditionerType.compare("Ifpack_ILU1") == 0)
	{
		PrecondType=ifPack;
		ifpackType_ = "ILU1";
		parOut("Preconditioning: Ifpack - Type: " + ifpackType_);
	}
	else if (preconditionerType.compare("Ifpack_ILU2") == 0)
	{
		PrecondType=ifPack;
		ifpackType_ = "ILU2";
		parOut("Preconditioning: Ifpack - Type: " + ifpackType_);
	}
	else if (preconditionerType.compare("Ifpack_IC0") == 0)
	{
		PrecondType=ifPack;
		ifpackType_ = "IC0";
		parOut("Preconditioning: Ifpack - Type: " + ifpackType_);
	}
	else if (preconditionerType.compare("Ifpack_IC1") == 0)
	{
		PrecondType=ifPack;
		ifpackType_ = "IC1";
		parOut("Preconditioning: Ifpack - Type: " + ifpackType_);
	}
	else if (preconditionerType.compare("Ifpack_IC2") == 0)
	{
		PrecondType=ifPack;
		ifpackType_ = "IC2";
		parOut("Preconditioning: Ifpack - Type: " + ifpackType_);
	}
	else if (preconditionerType.compare("Ifpack_Amesos") == 0)
	{
		PrecondType=ifPack;
		ifpackType_ = "Amesos";
		parOut("Preconditioning: Ifpack - Type: " + ifpackType_);
	}
	else if (preconditionerType.compare("Ifpack_Identity") == 0)
	{
		PrecondType=ifPack;
		ifpackType_ = "Identity";
		parOut("Preconditioning: Ifpack - Type: " + ifpackType_);
	}
	else if (preconditionerType.compare("Mimic Galerkin") == 0)
	{
		// nothing to do... this isn't really a preconditioner, but it's convinient to define it here...
	}
	else
	{
		TEUCHOS_TEST_FOR_EXCEPTION(true, std::runtime_error, "Preconditioner type not recognized!!");
	}
}

std::vector<std::string> split(const char *str, char c = ' ')
{
  std::vector<std::string> result;
  do
  {
    const char *begin = str;
    while(*str != c && *str)
        str++;
    result.push_back(std::string(begin, str));
  } while (0 != *str++);
  return result;
}

double ReducedOrderModelEvaluator::valAtTime(std::pair< bool, std::vector<double> > vals, double t) const
{
	if (vals.first)
	{
		double x1 = vals.second[0], y1 = vals.second[1], s = vals.second[2];
		return y1 + (t-x1)*s;
	}
	else
		return vals.second[0];
}

void ReducedOrderModelEvaluator::extract_DBC_data(Teuchos::RCP<Teuchos::ParameterList> my_DBC_params)
{
	for (auto it=my_DBC_params->begin(); it!=my_DBC_params->end(); it++)
	{
		std::string this_name = my_DBC_params->name(it);
		std::vector<std::string> token_name = split(this_name.c_str());

		int offset;
		bool time_varying = token_name[0].compare("Time") == 0 ? offset = 2 : offset = 0;
		bool sdbc = token_name[offset].compare("SDBC")==0;
		std::string name = token_name[offset+3];
		std::string DOF = token_name[offset+6];
		int dof=-1;
		if (DOF=="X")
			dof = 0;
		else if (DOF=="Y")
			dof = 1;
		else if (DOF=="Z")
			dof = 2;
		else if (DOF=="T")
			dof = 3;

		std::vector<double> vals;
		if (time_varying)
		{
			const RCP<Teuchos::ParameterList> this_data = sublist(my_DBC_params, this_name);
			Teuchos::Array<double> defaultData;
			Teuchos::Array<double> this_time_values = this_data->get("Time Values", defaultData);
			Teuchos::Array<double> this_BC_values = this_data->get("BC Values", defaultData);
			double x1 = this_time_values[0], x2 = this_time_values[1];
			double y1 = this_BC_values[0], y2 = this_BC_values[1];
			double s = (y2-y1)/(x2-x1);
			vals = {x1,y1,s};
		}
		else
		{
			vals = {my_DBC_params->get(this_name,0.0)};
		}

		std::vector<std::vector<int>> const &
		ns_nodes = app_->getDisc()->getNodeSets().find(name)->second;
		gen_DBC_data this_DBC_data = {sdbc, name, ns_nodes, dof, std::make_pair(time_varying, vals)};
		DBC_data_.push_back(this_DBC_data);
	}
}

RCP<Epetra_CrsMatrix> ReducedOrderModelEvaluator::MultiVector2CRSMatrix(RCP<Epetra_MultiVector> MV_E) const
{
	// caution... this almost certainly DOESN'T work in parallel

	Teuchos::RCP<Tpetra_MultiVector> MV = Petra::EpetraMultiVector_To_TpetraMultiVector(*MV_E, app_->getComm());
	auto rowMap = MV->getMap();
	Teuchos::RCP<const Tpetra_Map> colMap = Teuchos::rcp(new const Tpetra_Map(rowMap->getGlobalNumElements(), 0, app_->getComm(), Tpetra::LocallyReplicated));
	int numLocalRows = MV->getLocalLength();
	int numVectors = MV->getNumVectors();

	auto MV_view = MV->get2dViewNonConst();

	RCP<Tpetra_CrsMatrix> CRSM = rcp(new Tpetra_CrsMatrix(rowMap,colMap,numVectors,Tpetra::StaticProfile));
	CRSM->setAllToScalar(0.0);

	for (LO local_row = 0; local_row<numLocalRows; local_row++) {
		Teuchos::Array<LO> col(numVectors); Teuchos::Array<ST> val(numVectors);
		 for (LO global_col=0; global_col<numVectors; global_col++)
		 {
				col[global_col]=global_col; val[global_col]=MV_view[global_col][local_row];
		 }
		 CRSM->insertLocalValues(local_row, col(), val());
	}
	CRSM->fillComplete();

	return Petra::TpetraCrsMatrix_To_EpetraCrsMatrix(CRSM, app_->getEpetraComm());
}

void ReducedOrderModelEvaluator::reset_x_and_x_dot_init()
{
	reset_x_init();
	reset_x_dot_init();
}

void ReducedOrderModelEvaluator::reset_x_init()
{
	if (nonnull(fullOrderModel_->get_x_init())) {
		x_init_ = solutionSpace_->reduction(*fullOrderModel_->get_x_init());
	} else {
		x_init_ = null;
	}
}

void ReducedOrderModelEvaluator::reset_x_dot_init()
{
	if (nonnull(fullOrderModel_->get_x_dot_init())) {
		x_dot_init_ = solutionSpace_->linearReduction(*fullOrderModel_->get_x_dot_init());
	} else {
		x_dot_init_ = null;
	}
}

Teuchos::RCP<const EpetraExt::ModelEvaluator> ReducedOrderModelEvaluator::getFullOrderModel() const
{
	return fullOrderModel_;
}

Teuchos::RCP<const ReducedSpace> ReducedOrderModelEvaluator::getSolutionSpace() const
{
	return solutionSpace_;
}

const Epetra_Map &ReducedOrderModelEvaluator::componentMap() const
{
	return solutionSpace_->componentMap();
}

RCP<const Epetra_Map> ReducedOrderModelEvaluator::componentMapRCP() const
{
	// TODO more efficient
	return rcp(new Epetra_Map(this->componentMap()));
}

RCP<const Epetra_Map> ReducedOrderModelEvaluator::get_x_map() const
{
	return componentMapRCP();
}

RCP<const Epetra_Map> ReducedOrderModelEvaluator::get_f_map() const
{
	return componentMapRCP();
}

RCP<const Epetra_Map> ReducedOrderModelEvaluator::get_p_map(int l) const
{
	return fullOrderModel_->get_p_map(l);
}

RCP<const Teuchos::Array<std::string> > ReducedOrderModelEvaluator::get_p_names(int l) const
{
	return fullOrderModel_->get_p_names(l);
}

RCP<const Epetra_Map> ReducedOrderModelEvaluator::get_g_map(int j) const
{
	return fullOrderModel_->get_g_map(j);
}

RCP<const Epetra_Vector> ReducedOrderModelEvaluator::get_x_init() const
{
	return x_init_;
}

RCP<const Epetra_Vector> ReducedOrderModelEvaluator::get_x_dot_init() const
{
	return x_dot_init_;
}

RCP<const Epetra_Vector> ReducedOrderModelEvaluator::get_p_init(int l) const
{
	return fullOrderModel_->get_p_init(l);
}

double ReducedOrderModelEvaluator::get_t_init() const
{
	return fullOrderModel_->get_t_init();
}

double ReducedOrderModelEvaluator::getInfBound() const
{
	return fullOrderModel_->getInfBound();
}

RCP<const Epetra_Vector> ReducedOrderModelEvaluator::get_p_lower_bounds(int l) const
{
	return fullOrderModel_->get_p_lower_bounds(l);
}

RCP<const Epetra_Vector> ReducedOrderModelEvaluator::get_p_upper_bounds(int l) const
{
	return fullOrderModel_->get_p_upper_bounds(l);
}

double ReducedOrderModelEvaluator::get_t_lower_bound() const
{
	return fullOrderModel_->get_t_lower_bound();
}

double ReducedOrderModelEvaluator::get_t_upper_bound() const
{
	return fullOrderModel_->get_t_upper_bound();
}

RCP<Epetra_Operator> ReducedOrderModelEvaluator::create_W() const
{
	const RCP<Epetra_Operator> fullOrderOperator = fullOrderModel_->create_W();
	if (is_null(fullOrderOperator)) {
		return null;
	}

	return reducedOpFactory_->reducedJacobianNew();
}

RCP<Epetra_Operator> ReducedOrderModelEvaluator::create_DgDp_op(int j, int l) const
{
	return fullOrderModel_->create_DgDp_op(j, l);
}

EpetraExt::ModelEvaluator::InArgs ReducedOrderModelEvaluator::createInArgs() const
{
	const InArgs fullInArgs = fullOrderModel_->createInArgs();

	InArgsSetup result;

	result.setModelEvalDescription("MOR applied to " + fullInArgs.modelEvalDescription());

	result.set_Np(fullInArgs.Np());

	// Requires underlying full order model to accept a state input
	TEUCHOS_TEST_FOR_EXCEPT(!fullInArgs.supports(IN_ARG_x));
	const Tuple<EInArgsMembers, 5> optionalMembers = tuple(IN_ARG_x, IN_ARG_x_dot, IN_ARG_t, IN_ARG_alpha, IN_ARG_beta);
	for (Tuple<EInArgsMembers, 5>::const_iterator it = optionalMembers.begin(); it != optionalMembers.end(); ++it) {
		const EInArgsMembers member = *it;
		result.setSupports(member, fullInArgs.supports(member));
	}

	return result;
}

EpetraExt::ModelEvaluator::OutArgs ReducedOrderModelEvaluator::createOutArgs() const
{
	const OutArgs fullOutArgs = fullOrderModel_->createOutArgs();

	OutArgsSetup result;

	result.setModelEvalDescription("MOR applied to " + fullOutArgs.modelEvalDescription());

	result.set_Np_Ng(fullOutArgs.Np(), fullOutArgs.Ng());

	for (int j = 0; j < fullOutArgs.Ng(); ++j) {
		if (fullOutArgs.supports(OUT_ARG_DgDx, j).supports(DERIV_TRANS_MV_BY_ROW)) {
			result.setSupports(OUT_ARG_DgDx, j, DERIV_TRANS_MV_BY_ROW);
			result.set_DgDx_properties(j, fullOutArgs.get_DgDx_properties(j));
		}
	}

	for (int j = 0; j < fullOutArgs.Ng(); ++j) {
		if (fullOutArgs.supports(OUT_ARG_DgDx_dot, j).supports(DERIV_TRANS_MV_BY_ROW)) {
			result.setSupports(OUT_ARG_DgDx_dot, j, DERIV_TRANS_MV_BY_ROW);
			result.set_DgDx_dot_properties(j, fullOutArgs.get_DgDx_dot_properties(j));
		}
	}

	for (int l = 0; l < fullOutArgs.Np(); ++l) {
		if (fullOutArgs.supports(OUT_ARG_DfDp, l).supports(DERIV_MV_BY_COL)) {
			result.setSupports(OUT_ARG_DfDp, l, DERIV_MV_BY_COL);
			result.set_DfDp_properties(l, fullOutArgs.get_DfDp_properties(l));
		}
	}

	for (int j = 0; j < fullOutArgs.Ng(); ++j) {
		for (int l = 0; l < fullOutArgs.Np(); ++l) {
			result.setSupports(OUT_ARG_DgDp, j, l, fullOutArgs.supports(OUT_ARG_DgDp, j, l));
			result.set_DgDp_properties(j, l, fullOutArgs.get_DgDp_properties(j, l));
		}
	}

	const Tuple<EOutArgsMembers, 2> optionalMembers = tuple(OUT_ARG_f, OUT_ARG_W);
	for (Tuple<EOutArgsMembers, 2>::const_iterator it = optionalMembers.begin(); it != optionalMembers.end(); ++it) {
		const EOutArgsMembers member = *it;
		result.setSupports(member, fullOutArgs.supports(member));
	}

	result.set_W_properties(fullOutArgs.get_W_properties());

	return result;
}

void ReducedOrderModelEvaluator::DBC_MultiVector(const Teuchos::RCP<Epetra_MultiVector>& MV_E) const
{
	if (MV_E != Teuchos::null)
	{
		// zero out non-diagonal DBC rows (or SDBC rows/cols) of EITHER a square MultiVector or just the rows of a Vector
		Teuchos::RCP<Tpetra_MultiVector> MV = Petra::EpetraMultiVector_To_TpetraMultiVector(*MV_E, app_->getComm());

		auto MV_view = MV->get2dViewNonConst();

		auto row_map = MV->getMap();
		Teuchos::RCP<const Tpetra_Map> col_map = Teuchos::rcp(new const Tpetra_Map(row_map->getGlobalNumElements(), 0, app_->getComm(), Tpetra::LocallyReplicated));
		int num_row_entries = MV->getNumVectors();


		for (auto it=DBC_data_.begin(); it!=DBC_data_.end(); it++){
			auto& ns_nodes = it->ns_nodes;

			using IntVec = Tpetra::Vector<int, Tpetra_LO, Tpetra_GO, KokkosNode>;
			using Import = Tpetra::Import<Tpetra_LO, Tpetra_GO, KokkosNode>;
			Teuchos::RCP<const Import> import;

			auto domain_map = row_map; // we are assuming this!

			import = Teuchos::rcp(new Import(domain_map, col_map));

			IntVec row_is_dbc(row_map);
			IntVec col_is_dbc(col_map);
			auto row_is_DBC_data = row_is_dbc.get1dViewNonConst();

			for (size_t ns_node = 0; ns_node < ns_nodes.size(); ns_node++) {
				int const dof = ns_nodes[ns_node][it->dof];
				row_is_DBC_data[dof] = 1;
			}

			//Teuchos::RCP<IntVec> col_is_dbc = locallyReplicatedVector<IntVec>(Teuchos::rcpFromRef(row_is_dbc)); // same result...
			//auto col_is_DBC_data = col_is_dbc->get1dViewNonConst();
			col_is_dbc.doImport(row_is_dbc, *import, Tpetra::ADD);
			auto col_is_DBC_data = col_is_dbc.get1dViewNonConst();

			size_t const num_local_rows = MV->getLocalLength();
			for (auto local_row = 0; local_row < num_local_rows; ++local_row) {
				auto global_row = row_map->getGlobalElement(local_row);
				auto row_is_dbc = col_is_DBC_data[global_row] > 0.0;

				for (size_t global_col = 0; global_col < num_row_entries; ++global_col) {

					auto is_diagonal_entry = (global_col == global_row) && (num_row_entries > 1);
					if (is_diagonal_entry)
					{
						if (!(it->sdbc) && row_is_dbc)
						MV_view[global_col][local_row] = 1.0;
						continue;
					}

					auto col_is_dbc = (col_is_DBC_data[global_col] > 0.0) && (num_row_entries > 1) && (it->sdbc);
					if (row_is_dbc || col_is_dbc) {
						MV_view[global_col][local_row] = 0.0; // NOTE: this is the opposite of what you might expect...
					}
				}
			}
		}

		Petra::TpetraMultiVector_To_EpetraMultiVector(MV, *MV_E, app_->getEpetraComm());
	}
}

void ReducedOrderModelEvaluator::DBC_CRSMatrix(const RCP<Epetra_CrsMatrix>& CSRM_E) const
{
	if (CSRM_E != Teuchos::null)
	{
		// zero out non-diagonal DBC rows (or SDBC rows/cols) of a square CRSMatrix
		Teuchos::RCP<Tpetra_CrsMatrix> CSRM = Petra::EpetraCrsMatrix_To_TpetraCrsMatrix(*CSRM_E, app_->getComm());
		CSRM->resumeFill();
		auto row_map = CSRM->getRowMap();
		auto col_map = CSRM->getColMap();
		// we make this assumption, which lets us use both local row and column
		// indices into a single is_dbc vector
		ALBANY_ASSERT(col_map->isLocallyFitted(*row_map));

		for (auto it=DBC_data_.begin(); it!=DBC_data_.end(); it++){
			auto& ns_nodes = it->ns_nodes;

			Teuchos::Array<ST> entries;
			Teuchos::Array<LO> indices;

			using IntVec = Tpetra::Vector<int, Tpetra_LO, Tpetra_GO, KokkosNode>;
			using Import = Tpetra::Import<Tpetra_LO, Tpetra_GO, KokkosNode>;
			Teuchos::RCP<const Import> import;

			auto domain_map = row_map; // we are assuming this!

			import = Teuchos::rcp(new Import(domain_map, col_map));

			IntVec row_is_dbc(row_map);
			IntVec col_is_dbc(col_map);
			auto row_is_DBC_data = row_is_dbc.get1dViewNonConst();

			for (size_t ns_node = 0; ns_node < ns_nodes.size(); ns_node++) {
				int const dof = ns_nodes[ns_node][it->dof];
				row_is_DBC_data[dof] = 1;
			}
			col_is_dbc.doImport(row_is_dbc, *import, Tpetra::ADD);
			auto col_is_DBC_data = col_is_dbc.get1dView();

			auto min_local_row = row_map->getMinLocalIndex();
			auto max_local_row = row_map->getMaxLocalIndex();
			for (auto local_row = min_local_row; local_row <= max_local_row; ++local_row) {
				auto num_row_entries = CSRM->getNumEntriesInLocalRow(local_row);

				entries.resize(num_row_entries);
				indices.resize(num_row_entries);

				CSRM->getLocalRowCopy(local_row, indices(), entries(), num_row_entries);

				auto row_is_dbc = col_is_DBC_data[local_row] > 0.0;

				for (size_t row_entry = 0; row_entry < num_row_entries; ++row_entry) {
					auto local_col = indices[row_entry];
					auto is_diagonal_entry = local_col == local_row;
					if (is_diagonal_entry)
					{
						if (!(it->sdbc) && row_is_dbc)
							entries[row_entry] = 1.0;
						continue;
					}
					ALBANY_ASSERT(local_col >= col_map->getMinLocalIndex());
					ALBANY_ASSERT(local_col <= col_map->getMaxLocalIndex());
					auto col_is_dbc = (col_is_DBC_data[local_col] > 0.0) && (it->sdbc);
					if (row_is_dbc || col_is_dbc) {
						entries[row_entry] = 0.0;
					}
				}
				CSRM->replaceLocalValues(local_row, indices(), entries());
			}
		}

		CSRM->fillComplete();
		Petra::TpetraCrsMatrix_To_EpetraCrsMatrix(CSRM, *CSRM_E, app_->getEpetraComm());
		CSRM_E->FillComplete(true);
	}
}

const Tpetra::global_size_t INVALID =
	Teuchos::OrdinalTraits<Tpetra::global_size_t>::invalid();

void ReducedOrderModelEvaluator::printCRSMatrix(std::string filename, const Teuchos::RCP<Epetra_CrsMatrix> CRSM, int index) const
{
	std::string full_filename = outdir_ + filename + std::to_string(index) + ".mm";
	parOut("ReducedOrderModelEvaluator::evalModel writing file named: " + full_filename);
	TEUCHOS_TEST_FOR_EXCEPT(CRSM == Teuchos::null)
	Teuchos::RCP<Tpetra_CrsMatrix> CRSM_T = Petra::EpetraCrsMatrix_To_TpetraCrsMatrix(*CRSM, app_->getComm());
	Tpetra::MatrixMarket::Writer<Tpetra_CrsMatrix>::writeSparseFile(full_filename, CRSM_T);
	/*
	Tpetra::MatrixMarket::Writer<Tpetra_CrsMatrix>::writeSparseFile(full_filename, CRSM_T, true);

	std::string mystr = outdir_ + filename + "_Rmap" + std::to_string(index) + ".mm";
	Tpetra::MatrixMarket::Writer<Tpetra_CrsMatrix>::writeMapFile(mystr, *CRSM_T->getRowMap());
	std::string mystr2 = outdir_ + filename + "_Cmap" + std::to_string(index) + ".mm";
	Tpetra::MatrixMarket::Writer<Tpetra_CrsMatrix>::writeMapFile(mystr2, *CRSM_T->getColMap());
	*/
}

void ReducedOrderModelEvaluator::printConstCRSMatrix(std::string filename, const Teuchos::RCP<const Epetra_CrsMatrix> CRSM, int index) const
{
	printCRSMatrix(filename, Teuchos::rcp_const_cast<Epetra_CrsMatrix> (CRSM), index);
}

template <typename VectorType>
Teuchos::RCP<VectorType> ReducedOrderModelEvaluator::locallyReplicatedVector(Teuchos::RCP<VectorType> MV) const
{
	Teuchos::RCP<const Tpetra_Map> LR_map = Teuchos::rcp(new const Tpetra_Map(MV->getMap()->getGlobalNumElements(), 0, app_->getComm(), Tpetra::LocallyReplicated));

	// create importer from parallel map to serial map and populate serial solution MV_serial
	Teuchos::RCP<Tpetra_Import> importOperator = Teuchos::rcp(new Tpetra_Import(MV->getMap(), LR_map));
	Teuchos::RCP<VectorType> MV_LR = Teuchos::rcp(new VectorType(LR_map,MV->getNumVectors()));
	MV_LR->doImport(*MV, *importOperator, Tpetra::INSERT);

	return MV_LR;
}

void ReducedOrderModelEvaluator::multiplyInPlace(const Epetra_MultiVector &A, const Epetra_MultiVector M) const
{
	// calculate A <- M*A (i.e. preconditioning a matrix/vector A by M)

	Teuchos::RCP<Tpetra_MultiVector> M_T = Petra::EpetraMultiVector_To_TpetraMultiVector(M, app_->getComm());
	Teuchos::RCP<Tpetra_MultiVector> A_T = Petra::EpetraMultiVector_To_TpetraMultiVector(A, app_->getComm());

	Teuchos::RCP<Tpetra_MultiVector> temp = locallyReplicatedVector<Tpetra_MultiVector>(Teuchos::rcp(new Tpetra_MultiVector(*A_T))); // this is really important when running in parallel
	Teuchos::RCP<Tpetra_MultiVector> temp2 = Teuchos::rcp(new Tpetra_MultiVector(*A_T, Teuchos::View));

	const bool A_is_local = ! M_T->isDistributed ();
	const bool B_is_local = ! temp->isDistributed ();
	const bool C_is_local = ! temp2->isDistributed ();
	// for parallel, we need A_is_local = false, B_is_local = TRUE, and C_is_local = false,
	//                                                   THIS ^^^^ is the important one!!

	temp2->multiply(Teuchos::NO_TRANS,Teuchos::NO_TRANS,1.0,*M_T,*temp,0.0);

	Petra::TpetraMultiVector_To_EpetraMultiVector(A_T, const_cast<Epetra_MultiVector&>(A), app_->getEpetraComm());
}

template <typename VectorType>
Teuchos::RCP<VectorType> ReducedOrderModelEvaluator::serialVector(Teuchos::RCP<VectorType> MV) const
{
	// create serial map that puts the whole solution on processor 0
	int numMyElements = (MV->getMap()->getComm()->getRank() == 0) ? MV->getMap()->getGlobalNumElements() : 0;
	Teuchos::RCP<const Tpetra_Map> serial_map = Teuchos::rcp(new const Tpetra_Map(INVALID, numMyElements, 0, app_->getComm()));

	// create importer from parallel map to serial map and populate serial solution MV_serial
	Teuchos::RCP<Tpetra_Import> importOperator = Teuchos::rcp(new Tpetra_Import(MV->getMap(), serial_map));
	Teuchos::RCP<VectorType> MV_serial = Teuchos::rcp(new VectorType(serial_map,MV->getNumVectors()));
	MV_serial->doImport(*MV, *importOperator, Tpetra::INSERT);

	return MV_serial;
}

void ReducedOrderModelEvaluator::printMultiVectorT(std::string filename, const Teuchos::RCP<Tpetra_MultiVector> MV, int index, bool isDist) const
{
	// bool isDist = MV->isDistributed();

	std::string full_filename = outdir_ + filename + std::to_string(index) + ".mm";
	parOut("ReducedOrderModelEvaluator::evalModel writing file named: " + full_filename);

	if (isDist)
	{
		Teuchos::RCP<Tpetra_MultiVector> MV_serial = serialVector<Tpetra_MultiVector>(MV);
		Tpetra::MatrixMarket::Writer<Tpetra_CrsMatrix>::writeDenseFile(full_filename, MV_serial);
	}
	else
	{
		Tpetra::MatrixMarket::Writer<Tpetra_CrsMatrix>::writeDenseFile(full_filename, MV);
	}

	/*
	std::string mystr = outdir_ + filename + "_map" + std::to_string(index) + ".mm";
	Tpetra::MatrixMarket::Writer<Tpetra_CrsMatrix>::writeMapFile(mystr, *MV->getMap());
	//std::string mystr2 = outdir_ + filename + "_serial_map" + std::to_string(index) + ".mm";
	//Tpetra::MatrixMarket::Writer<Tpetra_CrsMatrix>::writeMapFile(mystr2, *MV_serial->getMap());
	*/
}

void ReducedOrderModelEvaluator::printMultiVector(std::string filename, const Teuchos::RCP<Epetra_MultiVector> MV, int index) const
{
	bool isDist = MV->DistributedGlobal();
	Teuchos::RCP<Tpetra_MultiVector> MV_T = Petra::EpetraMultiVector_To_TpetraMultiVector(*MV, app_->getComm());
	printMultiVectorT(filename, MV_T, index, isDist);
}

void ReducedOrderModelEvaluator::printConstMultiVector(std::string filename, const Teuchos::RCP<const Epetra_MultiVector> MV, int index) const
{
	printMultiVector(filename, Teuchos::rcp_const_cast<Epetra_MultiVector> (MV), index);
}

Teuchos::RCP<Epetra_MultiVector> ReducedOrderModelEvaluator::eye() const
{
	Teuchos::RCP<Tpetra_MultiVector> RB_T = Petra::EpetraMultiVector_To_TpetraMultiVector(*reducedOpFactory_->getReducedBasis(), app_->getComm());

	Teuchos::RCP<Tpetra_MultiVector> I = Teuchos::rcp(new Tpetra_MultiVector(RB_T->getMap(), RB_T->getGlobalLength(), true));
	int num_rows = I->getGlobalLength();
	int num_vecs = I->getNumVectors();
	TEUCHOS_ASSERT(num_rows == num_vecs);

	I->putScalar(0.0);
	for (int local_row = 0; local_row < I->getLocalLength(); local_row++){
		int global_row = I->getMap()->getGlobalElement(local_row);
		I->replaceLocalValue(local_row, global_row, 1.0);
	}

	Teuchos::RCP<Epetra_MultiVector> I_E = Teuchos::rcp(new Epetra_MultiVector(reducedOpFactory_->getReducedBasis()->Map(), reducedOpFactory_->getReducedBasis()->GlobalLength(), true));
	Petra::TpetraMultiVector_To_EpetraMultiVector(I, *I_E, app_->getEpetraComm());

	return I_E;
}

RCP<Epetra_CrsMatrix> ReducedOrderModelEvaluator::eyeFromCRSMatrix(RCP<Epetra_CrsMatrix> CRSM_E) const
{
	Teuchos::RCP<Tpetra_CrsMatrix> CRSM = Petra::EpetraCrsMatrix_To_TpetraCrsMatrix(*CRSM_E, app_->getComm());

	auto row_map = CRSM->getRowMap();
	auto col_map = CRSM->getColMap();

	// we make this assumption, which lets us use both local row and column indices
	// into a single is_dbc vector
	ALBANY_ASSERT(col_map->isLocallyFitted(*row_map));

	Teuchos::Array<ST> entries;
	Teuchos::Array<LO> indices;

	auto min_local_row = row_map->getMinLocalIndex();
	auto max_local_row = row_map->getMaxLocalIndex();

	CRSM->resumeFill();
	for (auto local_row = min_local_row; local_row <= max_local_row; ++local_row)
	{
		auto global_row = row_map->getGlobalElement(local_row);
		auto num_row_entries = CRSM->getNumEntriesInLocalRow(local_row);

		entries.resize(num_row_entries);
		indices.resize(num_row_entries);

		CRSM->getLocalRowCopy(local_row, indices(), entries(), num_row_entries);

		for (size_t row_entry = 0; row_entry < num_row_entries; ++row_entry)
		{
			auto local_col = indices[row_entry];
			auto global_col = col_map->getGlobalElement(local_col);

			auto is_diagonal_entry = (local_col == local_row); // should this be against global_row/col?

			ALBANY_ASSERT(local_col >= col_map->getMinLocalIndex());
			ALBANY_ASSERT(local_col <= col_map->getMaxLocalIndex());

			if (is_diagonal_entry)
			{
				entries[row_entry] = 1.0;
			}
			else
			{
				entries[row_entry] = 0.0;
			}

		}
		CRSM->replaceLocalValues(local_row, indices(), entries());
	}
	CRSM->fillComplete();

	return Petra::TpetraCrsMatrix_To_EpetraCrsMatrix(CRSM, app_->getEpetraComm());
}

void ReducedOrderModelEvaluator::nanCheck(const Teuchos::RCP<Epetra_Vector> &f) const
{
	Teuchos::RCP<Tpetra_Vector> f_T = Petra::EpetraVector_To_TpetraVectorNonConst(*f, app_->getComm());

	//	possibly... add a Jacobian check where we only go into it if isnan(J_T->getFrobeniusNorm())
	if (f_T != Teuchos::null && isnan(f_T->norm2()))
	{
		//Teuchos::RCP<Tpetra_Vector> f_T_serial = locallyReplicatedVector<Tpetra_Vector>(f_T);
		auto f_view = f_T->get1dViewNonConst();
		auto row_map = f_T->getMap();

		int num_nans = 0;
		auto num_local_rows = f_T->getLocalLength();
		for (auto local_row = 0; local_row < num_local_rows; ++local_row){
			if (isnan(f_view[local_row])){
				num_nans++;
				std::cout << "r(" << row_map->getGlobalElement(local_row) << ") = " << f_view[local_row] << std::endl;
			}
		}

		if (num_nans>0)
			printMultiVector("Rerr",f,0);
		std::string str = "The residual has " + std::to_string(num_nans) + " nan entries (see output for actual elements).  Check Rerr0.mm for actual residual in MatrixMarket format.  This might mean that you should take a close look at your model (e.g. material params).  Otherwise, set \"Run nan Check\" to false in the input and Albany will keep going by dropping the time-step and trying again. \n";
		TEUCHOS_TEST_FOR_EXCEPTION(num_nans>0, std::runtime_error, str);
	}
}

void ReducedOrderModelEvaluator::singularCheck(const Teuchos::RCP<Epetra_CrsMatrix> &J) const
{
	Teuchos::RCP<Tpetra_CrsMatrix> J_T = Petra::EpetraCrsMatrix_To_TpetraCrsMatrix(*J, app_->getComm());

	Teuchos::RCP<Tpetra_Vector> J_diag = Teuchos::rcp(new Tpetra_Vector(J_T->getRowMap()));
	J_T->getLocalDiagCopy(*J_diag);

	//Teuchos::RCP<Tpetra_Vector> J_diag_serial = locallyReplicatedVector<Tpetra_Vector>(J_diag);

	int num_zero_diags = 0;
	auto num_local_rows = J_diag->getLocalLength();

	auto J_diag_view = J_diag->get1dViewNonConst();
	auto row_map = J_diag->getMap();

	for (auto local_row=0; local_row<num_local_rows; local_row++){
		if (J_diag_view[local_row] == 0.0)
		{
			num_zero_diags++;
			std::cout << "J(" << row_map->getGlobalElement(local_row) << "," << row_map->getGlobalElement(local_row) << ") = " << J_diag_view[local_row] << std::endl;
		}
	}

	if (num_zero_diags>0)
		printCRSMatrix("Jerr",J,0);
	std::string str = "The Jacobian has " + std::to_string(num_zero_diags) + " zero diagonal elements (see output for actual elements) so it's probably singular.  Check Jerr0.mm for actual Jacobian in MatrixMarket format.\n";
	TEUCHOS_TEST_FOR_EXCEPTION(num_zero_diags>0, std::runtime_error, str);
}

double ReducedOrderModelEvaluator::getTime(const InArgs& inArgs) const
{
	if (!app_->getParamLib()->isParameter("Time"))
		return inArgs.get_t();
	else
		return app_->getParamLib()->getRealValue<PHAL::AlbanyTraits::Residual>("Time");
}

void ReducedOrderModelEvaluator::DBC_ROM_jac(const InArgs& inArgs, const OutArgs& outArgs, const bool SDBC) const
{
	// zero out non-diagonal DBC rows (or SDBC rows/cols) of Wr

	RCP<Epetra_CrsMatrix> W_r = rcp_dynamic_cast<Epetra_CrsMatrix>(outArgs.get_W());
	TEUCHOS_TEST_FOR_EXCEPT(is_null((W_r)));

	bool leave_BB_block = false; 	// set to true ONLY if you don't want to zero out the (1,1) block position
	// (corresponding to blocked DBC modes) of the reduced Jacobian Psi^T*J*Phi

	Teuchos::RCP<Tpetra_CrsMatrix> W_rT = Petra::EpetraCrsMatrix_To_TpetraCrsMatrix(*W_r, app_->getComm());
	auto row_map = W_rT->getRowMap();
	auto col_map = W_rT->getColMap();

	// we make this assumption, which lets us use both local row and column indices
	// into a single is_dbc vector
	ALBANY_ASSERT(col_map->isLocallyFitted(*row_map));

	Teuchos::Array<ST> entries;
	Teuchos::Array<LO> indices;

	auto min_local_row = row_map->getMinLocalIndex();
	auto max_local_row = row_map->getMaxLocalIndex();

	W_rT->resumeFill();
	for (auto local_row = min_local_row; local_row <= max_local_row; ++local_row)
	{
		auto global_row = row_map->getGlobalElement(local_row);
		auto num_row_entries = W_rT->getNumEntriesInLocalRow(local_row);

		entries.resize(num_row_entries);
		indices.resize(num_row_entries);

		W_rT->getLocalRowCopy(local_row, indices(), entries(), num_row_entries);

		//auto row_is_dbc = (local_row < num_dbc_modes_);
		auto row_is_dbc = (global_row < num_dbc_modes_);

		for (size_t row_entry = 0; row_entry < num_row_entries; ++row_entry)
		{
			auto local_col = indices[row_entry];
			auto global_col = col_map->getGlobalElement(local_col);

			auto is_diagonal_entry = (local_col == local_row); // should this be against global_row/col?
			//auto col_is_dbc = (local_col<num_dbc_modes_); // if truly local_col, then this should be global_col
			auto col_is_dbc = (global_col<num_dbc_modes_);
			bool leave_this_dof = leave_BB_block && row_is_dbc && col_is_dbc;

			if (is_diagonal_entry && !leave_this_dof)
			{
				if (!SDBC && row_is_dbc)
					entries[row_entry] = 1.0;
				continue;
			}
			ALBANY_ASSERT(local_col >= col_map->getMinLocalIndex());
			ALBANY_ASSERT(local_col <= col_map->getMaxLocalIndex());

			if ((row_is_dbc || (col_is_dbc && SDBC)) && !leave_this_dof)
				entries[row_entry] = 0.0;
		}
		W_rT->replaceLocalValues(local_row, indices(), entries());
	}
	W_rT->fillComplete();

	Petra::TpetraCrsMatrix_To_EpetraCrsMatrix(W_rT, *W_r, app_->getEpetraComm());
}


int ROME_call = 0;
int count_sol_MR = 0;
int count_res_MR = 0;
int count_jac_MR = 0;

bool recomputePreconditioner = true;  //should always be initialized to true
bool recomputePreconditionerStepStart = false;

void ReducedOrderModelEvaluator::evalModel(const InArgs &inArgs, const OutArgs &outArgs) const
{
	ROME_call++;
	if (outputTrace_ == true)
		parOut("ReducedOrderModelEvaluator::evalModel... starting call ", ROME_call);

	// Copy arguments to be able to modify x and x_dot
	InArgs fullInArgs = fullOrderModel_->createInArgs();
	{
		// Copy untouched supported inArgs content
		if (fullInArgs.supports(IN_ARG_t))     fullInArgs.set_t(inArgs.get_t());
		if (fullInArgs.supports(IN_ARG_alpha)) fullInArgs.set_alpha(inArgs.get_alpha());
		if (fullInArgs.supports(IN_ARG_beta))  fullInArgs.set_beta(inArgs.get_beta());
		for (int l = 0; l < fullInArgs.Np(); ++l) {
			fullInArgs.set_p(l, inArgs.get_p(l));
		}

		if (outputTrace_ == true)
			parOut("ReducedOrderModelEvaluator::evalModel... construct full solution");
		// x <- basis * x_r + x_origin
		TEUCHOS_TEST_FOR_EXCEPT(is_null(inArgs.get_x()));
		fullInArgs.set_x(solutionSpace_->expansion(*inArgs.get_x()));

		count_sol_MR++;
		if (writeSolution_reduced_)
		{
			printConstMultiVector("xr",inArgs.get_x(), count_sol_MR);
			printConstMultiVector("x",fullInArgs.get_x(), count_sol_MR);
		}

		// x_dot <- basis * x_dot_r
		if (inArgs.supports(IN_ARG_x_dot) && nonnull(inArgs.get_x_dot())) {
			fullInArgs.set_x_dot(solutionSpace_->linearExpansion(*inArgs.get_x_dot()));
		}
	}

	// Copy arguments to be able to modify f and W
	OutArgs fullOutArgs = fullOrderModel_->createOutArgs();

	const bool supportsResidual = fullOutArgs.supports(OUT_ARG_f);
	bool requestedResidual = supportsResidual && nonnull(outArgs.get_f());

	const bool supportsJacobian = fullOutArgs.supports(OUT_ARG_W);
	const bool requestedJacobian = supportsJacobian && nonnull(outArgs.get_W());

	if (outputTrace_ == true)
	{
		if (app_->getEpetraComm()->MyPID() == 0)
		{
			std::cout << "RunMode (ROME call " << ROME_call << ", step " << step_ << ", iter " << iter_ << "): ";
			std::cout << "requestedResidual = " << std::boolalpha << requestedResidual;
			std::cout << ", requestedJacobian = " << std::boolalpha << requestedJacobian << std::endl;
		}
	}


	if (outputTrace_ == true)
	{
		if (fullOutArgs.supports(OUT_ARG_f))
		{
			parOut("FOM supports Residual");
			if (nonnull(outArgs.get_f()))
				parOut("ROM f is nonnull, requestedResidual will be set to true");
			else
				parOut("ROM f is null, requestedResidual will be set to false");
		}
		else
		{
			parOut("FOM does not support Residual");
		}

		if (fullOutArgs.supports(OUT_ARG_W))
		{
			parOut("FOM supports Jacobian");
			if (nonnull(outArgs.get_W()))
				parOut("ROM W is nonnull, requestedJacobian will be set to true");
			else
				parOut("ROM W is null, requestedJacobian will be set to false");
		}
		else
		{
			parOut("FOM does not support Jacobian");
		}

		parOut("ROM Np = ",outArgs.Np());
		parOut("ROM Ng = ",outArgs.Ng());
		parOut("FOM Np = ",fullOutArgs.Np());
		parOut("FOM Ng = ",fullOutArgs.Ng());
	}
	for (int j = 0; j < outArgs.Ng(); ++j)
	{
		if (nonnull(outArgs.get_g(j)))
		{
			if (outputTrace_ == true)
				parOut("ROM g(j) is nonnull for j = ",j);
			//if (useInvJac_ || useScaling_ || usePreconditionerIfpack_)
			if (PrecondType!=none && PrecondType!=projSoln)
			{
				recomputePreconditioner = true;
				if (outputTrace_ == true)
					parOut("Setting recomputePreconditioner to true @ ROME call ",ROME_call);
			}
		}
		else
		{
			if (outputTrace_ == true)
				parOut("ROM g(j) is null for j = ",j);
		}
	}

	bool requestedAnyDfDp = false;
	for (int l = 0; l < outArgs.Np(); ++l) {
		const bool supportsDfDp = !outArgs.supports(OUT_ARG_DfDp, l).none();
		const bool requestedDfDp = supportsDfDp && (!outArgs.get_DfDp(l).isEmpty());
		if (requestedDfDp) {
			requestedAnyDfDp = true;
			break;
		}
	}
	const bool requestedProjection = requestedResidual || requestedAnyDfDp;
	const bool fullJacobianRequired =
			reducedOpFactory_->fullJacobianRequired(requestedProjection, requestedJacobian)
			&& !(step_ == 0 && isThermoMech_); // this is a (depreciated) way to not run into trouble with initial step on thermo-mechanical problems - see the note where isThermoMech_ is set for more info

	{
		// Prepare forwarded outArgs content (g and DgDp)
		for (int j = 0; j < outArgs.Ng(); ++j) {
			fullOutArgs.set_g(j, outArgs.get_g(j));
			for (int l = 0; l < inArgs.Np(); ++l) {
				if (!outArgs.supports(OUT_ARG_DgDp, j, l).none()) {
					fullOutArgs.set_DgDp(j, l, outArgs.get_DgDp(j, l));
				}
			}
		}

		// Prepare reduced residual (f_r)
		if (requestedResidual) {
			const Evaluation<Epetra_Vector> f_r = outArgs.get_f();
			const Evaluation<Epetra_Vector> f(
					rcp(new Epetra_Vector(*fullOrderModel_->get_f_map(), false)),
					f_r.getType());
			fullOutArgs.set_f(f);
		}

		if (fullJacobianRequired) {
			fullOutArgs.set_W(fullOrderModel_->create_W());
		}

		// Prepare reduced sensitivities DgDx_r (Only mv with gradient orientation is supported)
		for (int j = 0; j < outArgs.Ng(); ++j) {
			if (!outArgs.supports(OUT_ARG_DgDx, j).none()) {
				TEUCHOS_ASSERT(outArgs.supports(OUT_ARG_DgDx, j).supports(DERIV_TRANS_MV_BY_ROW));
				if (!outArgs.get_DgDx(j).isEmpty()) {
					TEUCHOS_ASSERT(
							nonnull(outArgs.get_DgDx(j).getMultiVector()) &&
							outArgs.get_DgDx(j).getMultiVectorOrientation() == DERIV_TRANS_MV_BY_ROW);
					const int g_size = fullOrderModel_->get_g_map(j)->NumGlobalElements();
					const RCP<Epetra_MultiVector> full_dgdx_mv = rcp(
							new Epetra_MultiVector(*fullOrderModel_->get_x_map(), g_size, false));
					const Derivative full_dgdx_deriv(full_dgdx_mv, DERIV_TRANS_MV_BY_ROW);
					fullOutArgs.set_DgDx(j, full_dgdx_deriv);
				}
			}
		}

		// Prepare reduced sensitivities DgDx_dot_r (Only mv with gradient orientation is supported)
		for (int j = 0; j < outArgs.Ng(); ++j) {
			if (!outArgs.supports(OUT_ARG_DgDx_dot, j).none()) {
				TEUCHOS_ASSERT(outArgs.supports(OUT_ARG_DgDx_dot, j).supports(DERIV_TRANS_MV_BY_ROW));
				if (!outArgs.get_DgDx_dot(j).isEmpty()) {
					TEUCHOS_ASSERT(
							nonnull(outArgs.get_DgDx_dot(j).getMultiVector()) &&
							outArgs.get_DgDx_dot(j).getMultiVectorOrientation() == DERIV_TRANS_MV_BY_ROW);
					const int g_size = fullOrderModel_->get_g_map(j)->NumGlobalElements();
					const RCP<Epetra_MultiVector> full_dgdx_dot_mv = rcp(
							new Epetra_MultiVector(*fullOrderModel_->get_x_map(), g_size, false));
					const Derivative full_dgdx_dot_deriv(full_dgdx_dot_mv, DERIV_TRANS_MV_BY_ROW);
					fullOutArgs.set_DgDx_dot(j, full_dgdx_dot_deriv);
				}
			}
		}

		// Prepare reduced sensitivities DfDp_r (Only mv with jacobian orientation is supported)
		for (int l = 0; l < outArgs.Np(); ++l) {
			if (!outArgs.supports(OUT_ARG_DfDp, l).none()) {
				TEUCHOS_ASSERT(outArgs.supports(OUT_ARG_DfDp, l).supports(DERIV_MV_BY_COL));
				if (!outArgs.get_DfDp(l).isEmpty()) {
					TEUCHOS_ASSERT(
							nonnull(outArgs.get_DfDp(l).getMultiVector()) &&
							outArgs.get_DfDp(l).getMultiVectorOrientation() == DERIV_MV_BY_COL);
					const int p_size = fullOrderModel_->get_p_map(l)->NumGlobalElements();
					const RCP<Epetra_MultiVector> full_dfdp_mv = rcp(
							new Epetra_MultiVector(*fullOrderModel_->get_f_map(), p_size, false));
					const Derivative full_dfdp_deriv(full_dfdp_mv, DERIV_MV_BY_COL);
					fullOutArgs.set_DfDp(l, full_dfdp_deriv);
				}
			}
		}
	}

	if (outputTrace_ == true)
	{
		parOut("ReducedOrderModelEvaluator::evalModel... run FOM");
		if (nonnull(fullOutArgs.get_f()))
			parOut("ReducedOrderModelEvaluator::evalModel... run FOM - full residual is nonnull");
		if (nonnull(fullOutArgs.get_W()))
			parOut("ReducedOrderModelEvaluator::evalModel... run FOM - full Jacobian is nonnull");
	}

	// (f, W) <- fullOrderModel(x, x_dot, ...)
	fullOrderModel_->evalModel(fullInArgs, fullOutArgs);

	double this_time = getTime(inArgs);
	bool SDBC = app_->getProblem()->useSDBCs();

	if ((prev_time_ != this_time) && SDBC){
		if (writeSolution_reduced_)
		{
			printConstMultiVector("xr_pre",inArgs.get_x(), count_sol_MR);
			printConstMultiVector("x_pre",fullInArgs.get_x(), count_sol_MR);
		}


		solutionSpace_->reduction(*fullInArgs.get_x(),const_cast<Epetra_Vector&> (*inArgs.get_x()));
		//const_cast<InArgs&>(inArgs).set_x(solutionSpace_->reduction(*fullInArgs.get_x())); // DOESN'T WORK!! changes the solution, but doesn't get out to the solver somehow!
		prev_time_ = this_time;

		if (writeSolution_reduced_)
		{
			printConstMultiVector("xr_post",inArgs.get_x(), count_sol_MR);
			printConstMultiVector("x_post",fullInArgs.get_x(), count_sol_MR);
		}
	}

	// (W * basis, W_r) <- W
	if (fullJacobianRequired) {
		if (outputTrace_ == true)
			parOut("ReducedOrderModelEvaluator::evalModel... multiply Jac*phi");

		const RCP<Epetra_CrsMatrix> W_temp = rcp_dynamic_cast<Epetra_CrsMatrix>(fullOutArgs.get_W());
		if (!apply_bcs_ && !prec_full_jac_){ // if we didn't handle DBCs in evalModel, we need to do that here...
			parOut("calling DBC_CRSMatrix on the Jacobian (before preconditioning)");
			DBC_CRSMatrix(W_temp);
	 	}
		if (run_singular_check_)
			singularCheck(W_temp);

		reducedOpFactory_->fullJacobianIs(*fullOutArgs.get_W());

		count_jac_MR++;
		if (outputTrace_ == true)
		{
			parOut("ReducedOrderModelEvaluator::evalModel... full Jacobian norm = ", W_temp->NormFrobenius());
		}
		if (writeJacobian_reduced_)
		{
			printCRSMatrix("J", W_temp,count_jac_MR);
		}
		if (PrecondType!=none)
		{
			if (recomputePreconditioner)
			{
				// use following line to specify that preconditioner is only computed at beginning of successful continuation step
				if (recomputePreconditionerStepStart)
					recomputePreconditioner = false;
				if (outputTrace_ == true)
					parOut("ReducedOrderModelEvaluator::evalModel... computing preconditioner");
				switch(PrecondType)
				{
				case identity:
				{
					Teuchos::RCP<Epetra_MultiVector> I = eye();
					reducedOpFactory_->setPreconditionerDirectly(*I);
					if (writePreconditioner_)
					{
						printConstMultiVector("M", reducedOpFactory_->getPreconditioner(),count_jac_MR);
					}
					break;
				}
				case scaling:
				{
					reducedOpFactory_->setScaling(*W_temp);
					//reducedOpFactory_->getScaling()->Print(std::cout);
					break;
				}
				case invJac:
				{
					reducedOpFactory_->setPreconditioner(*W_temp);
					if (writePreconditioner_)
					{
						printConstMultiVector("M", reducedOpFactory_->getPreconditioner(),count_jac_MR);
					}
					break;
				}
				case projSoln:
				{
					parOut("setting up projected solution preconditioning");
					reducedOpFactory_->setJacobian(*W_temp);
					if (writePreconditioner_)
					{
						printConstCRSMatrix("ProjJ", reducedOpFactory_->getJacobian(), count_jac_MR);
					}
					break;
				}
				case ifPack:
				{
					parOut("setting up Ifpack preconditioning");
					if (ifpackType_.compare("Identity") == 0)
					{
						reducedOpFactory_->setPreconditionerIfpack(*eyeFromCRSMatrix(W_temp), "Identity");
					}
					else
					{
						reducedOpFactory_->setPreconditionerIfpack(*W_temp, ifpackType_);
					}
					if (writePreconditioner_ || prec_full_jac_)
					{
						Teuchos::RCP<Epetra_MultiVector> M = eye(); // can pull into if statement for a bit of comp. gain
						reducedOpFactory_->applyPreconditionerIfpack(*M); // can pull into if statement for a bit of comp. gain
						if (writePreconditioner_)
						{
							printMultiVector("M", M, count_jac_MR);
						}
						if (prec_full_jac_){
							DBC_MultiVector(M);
							reducedOpFactory_->setPreconditionerDirectly(*M);
							if (writePreconditioner_)
								printMultiVector("M_post", M, count_jac_MR);
						}
					}
					break;
				}
				}
			}
		}
		if (!apply_bcs_ && prec_full_jac_){ // if we didn't handle DBCs in evalModel, we need to do that here...
			parOut("calling DBC_CRSMatrix on the Jacobian (after preconditioning)");
			DBC_CRSMatrix(W_temp);
			printCRSMatrix("J_post", W_temp, count_jac_MR);
		}
	}

	//Apply scaling to Jac*phi
	//if (useInvJac_ || useScaling_ || usePreconditionerIfpack_)
	if (fullJacobianRequired)
	{
		if (writeJacobian_reduced_)
		{
			printConstMultiVector("Phi", reducedOpFactory_->getReducedBasis(), count_jac_MR);
			printConstMultiVector("JPhi", reducedOpFactory_->getPremultipliedReducedBasis(), count_jac_MR);
			printConstMultiVector("Psi", reducedOpFactory_->getLeftBasis(), count_jac_MR);
		}
		if (PrecondType!=none && PrecondType!=projSoln)
		{
			switch (PrecondType)
			{
			case identity:
			{
				//
#if !precLBonly
				//reducedOpFactory_->applyPreconditioner(*reducedOpFactory_->getPremultipliedReducedBasis()); // DOESN'T WORK IN PARALLEL
				multiplyInPlace(*reducedOpFactory_->getPremultipliedReducedBasis(), *reducedOpFactory_->getPreconditioner());
#endif
				//reducedOpFactory_->applyPreconditioner(*reducedOpFactory_->getLeftBasis()); // DOESN'T WORK IN PARALLEL
				multiplyInPlace(*reducedOpFactory_->getLeftBasis(), *reducedOpFactory_->getPreconditioner());
				break;
			}
			case scaling:
			{
				reducedOpFactory_->applyScaling(*reducedOpFactory_->getPremultipliedReducedBasis());
				reducedOpFactory_->applyScaling(*reducedOpFactory_->getLeftBasis());
				break;
			}
			case invJac:
			{
#if !precLBonly
				//reducedOpFactory_->applyPreconditioner(*reducedOpFactory_->getPremultipliedReducedBasis()); // DOESN'T WORK IN PARALLEL
				multiplyInPlace(*reducedOpFactory_->getPremultipliedReducedBasis(), *reducedOpFactory_->getPreconditioner());
#endif
				//reducedOpFactory_->applyPreconditioner(*reducedOpFactory_->getLeftBasis()); // DOESN'T WORK IN PARALLEL
				multiplyInPlace(*reducedOpFactory_->getLeftBasis(), *reducedOpFactory_->getPreconditioner());
				break;
			}
			case ifPack:
			{
#if !precLBonly
				if (prec_full_jac_)
					multiplyInPlace(*reducedOpFactory_->getPremultipliedReducedBasis(), *reducedOpFactory_->getPreconditioner());
				else
					reducedOpFactory_->applyPreconditionerIfpack(*reducedOpFactory_->getPremultipliedReducedBasis());
#endif //precLBonly
				if (prec_full_jac_)
					multiplyInPlace(*reducedOpFactory_->getLeftBasis(), *reducedOpFactory_->getPreconditioner());
				else
					reducedOpFactory_->applyPreconditionerIfpack(*reducedOpFactory_->getLeftBasis());
				break;
			}
			}

			if (writePreconditioner_)
			{
				printConstMultiVector("MJPhi", reducedOpFactory_->getPremultipliedReducedBasis(), count_jac_MR);
				printConstMultiVector("MPsi", reducedOpFactory_->getLeftBasis(), count_jac_MR);
			}
		}
	}

	// f_r <- leftBasis^T * f
	if (requestedResidual) {
		if (outputTrace_ == true)
			parOut("ReducedOrderModelEvaluator::evalModel... multiply psiT*res");

		if (!apply_bcs_){ // if we didn't handle DBCs in evalModel, we need to do that here...
			parOut("calling DBC_MultiVector on R");
			DBC_MultiVector(fullOutArgs.get_f());
		}
		if (run_nan_check_)
			nanCheck(fullOutArgs.get_f());

		count_res_MR++;
		if (outputTrace_ == true)
		{
			double res_norm = 0.0;
			fullOutArgs.get_f()->Norm2(&res_norm);
			parOut("ReducedOrderModelEvaluator::evalModel... full residual norm = ", res_norm);
		}
		if (writeResidual_reduced_)
		{
			printMultiVector("R",fullOutArgs.get_f(), count_res_MR);
		}

		// apply preconditioner to full residual, f
		RCP<Epetra_Vector> prevF = solutionSpace_->linearReduction(*fullOutArgs.get_f()); // make a copy of the reduced full residual (i.e. Phi'*f) that will be used to replace the DBC rows of f_r later

		if (fullJacobianRequired)
		{
			switch(PrecondType)
			{
				case identity:
				{
#if !precLBonly
					//reducedOpFactory_->applyPreconditioner(*fullOutArgs.get_f());  // DOESN'T WORK IN PARALLEL
					multiplyInPlace(*fullOutArgs.get_f(), *reducedOpFactory_->getPreconditioner());
#endif
					break;
				}
				case scaling:
				{
					reducedOpFactory_->applyScaling(*fullOutArgs.get_f());
					break;
				}
				case invJac:
				{
#if !precLBonly
					//reducedOpFactory_->applyPreconditioner(*fullOutArgs.get_f());  // DOESN'T WORK IN PARALLEL
					multiplyInPlace(*fullOutArgs.get_f(), *reducedOpFactory_->getPreconditioner());
#endif
					break;
				}
				case projSoln:
				{
					reducedOpFactory_->applyJacobian(*fullOutArgs.get_f());
					break;
				}
				case ifPack:
				{
#if !precLBonly
					if (prec_full_jac_)
						multiplyInPlace(*fullOutArgs.get_f(), *reducedOpFactory_->getPreconditioner());
					else
						reducedOpFactory_->applyPreconditionerIfpack(*fullOutArgs.get_f());
#endif //precLBonly
					break;
				}
			}
			if (PrecondType!=none)
			{
				if (outputTrace_ == true)
				{
					double res_norm = 0.0;
					fullOutArgs.get_f()->Norm2(&res_norm);
					parOut("ReducedOrderModelEvaluator::evalModel... full residual norm (after preconditioner applied) = ", res_norm);
				}
			}
			if ((PrecondType!=none) && writePreconditioner_)
			{
				printMultiVector("MR",fullOutArgs.get_f(), count_res_MR);
			}
		}

		if (PrecondType == projSoln)
		{
			// f_r <- phi^T * ( inv(J)*f )
			reducedOpFactory_->leftProjection_ProjectedSol(*fullOutArgs.get_f(), *outArgs.get_f());
		}
		else
		{
			// f_r <- psi^T * f
			reducedOpFactory_->leftProjection(*fullOutArgs.get_f(), *outArgs.get_f());
		}

		if (outputTrace_ == true)
		{
			double res_norm = 0;
			outArgs.get_f()->Norm2(&res_norm);
			parOut("ReducedOrderModelEvaluator::evalModel... reduced residual norm (psiT*resid, pre DBC) = ", res_norm);
		}
		if (writeResidual_reduced_)
		{
			printMultiVector("Rr_pre", outArgs.get_f(), count_res_MR);
		}

		// replace DBC rows of fr with Phi_B*f
		{
			Teuchos::RCP<Tpetra_Vector> f_T = Petra::EpetraVector_To_TpetraVectorNonConst(*outArgs.get_f(),app_->getComm());
			Teuchos::RCP<Tpetra_Vector> prevF_T = Petra::EpetraVector_To_TpetraVectorNonConst(*prevF,app_->getComm());

			auto f_view = f_T->get1dViewNonConst();
			auto prevF_view = prevF_T->get1dViewNonConst();

			auto row_map = f_T->getMap();
			ALBANY_ASSERT(row_map->isSameAs(*prevF_T->getMap()));

			auto num_local_rows = f_T->getLocalLength();
			for (auto local_row = 0; local_row < num_local_rows; ++local_row){
				auto global_row = row_map->getGlobalElement(local_row);
				if (global_row < num_dbc_modes_)
					f_view[local_row] = prevF_view[local_row];
			}
			Petra::TpetraVector_To_EpetraVector(f_T,*outArgs.get_f(),app_->getEpetraComm());
		}

		if (outputTrace_ == true)
		{
			double res_norm = 0;
			outArgs.get_f()->Norm2(&res_norm);
			parOut("ReducedOrderModelEvaluator::evalModel... reduced residual norm (psiT*resid, post DBC) = ", res_norm);
		}
		if (writeResidual_reduced_)
		{
			printMultiVector("Rr", outArgs.get_f(), count_res_MR);
		}
	}

	// Wr <- leftBasis^T * W * basis
	if (requestedJacobian) {
		if (outputTrace_ == true)
			parOut("ReducedOrderModelEvaluator::evalModel... multiply psiT*(Jac*phi)");

		RCP<Epetra_CrsMatrix> W_r = rcp_dynamic_cast<Epetra_CrsMatrix>(outArgs.get_W());
		TEUCHOS_TEST_FOR_EXCEPT(is_null((W_r)));

		if (PrecondType==projSoln)
		{
			// set W_r to the identity matrix
			reducedOpFactory_->reducedJacobian_ProjectedSol(*W_r);
		}
		else
		{
			reducedOpFactory_->reducedJacobian(*W_r);
		}

		if (outputTrace_ == true)
		{
			parOut("ReducedOrderModelEvaluator::evalModel... reduced Jacobian norm (pre DBC) = ", W_r->NormFrobenius());
		}
		if (writeJacobian_reduced_)
		{
			std::string full_filename = outdir_ + "JEr_pre" + std::to_string(count_jacr_pl) + ".mm";
			EpetraExt::RowMatrixToMatrixMarketFile(full_filename.c_str(), *W_r);
			//EpetraExt::BlockMapToMatrixMarketFile("JEr_pre_Rmap.mm", W_r->RowMap());
			//EpetraExt::BlockMapToMatrixMarketFile("JEr_pre_Cmap.mm", W_r->ColMap());
			//printCRSMatrix("Jr_pre", W_r, count_jacr_pl); // Tpetra::MatrixMarket::Writer<Tpetra_CrsMatrix>::writeSparseFile fails here because the Jacobian is locally replicated (see issue #1021 on GitHub)
		}

		DBC_ROM_jac(inArgs,outArgs,SDBC);

		if (outputTrace_ == true)
		{
			parOut("ReducedOrderModelEvaluator::evalModel... reduced Jacobian norm (post DBC) = ", W_r->NormFrobenius());
		}
		if (writeJacobian_reduced_)
		{
			std::string full_filename = outdir_ + "JEr" + std::to_string(count_jacr_pl) + ".mm";
			EpetraExt::RowMatrixToMatrixMarketFile(full_filename.c_str(), *W_r);
			//EpetraExt::BlockMapToMatrixMarketFile("JEr_Rmap.mm", W_r->RowMap());
			//EpetraExt::BlockMapToMatrixMarketFile("JEr_Cmap.mm", W_r->ColMap());
			//printCRSMatrix("Jr", W_r, count_jacr_pl); // Tpetra::MatrixMarket::Writer<Tpetra_CrsMatrix>::writeSparseFile fails here because the Jacobian is locally replicated (see issue #1021 on GitHub)
			count_jacr_pl++;
		}
//TEUCHOS_TEST_FOR_EXCEPTION(true, std::runtime_error, "we'll just stop here...");
	}

	// (DgDx_r)^T <- basis^T * (DgDx)^T
	for (int j = 0; j < outArgs.Ng(); ++j) {
		if (!outArgs.supports(OUT_ARG_DgDx, j).none()) {
			const RCP<Epetra_MultiVector> dgdx_r_mv = outArgs.get_DgDx(j).getMultiVector();
			if (nonnull(dgdx_r_mv)) {
				if (outputTrace_ == true)
					parOut("ReducedOrderModelEvaluator::evalModel... compute DgDx_r");
				const RCP<const Epetra_MultiVector> full_dgdx_mv = fullOutArgs.get_DgDx(j).getMultiVector();
				solutionSpace_->linearReduction(*full_dgdx_mv, *dgdx_r_mv);
			}
		}
	}

	// (DgDx_dot_r)^T <- basis^T * (DgDx_dot)^T
	for (int j = 0; j < outArgs.Ng(); ++j) {
		if (!outArgs.supports(OUT_ARG_DgDx_dot, j).none()) {
			const RCP<Epetra_MultiVector> dgdx_dot_r_mv = outArgs.get_DgDx_dot(j).getMultiVector();
			if (nonnull(dgdx_dot_r_mv)) {
				if (outputTrace_ == true)
					parOut("ReducedOrderModelEvaluator::evalModel... compute DgDx_dot_r");
				const RCP<const Epetra_MultiVector> full_dgdx_dot_mv = fullOutArgs.get_DgDx_dot(j).getMultiVector();
				solutionSpace_->linearReduction(*full_dgdx_dot_mv, *dgdx_dot_r_mv);
			}
		}
	}

	// DfDp_r <- leftBasis^T * DfDp
	for (int l = 0; l < outArgs.Np(); ++l) {
		if (!outArgs.supports(OUT_ARG_DfDp, l).none()) {
			const RCP<Epetra_MultiVector> dfdp_r_mv = outArgs.get_DfDp(l).getMultiVector();
			if (nonnull(dfdp_r_mv)) {
				if (outputTrace_ == true)
					parOut("ReducedOrderModelEvaluator::evalModel... compute DfDp_r");
				const RCP<const Epetra_MultiVector> full_dfdp_mv = fullOutArgs.get_DfDp(l).getMultiVector();
				reducedOpFactory_->leftProjection(*full_dfdp_mv, *dfdp_r_mv);
			}
		}
	}

	if (outputTrace_ == true)
	{
		parOut("ReducedOrderModelEvaluator::evalModel... done");
	}

	if (requestedJacobian)
	{
		iter_++;
	}
	if (!requestedResidual && !requestedJacobian)
	{
		step_++;
		iter_=0;
	}
}

} // namespace MOR
