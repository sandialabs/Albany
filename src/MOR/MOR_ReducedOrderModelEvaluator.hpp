//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//
#ifndef MOR_REDUCEDORDERMODELEVALUATOR_HPP
#define MOR_REDUCEDORDERMODELEVALUATOR_HPP

#include "Albany_ModelEvaluator.hpp"

#include "Teuchos_RCP.hpp"

namespace MOR {

class ReducedSpace;
class ReducedOperatorFactory;

class ReducedOrderModelEvaluator : public EpetraExt::ModelEvaluator {
public:
	ReducedOrderModelEvaluator(const Teuchos::RCP<EpetraExt::ModelEvaluator> &fullOrderModel,
			const Teuchos::RCP<const ReducedSpace> &solutionSpace,
			const Teuchos::RCP<ReducedOperatorFactory> &reducedOpFactory,
			const bool* outputFlags,
			const std::string preconditionerType);

	// Overridden functions
	virtual Teuchos::RCP<const Epetra_Map> get_x_map() const;
	virtual Teuchos::RCP<const Epetra_Map> get_f_map() const;
	virtual Teuchos::RCP<const Epetra_Map> get_p_map(int l) const;
	virtual Teuchos::RCP<const Teuchos::Array<std::string> > get_p_names(int l) const;
	virtual Teuchos::RCP<const Epetra_Map> get_g_map(int j) const;

	virtual Teuchos::RCP<const Epetra_Vector> get_x_init() const;
	virtual Teuchos::RCP<const Epetra_Vector> get_x_dot_init() const;
	virtual Teuchos::RCP<const Epetra_Vector> get_p_init(int l) const;
	virtual double get_t_init() const;

	virtual double getInfBound() const;
	virtual Teuchos::RCP<const Epetra_Vector> get_p_lower_bounds(int l) const;
	virtual Teuchos::RCP<const Epetra_Vector> get_p_upper_bounds(int l) const;
	virtual double get_t_upper_bound() const;
	virtual double get_t_lower_bound() const;

	virtual Teuchos::RCP<Epetra_Operator> create_W() const;
	virtual Teuchos::RCP<Epetra_Operator> create_DgDp_op(int j, int l) const;

	virtual InArgs createInArgs() const;
	virtual OutArgs createOutArgs() const;

	virtual void evalModel(const InArgs &inArgs, const OutArgs &outArgs) const;

	// Additional functions
	Teuchos::RCP<const EpetraExt::ModelEvaluator> getFullOrderModel() const;
	Teuchos::RCP<const ReducedSpace> getSolutionSpace() const;

	double valAtTime(std::pair< bool, std::vector<double> > vals, double t) const;
	void extract_DBC_data(Teuchos::RCP<Teuchos::ParameterList> DBC_params);

	void reset_x_and_x_dot_init();
	void reset_x_init();
	void reset_x_dot_init();

	void printCRSMatrix(std::string filename, const Teuchos::RCP<Epetra_CrsMatrix> CRSM, int index) const;
	void printConstCRSMatrix(std::string filename, const Teuchos::RCP<const Epetra_CrsMatrix> CRSM, int index) const;
	void printMultiVectorT(std::string full_filename, const Teuchos::RCP<Tpetra_MultiVector> MV, int index, bool isDist) const;
	void printMultiVector(std::string full_filename, const Teuchos::RCP<Epetra_MultiVector> MV, int index) const;
	void printConstMultiVector(std::string filename, const Teuchos::RCP<const Epetra_MultiVector> MV, int index) const;
	Teuchos::RCP<Epetra_MultiVector> eye() const;
	double getTime(const InArgs& inArgs) const;
	void nanCheck(const Teuchos::RCP<Epetra_Vector> &f) const;
	void singularCheck(const Teuchos::RCP<Epetra_CrsMatrix> &J) const;
	template <typename VectorType>
	Teuchos::RCP<VectorType> serialVector(Teuchos::RCP<VectorType> MV) const;
	template <typename VectorType>
	Teuchos::RCP<VectorType> locallyReplicatedVector(Teuchos::RCP<VectorType> MV) const;
	void multiplyInPlace(const Epetra_MultiVector &A, const Epetra_MultiVector M) const;

	void DBC_MultiVector(const Teuchos::RCP<Epetra_MultiVector>& MV_E) const;
	void DBC_CRSMatrix(const Teuchos::RCP<Epetra_CrsMatrix>& CRSM_E) const;
	void DBC_ROM_jac(const InArgs& inArgs, const OutArgs& outArgs, const bool SDBC) const;

	Teuchos::RCP<Epetra_CrsMatrix> MultiVector2CRSMatrix(Teuchos::RCP<Epetra_MultiVector> MV_E) const;

private:
	Teuchos::RCP<EpetraExt::ModelEvaluator> fullOrderModel_;
	Teuchos::RCP<Albany::Application> app_;
	Teuchos::RCP<Teuchos::ParameterList> morParams_;
	bool apply_bcs_, prec_full_jac_, run_singular_check_, run_nan_check_, isThermoMech_;
	int num_dbc_modes_;
	std::string outdir_;
	Teuchos::RCP<const ReducedSpace> solutionSpace_;
	struct gen_DBC_data{bool sdbc; std::string name; std::vector<std::vector<int>> const & ns_nodes; int dof; std::pair< bool, std::vector<double> > vals;};
	std::vector<gen_DBC_data> DBC_data_;

	Teuchos::RCP<ReducedOperatorFactory> reducedOpFactory_;

	const Epetra_Map &componentMap() const;
	Teuchos::RCP<const Epetra_Map> componentMapRCP() const;

	Teuchos::RCP<Epetra_Vector> x_init_;
	Teuchos::RCP<Epetra_Vector> x_dot_init_;

	bool outputTrace_;
	bool writeJacobian_reduced_;
	bool writeResidual_reduced_;
	bool writeSolution_reduced_;
	bool writePreconditioner_;

	enum PrecondTypes {none, identity, scaling, invJac, projSoln, ifPack};
	PrecondTypes PrecondType=none;
	std::string ifpackType_;

	mutable double prev_time_;
	mutable int step_, iter_;

	// Disallow copy and assignment
	ReducedOrderModelEvaluator(const ReducedOrderModelEvaluator &);
	ReducedOrderModelEvaluator &operator=(const ReducedOrderModelEvaluator &);
};

} // namespace MOR

#endif /* MOR_REDUCEDORDERMODELEVALUATOR_HPP */
