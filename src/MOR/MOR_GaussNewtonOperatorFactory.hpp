//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//
#ifndef MOR_GAUSSNEWTONOPERATORFACTOR_HPP
#define MOR_GAUSSNEWTONOPERATORFACTOR_HPP

#include "MOR_ReducedOperatorFactory.hpp"

class Epetra_MultiVector;
class Epetra_CrsMatrix;
class Epetra_Operator;

#include "MOR_ReducedJacobianFactory.hpp"

#include "Teuchos_RCP.hpp"

#include "Teuchos_SerialDenseMatrix.hpp"  //JF
#include "Epetra_TsqrAdaptor.hpp"  //JF

namespace MOR {

template <typename Derived>
class GaussNewtonOperatorFactoryBase : public ReducedOperatorFactory {
public:
	explicit GaussNewtonOperatorFactoryBase(const Teuchos::RCP<const Epetra_MultiVector> &reducedBasis, bool PsiEqualsPhi, bool runWithQR);

	virtual bool fullJacobianRequired(bool residualRequested, bool jacobianRequested) const;

	virtual const Epetra_MultiVector &leftProjection(const Epetra_MultiVector &fullVec, Epetra_MultiVector &result) const;
	virtual const Epetra_MultiVector &leftProjection_ProjectedSol(const Epetra_MultiVector &fullVec, Epetra_MultiVector &result) const;

	virtual Teuchos::RCP<Epetra_CrsMatrix> reducedJacobianNew();
	virtual const Epetra_CrsMatrix &reducedJacobian(Epetra_CrsMatrix &result) const;
	virtual const Epetra_CrsMatrix &reducedJacobian_ProjectedSol(Epetra_CrsMatrix &result) const;

	virtual void fullJacobianIs(const Epetra_Operator &op);

	virtual Teuchos::RCP<const Epetra_MultiVector> getPremultipliedReducedBasis() const;
	virtual Teuchos::RCP<const Epetra_MultiVector> getReducedBasis() const;
	virtual Teuchos::RCP<const Epetra_MultiVector> getLeftBasis() const;

	virtual Teuchos::RCP<const Epetra_MultiVector> getScaling() const;
	virtual void setScaling(Epetra_CrsMatrix &jacobian) const;
	virtual void applyScaling(const Epetra_MultiVector &vector) const;

	virtual Teuchos::RCP<const Epetra_MultiVector> getPreconditioner() const;
	virtual void setPreconditionerDirectly(Epetra_MultiVector &vector) const;
	virtual void setPreconditioner(Epetra_CrsMatrix &jacobian) const;
	virtual void applyPreconditioner(const Epetra_MultiVector &vector) const;
	virtual void applyPreconditionerTwice(const Epetra_MultiVector &vector) const;

	virtual Teuchos::RCP<Ifpack_Preconditioner> getPreconditionerIfpack() const;
	virtual void setPreconditionerIfpack(Epetra_CrsMatrix &jacobian, std::string ifpackType) const;
	virtual void applyPreconditionerIfpack(const Epetra_MultiVector &vector) const;
	virtual void applyPreconditionerIfpackTwice(const Epetra_MultiVector &vector) const;

	virtual Teuchos::RCP<const Epetra_CrsMatrix> getJacobian() const;
	virtual void setJacobian(Epetra_CrsMatrix &jacobian) const;
	virtual void applyJacobian(const Epetra_MultiVector &vector) const;

protected:
	bool PsiEqualsPhi_, runWithQR_;
	void parOut(std::string text) const;

private:
	Teuchos::RCP<const Epetra_MultiVector> reducedBasis_;

	ReducedJacobianFactory jacobianFactory_;

	Teuchos::RCP<Epetra_MultiVector> scaling_;
	Teuchos::RCP<Epetra_MultiVector> preconditioner_;
	Teuchos::RCP<Epetra_MultiVector> leftbasis_;
	Teuchos::RCP<Epetra_MultiVector> Q_;
	Teuchos::RCP<Epetra_MultiVector> jacphi_int_;
	Teuchos::RCP<Teuchos::SerialDenseMatrix<int,double> > R_;
	Teuchos::RCP<Teuchos::ParameterList> tsqr_params_;
	Teuchos::RCP<Epetra::TsqrAdaptor> tsqr_adaptor_;
	mutable Teuchos::RCP<Ifpack_Preconditioner> preconditioner_ifpack_;
	mutable Teuchos::RCP<Epetra_CrsMatrix> jacobian_;

	Teuchos::RCP<const Epetra_MultiVector> getMyLeftBasis() const;
};

class GaussNewtonOperatorFactory : public GaussNewtonOperatorFactoryBase<GaussNewtonOperatorFactory> {
public:
	explicit GaussNewtonOperatorFactory(const Teuchos::RCP<const Epetra_MultiVector> &reducedBasis, bool PsiEqualsPhi, bool runWithQR);

	Teuchos::RCP<const Epetra_MultiVector> leftProjectorBasis() const;
};

class GaussNewtonMetricOperatorFactory : public GaussNewtonOperatorFactoryBase<GaussNewtonMetricOperatorFactory> {
public:
	GaussNewtonMetricOperatorFactory(const Teuchos::RCP<const Epetra_MultiVector> &reducedBasis,
			const Teuchos::RCP<const Epetra_Operator> &metric, bool PsiEqualsPhi, bool runWithQR);

	// Overridden
	virtual void fullJacobianIs(const Epetra_Operator &op);

	Teuchos::RCP<const Epetra_MultiVector> leftProjectorBasis() const;

private:
	Teuchos::RCP<const Epetra_Operator> metric_;

	Teuchos::RCP<Epetra_MultiVector> premultipliedLeftProjector_;

	void updatePremultipliedLeftProjector();
};

} // namespace MOR

#endif /* MOR_GAUSSNEWTONOPERATORFACTOR_HPP */
