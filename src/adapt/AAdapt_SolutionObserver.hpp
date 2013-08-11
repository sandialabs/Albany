#ifndef AADAPT_SOLUTIONOBSERVER_HPP
#define AADAPT_SOLUTIONOBSERVER_HPP

#include "Piro_SolutionObserverBase.hpp"

namespace AAdapt {

class SolutionObserver : public Piro::SolutionObserverBase<double, const Thyra::VectorBase<double> > {

public:

  SolutionObserver() {}

  void observeResponse(
      int j,
      const Teuchos::RCP<Thyra::ModelEvaluatorBase::OutArgs<double> >& outArgs,
      const Teuchos::RCP<Teuchos::Array<Teuchos::RCP<const Thyra::VectorBase<double> > > > &responses,
      const Teuchos::RCP<const Thyra::VectorBase<double> > &g);

  void set_g_vector(int j, const Teuchos::RCP<Thyra::VectorBase<double> >& g_j);

  virtual ~SolutionObserver() {}

private:

    Teuchos::RCP<Thyra::ModelEvaluatorBase::OutArgs<double> > outArgs;
    Teuchos::RCP<Teuchos::Array<Teuchos::RCP<const Thyra::VectorBase<double> > > > responses;

};

} // namespace Adapt

#endif /* AADAPT_SOLUTIONOBSERVER_HPP */
