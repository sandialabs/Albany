
#include "AAdapt_SolutionObserver.hpp"

namespace AAdapt {

void
SolutionObserver::observeResponse(
      int j,
      const Teuchos::RCP<Thyra::ModelEvaluatorBase::OutArgs<ST> >& outArgs_,
      const Teuchos::RCP<Teuchos::Array<Teuchos::RCP<const Thyra::VectorBase<ST> > > > &responses_,
      const Teuchos::RCP<const Thyra::VectorBase<ST> > &g)
{
      outArgs = outArgs_;
      responses = responses_;
}

void 
SolutionObserver::set_g_vector(
      int j, 
      const Teuchos::RCP<Thyra::VectorBase<ST> >& g_j)
{
      outArgs->set_g(j, g_j);
      (*responses)[j] = g_j;
}

} // namespace Adapt

