//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#ifndef AADAPT_SOLUTIONOBSERVER_HPP
#define AADAPT_SOLUTIONOBSERVER_HPP

#include "Albany_DataTypes.hpp"
#include "Piro_SolutionObserverBase.hpp"

namespace AAdapt {

class SolutionObserver : public Piro::SolutionObserverBase<ST, const Thyra::VectorBase<ST> > {

public:

  SolutionObserver() {}

  void observeResponse(
      int j,
      const Teuchos::RCP<Thyra::ModelEvaluatorBase::OutArgs<ST> >& outArgs,
      const Teuchos::RCP<Teuchos::Array<Teuchos::RCP<const Thyra::VectorBase<ST> > > > &responses,
      const Teuchos::RCP<const Thyra::VectorBase<ST> > &g);

  void set_g_vector(int j, const Teuchos::RCP<Thyra::VectorBase<ST> >& g_j);

  virtual ~SolutionObserver() {}

private:

    Teuchos::RCP<Thyra::ModelEvaluatorBase::OutArgs<ST> > outArgs;
    Teuchos::RCP<Teuchos::Array<Teuchos::RCP<const Thyra::VectorBase<ST> > > > responses;

};

} // namespace Adapt

#endif /* AADAPT_SOLUTIONOBSERVER_HPP */
