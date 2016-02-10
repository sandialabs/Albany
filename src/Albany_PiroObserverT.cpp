//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

//IK, 9/12/14: no Epetra!

#include "Albany_PiroObserverT.hpp"
#include "PHAL_AlbanyTraits.hpp"
#include "Teuchos_ScalarTraits.hpp"

#include <cstddef>

Albany::PiroObserverT::PiroObserverT(
    const Teuchos::RCP<Albany::Application> &app, 
    Teuchos::RCP<const Thyra::ModelEvaluator<double>> model) :
  impl_(app), 
  model_(model) 
  {}

void
Albany::PiroObserverT::observeSolution(const Thyra::VectorBase<ST> &solution)
{
  this->observeSolutionImpl(solution, Teuchos::ScalarTraits<ST>::zero());
}

void
Albany::PiroObserverT::observeSolution(
    const Thyra::VectorBase<ST> &solution,
    const ST stamp)
{
  this->observeSolutionImpl(solution, stamp);
}

void
Albany::PiroObserverT::observeSolution(
    const Thyra::VectorBase<ST> &solution,
    const Thyra::VectorBase<ST> &solution_dot,
    const ST stamp)
{
  this->observeSolutionImpl(solution, solution_dot, stamp);
}

void
Albany::PiroObserverT::observeSolution(
    const Thyra::MultiVectorBase<ST> &solution,
    const ST stamp)
{
  this->observeSolutionImpl(solution, stamp);
}

namespace { // anonymous

Teuchos::RCP<const Tpetra_Vector>
tpetraFromThyra(const Thyra::VectorBase<double> &v)
{
  // Create non-owning RCP to solution to use the Thyra -> Epetra converter
  // This is safe since we will not be creating any persisting relations
  const Teuchos::RCP<const Thyra::VectorBase<double> > v_nonowning_rcp =
    Teuchos::rcpFromRef(v);

  return ConverterT::getConstTpetraVector(v_nonowning_rcp);
}

Teuchos::RCP<const Tpetra_MultiVector>
tpetraMVFromThyraMV(const Thyra::MultiVectorBase<double> &v)
{
  // Create non-owning RCP to solution to use the Thyra -> Epetra converter
  // This is safe since we will not be creating any persisting relations
  const Teuchos::RCP<const Thyra::MultiVectorBase<double> > v_nonowning_rcp =
    Teuchos::rcpFromRef(v);

  return ConverterT::getConstTpetraMultiVector(v_nonowning_rcp);
}

} // anonymous namespace

void
Albany::PiroObserverT::observeSolutionImpl(
    const Thyra::VectorBase<ST> &solution,
    const ST defaultStamp)
{
  const Teuchos::RCP<const Tpetra_Vector> solution_tpetra =
    tpetraFromThyra(solution);

  this->observeTpetraSolutionImpl(
      *solution_tpetra,
      Teuchos::null,
      defaultStamp);
  
  // observe responses 
  if (model_ != Teuchos::null) {
    this->observeResponse(defaultStamp, Teuchos::rcpFromRef(solution));
   }
}

void
Albany::PiroObserverT::observeSolutionImpl(
    const Thyra::VectorBase<ST> &solution,
    const Thyra::VectorBase<ST> &solution_dot,
    const ST defaultStamp)
{
  const Teuchos::RCP<const Tpetra_Vector> solution_tpetra =
    tpetraFromThyra(solution);
  const Teuchos::RCP<const Tpetra_Vector> solution_dot_tpetra =
    tpetraFromThyra(solution_dot);

  this->observeTpetraSolutionImpl(
      *solution_tpetra,
      solution_dot_tpetra.ptr(),
      defaultStamp);

  // observe responses 
  if (model_ != Teuchos::null) {
    this->observeResponse(defaultStamp, Teuchos::rcpFromRef(solution), Teuchos::rcpFromRef(solution_dot));
   }
}

void
Albany::PiroObserverT::observeSolutionImpl(
    const Thyra::MultiVectorBase<ST> &solution,
    const ST defaultStamp)
{
  const Teuchos::RCP<const Tpetra_MultiVector> solution_tpetraMV =
    tpetraMVFromThyraMV(solution);

  impl_.observeSolutionT(defaultStamp, *solution_tpetraMV);

}

void
Albany::PiroObserverT::observeTpetraSolutionImpl(
    const Tpetra_Vector &solution,
    Teuchos::Ptr<const Tpetra_Vector> solution_dot,
    const ST defaultStamp)
{
  // Determine the stamp associated with the snapshot
  const ST stamp = impl_.getTimeParamValueOrDefault(defaultStamp);
  impl_.observeSolutionT(stamp, solution, solution_dot);
}

void 
Albany::PiroObserverT::observeResponse(
    const ST defaultStamp, 
    Teuchos::RCP<const Thyra::VectorBase<ST>> solution,
    Teuchos::RCP<const Thyra::VectorBase<ST>> solution_dot)
{
  Teuchos::RCP<Teuchos::FancyOStream> out(Teuchos::VerboseObjectBase::getDefaultOStream());
  std::map<int,std::string> m_response_index_to_name;

  // build out args and evaluate responses if they exist
  Thyra::ModelEvaluatorBase::OutArgs<double> outArgs = model_->createOutArgs();
  if(outArgs.Ng()>0) {
    // build the in arguments
    Thyra::ModelEvaluatorBase::InArgs<double> nominal_values = model_->getNominalValues();
    Thyra::ModelEvaluatorBase::InArgs<double> inArgs = model_->createInArgs();
    inArgs.setArgs(nominal_values); 
    inArgs.set_x(solution);
    if(inArgs.supports(Thyra::ModelEvaluatorBase::IN_ARG_x_dot))
      inArgs.set_x_dot(solution_dot);
    if (inArgs.supports(Thyra::ModelEvaluatorBase::IN_ARG_t)) { 
      const ST time = impl_.getTimeParamValueOrDefault(defaultStamp);
      inArgs.set_t(time);
    }
  
    // set up the output arguments, in this case only the responses
    for(int i=0;i<outArgs.Ng();i++)
      outArgs.set_g(i,Thyra::createMember(*model_->get_g_space(i)));

    // Solve the model
    model_->evalModel(inArgs, outArgs);
  
    std::size_t precision = 8;
    std::size_t value_width = precision + 7;
    *out << std::scientific << std::showpoint << std::setprecision(precision) << std::left;


    // Note that we don't have g_names support in thyra yet.  Once
    // this is added, we can print response names as well.
  
    for(int i=0;i<outArgs.Ng();i++) {
      std::stringstream ss;
      std::map<int,std::string>::const_iterator itr = m_response_index_to_name.find(i);
      if(itr!=m_response_index_to_name.end())
        ss << "   Response \"" << itr->second << "\" = ";
      else
        ss << "   Response[" << i << "] = ";

      Teuchos::RCP<Thyra::VectorBase<double> > g = outArgs.get_g(i);
      *out << ss.str(); // "   Response[" << i << "] = ";
      for(Thyra::Ordinal k=0;k<g->space()->dim();k++)
        *out << std::setw(value_width) << Thyra::get_ele(*g,k) << " ";
        *out << std::endl;
    }
  }
}

