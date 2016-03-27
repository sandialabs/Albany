//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

//IK, 9/12/14: no Epetra!

#include "Albany_PiroObserverT.hpp"
#include "PHAL_AlbanyTraits.hpp"
#include "Teuchos_ScalarTraits.hpp"
#include "Thyra_VectorStdOps.hpp"

#include <cstddef>

Albany::PiroObserverT::PiroObserverT(
    const Teuchos::RCP<Albany::Application> &app, 
    Teuchos::RCP<const Thyra::ModelEvaluator<double>> model) :
  impl_(app), 
  model_(model), 
  out(Teuchos::VerboseObjectBase::getDefaultOStream())
  {
    observe_responses_ = false; 
    if ((app->observeResponses() == true) && (model_ != Teuchos::null)) 
      observe_responses_ = true;
    stepper_counter_ = 0;  
    observe_responses_every_n_steps_ = app->observeResponsesFreq();  

    relative_responses = app->getMarkersForRelativeResponses();
    if(relative_responses.size()){
    	calculateRelativeResponses = true;
    }else{
    	calculateRelativeResponses = false;
    }
    firstResponseObtained = false;
  }

void
Albany::PiroObserverT::observeSolution(const Thyra::VectorBase<ST> &solution)
{
  this->observeSolutionImpl(solution, Teuchos::ScalarTraits<ST>::zero());
  stepper_counter_++;
}

void
Albany::PiroObserverT::observeSolution(
    const Thyra::VectorBase<ST> &solution,
    const ST stamp)
{
  this->observeSolutionImpl(solution, stamp);
  stepper_counter_++; 
}

void
Albany::PiroObserverT::observeSolution(
    const Thyra::VectorBase<ST> &solution,
    const Thyra::VectorBase<ST> &solution_dot,
    const ST stamp)
{
  this->observeSolutionImpl(solution, solution_dot, stamp);
  stepper_counter_++; 
}

void
Albany::PiroObserverT::observeSolution(
    const Thyra::MultiVectorBase<ST> &solution,
    const ST stamp)
{
  this->observeSolutionImpl(solution, stamp);
  stepper_counter_++; 
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
  if (observe_responses_ == true) {
    if (stepper_counter_ % observe_responses_every_n_steps_ == 0) 
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
  if (observe_responses_ == true) {
    if (stepper_counter_ % observe_responses_every_n_steps_ == 0) 
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
      *out << "Time = " << time << "\n";  
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

    //OG It seems that outArgs.Ng() always returns 1, so, there is 1 response vector only, Response[0].
    //This response vector contains different responses (min, max, norms) and it would be good
    //to have functionality to obtain relative responses only for some values. But it would require more
    //parameters in param list. Alternatively, one can rewrite the code below to use is_relative
    //as an array of markers for relative responses for Response[0] only. This is not the case
    //right now and if in the param list "Relative Responses"="{0}", the code below will compute
    //relative values for all terms in vector Response[0].
    if((!firstResponseObtained) && calculateRelativeResponses ){
	  storedResponses.resize(outArgs.Ng());
	  is_relative.resize(outArgs.Ng(), false);
    }

    for(int i=0;i<outArgs.Ng();i++) {
      std::stringstream ss;
      std::map<int,std::string>::const_iterator itr = m_response_index_to_name.find(i);
      if(itr!=m_response_index_to_name.end())
        ss << "         Response \"" << itr->second << "\" = ";
      else
        ss << "         Response[" << i << "] = ";

      //ss << "relative resp size? " << relative_responses.size() << "\n";
      //ss << "relative resp values? " << relative_responses << "\n";

      Teuchos::RCP<Thyra::VectorBase<double> > g = outArgs.get_g(i);
      *out << ss.str(); // "   Response[" << i << "] = ";
      for(Thyra::Ordinal k=0;k<g->space()->dim();k++)
        *out << std::setw(value_width) << Thyra::get_ele(*g,k) << " ";
      *out << std::endl;

      if(firstResponseObtained && calculateRelativeResponses )
      if(is_relative[i]){
    	  *out << "\n";
	      *out << "Relative Response[" << i << "] = ";
          for( int j = 0; j < storedResponses[i].size(); j++){
        	  double prevresp = storedResponses[i][j];
        	  if( std::abs(prevresp) > tol ){
        		  *out << std::setw(value_width) << (Thyra::get_ele(*g,j) - prevresp)/prevresp << " ";
        	  }else{
        		  *out << " N/A(int. value 0) ";
        	  }
          }
    	  *out << "\n";
      }

      if( (!firstResponseObtained) && calculateRelativeResponses ){
    	  for(int j = 0; j < relative_responses.size(); j++){
    		  unsigned int resp_index = relative_responses[j];
    		  if( (resp_index < outArgs.Ng()) )
    			  is_relative[resp_index] = true;
    	  }

      }
      //Save first responses for relative changes in st
      if( (!firstResponseObtained) && calculateRelativeResponses ){
      	  int gsize = g->space()->dim();
      	  storedResponses[i].resize(gsize);
      	  for (int j = 0; j < gsize; j++)
      		  storedResponses[i][j] = Thyra::get_ele(*g,j);
      }//end if !firstRessponseObtained
    }//end of loop over outArgs.Ng()
    firstResponseObtained = true;
  }
}

