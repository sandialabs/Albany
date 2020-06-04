//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#include "Albany_PiroObserver.hpp"
#include "PHAL_AlbanyTraits.hpp"
#include "Teuchos_ScalarTraits.hpp"
#include "Thyra_VectorStdOps.hpp"

#include <cstddef>


namespace Albany
{

PiroObserver::
PiroObserver(const Teuchos::RCP<Application> &app, 
              Teuchos::RCP<const Thyra_ModelEvaluator> model)
 : impl_(app) 
 , model_(model) 
 , out(Teuchos::VerboseObjectBase::getDefaultOStream())
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

void PiroObserver::
observeSolution(const Thyra_Vector &solution)
{
  this->observeSolutionImpl(solution, Teuchos::ScalarTraits<ST>::zero());
  stepper_counter_++;
}

void PiroObserver::
observeSolution(const Thyra_Vector &solution,
                const Thyra_MultiVector &solution_dxdp)
{
  this->observeSolutionImpl(solution, solution_dxdp, Teuchos::ScalarTraits<ST>::zero());
  stepper_counter_++;
}

void PiroObserver::
observeSolution(const Thyra_Vector &solution,
                const ST stamp)
{
  this->observeSolutionImpl(solution, stamp);
  stepper_counter_++; 
}

void PiroObserver::
observeSolution(const Thyra_Vector &solution,
                const Thyra_MultiVector &solution_dxdp, 
                const ST stamp)
{
  this->observeSolutionImpl(solution, solution_dxdp, stamp);
  stepper_counter_++; 
}

void PiroObserver::
observeSolution(const Thyra_Vector &solution,
                const Thyra_Vector &solution_dot,
                const ST stamp)
{
  this->observeSolutionImpl(solution, solution_dot, stamp);
  stepper_counter_++; 
}

void PiroObserver::
observeSolution(const Thyra_Vector &solution,
                const Thyra_MultiVector &solution_dxdp,
                const Thyra_Vector &solution_dot,
                const ST stamp)
{
  this->observeSolutionImpl(solution, solution_dxdp, solution_dot, stamp);
  stepper_counter_++; 
}

void PiroObserver::
observeSolution(const Thyra_Vector &solution,
                const Thyra_Vector &solution_dot,
                const Thyra_Vector &solution_dotdot,
                const ST stamp)
{
  this->observeSolutionImpl(solution, solution_dot, solution_dotdot, stamp);
  stepper_counter_++; 
}

void PiroObserver::
observeSolution(const Thyra_Vector &solution,
                const Thyra_MultiVector &solution_dxdp,
                const Thyra_Vector &solution_dot,
                const Thyra_Vector &solution_dotdot,
                const ST stamp)
{
  this->observeSolutionImpl(solution, solution_dxdp, solution_dot, solution_dotdot, stamp);
  stepper_counter_++; 
}

void PiroObserver::
observeSolution(const Thyra_MultiVector &solution,
                const ST stamp)
{
  this->observeSolutionImpl(solution, stamp);
  stepper_counter_++; 
}

void PiroObserver::
observeSolution(const Thyra_MultiVector &solution,
                const Thyra_MultiVector &solution_dxdp, const ST stamp)
{
  this->observeSolutionImpl(solution, solution_dxdp, stamp);
  stepper_counter_++; 
}

void PiroObserver::
parameterChanged(const std::string& param)
{
  impl_.parameterChanged(param);
}

void PiroObserver::
observeSolutionImpl(const Thyra_Vector &solution,
                    const ST defaultStamp)
{
  // Determine the stamp associated with the snapshot
  const ST stamp = impl_.getTimeParamValueOrDefault(defaultStamp);
  impl_.observeSolution(stamp, solution, Teuchos::null, 
                        Teuchos::null, Teuchos::null);
  
  // observe responses 
  if (observe_responses_ == true) {
    if (stepper_counter_ % observe_responses_every_n_steps_ == 0) 
      this->observeResponse(defaultStamp, Teuchos::rcpFromRef(solution));
   }
}

void PiroObserver::
observeSolutionImpl(const Thyra_Vector &solution, 
                    const Thyra_MultiVector &solution_dxdp, 
                    const ST defaultStamp)
{
  // Determine the stamp associated with the snapshot
  const ST stamp = impl_.getTimeParamValueOrDefault(defaultStamp);
  impl_.observeSolution(stamp, solution, Teuchos::constPtr(solution_dxdp), Teuchos::null, Teuchos::null);
  
  // observe responses 
  if (observe_responses_ == true) {
    if (stepper_counter_ % observe_responses_every_n_steps_ == 0) 
      this->observeResponse(defaultStamp, Teuchos::rcpFromRef(solution));
   }
}

void PiroObserver::
observeSolutionImpl(const Thyra_Vector &solution,
                    const Thyra_Vector &solution_dot,
                    const ST defaultStamp)
{
  // Determine the stamp associated with the snapshot
  const ST stamp = impl_.getTimeParamValueOrDefault(defaultStamp);
  impl_.observeSolution(stamp, solution, Teuchos::null, 
                        Teuchos::constPtr(solution_dot), Teuchos::null);

  // observe responses 
  if (observe_responses_ == true) {
    if (stepper_counter_ % observe_responses_every_n_steps_ == 0) 
      this->observeResponse(defaultStamp,
                            Teuchos::rcpFromRef(solution),
                            Teuchos::rcpFromRef(solution_dot));
   }
}

void PiroObserver::
observeSolutionImpl(const Thyra_Vector &solution,
                    const Thyra_MultiVector &solution_dxdp, 
                    const Thyra_Vector &solution_dot,
                    const ST defaultStamp)
{
  // Determine the stamp associated with the snapshot
  const ST stamp = impl_.getTimeParamValueOrDefault(defaultStamp);
  impl_.observeSolution(stamp, solution, Teuchos::constPtr(solution_dxdp), Teuchos::constPtr(solution_dot), Teuchos::null);

  // observe responses 
  if (observe_responses_ == true) {
    if (stepper_counter_ % observe_responses_every_n_steps_ == 0) 
      this->observeResponse(defaultStamp,
                            Teuchos::rcpFromRef(solution),
                            Teuchos::rcpFromRef(solution_dot));
   }
}

void PiroObserver::
observeSolutionImpl(const Thyra_Vector &solution,
                    const Thyra_Vector &solution_dot,
                    const Thyra_Vector &solution_dotdot,
                    const ST defaultStamp)
{
  // Determine the stamp associated with the snapshot
  const ST stamp = impl_.getTimeParamValueOrDefault(defaultStamp);
  impl_.observeSolution(stamp, solution, Teuchos::null, Teuchos::constPtr(solution_dot), Teuchos::constPtr(solution_dotdot));

  // observe responses 
  if (observe_responses_ == true) {
    if (stepper_counter_ % observe_responses_every_n_steps_ == 0) 
      this->observeResponse(defaultStamp,
                            Teuchos::rcpFromRef(solution),
                            Teuchos::rcpFromRef(solution_dot), 
                            Teuchos::rcpFromRef(solution_dotdot));
   }
}


void PiroObserver::
observeSolutionImpl(const Thyra_Vector &solution,
                    const Thyra_MultiVector &solution_dxdp,
                    const Thyra_Vector &solution_dot,
                    const Thyra_Vector &solution_dotdot,
                    const ST defaultStamp)
{
  // Determine the stamp associated with the snapshot
  const ST stamp = impl_.getTimeParamValueOrDefault(defaultStamp);
  impl_.observeSolution(stamp, solution, Teuchos::constPtr(solution_dxdp), 
                  Teuchos::constPtr(solution_dot), Teuchos::constPtr(solution_dotdot));

  // observe responses 
  if (observe_responses_ == true) {
    if (stepper_counter_ % observe_responses_every_n_steps_ == 0) 
      this->observeResponse(defaultStamp,
                            Teuchos::rcpFromRef(solution),
                            Teuchos::rcpFromRef(solution_dot), 
                            Teuchos::rcpFromRef(solution_dotdot));
   }
} 

void PiroObserver::
observeSolutionImpl(const Thyra_MultiVector &solution,
                    const ST defaultStamp)
{
  impl_.observeSolution(defaultStamp, solution, Teuchos::null);
}

void PiroObserver::
observeSolutionImpl(const Thyra_MultiVector &solution,
                    const Thyra_MultiVector &solution_dxdp, 
                    const ST defaultStamp)
{
  impl_.observeSolution(defaultStamp, solution, 
                        Teuchos::constPtr(solution_dxdp));
}


void PiroObserver::
observeResponse(const ST defaultStamp, 
                Teuchos::RCP<const Thyra_Vector> solution,
                Teuchos::RCP<const Thyra_Vector> solution_dot,
                Teuchos::RCP<const Thyra_Vector> /* solution_dotdot */)
{
  //IKT 5/10/17: note that this function takes solution_dotdot as an input 
  //argument but does not do anything with it yet.  This can be modified 
  //if desired.

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
          for( size_t j = 0; j < storedResponses[i].size(); j++){
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
    		  int resp_index = relative_responses[j];
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

} // namespace Albany
