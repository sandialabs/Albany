//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#include <fstream>
#include "Teuchos_TestForException.hpp"
#include "Phalanx_DataLayout.hpp"
#include "Sacado_ParameterRegistration.hpp"
#include "Albany_Utils.hpp"

namespace ANISO {

template <typename EvalT, typename Traits>
Time<EvalT, Traits>::Time(
    const Teuchos::ParameterList& p,
    const Teuchos::RCP<Albany::Layouts>& dl) :
  time (p.get<std::string>("Time Name"), dl->workset_scalar),
  deltaTime (p.get<std::string>("Delta Time Name"), dl->workset_scalar),
  timeValue(0.0) {

    // add time as sacado-ized parameter
    Teuchos::RCP<ParamLib> paramLib =
      p.get<Teuchos::RCP<ParamLib> >("Parameter Library");
    this->registerSacadoParameter("Time", paramLib);

    this->addEvaluatedField(time);
    this->addEvaluatedField(deltaTime);

    timeName = p.get<std::string>("Time Name")+"_old";;
    this->setName("Time"+PHX::typeAsString<EvalT>());
}

template <typename EvalT, typename Traits>
void Time<EvalT, Traits>::postRegistrationSetup(
    typename Traits::SetupData d,
    PHX::FieldManager<Traits>& fm) {
  this->utils.setFieldData(time, fm);
  this->utils.setFieldData(deltaTime, fm);
}

template <typename EvalT, typename Traits>
void Time<EvalT, Traits>::evaluateFields(
    typename Traits::EvalData workset) {
  time(0) = workset.current_time;
  Albany::MDArray timeOld = (*workset.stateArrayPtr)[timeName];
  deltaTime(0) = time(0) - timeOld(0);
}

template <typename EvalT, typename Traits>
typename Time<EvalT, Traits>::ScalarT&
Time<EvalT, Traits>::getValue(const std::string& n) {
  return timeValue;
}

}
