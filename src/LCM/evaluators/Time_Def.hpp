/********************************************************************\
*            Albany, Copyright (2010) Sandia Corporation             *
*                                                                    *
* Notice: This computer software was prepared by Sandia Corporation, *
* hereinafter the Contractor, under Contract DE-AC04-94AL85000 with  *
* the Department of Energy (DOE). All rights in the computer software*
* are reserved by DOE on behalf of the United States Government and  *
* the Contractor as provided in the Contract. You are authorized to  *
* use this computer software for Governmental purposes but it is not *
* to be released or distributed to the public. NEITHER THE GOVERNMENT*
* NOR THE CONTRACTOR MAKES ANY WARRANTY, EXPRESS OR IMPLIED, OR      *
* ASSUMES ANY LIABILITY FOR THE USE OF THIS SOFTWARE. This notice    *
* including this sentence must appear on any copies of this software.*
*    Questions to Andy Salinger, agsalin@sandia.gov                  *
\********************************************************************/


#include <fstream>
#include "Teuchos_TestForException.hpp"
#include "Phalanx_DataLayout.hpp"
#include "Sacado_ParameterRegistration.hpp"
#include "Albany_Utils.hpp"

namespace LCM {

template<typename EvalT, typename Traits>
Time<EvalT, Traits>::
Time(Teuchos::ParameterList& p) :
  time      (p.get<std::string>("Time Name"),
	     p.get<Teuchos::RCP<PHX::DataLayout> >("Workset Scalar Data Layout")),
  deltaTime (p.get<std::string>("Delta Time Name"),
	     p.get<Teuchos::RCP<PHX::DataLayout> >("Workset Scalar Data Layout")),
  timeValue(0.0)
{
  if (p.isType<bool>("Disable Transient"))
    enableTransient = !p.get<bool>("Disable Transient");
  else enableTransient = true;

  // Add Time as a Sacado-ized parameter
  Teuchos::RCP<ParamLib> paramLib = 
    p.get< Teuchos::RCP<ParamLib> >("Parameter Library", Teuchos::null);
  new Sacado::ParameterRegistration<EvalT, SPL_Traits>("Time", this, paramLib);

  this->addEvaluatedField(time);
  this->addEvaluatedField(deltaTime);

  timeName = p.get<std::string>("Time Name")+"_old";;
  this->setName("Time"+PHX::TypeString<EvalT>::value);
}

// **********************************************************************
template<typename EvalT, typename Traits>
void Time<EvalT, Traits>::
postRegistrationSetup(typename Traits::SetupData d,
                      PHX::FieldManager<Traits>& fm)
{
  this->utils.setFieldData(time,fm);
  this->utils.setFieldData(deltaTime,fm);
}

// **********************************************************************
template<typename EvalT, typename Traits>
void Time<EvalT, Traits>::
evaluateFields(typename Traits::EvalData workset)
{
  time(0) = workset.current_time;

  Albany::MDArray timeOld = (*workset.stateArrayPtr)[timeName];
  deltaTime(0) = time(0) - timeOld(0);

}

// **********************************************************************
template<typename EvalT,typename Traits>
typename Time<EvalT,Traits>::ScalarT& 
Time<EvalT,Traits>::getValue(const std::string &n)
{
  return timeValue;
}

// **********************************************************************
// **********************************************************************
}

