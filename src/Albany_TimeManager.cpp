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
#include "Albany_TimeManager.hpp"
#include "Sacado_ParameterRegistration.hpp"

void Albany::TimeManager::init(const Teuchos::RCP<ParamLib>& paramLib_)
{
  paramLib = paramLib_;

  CurrentTime = 0.0;
  PreviousTime = 0.0;

  std::string name = "Time";
  new Sacado::ParameterRegistration<PHAL::AlbanyTraits::Residual, SPL_Traits>(name, this, paramLib);
}

PHAL::AlbanyTraits::Residual::ScalarT& Albany::TimeManager::getValue(const std::string &n)
{
  return CurrentTime;
}

void Albany::TimeManager::updateTime()
{
  PreviousTime = CurrentTime;
}

void Albany::TimeManager::setTime(double current_time)
{
  CurrentTime = current_time;
}
