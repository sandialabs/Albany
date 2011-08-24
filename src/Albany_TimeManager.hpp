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


#ifndef ALBANY_TIMEMANAGER
#define ALBANY_TIMEMANAGER

#include "PHAL_AlbanyTraits.hpp"
#include "Sacado_ParameterAccessor.hpp"

namespace Albany {

  //! Class to manage time for NOX/LOCA/RYTHMOS/PIRO solvers

  class TimeManager :
    public Sacado::ParameterAccessor<PHAL::AlbanyTraits::Residual, SPL_Traits> 
  {
  public:

    TimeManager() {};
  
    ~TimeManager() {};
  
    //! Provide access to time parameter
    PHAL::AlbanyTraits::Residual::ScalarT& getValue(const std::string &n);

    //! initialize
    void init(const Teuchos::RCP<ParamLib>& paramLib);
    
    //! Set Previous_Time to Current_Time on successful step
    void updateTime();

    //! Set Time, either from Rythmos or from parameter
    void setTime(double current_time);

    //! Accessors for CurrentTime, DeltaTime
    double getCurrentTime() {return CurrentTime;};
    double getDeltaTime() {return CurrentTime - PreviousTime;};

  protected:

    //! Parameter library
    Teuchos::RCP<ParamLib> paramLib;
  
  private:

    //! Private to prohibit copying
    TimeManager(const TimeManager&);

    //! Private to prohibit copying
    TimeManager& operator=(const TimeManager&);

    PHAL::AlbanyTraits::Residual::ScalarT CurrentTime, PreviousTime;
  };

}
#endif
