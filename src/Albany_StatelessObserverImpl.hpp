//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#ifndef ALBANY_STATELESS_OBSERVER_IMPL_HPP
#define ALBANY_STATELESS_OBSERVER_IMPL_HPP

#include "Albany_Application.hpp"
#include "Albany_DataTypes.hpp"

#include "Teuchos_RCP.hpp"
#include "Teuchos_Ptr.hpp"

#include "Teuchos_Time.hpp"

namespace Albany {

/*! \brief Implementation to observe the solution without updating state
 *         information.
 */

class StatelessObserverImpl {
public:
  explicit StatelessObserverImpl(const Teuchos::RCP<Application> &app);

  RealType getTimeParamValueOrDefault(RealType defaultValue) const;

  void observeSolution (double stamp,
                        const Teuchos::RCP<const Thyra_Vector>& x,
                        const Teuchos::RCP<const Thyra_Vector>& x_dot,
                        const Teuchos::RCP<const Thyra_Vector>& x_dotdot,
                        const Teuchos::RCP<const Thyra_MultiVector>& x_dxdp);

protected:
  Teuchos::RCP<Application> app_;
  Teuchos::RCP<Teuchos::Time> solOutTime_;
  bool force_write_solution_; 
};

} // namespace Albany

#endif // ALBANY_STATELESS_OBSERVER_IMPL_HPP
