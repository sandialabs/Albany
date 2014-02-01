//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#if !defined(AAdapt_StratSolver_hpp)
#define AAdapt_StratSolver_hpp

#include <Teuchos_RCP.hpp>
#include <Teuchos_ParameterList.hpp>

#include <Phalanx.hpp>
#include <PHAL_Workset.hpp>
#include <PHAL_Dimension.hpp>

#include "AAdapt_AbstractAdapter.hpp"
#include "Albany_STKDiscretization.hpp"

namespace AAdapt {

///
/// This class implements a general parallel linear solver using the Stratimikos interface
///
class StratSolver {
  public:

    ///
    /// Constructor
    ///
    StratSolver(const Teuchos::RCP<Teuchos::ParameterList>& params);

    ///
    /// Destructor
    ///
    ~StratSolver();

    ///
    /// Valid parameters for the solver
    ///
    Teuchos::RCP<const Teuchos::ParameterList>
    getValidAdapterParameters() const;

  private:

    ///
    /// Prohibit default constructor
    ///
    StratSolver();

    ///
    /// Disallow copy and assignment
    ///
    StratSolver(const StratSolver&);
    StratSolver& operator=(const StratSolver&);

};

}

#endif //StratSolver_hpp
