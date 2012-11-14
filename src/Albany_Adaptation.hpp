//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//
#ifndef ALBANY_ADAPTATION_HPP
#define ALBANY_ADAPTATION_HPP

#include "Epetra_Vector.h"
#include "Teuchos_RCP.hpp"
#include "Teuchos_ParameterList.hpp"


namespace Albany {

class AbstractDiscretization;
class STKDiscretization;

class Adaptation {
public:

   Adaptation(const Teuchos::RCP<Teuchos::ParameterList>& appParams);

   void writeSolution(double stamp, const Epetra_Vector &solution);

//   explicit ExodusOutput(const Teuchos::RCP<AbstractDiscretization> &disc);

private:
   Teuchos::RCP<STKDiscretization> stkDisc_;

//   Teuchos::RCP<Teuchos::Time> exoOutTime_;

   // Disallow copy and assignment
   Adaptation(const Adaptation &);
   Adaptation &operator=(const Adaptation &);
};

}

#endif //ALBANY_ADAPTATION_HPP
