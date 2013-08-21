//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//
#ifndef ALBANY_EXODUSOUTPUT_HPP
#define ALBANY_EXODUSOUTPUT_HPP

#include "Albany_DataTypes.hpp"

#include "Epetra_Vector.h"

#include "Teuchos_RCP.hpp"

namespace Teuchos {
  class Time;
}

namespace Albany {

class AbstractDiscretization;
class STKDiscretization;

class ExodusOutput {
public:
   void writeSolution(double stamp, const Epetra_Vector &solution, const bool overlapped = false);

   void writeSolutionT(double stamp, const Tpetra_Vector &solutionT, const bool overlapped = false);

   explicit ExodusOutput(const Teuchos::RCP<AbstractDiscretization> &disc);

private:
   Teuchos::RCP<STKDiscretization> stkDisc_;

   Teuchos::RCP<Teuchos::Time> exoOutTime_;

   // Disallow copy and assignment
   ExodusOutput(const ExodusOutput &);
   ExodusOutput &operator=(const ExodusOutput &);
};

}

#endif //ALBANY_EXODUSOUTPUT_HPP
