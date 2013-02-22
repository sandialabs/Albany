//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#ifndef ALBANY_SIZEFIELD_HPP
#define ALBANY_SIZEFIELD_HPP

#include "AdaptTypes.h"
#include "MeshAdapt.h"
#include "AdaptUtil.h"
#include "PWLinearSField.h"

#include "Albany_FMDBDiscretization.hpp"

namespace Albany {

class SizeField : public PWLsfield {
public:
  SizeField(pMesh, Albany::FMDBDiscretization *disc, const Epetra_Vector& sol, const Epetra_Vector& ovlp_sol );
  ~SizeField();

  int computeSizeField();

private:

  Albany::FMDBDiscretization *disc;
  const Epetra_Vector& solution;
  const Epetra_Vector& ovlp_solution;

};

}

#endif

