//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#ifndef ALBANY_UNIFSIZEFIELD_HPP
#define ALBANY_UNIFSIZEFIELD_HPP

#include "Albany_FMDBDiscretization.hpp"
#include "Epetra_Vector.h"
#include "AdaptTypes.h"
#include "MeshAdapt.h"

namespace Albany {

class UnifSizeField {

public:
  UnifSizeField(Albany::FMDBDiscretization *disc);
  ~UnifSizeField();

  int computeSizeField(pPart part, pSField field);

  void setParams(const Epetra_Vector *sol, const Epetra_Vector *ovlp_sol, double element_size );
  void setError();


private:

  Albany::FMDBDiscretization *disc;
  const Epetra_Vector* solution;
  const Epetra_Vector* ovlp_solution;

  double elem_size;

};

}

#endif

