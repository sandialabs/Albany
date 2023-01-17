//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#ifndef PETRA_CONVERTERS_H
#define PETRA_CONVERTERS_H

#include "Teuchos_RCP.hpp"

#include "Albany_TpetraTypes.hpp"
#include "Albany_CommTypes.hpp"

#include "Epetra_Comm.h"
#include "Epetra_Vector.h"
#include "Epetra_CrsGraph.h"
#include "Epetra_CrsMatrix.h"
#include "Epetra_Export.h"
#include "Epetra_Import.h"
#include "Epetra_LocalMap.h"


namespace Petra {
//TpetraMap_To_EpetraMap: takes in Tpetra::Map object, converts it to its equivalent Epetra_Map object,
//and returns an RCP pointer to this Epetra_Map
Teuchos::RCP<const Epetra_Map> TpetraMap_To_EpetraMap(const Teuchos::RCP<const Tpetra_Map>& tpetraMap_,
                                                      const Teuchos::RCP<const Epetra_Comm>& comm_);

//EpetraMap_To_TpetraMap: takes in Epetra_Map object, converts it to its equivalent Tpetra::Map object,
//and returns an RCP pointer to this Tpetra::Map
Teuchos::RCP<const Tpetra_Map> EpetraMap_To_TpetraMap(const Teuchos::RCP<const Epetra_Map>& epetraMap_,
                                                      const Teuchos::RCP<const Teuchos::Comm<int> >& comm_);

//EpetraMap_To_TpetraMap: takes in Epetra_Map object, converts it to its equivalent Tpetra::Map object,
//and returns an RCP pointer to this Tpetra::Map
Teuchos::RCP<const Tpetra_Map> EpetraMap_To_TpetraMap(const Epetra_Map& epetraMap_,
                                                      const Teuchos::RCP<const Teuchos::Comm<int> >& comm_);

//EpetraMap_To_TpetraMap: takes in Epetra_Map object, converts it to its equivalent Tpetra::Map object,
//and returns an RCP pointer to this Tpetra::Map
Teuchos::RCP<const Tpetra_Map> EpetraMap_To_TpetraMap(const Epetra_BlockMap& epetraMap_,
                                                      const Teuchos::RCP<const Teuchos::Comm<int> >& comm_);

Teuchos::RCP<Epetra_CrsGraph> TpetraCrsGraph_To_EpetraCrsGraph(const Teuchos::RCP<const Tpetra_CrsGraph>& tpetraCrsGraph_,
                                                                const Teuchos::RCP<const Epetra_Comm>& comm);

//TpetraCrsMatrix_To_CrsMatrix: copies Tpetra::CrsMatrix object into its analogous
//Epetra_CrsMatrix object
void TpetraCrsMatrix_To_EpetraCrsMatrix(const Teuchos::RCP<const Tpetra_CrsMatrix>& tpetraCrsMatrix_,
                                        Epetra_CrsMatrix& epetraCrsMatrix_,
                                        const Teuchos::RCP<const Epetra_Comm>& comm_);

//TpetraCrsMatrix_To_CrsMatrix: takes in Tpetra::CrsMatrix object, converts it to its equivalent Epetra_CrsMatrix object,
//and returns an RCP pointer to this Eetra_CrsMatrix
Teuchos::RCP<Epetra_CrsMatrix> TpetraCrsMatrix_To_EpetraCrsMatrix(const Teuchos::RCP<const Tpetra_CrsMatrix>& tpetraCrsMatrix,
                                                                  const Teuchos::RCP<const Epetra_Comm>& comm);

//TpetraVector_To_EpetraVector: copies Tpetra::Vector object into its analogous
//Epetra_Vector object
void TpetraVector_To_EpetraVector(const Teuchos::RCP<const Tpetra_Vector>& tpetraVector_,
                                  Epetra_Vector& epetraVector_, const Teuchos::RCP<const Epetra_Comm>& comm_);

void TpetraVector_To_EpetraVector(const Teuchos::RCP<const Tpetra_Vector>& tpetraVector_,
                                  Teuchos::RCP<Epetra_Vector>& epetraVector_,
                                  const Teuchos::RCP<const Epetra_Comm>& comm_);

//EpetraVector_To_TpetraVectorConst: copies const Epetra_Vector to const Tpetra_Vector
Teuchos::RCP<const Tpetra_Vector> EpetraVector_To_TpetraVectorConst(const Epetra_Vector& epetraVector_,
                                                               const Teuchos::RCP<const Teuchos::Comm<int> >& commT_);

//EpetraVector_To_TpetraVectorNonConst: copies non-const Epetra_Vector to non-const Tpetra_Vector
Teuchos::RCP<Tpetra_Vector> EpetraVector_To_TpetraVectorNonConst(const Epetra_Vector& epetraVector_,
                                                               const Teuchos::RCP<const Teuchos::Comm<int> >& commT_);

//TpetraMultiVector_To_EpetraMultiVector: copies Tpetra::MultiVector object into its analogous
//Epetra_MultiVector object
void TpetraMultiVector_To_EpetraMultiVector(const Teuchos::RCP<const Tpetra_MultiVector>& tpetraMV_,
                                  Epetra_MultiVector& epetraMV_, const Teuchos::RCP<const Epetra_Comm>& comm_);

//EpetraMultiVector_To_TpetraMultiVectorConst: copies Epetra_MultiVector to const Tpetra_MultiVector
Teuchos::RCP<Tpetra_MultiVector> EpetraMultiVector_To_TpetraMultiVector(const Epetra_MultiVector& epetraMV_,
                                                               const Teuchos::RCP<const Teuchos::Comm<int> >& commT_);

//EpetraCrsMatrix_To_TpetraCrsMatrix: copies Epetra_CrsMatrix to its analogous Tpetra_CrsMatrix
Teuchos::RCP<Tpetra_CrsMatrix> EpetraCrsMatrix_To_TpetraCrsMatrix(const Epetra_CrsMatrix& epetraMatrix_,
                                                               const Teuchos::RCP<const Teuchos::Comm<int> >& commT_);


// Convenience class for conversions. One use case is to inherit from this class
// and implement situation-specific conversion functionality using concise
// wrapper methods.
class Converter {
protected:
public:
  Converter(const Teuchos::RCP<const Teuchos_Comm>& commT);
  virtual ~Converter () {}

  Teuchos::RCP<const Tpetra_Vector> e2t (const Epetra_Vector* ev) {
    return ev ?
      EpetraVector_To_TpetraVectorConst(*ev, commT_)
      : Teuchos::null;
  }
  Teuchos::RCP<Tpetra_Vector> e2t (Epetra_Vector* ev) {
    return ev ?
      EpetraVector_To_TpetraVectorNonConst(*ev, commT_)
      : Teuchos::null;
  }
  Teuchos::RCP<const Tpetra_Vector> e2t (const Epetra_Vector& ev) {
    return EpetraVector_To_TpetraVectorConst(ev, commT_);
  }
  Teuchos::RCP<Tpetra_Vector> e2t (Epetra_Vector& ev) {
    return EpetraVector_To_TpetraVectorNonConst(ev, commT_);
  }
  Teuchos::RCP<Tpetra_MultiVector> e2t (const Epetra_MultiVector* ev) {
    return ev ?
      EpetraMultiVector_To_TpetraMultiVector(*ev, commT_)
      : Teuchos::null;
  }
  Teuchos::RCP<Tpetra_MultiVector> e2t (const Epetra_MultiVector& ev) {
    return EpetraMultiVector_To_TpetraMultiVector(ev, commT_);
  }

  void t2e (const Teuchos::RCP<const Tpetra_Vector>& tv, Epetra_Vector& ev) {
    TpetraVector_To_EpetraVector(tv, ev, commE_);
  }
  void t2e (const Teuchos::RCP<const Tpetra_Vector>& tv, Epetra_Vector* ev) {
    if (ev) TpetraVector_To_EpetraVector(tv, *ev, commE_);
  }
  void t2e (const Teuchos::RCP<const Tpetra_MultiVector>& tv,
            Epetra_MultiVector& ev) {
    TpetraMultiVector_To_EpetraMultiVector(tv, ev, commE_);
  }
  void t2e (const Teuchos::RCP<const Tpetra_MultiVector>& tv,
            Epetra_MultiVector* ev) {
    if (ev) TpetraMultiVector_To_EpetraMultiVector(tv, *ev, commE_);
  }

  Teuchos::RCP<const Teuchos_Comm> commT_;
  Teuchos::RCP<const Epetra_Comm> commE_;
};

} // namespace Petra

#endif //PETRA_CONVERTERS
