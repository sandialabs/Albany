//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#ifndef ALBANY_PYUTILS_H
#define ALBANY_PYUTILS_H

// Get Albany configuration macros
#include "Albany_config.h"

#include <sstream>

#include "Albany_CommUtils.hpp"
#include "Albany_Macros.hpp"
#include "Albany_StateManager.hpp"
#include "Albany_ThyraTypes.hpp"
#include "Albany_TpetraTypes.hpp"
#include "Teuchos_RCP.hpp"

namespace PyAlbany
{

  //! Print ascii art and version information for PyAlbany
  void
  PrintPyHeader(std::ostream &os);

  /**
   * \brief correctIDs function
   * 
   * This function is used to recreate a contiguous map where the order of
   * the global IDs is kept.
   * 
   * \param comm [in] Communicator.
   * 
   * \param myVals [in/out] Array of global IDs which have to be reindexed to be contiguous.
   * 
   * \param indexbase [in] The used indexbase.
   */
  void
  correctIDs(const Teuchos::RCP<const Teuchos_Comm> &comm,
             const Teuchos::ArrayView<Tpetra_GO> &myVals,
             int indexbase);

  /**
   * \brief getCorrectedMap function
   * 
   * This function is used to communicate the Tpetra map used in Albany to Python.
   * 
   * \param t_map [in] Tpetra map which has to be communicated to Python.
   * 
   * \param correctGIDs [in] Boolean used to specify if a correction on the GID is performed.
   * 
   * The function returns an RCP to a map.
   */
  Teuchos::RCP<const Tpetra_Map> getCorrectedMap(Teuchos::RCP<const Tpetra_Map> t_map, bool correctGIDs);
} // namespace PyAlbany

#endif // ALBANY_PYUTILS_H
