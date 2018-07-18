//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

//IK, 9/12/14: no Epetra!

#ifndef ALBANY_ABSTRACTNODEFIELDCONT_HPP
#define ALBANY_ABSTRACTNODEFIELDCONT_HPP

#include "Albany_TpetraTypes.hpp"
#include "Teuchos_RCP.hpp"
#include <map>

namespace Albany {

/*!
 * \brief Abstract interface for an STK NodeField container
 *
 */

class AbstractNodeFieldContainer {

  public:

    AbstractNodeFieldContainer(){}
    virtual ~AbstractNodeFieldContainer(){}

    // MV version
    virtual void saveFieldVector(const Teuchos::RCP<const Tpetra_MultiVector>& mv, int offset) = 0;

};

typedef std::map<std::string, Teuchos::RCP<Albany::AbstractNodeFieldContainer> > NodeFieldContainer;

} // namespace Albany

#endif // ALBANY_ABSTRACTNODEFIELDCONT_HPP
