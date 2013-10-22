//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//
#ifndef MOR_GENERALIZEDCOORDINATESOUTPUT_HPP
#define MOR_GENERALIZEDCOORDINATESOUTPUT_HPP

#include "Epetra_Vector.h"

#include "Teuchos_RCP.hpp"

#include <string>
#include <deque>

namespace MOR {

class GeneralizedCoordinatesOutput {
public:
  explicit GeneralizedCoordinatesOutput(const std::string &filename);

  int vectorCount() const;
  void vectorAdd(const Epetra_Vector &);

  ~GeneralizedCoordinatesOutput(); // Non-trivial destructor

private:
  std::string filename_;

  std::deque<Epetra_Vector> projectionComponents_;

  // Disallow copy & assignment
  GeneralizedCoordinatesOutput(const GeneralizedCoordinatesOutput &);
  GeneralizedCoordinatesOutput &operator=(const GeneralizedCoordinatesOutput &);
};

} // end namespace MOR

#endif /* MOR_GENERALIZEDCOORDINATESOUTPUT_HPP */
