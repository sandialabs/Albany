//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//
#ifndef MOR_GENERALIZEDCOORDINATESOUTPUT_HPP
#define MOR_GENERALIZEDCOORDINATESOUTPUT_HPP

#include "Epetra_Vector.h"

#include "Teuchos_RCP.hpp"
#include "Teuchos_Array.hpp"

#include <string>
#include <deque>

namespace MOR {

class GeneralizedCoordinatesOutput {
public:
  GeneralizedCoordinatesOutput(const std::string &filename, const std::string &stampsFilename);

  int vectorCount() const;
  void vectorAdd(const Epetra_Vector &v);
  void stampedVectorAdd(double stamp, const Epetra_Vector &v);

  ~GeneralizedCoordinatesOutput(); // Non-trivial destructor

private:
  std::string filename_;
  std::string stampsFilename_;

  std::deque<Epetra_Vector> projectionComponents_;
  Teuchos::Array<double> stamps_;

  // Disallow copy & assignment
  GeneralizedCoordinatesOutput(const GeneralizedCoordinatesOutput &);
  GeneralizedCoordinatesOutput &operator=(const GeneralizedCoordinatesOutput &);
};

} // end namespace MOR

#endif /* MOR_GENERALIZEDCOORDINATESOUTPUT_HPP */
