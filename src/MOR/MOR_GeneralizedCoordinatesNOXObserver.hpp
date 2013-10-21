//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//
#ifndef MOR_GENERALIZEDCOORDINATESNOXOBSERVER_HPP
#define MOR_GENERALIZEDCOORDINATESNOXOBSERVER_HPP

#include "MOR_GeneralizedCoordinatesOutput.hpp"

#include "NOX_Epetra_Observer.H"

namespace MOR {

class GeneralizedCoordinatesNOXObserver : public NOX::Epetra::Observer {
public:
  explicit GeneralizedCoordinatesNOXObserver(const std::string filename);

  virtual void observeSolution(const Epetra_Vector& solution);
  virtual void observeSolution(const Epetra_Vector& solution, double time_or_param_val);

private:
  GeneralizedCoordinatesOutput impl_;
};

} // end namespace MOR

#endif /* MOR_GENERALIZEDCOORDINATESNOXOBSERVER_HPP */
