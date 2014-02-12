//*****************************************************************//
//    Albany 2.0:  Copyright 2013 Kitware Inc.                     //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#ifndef ALBANY_CATALYST_ADAPTER
#define ALBANY_CATALYST_ADAPTER

#include <string>
#include "Teuchos_ParameterList.hpp"

class Epetra_Vector;
class vtkCPPipeline;

namespace Albany {
namespace Catalyst {
class Decorator;

class Adapter
{
public:
  //! Singleton management: @{
  static Adapter * initialize(const Teuchos::RCP<Teuchos::ParameterList> &catalystParams);
  static Adapter * get();
  static void cleanup();
  //! @}

  //! Add a python script that specifies a coprocessing pipeline.
  bool addPythonScriptPipeline(const std::string &filename);

  //! Add a vtkCPPipeline coprocessing pipeline.
  bool addPipeline(vtkCPPipeline *pipeline);

  //! Update catalyst
  void update(int timeStep, double time, Decorator &decorator,
              const Epetra_Vector &soln);

  //! Validate parameter list
  static Teuchos::RCP<const Teuchos::ParameterList> getValidAdapterParameters();

private:
  static Adapter *instance;
  Adapter();
  ~Adapter();

  Adapter(const Adapter &); // Not implemented.
  void operator=(const Adapter &);   // Not implemented.

  class Private;
  Private * const d;
};

} // end namespace Catalyst
} // end namespace Albany

#endif // ALBANY_CATALYST_ADAPTER
