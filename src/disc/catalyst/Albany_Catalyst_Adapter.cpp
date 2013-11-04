//*****************************************************************//
//    Albany 2.0:  Copyright 2013 Kitware Inc.                     //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#include "Albany_Catalyst_Adapter.hpp"

#include "Albany_Catalyst_EpetraDataArray.hpp"
#include "Albany_Catalyst_Grid.hpp"

#include "Teuchos_Array.hpp"
#include "Teuchos_TestForException.hpp"

#include <vtkClientServerInterpreter.h>
#include <vtkClientServerInterpreterInitializer.h>
#include <vtkCPDataDescription.h>
#include <vtkCPInputDataDescription.h>
#include <vtkCPProcessor.h>
#include <vtkCPPythonScriptPipeline.h>
#include <vtkNew.h>
#include <vtkPointData.h>
#include <vtkPVInstantiator.h>

#include <iostream>

namespace Albany {
namespace Catalyst {

Adapter * Adapter::instance = NULL;

class Adapter::Private
{
public:
  // Used by Catalyst to create a dummy grid object:
  static vtkObjectBase* MakeGrid() { return Grid::New(); }
  Private() { processor->Initialize(); }
  ~Private() { processor->Finalize(); }
  vtkNew<vtkCPProcessor> processor;
};

Adapter::Adapter()
  : d(new Private)
{
}

Adapter::~Adapter()
{
  delete d;
}

Adapter *
Adapter::initialize(const Teuchos::RCP<Teuchos::ParameterList>& catalystParams)
{
  // Validate parameters against list for this specific class
  catalystParams->validateParameters(*getValidAdapterParameters(),0);

  if (Adapter::instance)
    delete Adapter::instance;
  Adapter::instance = new Adapter();

  // Register our Grid class with Catalyst so that it can be used in a pipeline.
  if (vtkClientServerInterpreterInitializer *intInit =
      vtkClientServerInterpreterInitializer::GetInitializer()) {
    if (vtkClientServerInterpreter *interp = intInit->GetGlobalInterpreter()) {
      interp->AddNewInstanceFunction("Grid", Private::MakeGrid);
    }
  }

  // Load pipeline file
  Teuchos::Array<std::string> files =
      catalystParams->get<Teuchos::Array<std::string> >("Pipeline Files");
  typedef Teuchos::Array<std::string>::const_iterator FileIterT;
  for (FileIterT it = files.begin(), itEnd = files.end(); it != itEnd; ++it)
    Adapter::instance->addPythonScriptPipeline(*it);

  return Adapter::instance;
}

Adapter * Adapter::get()
{
  TEUCHOS_TEST_FOR_EXCEPTION(!Adapter::instance, std::runtime_error,
                             "Albany::Catalyst::Adapter::get() called before "
                             "initialize()!" << std::endl);
  return Adapter::instance;
}

void Adapter::cleanup()
{
  delete Adapter::instance;
  Adapter::instance = NULL;
}

bool Adapter::addPythonScriptPipeline(const std::string &filename)
{
  vtkNew<vtkCPPythonScriptPipeline> pipeline;
  return pipeline->Initialize(filename.c_str()) != 0 &&
      this->addPipeline(pipeline.GetPointer());
}

bool Adapter::addPipeline(vtkCPPipeline *pipeline)
{
  return d->processor->AddPipeline(pipeline) != 0;
}

void Adapter::update(int timeStep, double time, Decorator &decorator,
                     const Epetra_Vector &soln)
{
  vtkNew<vtkCPDataDescription> desc;
  desc->AddInput("input");
  desc->SetTimeData(time, timeStep);
  if (d->processor->RequestDataDescription(desc.GetPointer())) {
    typedef vtkSmartPointer<vtkUnstructuredGridBase> GridRCP;
    GridRCP grid = GridRCP::Take(decorator.newVtkUnstructuredGrid());

    vtkNew<EpetraDataArray> pointScalars;
    pointScalars->SetEpetraVector(soln);
    pointScalars->SetName("Scalars_");
    grid->GetPointData()->SetScalars(pointScalars.GetPointer());

    desc->GetInputDescriptionByName("input")->SetGrid(grid);

    d->processor->CoProcess(desc.GetPointer());
  }
}

Teuchos::RCP<const Teuchos::ParameterList>
Adapter::getValidAdapterParameters()
{
  Teuchos::RCP<Teuchos::ParameterList> validPL =
      rcp(new Teuchos::ParameterList("ValidCatalystAdapterParams"));

  validPL->set<bool>("Interface Activated", false,
                     "Activates Catalyst if set to true");
  validPL->set<Teuchos::Array<std::string> >(
        "Pipeline Files", Teuchos::Array<std::string>(),
        "Filenames that contains Catalyst pipeline commands.");

  return validPL;
}

} // namespace Catalyst
} // namespace Albany
