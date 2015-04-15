//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#include <iostream>
#include <string>


#include "Albany_Utils.hpp"
#include "Albany_SolverFactory.hpp"
#include "Albany_Memory.hpp"

#include "Piro_PerformSolve.hpp"
#include "Teuchos_ParameterList.hpp"

#include "Teuchos_GlobalMPISession.hpp"
#include "Teuchos_TimeMonitor.hpp"
#include "Teuchos_VerboseObject.hpp"
#include "Teuchos_StandardCatchMacros.hpp"
#include "Epetra_Map.h"  //Needed for serial, somehow

//This header is for debug output -- writing of solution (xfinal) to MatrixMarket file
#include "EpetraExt_MultiVectorOut.h"
#include "EpetraExt_BlockMapOut.h"

#include "Phalanx_config.hpp"
#include "Phalanx.hpp"

#include "Kokkos_Core.hpp"

#ifdef ALBANY_PERIDIGM
#ifdef ALBANY_EPETRA
#include "PeridigmManager.hpp"
#endif
#endif

// Uncomment for run time nan checking
// This is set in the toplevel CMakeLists.txt file
//
//#define ALBANY_CHECK_FPE

#ifdef ALBANY_CHECK_FPE
#include <math.h>
//#include <Accelerate/Accelerate.h>
#include <xmmintrin.h>
#endif

// Global variable that denotes this is not the Tpetra executable
bool TpetraBuild = false;

#include "Thyra_EpetraThyraWrappers.hpp"

Teuchos::RCP<const Epetra_Vector>
epetraVectorFromThyra(
  const Teuchos::RCP<const Epetra_Comm> &comm,
  const Teuchos::RCP<const Thyra::VectorBase<double> > &thyra)
{
  Teuchos::RCP<const Epetra_Vector> result;
  if (Teuchos::nonnull(thyra)) {
    const Teuchos::RCP<const Epetra_Map> epetra_map = Thyra::get_Epetra_Map(*thyra->space(), comm);
    result = Thyra::get_Epetra_Vector(*epetra_map, thyra);
  }
  return result;
}

Teuchos::RCP<const Epetra_MultiVector>
epetraMultiVectorFromThyra(
  const Teuchos::RCP<const Epetra_Comm> &comm,
  const Teuchos::RCP<const Thyra::MultiVectorBase<double> > &thyra)
{
  Teuchos::RCP<const Epetra_MultiVector> result;
  if (Teuchos::nonnull(thyra)) {
    const Teuchos::RCP<const Epetra_Map> epetra_map = Thyra::get_Epetra_Map(*thyra->range(), comm);
    result = Thyra::get_Epetra_MultiVector(*epetra_map, thyra);
  }
  return result;
}

void epetraFromThyra(
  const Teuchos::RCP<const Epetra_Comm> &comm,
  const Teuchos::Array<Teuchos::RCP<const Thyra::VectorBase<double> > > &thyraResponses,
  const Teuchos::Array<Teuchos::Array<Teuchos::RCP<const Thyra::MultiVectorBase<double> > > > &thyraSensitivities,
  Teuchos::Array<Teuchos::RCP<const Epetra_Vector> > &responses,
  Teuchos::Array<Teuchos::Array<Teuchos::RCP<const Epetra_MultiVector> > > &sensitivities)
{
  responses.clear();
  responses.reserve(thyraResponses.size());
  typedef Teuchos::Array<Teuchos::RCP<const Thyra::VectorBase<double> > > ThyraResponseArray;
  for (ThyraResponseArray::const_iterator it_begin = thyraResponses.begin(),
      it_end = thyraResponses.end(),
      it = it_begin;
      it != it_end;
      ++it) {
    responses.push_back(epetraVectorFromThyra(comm, *it));
  }

  sensitivities.clear();
  sensitivities.reserve(thyraSensitivities.size());
  typedef Teuchos::Array<Teuchos::Array<Teuchos::RCP<const Thyra::MultiVectorBase<double> > > > ThyraSensitivityArray;
  for (ThyraSensitivityArray::const_iterator it_begin = thyraSensitivities.begin(),
      it_end = thyraSensitivities.end(),
      it = it_begin;
      it != it_end;
      ++it) {
    ThyraSensitivityArray::const_reference sens_thyra = *it;
    Teuchos::Array<Teuchos::RCP<const Epetra_MultiVector> > sens;
    sens.reserve(sens_thyra.size());
    for (ThyraSensitivityArray::value_type::const_iterator jt = sens_thyra.begin(),
        jt_end = sens_thyra.end();
        jt != jt_end;
        ++jt) {
        sens.push_back(epetraMultiVectorFromThyra(comm, *jt));
    }
    sensitivities.push_back(sens);
  }
}

int main(int argc, char *argv[]) {

  int status=0; // 0 = pass, failures are incremented
  bool success = true;

#ifdef ALBANY_DEBUG
  Teuchos::GlobalMPISession mpiSession(&argc, &argv);
#else // bypass printing process startup info
  Teuchos::GlobalMPISession mpiSession(&argc, &argv, NULL);
#endif

  Kokkos::initialize(argc, argv);

#ifdef ALBANY_CHECK_FPE
   // Catch FPEs. Follow Main_SolveT.cpp's approach to checking for floating
   // point exceptions.
   //_mm_setcsr(_MM_MASK_MASK &~ (_MM_MASK_OVERFLOW | _MM_MASK_INVALID | _MM_MASK_DIV_ZERO) );
   _MM_SET_EXCEPTION_MASK(_MM_GET_EXCEPTION_MASK() & ~_MM_MASK_INVALID);
#endif

  using Teuchos::RCP;
  using Teuchos::rcp;

  RCP<Teuchos::FancyOStream> out(Teuchos::VerboseObjectBase::getDefaultOStream());

  // Command-line argument for input file
  std::string xmlfilename;
  if(argc > 1){

    if(!strcmp(argv[1],"--help")){
      printf("albany [inputfile.xml]\n");
      exit(1);
    }
    else
      xmlfilename = argv[1];

  }
  else
    xmlfilename = "input.xml";

  try {

    RCP<Teuchos::Time> totalTime =
      Teuchos::TimeMonitor::getNewTimer("Albany: ***Total Time***");

    RCP<Teuchos::Time> setupTime =
      Teuchos::TimeMonitor::getNewTimer("Albany: Setup Time");
    Teuchos::TimeMonitor totalTimer(*totalTime); //start timer
    Teuchos::TimeMonitor setupTimer(*setupTime); //start timer

    RCP<const Teuchos_Comm> comm =
      Tpetra::DefaultPlatform::getDefaultPlatform().getComm();

    Albany::SolverFactory slvrfctry(xmlfilename, comm);
    RCP<Epetra_Comm> appComm = Albany::createEpetraCommFromTeuchosComm(comm);
    RCP<Albany::Application> app;
    const RCP<Thyra::ModelEvaluator<double> > solver =
      slvrfctry.createThyraSolverAndGetAlbanyApp(app, appComm, appComm);

    setupTimer.~TimeMonitor();

//    PHX::InitializeKokkosDevice();
   
    Teuchos::ParameterList &solveParams =
      slvrfctry.getAnalysisParameters().sublist("Solve", /*mustAlreadyExist =*/ false);
    // By default, request the sensitivities if not explicitly disabled
    solveParams.get("Compute Sensitivities", true);

    Teuchos::Array<Teuchos::RCP<const Thyra::VectorBase<double> > > thyraResponses;
    Teuchos::Array<Teuchos::Array<Teuchos::RCP<const Thyra::MultiVectorBase<double> > > > thyraSensitivities;

       // The PoissonSchrodinger_SchroPo and PoissonSchroMosCap1D tests seg fault as albanyApp is null -
       // For now, do not resize the response vectors. FIXME sort out this issue.
    if(Teuchos::nonnull(app))
      Piro::PerformSolveBase(*solver, solveParams, thyraResponses, thyraSensitivities, app->getAdaptSolMgr()->getSolObserver());
    else
      Piro::PerformSolveBase(*solver, solveParams, thyraResponses, thyraSensitivities);

    Teuchos::Array<Teuchos::RCP<const Epetra_Vector> > responses;
    Teuchos::Array<Teuchos::Array<Teuchos::RCP<const Epetra_MultiVector> > > sensitivities;
    epetraFromThyra(appComm, thyraResponses, thyraSensitivities, responses, sensitivities);

    const int num_p = solver->Np(); // Number of *vectors* of parameters
    const int num_g = solver->Ng(); // Number of *vectors* of responses

    *out << "Finished eval of first model: Params, Responses "
      << std::setprecision(12) << std::endl;

    Teuchos::ParameterList& parameterParams = slvrfctry.getParameters().sublist("Problem").sublist("Parameters");
    int num_param_vecs = (parameterParams.isType<int>("Number")) ?
        int(parameterParams.get("Number", 0) > 0) :
        parameterParams.get("Number of Parameter Vectors", 0);

    const Thyra::ModelEvaluatorBase::InArgs<double> nominal = solver->getNominalValues();
    double norm2;
    for (int i=0; i<num_p; i++) {
      const Teuchos::RCP<const Epetra_Vector> p_init = epetraVectorFromThyra(appComm, nominal.get_p(i));
      if(i < num_param_vecs)
        p_init->Print(*out << "\nParameter vector " << i << ":\n");
      else { //distributed parameters, we print only 2-norm
        p_init->Norm2(&norm2);
        *out << "\nDistributed Parameter " << i << ":  " << norm2 << " (two-norm)\n" << std::endl;
      }
    }

    for (int i=0; i<num_g-1; i++) {
      const RCP<const Epetra_Vector> g = responses[i];
      bool is_scalar = true;

      if (app != Teuchos::null)
        is_scalar = app->getResponse(i)->isScalarResponse();

      if (is_scalar) {
        g->Print(*out << "\nResponse vector " << i << ":\n");

        if (num_p == 0) {
          // Just calculate regression data
          status += slvrfctry.checkSolveTestResults(i, 0, g.get(), NULL);
        } else {
          for (int j=0; j<num_p; j++) {
            const RCP<const Epetra_MultiVector> dgdp = sensitivities[i][j];
            if (Teuchos::nonnull(dgdp)) {
              if(j < num_param_vecs) {
                dgdp->Print(*out << "\nSensitivities (" << i << "," << j << "): \n");
                status += slvrfctry.checkSolveTestResults(i, j, g.get(), dgdp.get());
              }
              else {
                const Epetra_Map serial_map(-1, 1, 0, dgdp.get()->Comm());
                Epetra_MultiVector norms(serial_map,dgdp->NumVectors());
              //  RCP<Albany::ScalarResponseFunction> response = rcp_dynamic_cast<Albany::ScalarResponseFunction>(app->getResponse(i));
               // int numResponses = response->numResponses();
                *out << "\nSensitivities (" << i << "," << j  << ") for Distributed Parameters:  (two-norm)\n";
                *out << "    ";
                for(int ir=0; ir<dgdp->NumVectors(); ++ir) {
                  (*dgdp)(ir)->Norm2(&norm2);
                  (*norms(ir))[0] = norm2;
                  *out << "    " << norm2;
                }
                *out << "\n" << std::endl;
                status += slvrfctry.checkSolveTestResults(i, j, g.get(), &norms);
              }
            }
          }
        }
      }
    }

    // Create debug output object
    Teuchos::ParameterList &debugParams =
      slvrfctry.getParameters().sublist("Debug Output", true);
    bool writeToMatrixMarketSoln = debugParams.get("Write Solution to MatrixMarket", false);
    bool writeToMatrixMarketDistrSolnMap = debugParams.get("Write Distributed Solution and Map to MatrixMarket", false);
    bool writeToCoutSoln = debugParams.get("Write Solution to Standard Output", false);


    const RCP<const Epetra_Vector> xfinal = responses.back();
    double mnv; xfinal->MeanValue(&mnv);
    *out << "Main_Solve: MeanValue of final solution " << mnv << std::endl;
    *out << "\nNumber of Failed Comparisons: " << status << std::endl;
    if (writeToCoutSoln == true) 
       std::cout << "xfinal: " << *xfinal << std::endl;

#ifdef ALBANY_PERIDIGM
#ifdef ALBANY_EPETRA
    *out << "\nPERIDIGM-ALBANY OPTIMIZATION-BASED COUPLING FINAL FUNCTIONAL VALUE = " << LCM::PeridigmManager::self().obcEvaluateFunctional()  << "\n" << std::endl;
#endif
#endif

    if (debugParams.get<bool>("Analyze Memory", false))
      Albany::printMemoryAnalysis(std::cout, comm);

    if (writeToMatrixMarketSoln == true) { 

      //create serial map that puts the whole solution on processor 0
      int numMyElements = (xfinal->Comm().MyPID() == 0) ? app->getDiscretization()->getMap()->NumGlobalElements() : 0;
      const Epetra_Map serial_map(-1, numMyElements, 0, xfinal->Comm());

      //create importer from parallel map to serial map and populate serial solution xfinal_serial
      Epetra_Import importOperator(serial_map, *app->getDiscretization()->getMap());
      Epetra_Vector xfinal_serial(serial_map);
      xfinal_serial.Import(*app->getDiscretization()->getSolutionField(), importOperator, Insert);

      //writing to MatrixMarket file
      EpetraExt::MultiVectorToMatrixMarketFile("xfinal.mm", xfinal_serial);
    }
    if (writeToMatrixMarketDistrSolnMap == true) {
      //writing to MatrixMarket file
      EpetraExt::MultiVectorToMatrixMarketFile("xfinal_distributed.mm", *xfinal);
      EpetraExt::BlockMapToMatrixMarketFile("xfinal_distributed_map.mm", *app->getDiscretization()->getMap());
    }
  }
  TEUCHOS_STANDARD_CATCH_STATEMENTS(true, std::cerr, success);
  if (!success) status+=10000;
  

  Teuchos::TimeMonitor::summarize(*out,false,true,false/*zero timers*/);

  Kokkos::finalize_all();
 
  return status;
}
