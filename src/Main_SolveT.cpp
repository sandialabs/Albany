//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#include <iostream>
#include <string>


#include "Albany_Utils.hpp"
#include "Albany_SolverFactory.hpp"

#include "Piro_PerformSolve.hpp"
#include "Teuchos_ParameterList.hpp"

#include "Teuchos_GlobalMPISession.hpp"
#include "Teuchos_TimeMonitor.hpp"
#include "Teuchos_VerboseObject.hpp"
#include "Teuchos_StandardCatchMacros.hpp"
#include "Epetra_Map.h"  //Needed for serial, somehow

// Uncomment for run time nan checking
// This is set in the toplevel CMakeLists.txt file
//#define ALBANY_CHECK_FPE

#ifdef ALBANY_CHECK_FPE
#include <math.h>
//#include <Accelerate/Accelerate.h>
#include <xmmintrin.h>
#endif


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


#include "Thyra_TpetraThyraWrappers.hpp"
#include "Albany_DataTypes.hpp"

void tpetraFromThyra(
  const Teuchos::Array<Teuchos::RCP<const Thyra::VectorBase<ST> > > &thyraResponses,
  const Teuchos::Array<Teuchos::Array<Teuchos::RCP<const Thyra::MultiVectorBase<ST> > > > &thyraSensitivities,
  Teuchos::Array<Teuchos::RCP<const Tpetra_Vector> > &responses,
  Teuchos::Array<Teuchos::Array<Teuchos::RCP<const Tpetra_MultiVector> > > &sensitivities)
{
  typedef Thyra::TpetraOperatorVectorExtraction<ST, int> ConverterT;

  responses.clear();
  responses.reserve(thyraResponses.size());
  typedef Teuchos::Array<Teuchos::RCP<const Thyra::VectorBase<ST> > > ThyraResponseArray;
  for (ThyraResponseArray::const_iterator it_begin = thyraResponses.begin(),
      it_end = thyraResponses.end(),
      it = it_begin;
      it != it_end;
      ++it) {
    responses.push_back(ConverterT::getConstTpetraVector(*it));
  }

  sensitivities.clear();
  sensitivities.reserve(thyraSensitivities.size());
  typedef Teuchos::Array<Teuchos::Array<Teuchos::RCP<const Thyra::MultiVectorBase<ST> > > > ThyraSensitivityArray;
  for (ThyraSensitivityArray::const_iterator it_begin = thyraSensitivities.begin(),
      it_end = thyraSensitivities.end(),
      it = it_begin;
      it != it_end;
      ++it) {
    ThyraSensitivityArray::const_reference sens_thyra = *it;
    Teuchos::Array<Teuchos::RCP<const Tpetra_MultiVector> > sens;
    sens.reserve(sens_thyra.size());
    for (ThyraSensitivityArray::value_type::const_iterator jt = sens_thyra.begin(),
        jt_end = sens_thyra.end();
        jt != jt_end;
        ++jt) {
        sens.push_back(ConverterT::getConstTpetraMultiVector(*jt));
    }
    sensitivities.push_back(sens);
  }
}


#include "Petra_Converters.hpp"

Teuchos::RCP<const Epetra_Vector>
epetraVectorFromTpetra(
  const Teuchos::RCP<const Epetra_Comm> &comm,
  const Teuchos::RCP<const Tpetra_Vector> &tpetra)
{
  Teuchos::RCP<Epetra_Vector> result;
  if (Teuchos::nonnull(tpetra)) {
    const Teuchos::RCP<const Epetra_Map> epetraMap =
      Petra::TpetraMap_To_EpetraMap(tpetra->getMap(), comm);
    result = Teuchos::rcp(new Epetra_Vector(*epetraMap, false));
    Petra::TpetraVector_To_EpetraVector(tpetra, *result, comm);
  }
  return result;
}

Teuchos::RCP<const Epetra_MultiVector>
epetraMultiVectorFromTpetra(
  const Teuchos::RCP<const Epetra_Comm> &comm,
  const Teuchos::RCP<const Tpetra_MultiVector> &tpetra)
{
  Teuchos::RCP<Epetra_MultiVector> result;
  if (Teuchos::nonnull(tpetra)) {
    const Teuchos::RCP<const Epetra_Map> epetraMap =
      Petra::TpetraMap_To_EpetraMap(tpetra->getMap(), comm);
    result = Teuchos::rcp(new Epetra_MultiVector(*epetraMap, tpetra->getNumVectors(), false));
    Petra::TpetraMultiVector_To_EpetraMultiVector(tpetra, *result, comm);
  }
  return result;
}

void
epetraFromTpetra(
  const Teuchos::RCP<const Epetra_Comm> &comm,
  const Teuchos::Array<Teuchos::RCP<const Tpetra_Vector> > &responsesT,
  const Teuchos::Array<Teuchos::Array<Teuchos::RCP<const Tpetra_MultiVector> > > &sensitivitiesT,
  Teuchos::Array<Teuchos::RCP<const Epetra_Vector> > &responses,
  Teuchos::Array<Teuchos::Array<Teuchos::RCP<const Epetra_MultiVector> > > &sensitivities)
{
  responses.clear();
  responses.reserve(responsesT.size());
  typedef Teuchos::Array<Teuchos::RCP<const Tpetra_Vector> > TpetraResponseArray;
  for (TpetraResponseArray::const_iterator it_begin = responsesT.begin(),
      it_end = responsesT.end(),
      it = it_begin;
      it != it_end;
      ++it) {
    responses.push_back(epetraVectorFromTpetra(comm, *it));
  }

  sensitivities.clear();
  sensitivities.reserve(sensitivitiesT.size());
  typedef Teuchos::Array<Teuchos::Array<Teuchos::RCP<const Tpetra_MultiVector> > > TpetraSensitivityArray;
  for (TpetraSensitivityArray::const_iterator it_begin = sensitivitiesT.begin(),
      it_end = sensitivitiesT.end(),
      it = it_begin;
      it != it_end;
      ++it) {
    TpetraSensitivityArray::const_reference sensT = *it;
    Teuchos::Array<Teuchos::RCP<const Epetra_MultiVector> > sens;
    sens.reserve(sensT.size());
    for (TpetraSensitivityArray::value_type::const_iterator jt = sensT.begin(),
        jt_end = sensT.end();
        jt != jt_end;
        ++jt) {
      sens.push_back(epetraMultiVectorFromTpetra(comm, *jt));
    }
    sensitivities.push_back(sens);
  }
}


int main(int argc, char *argv[]) {

  int status=0; // 0 = pass, failures are incremented
  bool success = true;
  Teuchos::GlobalMPISession mpiSession(&argc,&argv);

#ifdef ALBANY_CHECK_FPE
	_mm_setcsr(_MM_MASK_MASK &~
		(_MM_MASK_OVERFLOW | _MM_MASK_INVALID | _MM_MASK_DIV_ZERO) );
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

    Albany::SolverFactory slvrfctry(xmlfilename, Albany_MPI_COMM_WORLD);
    RCP<Epetra_Comm> appComm = Albany::createEpetraCommFromMpiComm(Albany_MPI_COMM_WORLD);
    RCP<Albany::Application> app;
    const RCP<Thyra::ResponseOnlyModelEvaluatorBase<ST> > solver =
      slvrfctry.createAndGetAlbanyAppT(app, appComm, appComm);

    setupTimer.~TimeMonitor();

    *out << "Before main solve" << std::endl;

    Teuchos::ParameterList &solveParams =
      slvrfctry.getAnalysisParameters().sublist("Solve", /*mustAlreadyExist =*/ false);
    // By default, request the sensitivities if not explicitly disabled
    solveParams.get("Compute Sensitivities", true);

    Teuchos::Array<Teuchos::RCP<const Thyra::VectorBase<ST> > > thyraResponses;
    Teuchos::Array<Teuchos::Array<Teuchos::RCP<const Thyra::MultiVectorBase<ST> > > > thyraSensitivities;
    Piro::PerformSolve(*solver, solveParams, thyraResponses, thyraSensitivities);

    *out << "After main solve" << std::endl;

    Teuchos::Array<Teuchos::RCP<const Tpetra_Vector> > responsesT;
    Teuchos::Array<Teuchos::Array<Teuchos::RCP<const Tpetra_MultiVector> > > sensitivitiesT;
    tpetraFromThyra(thyraResponses, thyraSensitivities, responsesT, sensitivitiesT);

    Teuchos::Array<Teuchos::RCP<const Epetra_Vector> > responses;
    Teuchos::Array<Teuchos::Array<Teuchos::RCP<const Epetra_MultiVector> > > sensitivities;
    epetraFromTpetra(appComm, responsesT, sensitivitiesT, responses, sensitivities);

    const int num_p = solver->Np(); // Number of *vectors* of parameters
    const int num_g = solver->Ng(); // Number of *vectors* of responses

    *out << "Finished eval of first model: Params, Responses "
      << std::setprecision(12) << std::endl;

    const Thyra::ModelEvaluatorBase::InArgs<double> nominal = solver->getNominalValues();
    for (int i=0; i<num_p; i++) {
      const Teuchos::RCP<const Epetra_Vector> p_init = epetraVectorFromThyra(appComm, nominal.get_p(i));
      p_init->Print(*out << "\nParameter vector " << i << ":\n");
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
              dgdp->Print(*out << "\nSensitivities (" << i << "," << j << "):!\n");
            }
            status += slvrfctry.checkSolveTestResults(i, j, g.get(), dgdp.get());
          }
        }
      }
    }

    const RCP<const Epetra_Vector> xfinal = responses.back();
    double mnv; xfinal->MeanValue(&mnv);
    *out << "Main_Solve: MeanValue of final solution " << mnv << std::endl;
    *out << "\nNumber of Failed Comparisons: " << status << std::endl;
  }
  TEUCHOS_STANDARD_CATCH_STATEMENTS(true, std::cerr, success);
  if (!success) status+=10000;

  Teuchos::TimeMonitor::summarize(*out,false,true,false/*zero timers*/);
  return status;
}
