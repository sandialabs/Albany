//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
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
#include "Teuchos_FancyOStream.hpp"
#include "Thyra_DefaultProductVector.hpp"
#include "Thyra_DefaultProductVectorSpace.hpp"

// Uncomment for run time nan checking
// This is set in the toplevel CMakeLists.txt file
//#define ALBANY_CHECK_FPE

#ifdef ALBANY_CHECK_FPE
#include <math.h>
//#include <Accelerate/Accelerate.h>
#include <xmmintrin.h>
#endif
//#define ALBANY_FLUSH_DENORMALS
#ifdef ALBANY_FLUSH_DENORMALS
#include <xmmintrin.h>
#include <pmmintrin.h>
#endif

#include "Albany_DataTypes.hpp"

#include "Phalanx_config.hpp"
#include "Phalanx.hpp"

#include "Kokkos_Core.hpp"

// Global variable that denotes this is the Tpetra executable
bool TpetraBuild = true;
const Tpetra::global_size_t INVALID = Teuchos::OrdinalTraits<Tpetra::global_size_t>::invalid ();

Teuchos::RCP<Tpetra_Vector>
createCombinedTpetraVector(
    Teuchos::Array<Teuchos::RCP<const Tpetra_Vector> > vecs)
{
  int  n_vecs = vecs.size();

  // Figure out how many local and global elements are in the
  // combined map by summing these quantities over each vector.
  LO local_num_elements = 0;
  GO global_num_elements = 0;

  for (int m = 0; m < n_vecs; ++m) {
    local_num_elements += vecs[m]->getMap()->getNodeNumElements();
    global_num_elements += vecs[m]->getMap()->getGlobalNumElements();
  }
  //Create global element indices array for combined map for this processor,
  //to be used to create the combined map.
  std::vector<GO> my_global_elements(local_num_elements);

  LO counter_local = 0;
  GO counter_global = 0;

  
  for (int m = 0; m < n_vecs; ++m) {
    LO local_num_elements_n = vecs[m]->getMap()->getNodeNumElements();
    GO global_num_elements_n = vecs[m]->getMap()->getGlobalNumElements();
    Teuchos::ArrayView<GO const> disc_global_elements = vecs[m]->getMap()->getNodeElementList();
 
    for (int l = 0; l < local_num_elements_n; ++l) {
      my_global_elements[counter_local + l] = counter_global + disc_global_elements[l];
    }
    counter_local += local_num_elements_n;
    counter_global += global_num_elements_n;
  }

  Teuchos::ArrayView<GO> const
  my_global_elements_AV =
      Teuchos::arrayView(&my_global_elements[0], local_num_elements);

  Teuchos::RCP<const Tpetra_Map>  combined_map = Teuchos::rcp(
      new Tpetra_Map(global_num_elements, my_global_elements_AV, 0, vecs[0]->getMap()->getComm()));
 
  Teuchos::RCP<Tpetra_Vector> combined_vec = Teuchos::rcp(new Tpetra_Vector(combined_map)); 
  
  counter_local = 0; 
  for (int m = 0; m < n_vecs; ++m) {
    int disc_local_elements_m = vecs[m]->getMap()->getNodeNumElements(); 
    Teuchos::RCP<Tpetra_Vector> temp = combined_vec->offsetViewNonConst(vecs[m]->getMap(), counter_local);
    Teuchos::Array<ST> array(vecs[m]->getMap()->getNodeNumElements());
    vecs[m]->get1dCopy(array);
    for (std::size_t i=0; i<vecs[m]->getMap()->getNodeNumElements(); ++i)
     temp->replaceLocalValue(i, array[i]);
    counter_local += disc_local_elements_m; 
  }

  return combined_vec;
}

// Overridden from Thyra::ModelEvaluator<ST>

//IKT: Similar to tpetraFromThyra but for thyra product vectors (currently only used in Coupled Schwarz problems). 
void tpetraFromThyraProdVec(
  const Teuchos::Array<Teuchos::RCP<const Thyra::VectorBase<ST> > > &thyraResponses,
  const Teuchos::Array<Teuchos::Array<Teuchos::RCP<const Thyra::MultiVectorBase<ST> > > > &thyraSensitivities,
  Teuchos::Array<Teuchos::RCP<const Tpetra_Vector> > &responses,
  Teuchos::Array<Teuchos::Array<Teuchos::RCP<const Tpetra_MultiVector> > > &sensitivities)
{
  Teuchos::RCP<Teuchos::FancyOStream> out(Teuchos::VerboseObjectBase::getDefaultOStream());
  responses.clear();
  //FIXME: right now only setting # of responses to 1 (solution vector) as printing of other 
  //responses does not work. 
  //responses.reserve(thyraResponses.size());
  responses.reserve(1); 
  typedef Teuchos::Array<Teuchos::RCP<const Thyra::VectorBase<ST> > > ThyraResponseArray;
  for (ThyraResponseArray::const_iterator it_begin = thyraResponses.begin(),
    it_end = thyraResponses.end(),
    it = it_begin;
    it != it_end;
    ++it) 
    {
    if (it == it_end-1) {
      Teuchos::RCP<const Thyra::ProductVectorBase<ST> > r_prod =
           Teuchos::nonnull(*it) ?
           Teuchos::rcp_dynamic_cast<const Thyra::ProductVectorBase<ST> >(*it,true) :
           Teuchos::null;
      if (r_prod != Teuchos::null) {
        //FIXME: productSpace()->numBlocks() when we have responses and >1 model does not work for some reason.  
        //Need to figure out why!
        const int nProdVecs = r_prod->productSpace()->numBlocks(); 
        //create Teuchos array of spaces / vectors, to be populated from the product vector
        Teuchos::Array<Teuchos::RCP<const Tpetra_Vector> > rs(nProdVecs); 
        for (int i=0; i<nProdVecs; i++) {
          rs[i] =  ConverterT::getConstTpetraVector(r_prod->getVectorBlock(i)); 
        }
        Teuchos::RCP<Tpetra_Vector> r_vec = createCombinedTpetraVector(rs); 
        responses.push_back(r_vec);
       }
     }
   }
  sensitivities.clear();
  sensitivities.reserve(thyraSensitivities.size());
  if (thyraSensitivities.size() > 0) 
    *out << "WARNING: For Thyra::ProductVectorBase, sensitivities are not yet supported! \n"; 
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
          Teuchos::RCP<const Thyra::ProductVectorBase<ST> > s_prod =
          Teuchos::nonnull(*jt) ?
          Teuchos::rcp_dynamic_cast<const Thyra::ProductVectorBase<ST> >(*jt,true) :
          Teuchos::null;
          //FIXME: ultimately we'll need to change this to set the sensitivity vector
          Teuchos::RCP<const Tpetra_Vector> s_vec = Teuchos::null; 
          sens.push_back(s_vec);
        }
     sensitivities.push_back(sens);
   }
}



void tpetraFromThyra(
  const Teuchos::Array<Teuchos::RCP<const Thyra::VectorBase<ST> > > &thyraResponses,
  const Teuchos::Array<Teuchos::Array<Teuchos::RCP<const Thyra::MultiVectorBase<ST> > > > &thyraSensitivities,
  Teuchos::Array<Teuchos::RCP<const Tpetra_Vector> > &responses,
  Teuchos::Array<Teuchos::Array<Teuchos::RCP<const Tpetra_MultiVector> > > &sensitivities)
{
  responses.clear();
  responses.reserve(thyraResponses.size());
  typedef Teuchos::Array<Teuchos::RCP<const Thyra::VectorBase<ST> > > ThyraResponseArray;
  for (ThyraResponseArray::const_iterator it_begin = thyraResponses.begin(),
      it_end = thyraResponses.end(),
      it = it_begin;
      it != it_end;
      ++it) {
    responses.push_back(Teuchos::nonnull(*it) ? ConverterT::getConstTpetraVector(*it) : Teuchos::null);
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
        sens.push_back(Teuchos::nonnull(*jt) ? ConverterT::getConstTpetraMultiVector(*jt) : Teuchos::null);
    }
    sensitivities.push_back(sens);
  }
}

int main(int argc, char *argv[]) {

  int status=0; // 0 = pass, failures are incremented
  bool success = true;

  Teuchos::GlobalMPISession mpiSession(&argc,&argv);
  Kokkos::initialize(argc, argv);

#ifdef ALBANY_FLUSH_DENORMALS
  _MM_SET_FLUSH_ZERO_MODE(_MM_FLUSH_ZERO_ON);
  _MM_SET_DENORMALS_ZERO_MODE(_MM_DENORMALS_ZERO_ON);
#endif

#ifdef ALBANY_CHECK_FPE
//	_mm_setcsr(_MM_MASK_MASK &~
//		(_MM_MASK_OVERFLOW | _MM_MASK_INVALID | _MM_MASK_DIV_ZERO) );
    _MM_SET_EXCEPTION_MASK(_MM_GET_EXCEPTION_MASK() & ~_MM_MASK_INVALID);
#endif

#ifdef ALBANY_64BIT_INT
// Albany assumes sizeof(long) is 64 bit ints

//	if(sizeof(long) != sizeof(long long)){
	if(sizeof(long) != 8){ // 8 bytes

		std::cerr << "Error: The 64 bit build of Albany assumes that sizeof(long) == 64 bits."
			<< " sizeof(long) = " << sizeof(long) << "; sizeof(long long) = " << sizeof(long long) << std::endl;

        exit(1);

    }

#endif

  using Teuchos::RCP;
  using Teuchos::rcp;

  RCP<Teuchos::FancyOStream> out(Teuchos::VerboseObjectBase::getDefaultOStream());

  // Command-line argument for input file
  Albany::CmdLineArgs cmd;
  cmd.parse_cmdline(argc, argv, *out);

  try {
    RCP<Teuchos::Time> totalTime =
      Teuchos::TimeMonitor::getNewTimer("Albany: ***Total Time***");

    RCP<Teuchos::Time> setupTime =
      Teuchos::TimeMonitor::getNewTimer("Albany: Setup Time");
    Teuchos::TimeMonitor totalTimer(*totalTime); //start timer
    Teuchos::TimeMonitor setupTimer(*setupTime); //start timer

    RCP<const Teuchos_Comm> comm =
      Tpetra::DefaultPlatform::getDefaultPlatform().getComm();

    // Connect vtune for performance profiling
    if (cmd.vtune) {
      Albany::connect_vtune(comm->getRank());
    }

    Albany::SolverFactory slvrfctry(cmd.xml_filename, comm);
    RCP<Albany::Application> app;
    const RCP<Thyra::ResponseOnlyModelEvaluatorBase<ST> > solver =
      slvrfctry.createAndGetAlbanyAppT(app, comm, comm);

    setupTimer.~TimeMonitor();

    Teuchos::ParameterList &solveParams =
      slvrfctry.getAnalysisParameters().sublist("Solve", /*mustAlreadyExist =*/ false);
    // By default, request the sensitivities if not explicitly disabled
    solveParams.get("Compute Sensitivities", true);

    Teuchos::Array<Teuchos::RCP<const Thyra::VectorBase<ST> > > thyraResponses;
    Teuchos::Array<Teuchos::Array<Teuchos::RCP<const Thyra::MultiVectorBase<ST> > > > thyraSensitivities;
    Piro::PerformSolve(*solver, solveParams, thyraResponses, thyraSensitivities);

    Teuchos::Array<Teuchos::RCP<const Tpetra_Vector> > responses;
    Teuchos::Array<Teuchos::Array<Teuchos::RCP<const Tpetra_MultiVector> > > sensitivities;

    //Check if thyraResponses are product vectors or regular vectors
    Teuchos::RCP<const Thyra::ProductVectorBase<ST> > r_prod; 
    if (thyraResponses.size() > 0) {
      r_prod = Teuchos::nonnull(thyraResponses[0]) ?
           Teuchos::rcp_dynamic_cast<const Thyra::ProductVectorBase<ST> >(thyraResponses[0],false) :
           Teuchos::null;
    }
    if (r_prod == Teuchos::null) tpetraFromThyra(thyraResponses, thyraSensitivities, responses, sensitivities);
    else tpetraFromThyraProdVec(thyraResponses, thyraSensitivities, responses, sensitivities);

    const int num_p = solver->Np(); // Number of *vectors* of parameters
    int num_g = solver->Ng();  // Number of *vectors* of responses
    if (r_prod != Teuchos::null && num_g > 0) { 
      *out << "WARNING: For Thyra::ProductVectorBase, printing of responses does not work yet!  " <<
              "No responses will be printed even though you requested " << num_g << " responses. \n"; 
      num_g = 1; 
    }

    *out << "Finished eval of first model: Params, Responses "
      << std::setprecision(12) << std::endl;


    Teuchos::ParameterList& parameterParams = slvrfctry.getParameters().sublist("Problem").sublist("Parameters");
    Teuchos::ParameterList& responseParams = slvrfctry.getParameters().sublist("Problem").sublist("Response Functions");


    int num_param_vecs =
       parameterParams.get("Number of Parameter Vectors", 0);
    bool using_old_parameter_list = false;
    if (parameterParams.isType<int>("Number")) {
      int numParameters = parameterParams.get<int>("Number");
      if (numParameters > 0) {
        num_param_vecs = 1;
        using_old_parameter_list = true;
      }
    }

    int num_response_vecs =
       responseParams.get("Number of Response Vectors", 0);
    bool using_old_response_list = false;
    if (responseParams.isType<int>("Number")) {
      int numParameters = responseParams.get<int>("Number");
      if (numParameters > 0) {
        num_response_vecs = 1;
        using_old_response_list = true;
      }
    }

    Teuchos::Array<Teuchos::RCP<Teuchos::Array<std::string> > > param_names;
    param_names.resize(num_param_vecs);
    for (int l = 0; l < num_param_vecs; ++l) {
      const Teuchos::ParameterList* pList =
        using_old_parameter_list ?
        &parameterParams :
        &(parameterParams.sublist(Albany::strint("Parameter Vector", l)));

      const int numParameters = pList->get<int>("Number");
      TEUCHOS_TEST_FOR_EXCEPTION(
          numParameters == 0,
          Teuchos::Exceptions::InvalidParameter,
          std::endl << "Error!  In Albany::ModelEvaluatorT constructor:  " <<
          "Parameter vector " << l << " has zero parameters!" << std::endl);

      param_names[l] = Teuchos::rcp(new Teuchos::Array<std::string>(numParameters));
      for (int k = 0; k < numParameters; ++k) {
        (*param_names[l])[k] =
          pList->get<std::string>(Albany::strint("Parameter", k));
      }
    }

    Teuchos::Array<Teuchos::RCP<Teuchos::Array<std::string> > > response_names;
    response_names.resize(num_response_vecs);
    for (int l = 0; l < num_response_vecs; ++l) {
      const Teuchos::ParameterList* pList =
        using_old_response_list ?
        &responseParams :
        &(responseParams.sublist(Albany::strint("Response Vector", l)));

      bool number_exists = pList->getEntryPtr("Number");

      if(number_exists){

        const int numParameters = pList->get<int>("Number");
        TEUCHOS_TEST_FOR_EXCEPTION(
          numParameters == 0,
          Teuchos::Exceptions::InvalidParameter,
          std::endl << "Error!  In Albany::ModelEvaluatorT constructor:  " <<
          "Response vector " << l << " has zero parameters!" << std::endl);

        response_names[l] = Teuchos::rcp(new Teuchos::Array<std::string>(numParameters));
        for (int k = 0; k < numParameters; ++k) {
          (*response_names[l])[k] =
            pList->get<std::string>(Albany::strint("Response", k));
        }
      }
      else response_names[l] = Teuchos::null;
    }

    const Thyra::ModelEvaluatorBase::InArgs<double> nominal = solver->getNominalValues();

    //Check if parameters are product vectors or regular vectors
    Teuchos::RCP<const Thyra::ProductVectorBase<ST> > p_prod;
    if (num_p > 0) { 
       p_prod =  Teuchos::nonnull(nominal.get_p(0)) ?
                        Teuchos::rcp_dynamic_cast<const Thyra::ProductVectorBase<ST> >(nominal.get_p(0),false) :
                        Teuchos::null;
      //If p_prod is not product vector, print out parameter vector by converting thyra vector to tpetra vector
      if (p_prod == Teuchos::null) { //Thyra vector case (default -- for everything except CoupledSchwarz right now)
        for (int i=0; i<num_p; i++) {
          Albany::printTpetraVector(*out << "\nParameter vector " << i << ":\n", *param_names[i],
             ConverterT::getConstTpetraVector(nominal.get_p(i)));
        }
      }
      else { //Thyra product vector case
        for (int i=0; i<num_p; i++) {
           Teuchos::RCP<const Thyra::ProductVectorBase<ST> > pT =
                Teuchos::rcp_dynamic_cast<const Thyra::ProductVectorBase<ST> >(
                nominal.get_p(i), true);
           //IKT: note that we are assuming the parameters are all the same for all the models 
           //that are being coupled (in CoupledSchwarz) so we print the parameters from the 0th 
           //model only.  LOCA does not populate p for more than 1 model at the moment so we cannot
           //allow for different parameters in different models.
           Teuchos::RCP<const Tpetra_Vector> p = Teuchos::rcp_dynamic_cast<const ThyraVector>(
                                       pT->getVectorBlock(0),true)->getConstTpetraVector();
           Albany::printTpetraVector(*out << "\nParameter vector " << i << ":\n", *param_names[i], p); 
        }
      }
    }

    for (int i=0; i<num_g-1; i++) {
      const RCP<const Tpetra_Vector> g = responses[i];
      bool is_scalar = true;

      if (app != Teuchos::null)
        is_scalar = app->getResponse(i)->isScalarResponse();

      if (is_scalar) {

        if(response_names[i] != Teuchos::null) {
          *out << "\n Response vector " << i << ": " << *response_names[i] << "\n"; 
          Albany::printTpetraVector(*out, g);
        }
        else {
          *out << "\n Response vector " << i << ":\n"; 
          Albany::printTpetraVector(*out, g);
        }

        if (num_p == 0) {
          // Just calculate regression data
          status += slvrfctry.checkSolveTestResultsT(i, 0, g.get(), NULL);
        } 
        else {
          if (sensitivities[0][0] != Teuchos::null) {
            for (int j=0; j<num_p; j++) {
              const RCP<const Tpetra_MultiVector> dgdp = sensitivities[i][j];
              if (Teuchos::nonnull(dgdp)) {
                Albany::printTpetraVector(*out << "\nSensitivities (" << i << "," << j << "):!\n", dgdp);
              }
              status += slvrfctry.checkSolveTestResultsT(i, j, g.get(), dgdp.get());
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

    const RCP<const Tpetra_Vector> xfinal = responses.back();
    double mnv = xfinal->meanValue();
    *out << "\nMain_Solve: MeanValue of final solution " << mnv << std::endl;
    *out << "\nNumber of Failed Comparisons: " << status << std::endl;
    if (writeToCoutSoln == true) { 
       Albany::printTpetraVector(*out << "\nxfinal:\n", xfinal);
    }

    if (debugParams.get<bool>("Analyze Memory", false))
      Albany::printMemoryAnalysis(std::cout, comm);

    if (writeToMatrixMarketSoln == true) { 

      //create serial map that puts the whole solution on processor 0
      int numMyElements = (xfinal->getMap()->getComm()->getRank() == 0) ? xfinal->getMap()->getGlobalNumElements() : 0;

     Teuchos::RCP<const Tpetra_Map> serial_map = Teuchos::rcp(new const Tpetra_Map(INVALID, numMyElements, 0, comm));
 
      //create importer from parallel map to serial map and populate serial solution xfinal_serial
      Teuchos::RCP<Tpetra_Import> importOperator = Teuchos::rcp(new Tpetra_Import(xfinal->getMap(), serial_map)); 
      Teuchos::RCP<Tpetra_Vector> xfinal_serial = Teuchos::rcp(new Tpetra_Vector(serial_map)); 
      xfinal_serial->doImport(*xfinal, *importOperator, Tpetra::INSERT);

      //writing to MatrixMarket file
       Tpetra_MatrixMarket_Writer::writeDenseFile("xfinal.mm", xfinal_serial);
    }
    if (writeToMatrixMarketDistrSolnMap == true) {
      //writing to MatrixMarket file
      Tpetra_MatrixMarket_Writer::writeDenseFile("xfinal_distributed.mm", *xfinal);
      Tpetra_MatrixMarket_Writer::writeMapFile("xfinal_distributed_map.mm", *xfinal->getMap());
    }
  }
  TEUCHOS_STANDARD_CATCH_STATEMENTS(true, std::cerr, success);
  if (!success) status+=10000;

  Teuchos::TimeMonitor::summarize(*out,false,true,false/*zero timers*/);

  Kokkos::finalize_all();

  return status;
}
