//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#include <fstream>
#include "Teuchos_TestForException.hpp"
#include "Teuchos_CommHelpers.hpp"
#include "Albany_Utils.hpp"

template<typename EvalT, typename Traits>
QCAD::ResponseFieldIntegral<EvalT, Traits>::
ResponseFieldIntegral(Teuchos::ParameterList& p,
		      const Teuchos::RCP<Albany::Layouts>& dl) :
  coordVec("Coord Vec", dl->qp_vector),
  weights("Weights", dl->qp_scalar)
{
  //! get and validate Response parameter list
  Teuchos::ParameterList* plist = 
    p.get<Teuchos::ParameterList*>("Parameter List");
  Teuchos::RCP<const Teuchos::ParameterList> reflist = 
    this->getValidResponseParameters();
  plist->validateParameters(*reflist,0);

  //! parameters passed down from problem
  Teuchos::RCP<Teuchos::ParameterList> paramsFromProblem = 
    p.get< Teuchos::RCP<Teuchos::ParameterList> >("Parameters From Problem");
  if(paramsFromProblem != Teuchos::null) {

    // Material database 
    materialDB = paramsFromProblem->get< Teuchos::RCP<QCAD::MaterialDatabase> >("MaterialDB");

    // Length unit in meters
    length_unit_in_m = paramsFromProblem->get<double>("Length unit in m");
  }
  else {
    materialDB = Teuchos::null;
    length_unit_in_m = 1.0e-6; //default length unit = microns (backward compat)
  }
       
  //! number of quad points per cell
  Teuchos::RCP<PHX::DataLayout> scalar_dl = dl->qp_scalar;
  numQPs = scalar_dl->dimension(1);
  
  //! obtain number of dimensions
  Teuchos::RCP<PHX::DataLayout> vector_dl = dl->qp_vector;
  std::vector<PHX::DataLayout::size_type> dims;
  vector_dl->dimensions(dims);
  numDims = dims[2];

  //! Initialize Region
  opRegion  = Teuchos::rcp( new QCAD::MeshRegion<EvalT, Traits>("Coord Vec","Weights",*plist,materialDB,dl) );

  //! User-specified parameters
  std::string fieldName;

  fieldName = plist->get<std::string>("Field Name","");
  if(fieldName.length() > 0) {
    fieldNames.push_back(fieldName);
    conjugateFieldFlag.push_back(plist->get<bool>("Conjugate Field",false));

    fieldName = plist->get<std::string>("Field Name Im","");
    fieldNames_Imag.push_back(fieldName);
    if(fieldName.length() > 0) fieldIsComplex.push_back(true);
    else fieldIsComplex.push_back(false);
  }

  for(int i=1; i < QCAD::MAX_FIELDNAMES_IN_INTEGRAL; i++) {
    fieldName = plist->get<std::string>(Albany::strint("Field Name",i),"");
    if(fieldName.length() > 0) {
      fieldNames.push_back(fieldName);
      conjugateFieldFlag.push_back(plist->get<bool>(Albany::strint("Conjugate Field",i),false));

      fieldName = plist->get<std::string>(Albany::strint("Field Name Im",i),"");
      fieldNames_Imag.push_back(fieldName);
      if(fieldName.length() > 0) fieldIsComplex.push_back(true);
      else fieldIsComplex.push_back(false);
    }
    else break;
  }
  bReturnImagPart = plist->get<bool>("Return Imaginary Part",false);
  
  std::string integrandLinLengthUnit; // linear length unit of integrand (e.g. "cm" for integrand in cm^-3)
  integrandLinLengthUnit = plist->get<std::string>("Integrand Length Unit","cm");
  bPositiveOnly = plist->get<bool>("Positive Return Only",false);

  //! compute scaling factor based on number of dimensions and units
  double integrand_length_unit_in_m;
  if( integrandLinLengthUnit == "m" )       integrand_length_unit_in_m = 1.0;
  else if( integrandLinLengthUnit == "cm" ) integrand_length_unit_in_m = 1e-2;
  else if( integrandLinLengthUnit == "um" ) integrand_length_unit_in_m = 1e-6;
  else if( integrandLinLengthUnit == "nm" ) integrand_length_unit_in_m = 1e-9;
  else if( integrandLinLengthUnit == "mesh" ) integrand_length_unit_in_m = length_unit_in_m;
  else integrand_length_unit_in_m = length_unit_in_m;  // assume same unit as mesh (e.g. if unit string is blank)
  
  double X0 = length_unit_in_m / integrand_length_unit_in_m; // length scaling to get to integrand's lenght unit

  if (numDims == 1)       scaling = X0; 
  else if (numDims == 2)  scaling = X0*X0; 
  else if (numDims == 3)  scaling = X0*X0*X0; 
  else 
    TEUCHOS_TEST_FOR_EXCEPTION (true, Teuchos::Exceptions::InvalidParameter, std::endl 
				<< "Error! Invalid number of dimensions: " << numDims << std::endl);


  //! add dependent fields (all fields assumed scalar qp)
  std::vector<std::string>::const_iterator it;
  //for(it = fieldNames.begin(); it != fieldNames.end(); ++it) {
  for(std::size_t i=0; i<fieldNames.size(); i++) {
    PHX::MDField<ScalarT,Cell,QuadPoint> f(fieldNames[i], scalar_dl);
    fields.push_back(f); this->addDependentField(f);

    PHX::MDField<ScalarT,Cell,QuadPoint> fi(fieldNames_Imag[i], scalar_dl);
    fields_Imag.push_back(fi);

    if(fieldIsComplex[i]) this->addDependentField(fi);
  }

  this->addDependentField(coordVec);
  this->addDependentField(weights);
  opRegion->addDependentFields(this);

  //TODO: make name unique? Is this needed for anything?
  this->setName(fieldName+" Response Field Integral"+PHX::TypeString<EvalT>::value);
  
  // Setup scatter evaluator
  p.set("Stand-alone Evaluator", false);
  std::string local_response_name = 
    fieldName + " Local Response Field Integral";
  std::string global_response_name = 
    fieldName + " Global Response Field Integral";
  PHX::Tag<ScalarT> local_response_tag(local_response_name, 
				       dl->cell_scalar);
  PHX::Tag<ScalarT> global_response_tag(global_response_name, 
					dl->workset_scalar);
  p.set("Local Response Field Tag", local_response_tag);
  p.set("Global Response Field Tag", global_response_tag);
  PHAL::SeparableScatterScalarResponse<EvalT,Traits>::setup(p,dl);
}

// **********************************************************************
template<typename EvalT, typename Traits>
void QCAD::ResponseFieldIntegral<EvalT, Traits>::
postRegistrationSetup(typename Traits::SetupData d,
                      PHX::FieldManager<Traits>& fm)
{
  typename std::vector<PHX::MDField<ScalarT,Cell,QuadPoint> >::iterator it;
  //for(it = fields.begin(); it != fields.end(); ++it)
  for(std::size_t i=0; i<fields.size(); i++) {
    this->utils.setFieldData(fields[i],fm);
    if(fieldIsComplex[i]) this->utils.setFieldData(fields_Imag[i],fm);
  }

  this->utils.setFieldData(coordVec,fm);
  this->utils.setFieldData(weights,fm);
  opRegion->postRegistrationSetup(fm);
  PHAL::SeparableScatterScalarResponse<EvalT,Traits>::postRegistrationSetup(d,fm);
}

// **********************************************************************
template<typename EvalT, typename Traits>
void QCAD::ResponseFieldIntegral<EvalT, Traits>::
preEvaluate(typename Traits::PreEvalData workset)
{
  for (typename PHX::MDField<ScalarT>::size_type i=0; 
       i<this->global_response.size(); i++)
    this->global_response[i] = 0.0;

  // Do global initialization
  PHAL::SeparableScatterScalarResponse<EvalT,Traits>::preEvaluate(workset);
}

// **********************************************************************
template<typename EvalT, typename Traits>
void QCAD::ResponseFieldIntegral<EvalT, Traits>::
evaluateFields(typename Traits::EvalData workset)
{
  // Zero out local response
  for (typename PHX::MDField<ScalarT>::size_type i=0; 
       i<this->local_response.size(); i++)
    this->local_response[i] = 0.0;

  typename std::vector<PHX::MDField<ScalarT,Cell,QuadPoint> >::const_iterator it;

  if(opRegion->elementBlockIsInRegion(workset.EBName)) {

    ScalarT term, val; //, dbI = 0.0;
    std::size_t n, max, nExtraMinuses, nOneBits, nBits = fields.size();

    //DEBUG
    //std::size_t nContrib1 = 0, nContrib2 = 0;
    //ScalarT dbMaxRe[10], dbMaxIm[10];
    //for(std::size_t i=0; i<10; i++) dbMaxRe[i] = dbMaxIm[i] = 0.0;

    for (std::size_t cell=0; cell < workset.numCells; ++cell) {
      if(!opRegion->cellIsInRegion(cell)) continue;

      for (std::size_t qp=0; qp < numQPs; ++qp) {
	val = 0.0;

	//Loop over all possible combinations of Re/Im parts which form product terms and 
	// add the relevant ones (depending on whether we're returning the overall real or
	// imaginary part of the integral) to get the integrand value for this (cell,qp).
	// We do this by mapping the Re/Im choice onto a string of N bits, where N is the 
	// number of fields being multiplied together. (0 = RePart, 1 = ImPart)

	//nContrib1++; //DEBUG

	//for(it = fields.begin(); it != fields.end(); ++it)
	max = (std::size_t)std::pow(2.,(int)nBits);
//	max = pow(2.0,static_cast<int>(nBits));
	for(std::size_t i=0; i<max; i++) {

	  // Count the number of 1 bits, and exit early if 
	  //  there's a 1 bit for a field that is not complex
	  nOneBits = nExtraMinuses = 0;
	  for(n=0; n<nBits; n++) {
	    if( (0x1 << n) & i ) { // if n-th bit of i is set (use Im part of n-th field)
	      if(!fieldIsComplex[n]) break;
	      if(conjugateFieldFlag[n]) nExtraMinuses++;
	      nOneBits++;
	    }	
	  }
	  if(n < nBits) continue;  // we exited early, signaling this product can't contribute

	  //check if this combination of Re/Im parts contributes to the overall Re or Im part we return
	  if( (bReturnImagPart && nOneBits % 2) || (!bReturnImagPart && nOneBits % 2 == 0)) {
	    term = (nOneBits % 4 >= 2) ? -1.0 : 1.0; //apply minus sign if nOneBits % 4 == 2 (-1) or == 3 (-i)
	    if(nExtraMinuses % 2) term *= -1.0;      //apply minus sign due to conjugations
	    //nContrib2++;

	    //multiply fields together
	    for(std::size_t m=0; m<nBits; m++) { 
	      if( (0x1 << m) & i ) {
		term *= fields_Imag[m](cell,qp);
		//if( abs(fields_Imag[m](cell,qp)) > dbMaxIm[m]) dbMaxIm[m] = abs(fields_Imag[m](cell,qp));
	      }	
	      else {
		term *= fields[m](cell,qp);
		//if( abs(fields[m](cell,qp)) > dbMaxRe[m]) dbMaxRe[m] = abs(fields[m](cell,qp));
	      }
	    }

	    val += term;  //add term to overall integrand
	  }
	}
	val *= weights(cell,qp) * scaling; //multiply integrand by volume

	//dbI += val; //DEBUG
        this->local_response(cell) += val;
	this->global_response(0) += val;
      }
    }

    //DEBUG
    /*if(fieldNames.size() > 1) {
      std::cout << "DB: " << (bReturnImagPart == true ? "Im" : "Re") << " Field Integral - int(";
      for(std::size_t i=0; i<fieldNames.size(); i++) 
	std::cout << fieldNames[i] << "," << (conjugateFieldFlag[i] ? "-" : "") << (fieldIsComplex[i] ? fieldNames_Imag[i] : "X") << " * ";
      std::cout << " dV) -- I += " << dbI << "  (ebName = " << workset.EBName << 
	" contrib1=" << nContrib1 << " contrib2=" << nContrib2 << ")" << std::endl;
      std::cout << "DB MAX of Fields Re: " << dbMaxRe[0] << "," << dbMaxRe[1] << "," << dbMaxRe[2] << "," << dbMaxRe[3] << std::endl;
      std::cout << "DB MAX of Fields Im: " << dbMaxIm[0] << "," << dbMaxIm[1] << "," << dbMaxIm[2] << "," << dbMaxIm[3] << std::endl;
      }*/
  }

  // Do any local-scattering necessary
  PHAL::SeparableScatterScalarResponse<EvalT,Traits>::evaluateFields(workset);
}

// **********************************************************************
template<typename EvalT, typename Traits>
void QCAD::ResponseFieldIntegral<EvalT, Traits>::
postEvaluate(typename Traits::PostEvalData workset)
{
  // Add contributions across processors
  Teuchos::RCP< Teuchos::ValueTypeSerializer<int,ScalarT> > serializer =
    workset.serializerManager.template getValue<EvalT>();

  // we cannot pass the same object for both the send and receive buffers in reduceAll call
  // creating a copy of the global_response, not a view
  std::vector<ScalarT> partial_vector(&this->global_response[0],&this->global_response[0]+this->global_response.size()); //needed for allocating new storage
  PHX::MDField<ScalarT> partial_response(this->global_response);
  partial_response.setFieldData(Teuchos::ArrayRCP<ScalarT>(partial_vector.data(),0,partial_vector.size(),false));

  Teuchos::reduceAll(
    *workset.comm, *serializer, Teuchos::REDUCE_SUM,
    this->global_response.size(), &partial_response[0],
    &this->global_response[0]);

  if (bPositiveOnly && this->global_response[0] < 1e-6) {
    this->global_response[0] = 1e+100;
  }
  
  // Do global scattering
  PHAL::SeparableScatterScalarResponse<EvalT,Traits>::postEvaluate(workset);
}

// **********************************************************************
template<typename EvalT,typename Traits>
Teuchos::RCP<const Teuchos::ParameterList>
QCAD::ResponseFieldIntegral<EvalT,Traits>::getValidResponseParameters() const
{
  Teuchos::RCP<Teuchos::ParameterList> validPL =
     	rcp(new Teuchos::ParameterList("Valid ResponseFieldIntegral Params"));;
  Teuchos::RCP<const Teuchos::ParameterList> baseValidPL =
    PHAL::SeparableScatterScalarResponse<EvalT,Traits>::getValidResponseParameters();
  validPL->setParameters(*baseValidPL);

  Teuchos::RCP<const Teuchos::ParameterList> regionValidPL =
    QCAD::MeshRegion<EvalT,Traits>::getValidParameters();
  validPL->setParameters(*regionValidPL);

  validPL->set<std::string>("Name", "", "Name of response function");
  validPL->set<int>("Phalanx Graph Visualization Detail", 0, "Make dot file to visualize phalanx graph");

  validPL->set<std::string>("Field Name", "", "Name of Field to integrate");
  validPL->set<std::string>("Field Name Im", "", "Name of Field to integrate");
  validPL->set<bool>("Conjugate Field", false, "Whether a (complex-valued) field should be conjugated in product of fields");
  for(int i=1; i < QCAD::MAX_FIELDNAMES_IN_INTEGRAL; i++)
    validPL->set<std::string>(Albany::strint("Field Name",i), "", "Name of Field to integrate (multiplied into integrand)");
  for(int i=1; i < QCAD::MAX_FIELDNAMES_IN_INTEGRAL; i++)
    validPL->set<std::string>(Albany::strint("Field Name Im",i), "", "Name of Imaginar part of Field to integrate (multiplied into integrand)");
  for(int i=1; i < QCAD::MAX_FIELDNAMES_IN_INTEGRAL; i++)
    validPL->set<bool>(Albany::strint("Conjugate Field",i), false, "Whether field should be conjugated in product of fields");

  validPL->set<std::string>("Integrand Length Unit","cm","Linear length unit of integrand, e.g. cm for integrand in cm^-3.  Can be m, cm, um, nm, or mesh.");
  validPL->set<bool>("Positive Return Only",false);
  validPL->set<bool>("Return Imaginary Part",false,"True return imaginary part of integral, False returns real part");


  validPL->set<std::string>("Description", "", "Description of this response used by post processors");
  
  return validPL;
}

// **********************************************************************

