//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#include <fstream>
#include "Teuchos_TestForException.hpp"
#include "Phalanx_DataLayout.hpp"
#include "Sacado_ParameterRegistration.hpp"

const int MAX_MESH_REGIONS = 30;
const int MAX_POINT_CHARGES = 10;
const int MAX_CLOUD_CHARGES = 10;

template<typename EvalT, typename Traits>
QCAD::PoissonSource<EvalT, Traits>::
PoissonSource(Teuchos::ParameterList& p,
                 const Teuchos::RCP<Albany::Layouts>& dl) :
  coordVec(p.get<std::string>("Coordinate Vector Name"), dl->qp_vector),
  coordVecAtVertices(p.get<std::string>("Coordinate Vector Name"), dl->vertices_vector),
  weights("Weights", dl->qp_scalar),
  potential(p.get<std::string>("Variable Name"), dl->qp_scalar),
  temperatureField(p.get<std::string>("Temperature Name"), dl->shared_param),
  poissonSource(p.get<std::string>("Source Name"), dl->qp_scalar),
  chargeDensity("Charge Density",dl->qp_scalar),
  electronDensity("Electron Density",dl->qp_scalar),
  artCBDensity("Artificial Conduction Band Density",dl->qp_scalar),
  holeDensity("Hole Density",dl->qp_scalar),
  electricPotential("Electric Potential",dl->qp_scalar),
  ionizedDopant("Ionized Dopant",dl->qp_scalar),
  conductionBand("Conduction Band",dl->qp_scalar),
  valenceBand("Valence Band",dl->qp_scalar),
  approxQuanEDen("Approx Quantum EDensity",dl->qp_scalar),
  bRealEigenvectors(false)
{
  // Material database
  materialDB = p.get< Teuchos::RCP<QCAD::MaterialDatabase> >("MaterialDB");

  Teuchos::ParameterList* psList = p.get<Teuchos::ParameterList*>("Parameter List");

  Teuchos::RCP<const Teuchos::ParameterList> reflist = 
      this->getValidPoissonSourceParameters();
  psList->validateParameters(*reflist,0);

  std::vector<PHX::DataLayout::size_type> dims;
  dl->qp_vector->dimensions(dims);
  numQPs  = dims[1];
  numDims = dims[2];
  numNodes = dl->node_scalar->dimension(1);

  // get values from the input .xml and use default values if not provided
  factor = psList->get("Factor", 1.0);
  device = psList->get("Device", "defaultdevice");
  nonQuantumRegionSource = psList->get("Non Quantum Region Source", "semiclassical");
  quantumRegionSource    = psList->get("Quantum Region Source", "semiclassical"); 
  imagPartOfCoulombSrc   = psList->get<bool>("Imaginary Part of Coulomb Source", false); 
  carrierStatistics = psList->get("Carrier Statistics", "Boltzmann Statistics");
  incompIonization = psList->get("Incomplete Ionization", "False");
  bUsePredictorCorrector = psList->get<bool>("Use predictor-corrector method",false);
  bIncludeVxc = psList->get<bool>("Include exchange-correlation potential",false);
  fixedQuantumOcc = psList->get<double>("Fixed Quantum Occupation",-1.0);

  // find element blocks and voltages applied on them
  std::string preName = "DBC on NS "; 
  std::string postName = " for DOF Phi";
  std::size_t preLen = preName.length();
  std::size_t postLen = postName.length();  

  // look through DBCs and pull out any element block names we find (for setting the Fermi level below)
  const Teuchos::ParameterList& dbcPList = *(p.get<Teuchos::ParameterList*>("Dirichlet BCs ParameterList"));
  for (Teuchos::ParameterList::ConstIterator model_it = dbcPList.begin();
         model_it != dbcPList.end(); ++model_it)
  {
    // retrieve the nodeset name
    const std::string& dbcName = model_it->first;
    std::size_t nsNameLen = dbcName.length() - preLen - postLen;
    std::string nsName = dbcName.substr(preLen, nsNameLen);
    
    // obtain the element block associated with this nodeset (should be ohmic, NOT contact on insulator) 
    const std::string& ebName = materialDB->getNodeSetParam<std::string>(nsName,"elementBlock","");
    
    // map the element block to its applied voltage
    if (ebName.length() > 0)
    {
      double dbcValue = dbcPList.get<double>(dbcName); 
      mapDBCValue_eb[ebName] = dbcValue;
      mapDBCValue_ns[nsName] = dbcValue;
      //std::cout << "ebName = " << ebName << ", value = " << mapDBCValue_eb[ebName] << std::endl;  
    }
  }

  // specific values for "1D MOSCapacitor"
  if (device == "1D MOSCapacitor") 
  {
    oxideWidth = psList->get("Oxide Width", 0.);
    siliconWidth = psList->get("Silicon Width", 0.);
    dopingAcceptor = psList->get("Acceptor Doping", 1e14);
    acceptorActE = psList->get("Acceptor Activation Energy", 0.045);
  }
  
  // passed down from main list
  length_unit_in_m = p.get<double>("Length unit in m");
  energy_unit_in_eV = p.get<double>("Energy unit in eV");

  if(quantumRegionSource == "schrodinger" || 
     quantumRegionSource == "coulomb" || 
     quantumRegionSource == "ci") {
    std::string evecFieldRoot = p.get<std::string>("Eigenvector field name root");
    nEigenvectors = psList->get<int>("Eigenvectors to Import");
    bRealEigenvectors = psList->get<bool>("Eigenvectors are Real", false);

    char buf[200];

    eigenvector_Re.resize(nEigenvectors);
    for (int k = 0; k < nEigenvectors; ++k) {
      sprintf(buf, "%s_Re%d", evecFieldRoot.c_str(), k);
      PHX::MDField<ScalarT,Cell,QuadPoint> fr(buf,dl->qp_scalar);
      eigenvector_Re[k] = fr; this->addDependentField(eigenvector_Re[k]);
    }

    if(!bRealEigenvectors) {
      eigenvector_Im.resize(nEigenvectors);
      for (int k = 0; k < nEigenvectors; ++k) {
	sprintf(buf, "%s_Im%d", evecFieldRoot.c_str(), k);
	PHX::MDField<ScalarT,Cell,QuadPoint> fi(buf,dl->qp_scalar);
	eigenvector_Im[k] = fi; this->addDependentField(eigenvector_Im[k]);
      }
    }
  }
  else {
    nEigenvectors = 0;
  }

  // Defaults
  prevDensityMixingFactor = -1.0; //Flag that factor is unset... - sometimes this factor isn't set correctly within Albany framework, so HACK here

  // Add factor as a Sacado-ized parameter
  Teuchos::RCP<ParamLib> paramLib =
    p.get< Teuchos::RCP<ParamLib> >("Parameter Library", Teuchos::null);
  this->registerSacadoParameter("Poisson Source Factor", paramLib);


  // Add parameters from material database as Sacado params
  std::vector<std::string> dopingParamNames = materialDB->getAllMatchingParams<std::string>("Doping Parameter Name");
  std::vector<std::string> chargeParamNames = materialDB->getAllMatchingParams<std::string>("Charge Parameter Name");
  
  std::vector<std::string>::iterator s;
  for(s = dopingParamNames.begin(); s != dopingParamNames.end(); s++) {
    if( psList->isParameter(*s) ) {
      materialParams[*s] = psList->get<double>(*s);
      this->registerSacadoParameter(*s, paramLib);
    }
  }
  for(s = chargeParamNames.begin(); s != chargeParamNames.end(); s++) {
    if( psList->isParameter(*s) ) {
      materialParams[*s] = psList->get<double>(*s);
      this->registerSacadoParameter(*s, paramLib);
    }
  }

  //Add Mesh Region Parameters (factors which multiply RHS 
  //  of Poisson equation in a given mesh region)
  for(int i=0; i<MAX_MESH_REGIONS; i++) {
    std::string subListName = Albany::strint("Mesh Region",i);
    if( psList->isSublist(subListName) ) {
      std::string factorName = Albany::strint("Mesh Region Factor",i);
      this->registerSacadoParameter(factorName, paramLib);

      // Validate sublist
      Teuchos::RCP<const Teuchos::ParameterList> regionreflist = 
	QCAD::MeshRegion<EvalT, Traits>::getValidParameters();
      Teuchos::ParameterList refsublist(*regionreflist);
      refsublist.set<double>("Factor Value",1.0,"Initial value of the factor corresponding to this mesh region");
      psList->sublist(subListName).validateParameters(refsublist,0);

      // Create MeshRegion object
      Teuchos::RCP<QCAD::MeshRegion<EvalT, Traits> > region = 
	Teuchos::rcp( new QCAD::MeshRegion<EvalT, Traits>(p.get<std::string>("Coordinate Vector Name"),
							  "Weights",psList->sublist(subListName),materialDB,dl) );

      ScalarT value = psList->sublist(subListName).get<double>("Factor Value",1.0);
      meshRegionList.push_back(region);
      meshRegionFactors.push_back( value );
    }
    else break;
  }

  //Add Point Charges (later add the charge as a Sacado param?)
  numWorksetsScannedForPtCharges = 0;
  Teuchos::RCP<Teuchos::ParameterList> ptChargeValidPL =
     	rcp(new Teuchos::ParameterList("Valid Point Charge Params"));
  ptChargeValidPL->set<double>("X", 0.0, "x-coordinate of point charge");
  ptChargeValidPL->set<double>("Y", 0.0, "y-coordinate of point charge");
  ptChargeValidPL->set<double>("Z", 0.0, "z-coordinate of point charge");
  ptChargeValidPL->set<double>("Charge", 1.0, "Amount of charge in units of the elementary charge (default = +1)");

  for(int i=0; i<MAX_POINT_CHARGES; i++) {
    std::string subListName = Albany::strint("Point Charge",i);
    if( psList->isSublist(subListName) ) {

      // Validate sublist
      psList->sublist(subListName).validateParameters(*ptChargeValidPL,0);

      // Fill PointCharge struct and add to vector (list)
      // QCAD::PoissonSource<EvalT, Traits>::PointCharge ptCharge;
      PointCharge ptCharge;
      ptCharge.position[0] = psList->sublist(subListName).get<double>("X",0.0);
      ptCharge.position[1] = psList->sublist(subListName).get<double>("Y",0.0);
      ptCharge.position[2] = psList->sublist(subListName).get<double>("Z",0.0);
      ptCharge.charge = psList->sublist(subListName).get<double>("Charge",+1.0);
      ptCharge.iWorkset = ptCharge.iCell = -1;  // indicates workset & cell are unknown
      
      pointCharges.push_back(ptCharge);

      // Sacado-ization
      std::stringstream s1; s1 << "Point Charge " << i << " X";
      this->registerSacadoParameter(s1.str(), paramLib);
      std::stringstream s2; s2 << "Point Charge " << i << " Y";
      this->registerSacadoParameter(s2.str(), paramLib);
      std::stringstream s3; s3 << "Point Charge " << i << " Z";
      this->registerSacadoParameter(s3.str(), paramLib);
      std::stringstream s4; s4 << "Point Charge " << i << " Charge";
      this->registerSacadoParameter(s4.str(), paramLib);
    }
    else break;
  }

  //Add Cloud Charges 
  Teuchos::RCP<Teuchos::ParameterList> clChargeValidPL =
     	rcp(new Teuchos::ParameterList("Valid Cloud Charge Params"));
  clChargeValidPL->set<double>("X", 0.0, "x-coordinate of point charge");
  clChargeValidPL->set<double>("Y", 0.0, "y-coordinate of point charge");
  clChargeValidPL->set<double>("Z", 0.0, "z-coordinate of point charge");
  clChargeValidPL->set<double>("Amplitude", 1.0, "Amplitude of Cloud Charge");
  clChargeValidPL->set<double>("Width", 1.0, "Gaussian Width of Cloud Charge");
  clChargeValidPL->set<double>("Cutoff", 1.0, "Hard-Zero Cutoff for Cloud Charge");

  for(int i=0; i<MAX_CLOUD_CHARGES; i++) {
    std::string subListName = Albany::strint("Cloud Charge",i);
    if( psList->isSublist(subListName) ) {

      // Validate sublist
      psList->sublist(subListName).validateParameters(*clChargeValidPL,0);

      // Fill CloudCharge struct and add to vector (list)
      // QCAD::PoissonSource<EvalT, Traits>::CloudCharge clCharge;
      //   No Defaults for cloud parameters: X,Y,Z,Amplitude,Width,Cutoff
      CloudCharge clCharge;
      clCharge.position[0]               = psList->sublist(subListName).get<double>("X");
      if(numDims>1) clCharge.position[1] = psList->sublist(subListName).get<double>("Y");
      if(numDims>2) clCharge.position[2] = psList->sublist(subListName).get<double>("Z");
      clCharge.amplitude                 = psList->sublist(subListName).get<double>("Amplitude");
      clCharge.width                     = psList->sublist(subListName).get<double>("Width");
      clCharge.cutoff                    = psList->sublist(subListName).get<double>("Cutoff");

      // Sacado-ization
      std::stringstream s1; s1 << "Cloud Charge " << i << " Amplitude";
      this->registerSacadoParameter(s1.str(), paramLib);
      std::stringstream s2; s2 << "Cloud Charge " << i << " X";
      this->registerSacadoParameter(s2.str(), paramLib);
      if(numDims > 1) {
	std::stringstream ss; ss << "Cloud Charge " << i << " Y";
	this->registerSacadoParameter(ss.str(), paramLib);
      }
      if(numDims > 2) {
	std::stringstream ss; ss << "Cloud Charge " << i << " Z";
	this->registerSacadoParameter(ss.str(), paramLib);
      }
      std::stringstream s3; s3 << "Cloud Charge " << i << " Width";
      this->registerSacadoParameter(s3.str(), paramLib);
      std::stringstream s4; s4 << "Cloud Charge " << i << " Cutoff";
      this->registerSacadoParameter(s4.str(), paramLib);
      
      cloudCharges.push_back(clCharge);
    }
    else break;
  }


  if(quantumRegionSource == "schrodinger") {
    this->registerSacadoParameter("Previous Quantum Density Mixing Factor", paramLib);
  }
  else if(quantumRegionSource == "coulomb") {
    //Add Sacado parameters to set indices of eigenvectors to be multipled together
    this->registerSacadoParameter("Source Eigenvector 1", paramLib);
    this->registerSacadoParameter("Source Eigenvector 2", paramLib);
  }

  this->addDependentField(potential);
  this->addDependentField(coordVec);
  this->addDependentField(coordVecAtVertices);
  this->addDependentField(weights);
  this->addDependentField(temperatureField);
    
  typename std::vector< Teuchos::RCP<MeshRegion<EvalT, Traits> > >::iterator it;
  for(it = meshRegionList.begin(); it != meshRegionList.end(); it++)
    (*it)->addDependentFields(this);

  this->addEvaluatedField(poissonSource);
  this->addEvaluatedField(chargeDensity);
  this->addEvaluatedField(electronDensity);
  this->addEvaluatedField(artCBDensity);
  this->addEvaluatedField(holeDensity);
  this->addEvaluatedField(electricPotential);
  this->addEvaluatedField(ionizedDopant);
  this->addEvaluatedField(conductionBand);
  this->addEvaluatedField(valenceBand);
  this->addEvaluatedField(approxQuanEDen);
  
  this->setName("Poisson Source" );
}

// **********************************************************************
template<typename EvalT, typename Traits>
void QCAD::PoissonSource<EvalT, Traits>::
postRegistrationSetup(typename Traits::SetupData d,
                      PHX::FieldManager<Traits>& fm)
{
  this->utils.setFieldData(poissonSource,fm);
  this->utils.setFieldData(potential,fm);
  this->utils.setFieldData(coordVec,fm);
  this->utils.setFieldData(coordVecAtVertices,fm);
  this->utils.setFieldData(weights,fm);
  this->utils.setFieldData(temperatureField,fm);

  this->utils.setFieldData(chargeDensity,fm);
  this->utils.setFieldData(electronDensity,fm);
  this->utils.setFieldData(artCBDensity,fm);
  this->utils.setFieldData(holeDensity,fm);
  this->utils.setFieldData(electricPotential,fm);

  this->utils.setFieldData(ionizedDopant,fm);
  this->utils.setFieldData(conductionBand,fm);
  this->utils.setFieldData(valenceBand,fm);
  this->utils.setFieldData(approxQuanEDen,fm);

  for (int k = 0; k < nEigenvectors; ++k)
    this->utils.setFieldData(eigenvector_Re[k],fm);

  if(!bRealEigenvectors) {
    for (int k = 0; k < nEigenvectors; ++k)
      this->utils.setFieldData(eigenvector_Im[k],fm);
  }

  typename std::vector< Teuchos::RCP<MeshRegion<EvalT, Traits> > >::iterator it;
  for(it = meshRegionList.begin(); it != meshRegionList.end(); it++)
    (*it)->postRegistrationSetup(fm);

}

// **********************************************************************
template<typename EvalT, typename Traits>
void QCAD::PoissonSource<EvalT, Traits>::
evaluateFields(typename Traits::EvalData workset)
{
  if (device == "elementblocks") evaluateFields_elementblocks(workset);
  
  else if (device == "1D MOSCapacitor") evaluateFields_moscap1d(workset);

  //! otherwise, run the /examples/Poisson/input_test2D device  
  else evaluateFields_default(workset);
}

// **********************************************************************
template<typename EvalT,typename Traits>
typename QCAD::PoissonSource<EvalT,Traits>::ScalarT& 
QCAD::PoissonSource<EvalT,Traits>::getValue(const std::string &n)
{
  if(n == "Poisson Source Factor") return factor;
  else if( materialParams.find(n) != materialParams.end() ) return materialParams[n];
  else if( n == "Source Eigenvector 1") return sourceEvecInds[0];
  else if( n == "Source Eigenvector 2") return sourceEvecInds[1];
  else if( n == "Previous Quantum Density Mixing Factor") return prevDensityMixingFactor;
  else {
    int nRegions = meshRegionFactors.size();
    for(int i=0; i<nRegions; i++)
      if( n == Albany::strint("Mesh Region Factor",i) ) return meshRegionFactors[i];

    for( std::size_t i=0; i < cloudCharges.size(); ++i) {
      std::stringstream s1; s1 << "Cloud Charge " << i << " Amplitude";
      if( n == s1.str()) return cloudCharges[i].amplitude;
      std::stringstream s2; s2 << "Cloud Charge " << i << " X";
      if( n == s2.str()) return cloudCharges[i].position[0];
      if(numDims > 1) { 
	std::stringstream ss; ss << "Cloud Charge " << i << " Y";
	if( n == ss.str()) return cloudCharges[i].position[1];
      } 
      if(numDims > 2) { 
	std::stringstream ss; ss << "Cloud Charge " << i << " Z";
	if( n == ss.str()) return cloudCharges[i].position[2];
      }
      std::stringstream s3; s3 << "Cloud Charge " << i << " Width";
      if( n == s3.str()) return cloudCharges[i].width;
      std::stringstream s4; s4 << "Cloud Charge " << i << " Cutoff";
      if( n == s4.str()) return cloudCharges[i].cutoff;
    }

    for( std::size_t i=0; i < pointCharges.size(); ++i) {
      std::stringstream s2; s2 << "Point Charge " << i << " X";
      if( n == s2.str()) return pointCharges[i].position_param[0];
      if(numDims > 1) { 
	std::stringstream ss; ss << "Point Charge " << i << " Y";
	if( n == ss.str()) return pointCharges[i].position_param[1];
      } 
      if(numDims > 2) { 
	std::stringstream ss; ss << "Point Charge " << i << " Z";
	if( n == ss.str()) return pointCharges[i].position_param[2];
      }
      std::stringstream s1; s1 << "Point Charge " << i << " Charge";
      if( n == s1.str()) return pointCharges[i].charge;
    }

    TEUCHOS_TEST_FOR_EXCEPT(true); 
    return factor; //dummy so all control paths return
  }
}

// **********************************************************************
template<typename EvalT,typename Traits>
Teuchos::RCP<const Teuchos::ParameterList>
QCAD::PoissonSource<EvalT,Traits>::getValidPoissonSourceParameters() const
{
  Teuchos::RCP<Teuchos::ParameterList> validPL =
       rcp(new Teuchos::ParameterList("Valid Poisson Problem Params"));;

  validPL->set<double>("Factor", 1.0, "Constant multiplier in source term");
  validPL->set<std::string>("Device", "defaultdevice", "Switch between different device models");
  validPL->set<std::string>("Non Quantum Region Source", "semiclassical", "Source type for non-quantum regions");
  validPL->set<std::string>("Quantum Region Source", "semiclassical", "Source type for quantum regions");
  validPL->set<bool>("Imaginary Part Of Coulomb Source",false,"Whether to use imag or real part of coulomb quantum region source");
  //validPL->set<double>("Donor Doping", 1e14, "Doping for nsilicon element blocks [cm^-3]");
  validPL->set<double>("Acceptor Doping", 1e14, "Doping for psilicon element blocks [cm^-3]");
  validPL->set<std::string>("Carrier Statistics", "Boltzmann Statistics", "Carrier statistics");
  validPL->set<std::string>("Incomplete Ionization", "False", "Partial ionization of dopants");
  //validPL->set<double>("Donor Activation Energy", 0.045, "Donor activation energy [eV]");
  validPL->set<double>("Acceptor Activation Energy", 0.045, "Acceptor activation energy [eV]");
  validPL->set<int>("Eigenvectors to Import", 0, "Number of eigenvectors to take from eigendata information");
  validPL->set<bool>("Eigenvectors are Real", false, "Whether eigenvectors contain imaginary parts (which should be imported)");
  validPL->set<bool>("Use predictor-corrector method",false, "Enable use of predictor-corrector method for S-P iterations");
  validPL->set<bool>("Include exchange-correlation potential",false, "Include the exchange correlation term in the output potential state");
  validPL->set<bool>("Imaginary Part of Coulomb Source",false,"When 'Quantum Region Source' equals 'coulomb', whether to use imaginary or real part as source term.");
  validPL->set<double>("Fixed Quantum Occupation",-1.0, "The fixed number of quantum orbitals (one orbital == spin * valley degeneracy e-) to fill (non-equilibrium).");

  validPL->set<double>("Oxide Width", 0., "Oxide width for 1D MOSCapacitor device");
  validPL->set<double>("Silicon Width", 0., "Silicon width for 1D MOSCapacitor device");
  
  std::vector<std::string> dopingParamNames = materialDB->getAllMatchingParams<std::string>("Doping Parameter Name");
  std::vector<std::string> chargeParamNames = materialDB->getAllMatchingParams<std::string>("Charge Parameter Name");
  std::vector<std::string>::iterator s;
  for(s = dopingParamNames.begin(); s != dopingParamNames.end(); s++)
    validPL->set<double>( *s, 0.0, "Doping Parameter [cm^-3]");
  for(s = chargeParamNames.begin(); s != chargeParamNames.end(); s++)
    validPL->set<double>( *s, 0.0, "Charge Parameter [cm^-3]");

  for(int i=0; i<MAX_MESH_REGIONS; i++) {
    std::string subListName = Albany::strint("Mesh Region",i);
    validPL->sublist(subListName, false, "Sublist defining a mesh region");
  }

  for(int i=0; i<MAX_POINT_CHARGES; i++) {
    std::string subListName = Albany::strint("Point Charge",i);
    validPL->sublist(subListName, false, "Sublist defining a point charge");
  }
  for(int i=0; i<MAX_CLOUD_CHARGES; i++) {
    std::string subListName = Albany::strint("Cloud Charge",i);
    validPL->sublist(subListName, false, "Sublist defining a cloud charge");
  }
  
  return validPL;
}


// *****************************************************************************
template<typename EvalT, typename Traits>
void QCAD::PoissonSource<EvalT, Traits>::
evaluateFields_elementblocks(typename Traits::EvalData workset)
{
  using std::string;

  ScalarT mrsFromEBTest = 1.0;          // mesh region scaling factor from element block tests
  ScalarT scaleFactor = factor / energy_unit_in_eV; // overall scaling of RHS

  bool   isQuantum     =  materialDB->getElementBlockParam<bool>(workset.EBName,"quantum",false);
  string matrlCategory = materialDB->getElementBlockParam<string>(workset.EBName,"Category");
  string sourceName    = isQuantum ? quantumRegionSource : nonQuantumRegionSource;


  //mesh region scaling by element block
  std::size_t nRegions = meshRegionList.size();
  std::vector<bool> bEBInRegion(nRegions,false);
  for(std::size_t i=0; i<nRegions; i++) {    
    if(meshRegionList[i]->elementBlockIsInRegion(workset.EBName)) {
      mrsFromEBTest *= meshRegionFactors[i];
      bEBInRegion[i] = true;
    }
  }
  
  //! function pointer to source calc member function
  void (QCAD::PoissonSource<EvalT,Traits>::*sourceCalc) (const typename Traits::EvalData workset, std::size_t cell, std::size_t qp,
							 const ScalarT& scaleFactor, const PoissonSourceSetupInfo& setup_info);
        
  //special case of metals, which always have source == "none", since they have no charge
  if(matrlCategory == "Metal")  sourceName = "none";

  if(sourceName == "semiclassical")    sourceCalc = &QCAD::PoissonSource<EvalT,Traits>::source_semiclassical;
  else if(sourceName == "none")        sourceCalc = &QCAD::PoissonSource<EvalT,Traits>::source_none;
  else if(sourceName == "schrodinger") sourceCalc = &QCAD::PoissonSource<EvalT,Traits>::source_quantum;
  else if(sourceName == "ci")          sourceCalc = &QCAD::PoissonSource<EvalT,Traits>::source_quantum;
  else if(sourceName == "coulomb")     sourceCalc = &QCAD::PoissonSource<EvalT,Traits>::source_coulomb;
  else if(sourceName == "testcoulomb") sourceCalc = &QCAD::PoissonSource<EvalT,Traits>::source_testcoulomb;
  else {
    TEUCHOS_TEST_FOR_EXCEPTION (true, Teuchos::Exceptions::InvalidParameter,
	std::endl << "Error!  Unknown source name: " << sourceName << "!"<< std::endl);
  }

  PoissonSourceSetupInfo setup_info = source_setup(sourceName, matrlCategory, workset);
  for (std::size_t cell=0; cell < workset.numCells; ++cell)
  {
    scaleFactor = getCellScaleFactor(cell, bEBInRegion, mrsFromEBTest*factor / energy_unit_in_eV);
    for (std::size_t qp=0; qp < numQPs; ++qp)
      (this->*sourceCalc)(workset, cell, qp, scaleFactor, setup_info);
  }

  //point charges
  if(pointCharges.size() > 0)
    source_pointcharges(workset);

  //cloud charges
  if(cloudCharges.size() > 0)
    source_cloudcharges(workset);

}


// **********************************************************************
template<typename EvalT, typename Traits>
void QCAD::PoissonSource<EvalT, Traits>::
evaluateFields_default(typename Traits::EvalData workset)
{
  ScalarT charge;

  for (std::size_t cell=0; cell < workset.numCells; ++cell) 
  {
    for (std::size_t qp=0; qp < numQPs; ++qp) 
    {
      MeshScalarT coord1 = coordVec(cell,qp,1);
      const ScalarT& phi = potential(cell,qp);

      switch (numDims) {
      case 2:
        if (coord1<0.8) charge = (coord1*coord1);
        else charge = 3.0;
        charge *= (1.0 + exp(-phi));
        chargeDensity(cell, qp) = charge;
        break;
      default: TEUCHOS_TEST_FOR_EXCEPT(true);
      }

      // do not scale the default device since the DBC is not scaled
      poissonSource(cell, qp) = factor*charge;
      
      // set all states to 0 except electricPotential 
      chargeDensity(cell, qp) = 0.0;
      electronDensity(cell, qp) = 0.0;  
      holeDensity(cell, qp) = 0.0;      
      electricPotential(cell, qp) = phi; // no scaling
      ionizedDopant(cell, qp) = 0.0;
      conductionBand(cell, qp) = 0.0; 
      valenceBand(cell, qp) = 0.01; 
      approxQuanEDen(cell,qp) = 0.0;
      artCBDensity(cell, qp) = 0.0;
    }
  }
}


// **********************************************************************
template<typename EvalT, typename Traits>
void QCAD::PoissonSource<EvalT, Traits>::
evaluateFields_moscap1d(typename Traits::EvalData workset)
{
  //Note: the moscap1d test structure always outputs values in [eV] (or [V]) and ignores the "Energy Unit In Electron Volts" input parameter
  ScalarT temperature = temperatureField(0); //get shared temperature parameter from field

  // Scaling factors
  double X0 = length_unit_in_m/1e-2; // length scaling to get to [cm] (structure dimension in [um])
  ScalarT V0 = kbBoltz*temperature/1.0; // kb*T/q in [V], scaling for potential        
  ScalarT Lambda2 = eps0/(eleQ*X0*X0); // derived scaling factor
  
  //! Constant energy reference for heterogeneous structures
  ScalarT qPhiRef;
  {
    std::string refMtrlName, category;
    refMtrlName = materialDB->getParam<std::string>("Reference Material");
    category = materialDB->getMaterialParam<std::string>(refMtrlName,"Category");
    if (category == "Semiconductor") 
    {
      // Same qPhiRef needs to be used for the entire structure
      double mdn = materialDB->getMaterialParam<double>(refMtrlName,"Electron DOS Effective Mass");
      double mdp = materialDB->getMaterialParam<double>(refMtrlName,"Hole DOS Effective Mass");
      double Chi = materialDB->getMaterialParam<double>(refMtrlName,"Electron Affinity");
      double Eg0 = materialDB->getMaterialParam<double>(refMtrlName,"Zero Temperature Band Gap");
      double alpha = materialDB->getMaterialParam<double>(refMtrlName,"Band Gap Alpha Coefficient");
      double beta = materialDB->getMaterialParam<double>(refMtrlName,"Band Gap Beta Coefficient");
      ScalarT Eg = Eg0-alpha*pow(temperature,2.0)/(beta+temperature); // in [eV]
    
      ScalarT kbT = kbBoltz*temperature;      // in [eV]
      ScalarT Eic = -Eg/2. + 3./4.*kbT*log(mdp/mdn);  // (Ei-Ec) in [eV]
      qPhiRef = Chi - Eic;  // (Evac-Ei) in [eV] where Evac = vacuum level
    }
    else 
    {
      TEUCHOS_TEST_FOR_EXCEPTION (true, Teuchos::Exceptions::InvalidParameter, std::endl 
			  << "Error!  Invalid category " << category 
			  << " for reference material !" << std::endl);
    }
  }  
  
  for (std::size_t cell = 0; cell < workset.numCells; ++cell) 
  {
    for (std::size_t qp = 0; qp < numQPs; ++qp) 
    {
      MeshScalarT coord0 = coordVec(cell,qp,0);
      
     // Silicon region
     if ( (coord0 > oxideWidth) && (coord0 <= (oxideWidth + siliconWidth)) )
     {
      const std::string matName = "Silicon";
        
      //! temperature-independent material parameters
      double mdn = materialDB->getMaterialParam<double>(matName,"Electron DOS Effective Mass");
      double mdp = materialDB->getMaterialParam<double>(matName,"Hole DOS Effective Mass");
      double Tref = materialDB->getMaterialParam<double>(matName,"Reference Temperature");
    
      double Chi = materialDB->getMaterialParam<double>(matName,"Electron Affinity");
      double Eg0 = materialDB->getMaterialParam<double>(matName,"Zero Temperature Band Gap");
      double alpha = materialDB->getMaterialParam<double>(matName,"Band Gap Alpha Coefficient");
      double beta = materialDB->getMaterialParam<double>(matName,"Band Gap Beta Coefficient");
    
      // constant prefactor in calculating Nc and Nv in [cm-3]
      double NcvFactor = 2.0*pow((kbBoltz*eleQ*m0*Tref)/(2*pi*pow(hbar,2)),3./2.)*1e-6;
            // eleQ converts kbBoltz in [eV/K] to [J/K], 1e-6 converts [m-3] to [cm-3]
    
      //! strong temperature-dependent material parameters
      ScalarT Nc;  // conduction band effective DOS in [cm-3]
      ScalarT Nv;  // valence band effective DOS in [cm-3]
      ScalarT Eg;  // band gap at T [K] in [eV]
      //ScalarT ni;  // intrinsic carrier concentration in [cm-3]
    
      Nc = NcvFactor*pow(mdn,1.5)*pow(temperature/Tref,1.5);  // in [cm-3]
      Nv = NcvFactor*pow(mdp,1.5)*pow(temperature/Tref,1.5); 
      Eg = Eg0-alpha*pow(temperature,2.0)/(beta+temperature); // in [eV]
    
      ScalarT kbT = kbBoltz*temperature;      // in [eV]
      //ni = sqrt(Nc*Nv)*exp(-Eg/(2.0*kbT));    // in [cm-3]
    
      // argument offset in calculating electron and hole density
      ScalarT eArgOffset = (-qPhiRef+Chi)/kbT;
      ScalarT hArgOffset = (qPhiRef-Chi-Eg)/kbT;
 
      //! parameters for computing exchange-correlation potential
      double ml = materialDB->getMaterialParam<double>(matName,"Longitudinal Electron Effective Mass");
      double mt = materialDB->getMaterialParam<double>(matName,"Transverse Electron Effective Mass");        
      double invEffMass = (2.0/mt + 1.0/ml) / 3.0;
      double averagedEffMass = 1.0 / invEffMass;
      double relPerm = materialDB->getMaterialParam<double>(matName,"Permittivity");
   
      //! function pointer to carrier statistics member function
      ScalarT (QCAD::PoissonSource<EvalT,Traits>::*carrStat) (const ScalarT);
    
      if (carrierStatistics == "Boltzmann Statistics")
        carrStat = &QCAD::PoissonSource<EvalT,Traits>::computeMBStat;  

      else if (carrierStatistics == "Fermi-Dirac Statistics")
        carrStat = &QCAD::PoissonSource<EvalT,Traits>::computeFDIntOneHalf;

      else if (carrierStatistics == "0-K Fermi-Dirac Statistics")
        carrStat = &QCAD::PoissonSource<EvalT,Traits>::computeZeroKFDInt;

      else TEUCHOS_TEST_FOR_EXCEPTION (true, Teuchos::Exceptions::InvalidParameter,
        std::endl << "Error!  Unknown carrier statistics ! " << std::endl);

      //! function pointer to ionized dopants member function
      ScalarT (QCAD::PoissonSource<EvalT,Traits>::*ionDopant) (const std::string, const ScalarT&); 
    
      if (incompIonization == "False")
        ionDopant = &QCAD::PoissonSource<EvalT,Traits>::fullDopants; 
    
      else if (incompIonization == "True")
        ionDopant = &QCAD::PoissonSource<EvalT,Traits>::ionizedDopants;
    
      else TEUCHOS_TEST_FOR_EXCEPTION (true, Teuchos::Exceptions::InvalidParameter,
        std::endl << "Error!  Invalid incomplete ionization option ! " << std::endl);

      //! get doping concentration and activation energy
      const std::string dopantType = "Acceptor";
      ScalarT inArg, dopingConc, dopantActE;
      dopingConc = dopingAcceptor; 
      dopantActE = acceptorActE;

      if(dopantType == "Donor") 
        inArg = eArgOffset + dopantActE/kbT;
      else if(dopantType == "Acceptor") 
        inArg = hArgOffset + dopantActE/kbT;
      else TEUCHOS_TEST_FOR_EXCEPTION (true, Teuchos::Exceptions::InvalidParameter,
	       std::endl << "Error!  Unknown dopant type " << dopantType << "!"<< std::endl);

      //! Schrodinger source for electrons
      if(quantumRegionSource == "schrodinger")
      {
#if defined(ALBANY_EPETRA)
        // retrieve Previous Poisson Potential
        ScalarT prevPhi = 0.0, approxEDensity = 0.0;
        
        if(bUsePredictorCorrector) {
          // compute the approximate quantum electron density using predictor-corrector method
	  Albany::MDArray prevPhiArray = (*workset.stateArrayPtr)["PS Previous Poisson Potential"];
	  prevPhi = prevPhiArray(cell,qp);
          approxEDensity = eDensityForPoissonSchrodinger(workset, cell, qp, prevPhi, true, 0.0, -1.0);
	}
        else  // otherwise, use the exact quantum density
          approxEDensity = eDensityForPoissonSchrodinger(workset, cell, qp, 0.0, false, 0.0, -1.0);

        // compute the exact quantum electron density
        ScalarT eDensity = eDensityForPoissonSchrodinger(workset, cell, qp, 0.0, false, 0.0, -1.0);
          
        // obtain the scaled potential
        const ScalarT& unscaled_phi = potential(cell,qp);
        ScalarT phi = unscaled_phi / V0; 
           
        // compute the hole density treated as classical
        ScalarT hDensity = Nv*(this->*carrStat)(-phi+hArgOffset); 

        // obtain the ionized dopants
        ScalarT ionN  = 0.0;
        if (dopantType == "Donor")  // function takes care of sign
          ionN = (this->*ionDopant)(dopantType,phi+inArg)*dopingConc;
        else if (dopantType == "Acceptor")
          ionN = (this->*ionDopant)(dopantType,-phi+inArg)*dopingConc;
        else 
          ionN = 0.0;
              
        // the scaled full RHS
        ScalarT charge; 
        charge = 1.0/Lambda2 * (hDensity- approxEDensity + ionN);
        poissonSource(cell, qp) = factor*charge;

        // output states
        chargeDensity(cell, qp) = hDensity -eDensity +ionN;
        electronDensity(cell, qp) = eDensity;
        holeDensity(cell, qp) = hDensity;
        electricPotential(cell, qp) = phi*V0 - qPhiRef;
        ionizedDopant(cell, qp) = ionN;
        approxQuanEDen(cell,qp) = approxEDensity;
        artCBDensity(cell, qp) = ( eDensity > 1e-6 ? eDensity : -Nc*(this->*carrStat)( -(phi+eArgOffset) ));
        
        if (bIncludeVxc)  // include Vxc
        {
          ScalarT Vxc = computeVxcLDA(relPerm, averagedEffMass, approxEDensity);
          conductionBand(cell, qp) = qPhiRef-Chi-phi*V0 +Vxc; // [eV]
	  electricPotential(cell, qp) = phi*V0 + Vxc - qPhiRef; //add xc correction to electric potential (used in CI delta_ij computation)
          // conductionBand(cell, qp) = qPhiRef -Chi -0.5*(phi*V0 +prevPhi) +Vxc; // [eV]
        }
        else { // not include Vxc
          conductionBand(cell, qp) = qPhiRef-Chi-phi*V0; // [eV]
	  electricPotential(cell, qp) = phi*V0 - qPhiRef;
          // conductionBand(cell, qp) = qPhiRef -Chi -0.5*(phi*V0 +prevPhi); // [eV]
	}
        
        valenceBand(cell, qp) = conductionBand(cell,qp)-Eg;
#endif
      }

      //! calculate the classical charge (RHS) for Poisson equation
      else
      {
        // obtain the scaled potential
        const ScalarT& unscaled_phi = potential(cell,qp);  //[V]
        ScalarT phi = unscaled_phi / V0; 
          
        // obtain the ionized dopants
        ScalarT ionN;
        if (dopantType == "Donor")  // function takes care of sign
          ionN = (this->*ionDopant)(dopantType,phi+inArg)*dopingConc;
        else if (dopantType == "Acceptor")
          ionN = (this->*ionDopant)(dopantType,-phi+inArg)*dopingConc;
        else 
          ionN = 0.0; 

        // the scaled full RHS
        ScalarT charge, eDensity, hDensity;
        eDensity = Nc*(this->*carrStat)(phi+eArgOffset);
        hDensity = Nv*(this->*carrStat)(-phi+hArgOffset);
        charge = 1.0/Lambda2 * (hDensity - eDensity + ionN);
        poissonSource(cell, qp) = factor*charge;
          
        // output states
        chargeDensity(cell, qp) = charge*Lambda2;
        electronDensity(cell, qp) = eDensity;
        holeDensity(cell, qp) = hDensity;
        electricPotential(cell, qp) = phi*V0 - qPhiRef;
        ionizedDopant(cell, qp) = ionN;
        conductionBand(cell, qp) = qPhiRef-Chi-phi*V0; // [eV]
        valenceBand(cell, qp) = conductionBand(cell,qp)-Eg;
        approxQuanEDen(cell,qp) = 0.0; 
        artCBDensity(cell, qp) = ( eDensity > 1e-6 ? eDensity 
				     : -Nc*(this->*carrStat)( -(phi+eArgOffset) ));
      }
      
     } // end of if ( (coord0 > oxideWidth) ...)


     // Oxide region
     else if ((coord0 >= 0) && (coord0 <= oxideWidth))
     {
      const std::string matName = "SiliconDioxide" ;  
      double Eg = materialDB->getMaterialParam<double>(matName,"Band Gap",0.0);
      double Chi = materialDB->getMaterialParam<double>(matName,"Electron Affinity",0.0);

      //! parameters for computing exchange-correlation potential
      double ml = materialDB->getMaterialParam<double>(matName,"Longitudinal Electron Effective Mass");
      double mt = materialDB->getMaterialParam<double>(matName,"Transverse Electron Effective Mass");        
      double invEffMass = (2.0/mt + 1.0/ml) / 3.0;
      double averagedEffMass = 1.0 / invEffMass; 
      double relPerm = materialDB->getMaterialParam<double>(matName,"Permittivity");
     
      ScalarT fixedCharge = 0.0; // [cm^-3]

      //! Schrodinger source for electrons
      if(quantumRegionSource == "schrodinger")
      {
#if defined(ALBANY_EPETRA)
        // retrieve Previous Poisson Potential
        ScalarT prevPhi = 0.0, approxEDensity = 0.0;
        
        if (bUsePredictorCorrector) {
	  Albany::MDArray prevPhiArray = (*workset.stateArrayPtr)["PS Previous Poisson Potential"];
	  prevPhi = prevPhiArray(cell,qp);

          // compute the approximate quantum electron density using predictor-corrector method
          approxEDensity = eDensityForPoissonSchrodinger(workset, cell, qp, prevPhi, true, 0.0, -1.0);
	}
        else
          approxEDensity = eDensityForPoissonSchrodinger(workset, cell, qp, 0.0, false, 0.0, -1.0);

        // compute the exact quantum electron density
        ScalarT eDensity = eDensityForPoissonSchrodinger(workset, cell, qp, 0.0, false, 0.0, -1.0);

        // obtain the scaled potential
        const ScalarT& unscaled_phi = potential(cell,qp);
        ScalarT phi = unscaled_phi / V0; 

        //(No other classical density in insulator)
              
        // the scaled full RHS
        ScalarT charge;
        charge = 1.0/Lambda2 * (-approxEDensity + fixedCharge);
        poissonSource(cell, qp) = factor*charge;

        // output states
        chargeDensity(cell, qp) = -eDensity + fixedCharge; 
        electronDensity(cell, qp) = eDensity;  // quantum electrons in an insulator
        holeDensity(cell, qp) = 0.0;           // no holes in an insulator
        ionizedDopant(cell, qp) = 0.0;
        approxQuanEDen(cell,qp) = approxEDensity;
        artCBDensity(cell, qp) = eDensity;

        if (bIncludeVxc)  // include Vxc
        {
          ScalarT Vxc = computeVxcLDA(relPerm, averagedEffMass, approxEDensity);
          conductionBand(cell, qp) = qPhiRef-Chi-phi*V0 +Vxc; // [eV]
	  electricPotential(cell, qp) = phi*V0 + Vxc - qPhiRef; //add xc correction to electric potential (used in CI delta_ij computation)
          // conductionBand(cell, qp) = qPhiRef -Chi -0.5*(phi*V0 +prevPhi) +Vxc; // [eV]
        }
        else  { // not include Vxc
          conductionBand(cell, qp) = qPhiRef-Chi-phi*V0; // [eV]
	  electricPotential(cell, qp) = phi*V0 - qPhiRef;
          // conductionBand(cell, qp) = qPhiRef -Chi -0.5*(phi*V0 +prevPhi); // [eV]
	}
        
        valenceBand(cell, qp) = conductionBand(cell,qp)-Eg;
#endif
      }
    
      else  // use semiclassical source
      {  
        const ScalarT& unscaled_phi = potential(cell,qp);
        ScalarT phi = unscaled_phi / V0; 
          
        // the scaled full RHS
        ScalarT charge; 
        charge = 1.0/Lambda2 * fixedCharge;  // only fixed charge in an insulator
        poissonSource(cell, qp) = factor*charge;
	  
        chargeDensity(cell, qp) = fixedCharge; // fixed space charge in an insulator
        electronDensity(cell, qp) = 0.0;       // no electrons in an insulator
        holeDensity(cell, qp) = 0.0;           // no holes in an insulator
        electricPotential(cell, qp) = phi*V0 - qPhiRef;
        ionizedDopant(cell, qp) = 0.0;
        conductionBand(cell, qp) = qPhiRef-Chi-phi*V0; // [eV]
        valenceBand(cell, qp) = conductionBand(cell,qp)-Eg;
        approxQuanEDen(cell,qp) = 0.0;
        artCBDensity(cell, qp) = 0.0;
      }
     
     } // end of else if ((coord0 >= 0) ...)
     
     else 
      TEUCHOS_TEST_FOR_EXCEPTION (true, Teuchos::Exceptions::InvalidParameter,
	       std::endl << "Error!  x-coord:" << coord0 << "is outside the oxideWidth" << 
	       " + siliconWidth range: " << oxideWidth + siliconWidth << "!"<< std::endl);

    }  // end of loop over QPs
    
  }  // end of loop over cells
  
}





//! ----------------- Poisson source setup and fill functions ---------------------


// **********************************************************************
template<typename EvalT, typename Traits>
typename QCAD::PoissonSource<EvalT,Traits>::PoissonSourceSetupInfo 
QCAD::PoissonSource<EvalT, Traits>::source_setup(const std::string& sourceName, const std::string& mtrlCategory, 
						 const typename Traits::EvalData workset)
{
  PoissonSourceSetupInfo ret;

  double X0 = length_unit_in_m/1e-2; // length scaling to get to [cm] (structure dimension in [um] usually)
  ScalarT temperature = temperatureField(0); //get shared temperature parameter from field
  ret.kbT = kbBoltz*temperature / energy_unit_in_eV; // in [myV]

  //! Constant energy reference for heterogeneous structures
  ret.qPhiRef = getReferencePotential(workset); // in [myV]

  ret.V0 = kbBoltz*temperature / energy_unit_in_eV; // kb*T in desired energy unit ( or kb*T/q in desired voltage unit) [myV]
  ret.Lambda2 = eps0/(eleQ*X0*X0);  // derived scaling factor

  if(mtrlCategory == "Semiconductor") {

    //! temperature-independent material parameters
    double mdn = materialDB->getElementBlockParam<double>(workset.EBName,"Electron DOS Effective Mass");
    double mdp = materialDB->getElementBlockParam<double>(workset.EBName,"Hole DOS Effective Mass");
    double Tref = materialDB->getElementBlockParam<double>(workset.EBName,"Reference Temperature"); // in [K]
    
    ret.Chi = materialDB->getElementBlockParam<double>(workset.EBName,"Electron Affinity") / energy_unit_in_eV; // in [myV]
    double Eg0 = materialDB->getElementBlockParam<double>(workset.EBName,"Zero Temperature Band Gap") / energy_unit_in_eV; // in [myV]
    double alpha = materialDB->getElementBlockParam<double>(workset.EBName,"Band Gap Alpha Coefficient"); // in [eV]
    double beta = materialDB->getElementBlockParam<double>(workset.EBName,"Band Gap Beta Coefficient"); // in [K]
    
    //! constant prefactor in calculating Nc and Nv in [cm-3]
    double NcvFactor = 2.0*pow((kbBoltz*eleQ*m0*Tref)/(2*pi*pow(hbar,2)),3./2.)*1e-6;
            // eleQ converts kbBoltz in [eV/K] to [J/K], 1e-6 converts [m-3] to [cm-3]
                
    ret.Nc = NcvFactor*pow(mdn,1.5)*pow(temperature/Tref,1.5);  // in [cm-3]
    ret.Nv = NcvFactor*pow(mdp,1.5)*pow(temperature/Tref,1.5); 
    ret.Eg = Eg0 - (alpha*pow(temperature,2.0)/(beta+temperature)) / energy_unit_in_eV; // in [myV]
    

    //ni = sqrt(Nc*Nv)*exp(-Eg/(2.0*kbT));    // in [cm-3]

    //! parameters for computing exchange-correlation potential
    const std::string& condBandMin = materialDB->getElementBlockParam<std::string>(workset.EBName,"Conduction Band Minimum");
    double ml = materialDB->getElementBlockParam<double>(workset.EBName,"Longitudinal Electron Effective Mass");
    double mt = materialDB->getElementBlockParam<double>(workset.EBName,"Transverse Electron Effective Mass");        
    if ((condBandMin == "Gamma Valley") && (std::abs(ml-mt) > 1e-10))
      TEUCHOS_TEST_FOR_EXCEPTION (true, std::logic_error, "Gamma Valley's longitudinal and "
        << "transverse electron effective mass must be equal ! "
        << "Please check the values in materials.xml" << std::endl);
    
    double invEffMass = (2.0/mt + 1.0/ml) / 3.0;
    ret.averagedEffMass = 1.0 / invEffMass; 
    ret.relPerm = materialDB->getElementBlockParam<double>(workset.EBName,"Permittivity");
    
    //! argument offset in calculating electron and hole density
    ret.eArgOffset = (-ret.qPhiRef+ret.Chi)/ret.kbT;
    ret.hArgOffset = (ret.qPhiRef-ret.Chi-ret.Eg)/ret.kbT;
        
    if (carrierStatistics == "Boltzmann Statistics")
      ret.carrStat = &QCAD::PoissonSource<EvalT,Traits>::computeMBStat;  

    else if (carrierStatistics == "Fermi-Dirac Statistics")
      ret.carrStat = &QCAD::PoissonSource<EvalT,Traits>::computeFDIntOneHalf;

    else if (carrierStatistics == "0-K Fermi-Dirac Statistics")
      ret.carrStat = &QCAD::PoissonSource<EvalT,Traits>::computeZeroKFDInt;

    else TEUCHOS_TEST_FOR_EXCEPTION (true, Teuchos::Exceptions::InvalidParameter,
        std::endl << "Error!  Unknown carrier statistics ! " << std::endl);
    
    if (incompIonization == "False")
      ret.ionDopant = &QCAD::PoissonSource<EvalT,Traits>::fullDopants; 
    
    else if (incompIonization == "True")
      ret.ionDopant = &QCAD::PoissonSource<EvalT,Traits>::ionizedDopants;
    
    else TEUCHOS_TEST_FOR_EXCEPTION (true, Teuchos::Exceptions::InvalidParameter,
        std::endl << "Error!  Invalid incomplete ionization option ! " << std::endl);

    //! obtain the fermi energy in a given element block

    //  Each element block must have an associated Fermi level, otherwise
    //  the Poisson source term cannot be computed.  Since QCAD does not implement
    //  drift-diffusion equations, this Fermi level must be specified directly.  This 
    //  can be done in several ways:
    //   1) a nodeset with a DBC (e.g. a contact) specifies "elementBlock" in its materials db list
    //   2) an element block itself specifies "contactNodeset" in its materials db list
    //   3) if neither 1) nor 2) hold, a default value of zero is used as the Fermi level.

    ret.fermiE = 0.0;  // default, [myV]
    if (mapDBCValue_eb.count(workset.EBName) > 0) 
    {
      ret.fermiE = -1.0*mapDBCValue_eb[workset.EBName] / energy_unit_in_eV; // [myV] (DBCs are in volts, regardless of desired output energy unit)
      // std::cout << "EBName = " << workset.EBName << ", ret.fermiE = " << ret.fermiE << std::endl; 
    }
    else if(materialDB->isElementBlockParam(workset.EBName, "contactNodeset"))
    {
      std::string nsName = materialDB->getElementBlockParam<std::string>(workset.EBName, "contactNodeset");
      if(mapDBCValue_ns.count(nsName) > 0)
	ret.fermiE = -1.0*mapDBCValue_ns[nsName] / energy_unit_in_eV; // [myV]
    }

    //! get doping concentration and activation energy
    //** Note: doping profile unused currently
    ret.fixedChargeType = materialDB->getElementBlockParam<std::string>(workset.EBName,"Dopant Type","None");
    ret.fixedChargeConc = 0.0; //only applies to insulators
    std::string dopingProfile;

    if(ret.fixedChargeType != "None") {
      double dopantActE;
      dopingProfile = materialDB->getElementBlockParam<std::string>(workset.EBName,"Doping Profile","Constant");
      dopantActE = materialDB->getElementBlockParam<double>(workset.EBName,"Dopant Activation Energy",0.045) / energy_unit_in_eV; // [myV]
    
      if( materialDB->isElementBlockParam(workset.EBName, "Doping Value") ) 
        ret.dopingConc = materialDB->getElementBlockParam<double>(workset.EBName,"Doping Value");
      else if( materialDB->isElementBlockParam(workset.EBName, "Doping Parameter Name") ) {
        double scl = materialDB->getElementBlockParam<double>(workset.EBName,"Doping Parameter Scaling", 1.0);
        ret.dopingConc = materialParams[ materialDB->getElementBlockParam<std::string>(workset.EBName,"Doping Parameter Name") ] * scl;
      }
      else TEUCHOS_TEST_FOR_EXCEPTION (true, Teuchos::Exceptions::InvalidParameter,
        std::endl << "Error!  Unknown dopant concentration for " << workset.EBName << "!"<< std::endl);

      if(ret.fixedChargeType == "Donor") 
        ret.inArg = ret.eArgOffset + dopantActE/ret.kbT + ret.fermiE/ret.kbT;
      else if(ret.fixedChargeType == "Acceptor") 
        ret.inArg = ret.hArgOffset + dopantActE/ret.kbT - ret.fermiE/ret.kbT;
      else TEUCHOS_TEST_FOR_EXCEPTION (true, Teuchos::Exceptions::InvalidParameter,
	       std::endl << "Error!  Unknown dopant type " << ret.fixedChargeType << "!"<< std::endl);
    }
    else {
      dopingProfile = "Constant";
      ret.dopingConc = 0.0;
      ret.inArg = 0.0;
    }

  } // end "Semiconductor" setup
  
  else if(mtrlCategory == "Insulator")
  {  
    ret.Eg = materialDB->getElementBlockParam<double>(workset.EBName,"Band Gap",0.0) / energy_unit_in_eV; // [myV]
    ret.Chi = materialDB->getElementBlockParam<double>(workset.EBName,"Electron Affinity",0.0) / energy_unit_in_eV; // [myV]
    ret.carrStat = &QCAD::PoissonSource<EvalT,Traits>::computeZeroStat; //always zero  

    //Unused in insulator.  Set as zero
    ret.Nc = ret.Nv = 0.0;
    ret.hArgOffset = ret.eArgOffset = 0.0;
    ret.fermiE = 0.0;
    ret.dopingConc = 0.0; //only applies to semiconductors

    //! Fixed charge in insulator
    if( materialDB->isElementBlockParam(workset.EBName, "Charge Value") ) {
      ret.fixedChargeType = "Constant";
      ret.fixedChargeConc = materialDB->getElementBlockParam<double>(workset.EBName,"Charge Value");
      //std::cout << "DEBUG: applying fixed charge " << ret.fixedChargeConc << " to element block '" << workset.EBName << "'" << std::endl;
    }
    else if( materialDB->isElementBlockParam(workset.EBName, "Charge Parameter Name") ) { 
      double scl = materialDB->getElementBlockParam<double>(workset.EBName,"Charge Parameter Scaling", 1.0);
      ret.fixedChargeType = "Constant";
      ret.fixedChargeConc = materialParams[ materialDB->getElementBlockParam<std::string>(workset.EBName,"Charge Parameter Name") ] * scl;
      //std::cout << "DEBUG: applying fixed charge " << ret.fixedChargeConc << " to element block '" << workset.EBName << "' via param" << std::endl;
    }
    else {
      ret.fixedChargeType = "None";
      ret.fixedChargeConc = 0.0; 
    }

    //! parameters for computing exchange-correlation potential
    double ml = materialDB->getElementBlockParam<double>(workset.EBName,"Longitudinal Electron Effective Mass");
    double mt = materialDB->getElementBlockParam<double>(workset.EBName,"Transverse Electron Effective Mass");        
    if (std::abs(ml-mt) > 1e-10) 
      TEUCHOS_TEST_FOR_EXCEPTION (true, std::logic_error, "Insulator's longitudinal and "
	       << "transverse electron effective mass must be equal ! "
	       << "Please check the values in materials.xml" << std::endl);

    double invEffMass = (2.0/mt + 1.0/ml) / 3.0;
    ret.averagedEffMass = 1.0 / invEffMass; 
    ret.relPerm = materialDB->getElementBlockParam<double>(workset.EBName,"Permittivity");
  } // end "Insulator" setup


  else if(mtrlCategory == "Metal")
  {  
    // Use work function where semiconductor and insulator use electron affinity
    ret.Chi = materialDB->getElementBlockParam<double>(workset.EBName,"Work Function") / energy_unit_in_eV; // [myV]
    ret.Eg = 0.0;  //no bandgap in metals
  } // end "Metal" setup

  else {
    TEUCHOS_TEST_FOR_EXCEPTION (true, Teuchos::Exceptions::InvalidParameter,
		        std::endl << "Error!  Unknown material category " 
			<< mtrlCategory << "!" << std::endl);
  }

  // Previous electron density -- used for daming PS iterations
  ret.prevDensityFactor = QCAD::EvaluatorTools<EvalT,Traits>::getDoubleValue( prevDensityMixingFactor );

  //HACK
  static double lastNonDefaultFactor = -1.0;
  if(ret.prevDensityFactor < 0.0) { //factor was not set by parameters, which is probably an interal Albany bug, so set using last non-default param
    ret.prevDensityFactor = lastNonDefaultFactor;
    //std::cout << "WARNING: prevDensityMixingFactor was not set via a parameter - setting to last non-default = " << lastNonDefaultFactor << std::endl;
  }
  if(ret.prevDensityFactor != lastNonDefaultFactor) {
    std::cout << "DEBUG: setup prevDensityMixingFactor = " << ret.prevDensityFactor 
	      << " (lastDef = " << lastNonDefaultFactor << ", scalarT = " << prevDensityMixingFactor << ")" << std::endl;
    lastNonDefaultFactor = ret.prevDensityFactor;
  }
  //END HACK

  if(ret.prevDensityFactor > 1e-8)
    ret.prevDensityArray = (*workset.stateArrayPtr)["PS Previous Electron Density"];

  ret.quantum_edensity_fn = NULL; //default

  //Fill additional members for quantum and coulomb sources
  if(sourceName == "schrodinger") {
#if defined(ALBANY_EPETRA)
    if(bUsePredictorCorrector) // retrieve Previous Poisson Potential
      ret.prevPhiArray = (*workset.stateArrayPtr)["PS Previous Poisson Potential"]; //assumed in [myV]

    ret.quantum_edensity_fn = &QCAD::PoissonSource<EvalT,Traits>::eDensityForPoissonSchrodinger;
#endif
  }
  else if(sourceName == "ci") {
#if defined(ALBANY_EPETRA)
    if(bUsePredictorCorrector) // retrieve Previous Poisson Potential
      ret.prevPhiArray = (*workset.stateArrayPtr)["PS Previous Poisson Potential"]; //assumed in [myV]

    ret.quantum_edensity_fn = &QCAD::PoissonSource<EvalT,Traits>::eDensityForPoissonCI;
#endif
  }
  else if(sourceName == "coulomb")  {
    //RHS == evec[i] * evec[j]
    ret.sourceEvec1 = (int)QCAD::EvaluatorTools<EvalT,Traits>::getDoubleValue( sourceEvecInds[0] );
    ret.sourceEvec2 = (int)QCAD::EvaluatorTools<EvalT,Traits>::getDoubleValue( sourceEvecInds[1] );
    
    //int valleyDegeneracyFactor = materialDB->getElementBlockParam<int>(workset.EBName,"Number of conduction band min",2);
    // scale so electron density is in [cm^-3] (assume 3D? Suzey?) as expected of RHS of Poisson eqn
    ret.coulombPrefactor = 1.0/pow(X0,(int)numDims);
  }

  return ret;
}


// **********************************************************************
template<typename EvalT, typename Traits>
void QCAD::PoissonSource<EvalT, Traits>::
source_semiclassical(const typename Traits::EvalData workset, std::size_t cell, std::size_t qp, const ScalarT& scaleFactor,
		     const PoissonSourceSetupInfo& setup_info)
{
  // -- Semiconductor
  const ScalarT& unscaled_phi = potential(cell,qp);  // [myV]
  ScalarT phi = unscaled_phi / setup_info.V0; 
          
  // obtain the ionized dopants (in semiconductor) or fixed charge (in insulator)
  ScalarT fixedCharge;
  if (setup_info.fixedChargeType == "Donor")  // function takes care of sign
    fixedCharge = (this->*(setup_info.ionDopant))("Donor",phi + setup_info.inArg)*setup_info.dopingConc;
  else if (setup_info.fixedChargeType == "Acceptor")
    fixedCharge = (this->*(setup_info.ionDopant))("Acceptor",-phi + setup_info.inArg)*setup_info.dopingConc;
  else if (setup_info.fixedChargeType == "Constant")
    fixedCharge = setup_info.fixedChargeConc;
  else 
    fixedCharge = 0.0; 
  
  // the scaled full RHS
  ScalarT charge, eDensity, hDensity; 
  eDensity = setup_info.Nc*(this->*(setup_info.carrStat))(phi + setup_info.eArgOffset + setup_info.fermiE/setup_info.kbT);
  hDensity = setup_info.Nv*(this->*(setup_info.carrStat))(-phi + setup_info.hArgOffset - setup_info.fermiE/setup_info.kbT);

  // Mix with previous density (to damp S-P iterations)
  if(setup_info.prevDensityFactor > 1e-8) { 
    ScalarT prevDensity = setup_info.prevDensityArray(cell,qp); 
    //eDensity = eDensity * (1 - setup_info.prevDensityFactor) + setup_info.prevDensityFactor * prevDensity;
  }

  charge = 1.0/setup_info.Lambda2 * (hDensity - eDensity + fixedCharge);
  poissonSource(cell, qp) = scaleFactor*charge;
  
  //DEBUG
  /*if(isnan( QCAD::EvaluatorTools<EvalT,Traits>::getDoubleValue( poissonSource(cell,qp) )))
    std::cout << "semicl source("<<cell<<","<<qp<<") is NAN" << std::endl;
  if(isinf( QCAD::EvaluatorTools<EvalT,Traits>::getDoubleValue( poissonSource(cell,qp) )))
    std::cout << "semicl source("<<cell<<","<<qp<<") is INF -- h=" << hDensity << ", e="<< eDensity << ", i=" << fixedCharge
	      << ", phi=" << phi << ", eArg=" << setup_info.eArgOffset << ", fermiE=" << setup_info.fermiE 
	      << ", kbT=" << setup_info.kbT << std::endl;
  */
  
  // output states
  chargeDensity(cell, qp) = hDensity - eDensity + fixedCharge;
  electronDensity(cell, qp) = eDensity;
  holeDensity(cell, qp) = hDensity;
  electricPotential(cell, qp) = phi*setup_info.V0 - setup_info.qPhiRef; // [myV]
  ionizedDopant(cell, qp) = fixedCharge;
  conductionBand(cell, qp) = setup_info.qPhiRef-setup_info.Chi-phi*setup_info.V0; // [myV]
  valenceBand(cell, qp) = conductionBand(cell,qp)-setup_info.Eg; // [myV]
  approxQuanEDen(cell,qp) = 0.0; 
  artCBDensity(cell, qp) = ( eDensity > 1e-6 ? eDensity
			     : -setup_info.Nc*(this->*(setup_info.carrStat))( -(phi+setup_info.eArgOffset) ));
}


// **********************************************************************
template<typename EvalT, typename Traits>
void QCAD::PoissonSource<EvalT, Traits>::
source_none(const typename Traits::EvalData workset, std::size_t cell, std::size_t qp, const ScalarT& scaleFactor,
		     const PoissonSourceSetupInfo& setup_info)
{
  const ScalarT& unscaled_phi = potential(cell,qp);  //[myV]
  ScalarT phi = unscaled_phi / setup_info.V0;  //[unitless]
  
  // the scaled full RHS
  ScalarT charge = 0.0;  // no charge in this RHS mode
  poissonSource(cell, qp) = scaleFactor*charge;
  
  chargeDensity(cell, qp) = 0.0;         // no charge in this RHS mode
  electronDensity(cell, qp) = 0.0;       // no electron
  holeDensity(cell, qp) = 0.0;           // no holes
  electricPotential(cell, qp) = phi*setup_info.V0 - setup_info.qPhiRef;
  ionizedDopant(cell, qp) = 0.0;
  conductionBand(cell, qp) = setup_info.qPhiRef-setup_info.Chi-phi*setup_info.V0; // [myV]
  valenceBand(cell, qp) = conductionBand(cell,qp)-setup_info.Eg;
  approxQuanEDen(cell,qp) = 0.0;
  artCBDensity(cell, qp) = 0.0;
}


// **********************************************************************
template<typename EvalT, typename Traits>
void QCAD::PoissonSource<EvalT, Traits>::
source_quantum(const typename Traits::EvalData workset, std::size_t cell, std::size_t qp, const ScalarT& scaleFactor,
		     const PoissonSourceSetupInfo& setup_info)
{
  // -- Semiconductor, but see "insulator" comments
  ScalarT approxEDensity = 0.0;
          
  if(bUsePredictorCorrector) {
    // compute the approximate quantum electron density using predictor-corrector method
    ScalarT prevPhi = setup_info.prevPhiArray(cell,qp); 
    approxEDensity = (this->*(setup_info.quantum_edensity_fn))(workset, cell, qp, prevPhi, true, setup_info.fermiE, fixedQuantumOcc);
  }
  else  // otherwise, use the exact quantum density
    approxEDensity = (this->*(setup_info.quantum_edensity_fn))(workset, cell, qp, 0.0, false, setup_info.fermiE, fixedQuantumOcc);

  // compute the exact quantum electron density
  ScalarT eDensity = (this->*(setup_info.quantum_edensity_fn))(workset, cell, qp, 0.0, false, setup_info.fermiE, fixedQuantumOcc);

  // Mix with previous density (to damp S-P iterations)
  if(setup_info.prevDensityFactor > 1e-8) { 
    ScalarT prevDensity = setup_info.prevDensityArray(cell,qp); 
    approxEDensity = approxEDensity * (1.0 - setup_info.prevDensityFactor) + setup_info.prevDensityFactor * prevDensity;
    eDensity = eDensity * (1.0 - setup_info.prevDensityFactor) + setup_info.prevDensityFactor * prevDensity;
  }

  // obtain the scaled potential
  const ScalarT& unscaled_phi = potential(cell,qp); //[myV]
  ScalarT phi = unscaled_phi / setup_info.V0; 
           
  // compute the hole density treated as classical
  ScalarT hDensity = setup_info.Nv*(this->*(setup_info.carrStat))(-phi + setup_info.hArgOffset);

  // obtain the ionized dopants (in semiconductor) or fixed charge (in insulator)
  ScalarT fixedCharge  = 0.0;
  if (setup_info.fixedChargeType == "Donor")  // function takes care of sign
    fixedCharge = (this->*(setup_info.ionDopant))("Donor",phi + setup_info.inArg)*setup_info.dopingConc;
  else if (setup_info.fixedChargeType == "Acceptor")
    fixedCharge = (this->*(setup_info.ionDopant))("Acceptor",-phi + setup_info.inArg)*setup_info.dopingConc;
  else if (setup_info.fixedChargeType == "Constant")
    fixedCharge = setup_info.fixedChargeConc;
  else 
    fixedCharge = 0.0; 
              
  // the scaled full RHS
  ScalarT charge; 
  charge = 1.0/setup_info.Lambda2*(hDensity- approxEDensity + fixedCharge);
  poissonSource(cell, qp) = scaleFactor*charge;

  //DEBUG
  /*if(isnan( QCAD::EvaluatorTools<EvalT,Traits>::getDoubleValue( poissonSource(cell,qp) )))
    std::cout << "quantum source("<<cell<<","<<qp<<") is NAN" << std::endl;
    if(isinf( QCAD::EvaluatorTools<EvalT,Traits>::getDoubleValue( poissonSource(cell,qp) )))
    std::cout << "quantum source("<<cell<<","<<qp<<") is INF" << std::endl;
  */

  // output states
  chargeDensity(cell, qp) = hDensity -eDensity +fixedCharge;
  electronDensity(cell, qp) = eDensity;
  holeDensity(cell, qp) = hDensity;
  //electricPotential(cell, qp) = phi*V0 - qPhiRef; //electric potenial == solution shifted so "reference" == 0
  ionizedDopant(cell, qp) = fixedCharge;
  approxQuanEDen(cell,qp) = approxEDensity;
  artCBDensity(cell, qp) = ( eDensity > 1e-6 ? eDensity : -setup_info.Nc*(this->*(setup_info.carrStat))( -(phi+setup_info.eArgOffset) ));

  if (bIncludeVxc)  // include Vxc
  {
    ScalarT Vxc = computeVxcLDA(setup_info.relPerm, setup_info.averagedEffMass, approxEDensity) / energy_unit_in_eV; // [myV]
    conductionBand(cell, qp) = setup_info.qPhiRef -setup_info.Chi -phi*setup_info.V0 +Vxc; // [myV]
              
    // Suzey: need to be discussed (April 06, 2012) ? 
    electricPotential(cell, qp) = phi*setup_info.V0 + Vxc- setup_info.qPhiRef; //add xc correction to electric potential (used in CI delta_ij computation)
  }
  else  // not include Vxc
  {
    conductionBand(cell, qp) = setup_info.qPhiRef -setup_info.Chi -phi*setup_info.V0; // [myV]
    electricPotential(cell, qp) = phi*setup_info.V0 - setup_info.qPhiRef; // [myV]
  }
  valenceBand(cell, qp) = conductionBand(cell,qp)-setup_info.Eg;
}


// **********************************************************************
template<typename EvalT, typename Traits>
void QCAD::PoissonSource<EvalT, Traits>::
source_coulomb(const typename Traits::EvalData workset, std::size_t cell, std::size_t qp, const ScalarT& scaleFactor,
		     const PoissonSourceSetupInfo& setup_info)
{
  // obtain the scaled potential
  const ScalarT& unscaled_phi = potential(cell,qp); //[V]
  ScalarT phi = unscaled_phi / setup_info.V0; 

  // the scaled full RHS   note: wavefunctions are assumed normalized by the mass matrix (i.e. integral( Psi_i^2 dR ) == 1)
  // Source term is conj(evec_j) * evec_i  
  // TODO: double-check this is correct, seeing as eigenvectors are normalized by mass matrix
  ScalarT charge;
  int i = setup_info.sourceEvec1;
  int j = setup_info.sourceEvec2;
  if(imagPartOfCoulombSrc) {
    if(!bRealEigenvectors) // if eigenvectors are all real, then there is no imaginary part of coulomb source
      charge = - setup_info.coulombPrefactor * ( eigenvector_Re[i](cell,qp) * eigenvector_Im[j](cell,qp) - 
						 eigenvector_Im[i](cell,qp) * eigenvector_Re[j](cell,qp));
    else charge = 0.0;
  }
  else {
    if(bRealEigenvectors)
      charge = - setup_info.coulombPrefactor * ( eigenvector_Re[i](cell,qp) * eigenvector_Re[j](cell,qp) );
    else
      charge = - setup_info.coulombPrefactor * ( eigenvector_Re[i](cell,qp) * eigenvector_Re[j](cell,qp) + 
						 eigenvector_Im[i](cell,qp) * eigenvector_Im[j](cell,qp));
  }


  poissonSource(cell, qp) = scaleFactor * 1.0/setup_info.Lambda2 * charge;

  chargeDensity(cell, qp) = charge;
  electronDensity(cell, qp) = charge;
  holeDensity(cell, qp) = 0.0;
  electricPotential(cell, qp) = phi*setup_info.V0 - setup_info.qPhiRef;

  //never include Vxc
  conductionBand(cell, qp) = setup_info.qPhiRef -setup_info.Chi -phi*setup_info.V0; // [eV]
  valenceBand(cell, qp) = conductionBand(cell,qp)-setup_info.Eg;
}

// **********************************************************************
template<typename EvalT, typename Traits>
void QCAD::PoissonSource<EvalT, Traits>::
source_testcoulomb(const typename Traits::EvalData workset, std::size_t cell, std::size_t qp, const ScalarT& scaleFactor,
		     const PoissonSourceSetupInfo& setup_info)
{
  // obtain the scaled potential
  const ScalarT& unscaled_phi = potential(cell,qp); //[myV]
  ScalarT phi = unscaled_phi / setup_info.V0; 
  MeshScalarT coord0 = coordVec(cell,qp,0);
  MeshScalarT coord1 = coordVec(cell,qp,1);
  MeshScalarT coord2 = coordVec(cell,qp,2);

  // the scaled full RHS   Source term is x^2 + y^2 + z^2 (for debugging purposes)
  ScalarT charge = setup_info.coulombPrefactor * ( exp(-(coord0*coord0 + coord1*coord1 + coord2*coord2)));

  poissonSource(cell, qp) = scaleFactor * 1.0/setup_info.Lambda2 * charge; //sign??

  chargeDensity(cell, qp) = charge;
  electronDensity(cell, qp) = charge;
  holeDensity(cell, qp) = 0.0;
  electricPotential(cell, qp) = phi*setup_info.V0 - setup_info.qPhiRef;

  //never include Vxc
  conductionBand(cell, qp) = setup_info.qPhiRef-setup_info.Chi-phi*setup_info.V0; // [myV]
  valenceBand(cell, qp) = conductionBand(cell,qp)-setup_info.Eg;
}




//! ----------------- Carrier statistics functions ---------------------


// **********************************************************************
template<typename EvalT,typename Traits>
inline typename QCAD::PoissonSource<EvalT,Traits>::ScalarT
QCAD::PoissonSource<EvalT,Traits>::computeMBStat(const ScalarT x)
{
   return exp(x);
}


// **********************************************************************
template<typename EvalT,typename Traits>
inline typename QCAD::PoissonSource<EvalT,Traits>::ScalarT
QCAD::PoissonSource<EvalT,Traits>::computeFDIntOneHalf(const ScalarT x)
{
   // Use the approximate 1/2 FD integral by D. Bednarczyk and J. Bednarczyk, 
   // "The approximation of the Fermi-Dirac integral F_{1/2}(x),"
   // Physics Letters A, vol.64, no.4, pp.409-410, 1978. The approximation 
   // has error < 4e-3 in the entire x range.  
   
   ScalarT fdInt; 
   if (x >= -50.0)
   {
     fdInt = pow(x,4.) + 50. + 33.6*x*(1.-0.68*exp(-0.17*pow((x+1.),2.0)));
     fdInt = pow((exp(-x) + (3./4.*sqrt(pi)) * pow(fdInt, -3./8.)),-1.0);
   }      
   else
     fdInt = exp(x); // for x<-50, the 1/2 FD integral is well approximated by exp(x)
     
   return fdInt;
}


// **********************************************************************
template<typename EvalT,typename Traits>
inline typename QCAD::PoissonSource<EvalT,Traits>::ScalarT
QCAD::PoissonSource<EvalT,Traits>::computeZeroKFDInt(const ScalarT x)
{
   ScalarT zeroKFDInt;
   if (x > 0.0) 
     zeroKFDInt = 4./3./sqrt(pi)*pow(x, 3./2.);
   else
     zeroKFDInt = 0.0;
      
   return zeroKFDInt;
}

// **********************************************************************
template<typename EvalT,typename Traits>
inline typename QCAD::PoissonSource<EvalT,Traits>::ScalarT
QCAD::PoissonSource<EvalT,Traits>::computeZeroStat(const ScalarT x)
{
   return 0;
}




//! ----------------- Activated dopant concentration functions ---------------------


// **********************************************************************
template<typename EvalT,typename Traits>
inline typename QCAD::PoissonSource<EvalT,Traits>::ScalarT
QCAD::PoissonSource<EvalT,Traits>::fullDopants(const std::string dopType, const ScalarT &x)
{
  ScalarT ionDopants;

  // fully ionized (create function to use function pointer)
  if (dopType == "Donor")
    ionDopants = 1.0;
  else if (dopType == "Acceptor")
    ionDopants = -1.0;
  else if (dopType == "None")
    ionDopants = 0.0;
  else
  {
    ionDopants = 0.0;
    TEUCHOS_TEST_FOR_EXCEPTION (true, Teuchos::Exceptions::InvalidParameter,
       std::endl << "Error!  Unknown dopant type " << dopType << "!"<< std::endl);
  }

  return ionDopants;  
}


// **********************************************************************
template<typename EvalT,typename Traits>
typename QCAD::PoissonSource<EvalT,Traits>::ScalarT
QCAD::PoissonSource<EvalT,Traits>::ionizedDopants(const std::string dopType, const ScalarT &x)
{
  ScalarT ionDopants;
  
  if (dopType == "Donor")
  {
    if (x > MAX_EXPONENT)
      ionDopants = 0.5 * exp(-x);  // use Boltzman statistics for large positive x, 
    else                           // as the Fermi statistics leads to bad derivative.
      ionDopants = 1.0 / (1. + 2.*exp(x));  
  }  

  else if (dopType == "Acceptor")
  {
    if (x > MAX_EXPONENT)
      ionDopants = -0.25 * exp(-x);
    else
      ionDopants = -1.0 / (1. + 4.*exp(x));
  }  

  else if (dopType == "None")
    ionDopants = 0.0;
  else
  {
    ionDopants = 0.0;
    TEUCHOS_TEST_FOR_EXCEPTION (true, Teuchos::Exceptions::InvalidParameter,
       std::endl << "Error!  Unknown dopant type " << dopType << "!"<< std::endl);
  }
   
  return ionDopants; 
}




//! ----------------- Quantum electron density functions ---------------------


#if defined(ALBANY_EPETRA)
// *****************************************************************************
template<typename EvalT, typename Traits>
typename QCAD::PoissonSource<EvalT,Traits>::ScalarT
QCAD::PoissonSource<EvalT,Traits>::eDensityForPoissonSchrodinger 
  (typename Traits::EvalData workset, std::size_t cell, std::size_t qp, 
   const ScalarT prevPhi, const bool bUsePredCorr, const double Ef, const double fixedOcc)
{
  // Use the predictor-corrector method proposed by A. Trellakis, A. T. Galick,  
  // and U. Ravaioli, "Iteration scheme for the solution of the two-dimensional  
  // Schrodinger-Poisson equations in quantum structures", J. Appl. Phys. 81, 7880 (1997). 
  // If bUsePredCorr = true, use the predictor-corrector approach.
  // If bUsePredCorr = false, calculate the exact quantum electron density.
  // Fermi-Dirac distribution is used in computing electron density.  
  
  // unit conversion factor
  double eVPerJ = 1.0/eleQ; 
  double cm2Perm2 = 1.0e4; 
  
  // For Delta2-band in Silicon, valley degeneracy factor = 2
  int valleyDegeneracyFactor = materialDB->getElementBlockParam<int>(workset.EBName,"Number of conduction band min",2);

  // Scaling factors
  double X0 = length_unit_in_m/1e-2; // length scaling to get to [cm] (structure dimension in [um])

  ScalarT temperature = temperatureField(0); //get shared temperature parameter from field
  ScalarT kbT_eV = kbBoltz*temperature;  // in [eV]
  ScalarT kbT = kbBoltz*temperature / energy_unit_in_eV;  // in [myV]
  ScalarT eDensity = 0.0; 

  // assume eigenvalues are in [myV] units (units of energy_unit_in_eV * eV)
  const std::vector<double>& neg_eigenvals = *(workset.eigenDataPtr->eigenvalueRe); 
  std::vector<double> eigenvals( neg_eigenvals );
  for(unsigned int i=0; i<eigenvals.size(); ++i) eigenvals[i] *= -1; //apply minus sign (b/c of eigenval convention)


  // determine deltaPhi used in computing quantum electron density
  ScalarT deltaPhi = 0.0;  // 0 by default
  if (bUsePredCorr)  // true: apply the p-c method
  {
    const ScalarT& phi = potential(cell,qp); //[myV]
    deltaPhi = (phi - prevPhi) / (kbT /1.0);  //[unitless]
  }
  else
    deltaPhi = 0.0;  // false: do not apply the p-c method

  ScalarT eDenPrefactor;
  std::vector<ScalarT> occ( eigenvals.size(), 0.0); //occupation of ith eigenstate
  
  // compute quantum "occupation" according to dimensionality, which is just the coefficient 
  //   of each (wavefunction)^2 term in the density, as well as a prefactor.
  switch (numDims)
  {
    case 1: // 1D wavefunction (1D confinement)
    {  
      // m11 = in-plane effective mass (y-z plane when the 1D wavefunc. is along x)
      // For Delta2-band (or valley), m11 is the transverse effective mass (0.19). 
      
      double m11 = materialDB->getElementBlockParam<double>(workset.EBName,"Transverse Electron Effective Mass");
        
      // 2D density of states in [#/(eV.cm^2)] where 2D is the unconfined y-z plane
      // dos2D below includes the spin degeneracy of 2
      double dos2D = m11*m0/(pi*hbar*hbar*eVPerJ*cm2Perm2); 
        
      // subband-independent prefactor in calculating electron density
      // X0 is used to scale wavefunc. squared from [um^-1] or [nm^-1] to [cm^-1]
      eDenPrefactor = valleyDegeneracyFactor*dos2D*kbT_eV/X0; 

      // loop over eigenvalues to compute the occupation
      for(int i = 0; i < nEigenvectors; i++) 
      {
        ScalarT tmpArg = (Ef-eigenvals[i])/kbT + deltaPhi;
        ScalarT logFunc; 
        if (tmpArg > MAX_EXPONENT)
          logFunc = tmpArg;  // exp(tmpArg) blows up for large positive tmpArg, leading to bad derivative
        else
          logFunc = log(1.0 + exp(tmpArg));
	occ[i] = logFunc;
      }
      break; 
    }  // end of case 1 block


    case 2: // 2D wavefunction (2D confinement)
    {
      // mUnconfined = effective mass in the unconfined direction (z dir. when the 2D wavefunc. is in x-y plane)
      // For Delta2-band and assume SiO2/Si interface parallel to [100] plane, mUnconfined=0.19. 
      double mUnconfined = materialDB->getElementBlockParam<double>(workset.EBName,"Transverse Electron Effective Mass");
        
      // n1D below is a factor that is part of the line electron density 
      // in the unconfined dir. and includes spin degeneracy and in unit of [cm^-1]
      ScalarT n1D = sqrt(2.0*mUnconfined*m0*kbT_eV/(pi*hbar*hbar*eVPerJ*cm2Perm2));
        
      // subband-independent prefactor in calculating electron density
      // X0^2 is used to scale wavefunc. squared from [um^-2] or [nm^-2] to [cm^-2]
      eDenPrefactor = valleyDegeneracyFactor*n1D/pow(X0,2.);

      // loop over eigenvalues to compute the occupation
      for(int i=0; i < nEigenvectors; i++) 
      {
        ScalarT inArg = (Ef-eigenvals[i])/kbT + deltaPhi;
	occ[i] = computeFDIntMinusOneHalf(inArg);
      }
      break;
    }  // end of case 2 block    


    case 3: // 3D wavefunction (3D confinement)
    { 
      //degeneracy factor
      int spinDegeneracyFactor = 2;
      double degeneracyFactor = spinDegeneracyFactor * valleyDegeneracyFactor;
        
      // subband-independent prefactor in calculating electron density
      // X0^3 is used to scale wavefunc. squared from [um^-3] or [nm^-3] to [cm^-3]
      eDenPrefactor = degeneracyFactor/pow(X0,3.);

      // loop over eigenvalues to compute occupation
      for(int i = 0; i < nEigenvectors; i++) 
      {
        // ScalarT tmpArg = (eigenvals[i]-Ef)/kbT + deltaPhi;
        // It is critical to use -deltaPhi for 3D (while +deltaPhi for 1D and 2D) as 
        // derived theoretically. +deltaPhi leads to serious oscillations (tested).  
        
        ScalarT tmpArg = (eigenvals[i]-Ef)/kbT - deltaPhi;
        ScalarT fermiFactor; 
        
        if (tmpArg > MAX_EXPONENT) 
          fermiFactor = exp(-tmpArg);  // use Boltzmann statistics for large positive tmpArg,
        else                           // as the Fermi statistics leads to bad derivative        
          fermiFactor = 1.0/( exp(tmpArg) + 1.0 ); 
	occ[i] = fermiFactor;
      }
      break;
    }  // end of case 3 block 
      
    default:
      TEUCHOS_TEST_FOR_EXCEPTION (true, Teuchos::Exceptions::InvalidParameter,
        std::endl << "Error!  Invalid number of dimensions " << numDims << "!"<< std::endl);
      eDenPrefactor = 0.0; //to avoid compiler warning
      break; 
      
  }  // end of switch (numDims) 

  //Enforce a fixed occupation (non-equilibrium) if desired (indicated by fixedOcc > 0)
  if(fixedOcc > -1e-6) {
    double occLeft = fixedOcc;
    for(int i = 0; i < nEigenvectors; i++) {
      occ[i] = std::min(1.0, occLeft);
      occLeft -= QCAD::EvaluatorTools<EvalT,Traits>::getDoubleValue(occ[i]);
    }
  }

  // loop over eigenvalues to compute electron density [cm^-3]
  for(int i = 0; i < nEigenvectors; i++) 
  {
    // note: wavefunctions are assumed normalized here 
    ScalarT wfSquared;
    if(bRealEigenvectors)
      wfSquared = ( eigenvector_Re[i](cell,qp)*eigenvector_Re[i](cell,qp) );
    else
      wfSquared = ( eigenvector_Re[i](cell,qp)*eigenvector_Re[i](cell,qp) + 
		    eigenvector_Im[i](cell,qp)*eigenvector_Im[i](cell,qp) );
    eDensity += wfSquared*occ[i];
  }
  eDensity = eDenPrefactor*eDensity; // in [cm^-3]

  return eDensity; 
}


//Similar as eDensityForPoissonSchrodinger but assume |wf|^2 values are given in eigenvector_Re[.] instead of eigenvector
// Note: sum of |wf|^2 values == N where N is the number of electrons in the wave function, not necessarily == 1
template<typename EvalT, typename Traits>
typename QCAD::PoissonSource<EvalT,Traits>::ScalarT
QCAD::PoissonSource<EvalT,Traits>::eDensityForPoissonCI 
  (typename Traits::EvalData workset, std::size_t cell, std::size_t qp, 
   const ScalarT prevPhi, const bool bUsePredCorr, const double Ef, const double fixedOcc)
{
  // Use the predictor-corrector method proposed by A. Trellakis, A. T. Galick,  
  // and U. Ravaioli, "Iteration scheme for the solution of the two-dimensional  
  // Schrodinger-Poisson equations in quantum structures", J. Appl. Phys. 81, 7880 (1997). 
  // If bUsePredCorr = true, use the predictor-corrector approach.
  // If bUsePredCorr = false, calculate the exact quantum electron density.
  // Fermi-Dirac distribution is used in computing electron density.  
  
  // unit conversion factor
  //double eVPerJ = 1.0/eleQ; 
  //double cm2Perm2 = 1.0e4; 

  // For Delta2-band in Silicon, valley degeneracy factor = 2
  int valleyDegeneracyFactor = materialDB->getElementBlockParam<int>(workset.EBName,"Number of conduction band min",2);

  // Scaling factors
  double X0 = length_unit_in_m/1e-2; // length scaling to get to [cm] (structure dimension in [um])

  ScalarT temperature = temperatureField(0); //get shared temperature parameter from field in [K]
  ScalarT kbT = kbBoltz*temperature / energy_unit_in_eV;  // in [myV]
  ScalarT eDensity = 0.0; 
  //double Ef = 0.0;  //Fermi energy == 0

  const std::vector<double>& neg_eigenvals = *(workset.eigenDataPtr->eigenvalueRe);
  std::vector<double> eigenvals( neg_eigenvals );
  int nCIEvals = eigenvals.size(); //not necessarily == nEigenvectors, since CI could have not converged as many as requested
  int nEvals = std::min(nCIEvals, nEigenvectors); // the number of eigen-pairs to use (we don't gather more than nEigenvectors)

  //I don't believe CI eigenvalues are negated...
  //for(unsigned int i=0; i<nEvals; ++i) eigenvals[i] *= -1; //apply minus sign (b/c of eigenval convention)

  //Note: NO predictor corrector method used here yet -- need to understand what's going on better first

  ScalarT eDenPrefactor;
  std::vector<ScalarT> occ( nEvals, 0.0); //occupation of ith eigenstate
  
  // compute quantum electron density according to dimensionality
  switch (numDims)
  {
    //No 1D case -- deal with this later

    case 2: // 2D wavefunction (2D confinement)
    {
      // ** Assume for now that 2D case means there is confinement along the 3rd dimension such that only a single, completely non-degenerate
      //  level exists in the third dimension.  In addition, the CI accounts for spin degeneracy, so there should be no spin degeneracy factors.
      //  Together, I think this means n1D == 1.0 below. **

      // mUnconfined = effective mass in the unconfined direction (z dir. when the 2D wavefunc. is in x-y plane)
      // For Delta2-band and assume SiO2/Si interface parallel to [100] plane, mUnconfined=0.19. 
      //double mUnconfined = materialDB->getElementBlockParam<double>(workset.EBName,"Transverse Electron Effective Mass");
        
      // n1D below is a factor that is part of the line electron density 
      // in the unconfined dir. and includes spin degeneracy and in unit of [cm^-1]
      ScalarT n1D = 1.0; //sqrt(2.0*mUnconfined*m0*kbT_eV/(pi*hbar*hbar*eVPerJ*cm2Perm2));
        
      // subband-independent prefactor in calculating electron density
      // X0^2 is used to scale wavefunc. squared from [um^-2] or [nm^-2] to [cm^-2]
      // Note: 1/energy_unit_in_eV factor ==> correct [myV] units for LHS of Poisson
      eDenPrefactor = valleyDegeneracyFactor*n1D/pow(X0,2.) / energy_unit_in_eV;
      
      // Get Z = sum( exp(-E_i/kT) )
      ScalarT Z = 0.0;
      for(int i=0; i < nEvals; i++) Z += exp(-eigenvals[i]/kbT);

      for(int i=0; i < nEvals; i++) 
      {
        ScalarT wfOcc = exp(-eigenvals[i]/kbT) / Z;
        occ[i] = wfOcc;
      }
      break;
    }  // end of case 2 block    


    case 3: // 3D wavefunction (3D confinement)
    { 
      //degeneracy factor
      double degeneracyFactor = valleyDegeneracyFactor; //CI includes spin degeneracy
        
      // subband-independent prefactor in calculating electron density
      // X0^3 is used to scale wavefunc. squared from [um^-3] or [nm^-3] to [cm^-3]
      eDenPrefactor = degeneracyFactor/pow(X0,3.);

      // Get Z = sum( exp(-E_i/kT) )
      //ScalarT Z = 0.0;
      //for(int i=0; i < nEvals; i++) Z += exp(-eigenvals[i]/kbT);

      for(int i = 0; i < nEvals; i++) 
      {
        ScalarT oneOverOcc = 0.0;  // get 1/(exp(-eigenvals[i]/kbT) / Z) as sum, then invert (avoids inf issues)
	for(int j=0; j < nEvals; j++) oneOverOcc += exp(-(eigenvals[j]-eigenvals[i])/kbT);

	ScalarT wfOcc = 0.0; //exp(-eigenvals[i]/kbT) / Z;
	if(!std::isinf(QCAD::EvaluatorTools<EvalT,Traits>::getDoubleValue(oneOverOcc)))
	  wfOcc = 1/oneOverOcc; //otherwise just leave as zero since denom is infinite
	occ[i] = wfOcc;
      }
      break;
    }  // end of case 3 block 
      
    default:
      TEUCHOS_TEST_FOR_EXCEPTION (true, Teuchos::Exceptions::InvalidParameter,
        std::endl << "Error!  Invalid number of dimensions " << numDims << "!"<< std::endl);
      break; 
      
  }  // end of switch (numDims) 

  //Enforce a fixed occupation (non-equilibrium) if desired (indicated by fixedOcc > 0)
  if(fixedOcc > -1e-6) {
    double occLeft = fixedOcc;
    for(int i = 0; i < nEigenvectors; i++) {
      occ[i] = std::min(1.0, occLeft);
      occLeft -= QCAD::EvaluatorTools<EvalT,Traits>::getDoubleValue(occ[i]);
    }
  }

  // loop over eigenvalues to compute electron density [cm^-3]
  for(int i = 0; i < nEvals; i++) 
  {
    ScalarT wfSquared = ( eigenvector_Re[i](cell,qp) );
    eDensity += wfSquared * occ[i];
  }
  eDensity = eDenPrefactor*eDensity; // in [cm^-3]

  return eDensity; 
}

#endif



//! ----------------- Point charge functions ---------------------

template<typename EvalT, typename Traits>
void QCAD::PoissonSource<EvalT,Traits>::source_pointcharges(typename Traits::EvalData workset)
{
  // Scaling factors
  double X0 = length_unit_in_m/1e-2; // length scaling to get to [cm] (structure dimension in [um] usually)
  ScalarT Lambda2 = eps0/(eleQ*X0*X0);  


  // Copy params to positions
  for( std::size_t i=0; i < pointCharges.size(); ++i) {
#ifdef ALBANY_MESH_TANFAD
    pointCharges[i].position[0] = pointCharges[i].position_param[0];
    if(numDims > 1) pointCharges[i].position[1] = pointCharges[i].position_param[1];
    if(numDims > 2) pointCharges[i].position[2] = pointCharges[i].position_param[2];
#else
    pointCharges[i].position[0] = Albany::ADValue(pointCharges[i].position_param[0]);
    if(numDims > 1) pointCharges[i].position[1] = Albany::ADValue(pointCharges[i].position_param[1]);
    if(numDims > 2) pointCharges[i].position[2] = Albany::ADValue(pointCharges[i].position_param[2]);
#endif
  }

  //! find cells where point charges reside if we haven't searched yet (search only occurs once)
  if(numWorksetsScannedForPtCharges <= workset.wsIndex) {
    TEUCHOS_TEST_FOR_EXCEPTION ( !(numDims == 2 || ((numNodes == 4 || numNodes == 8) && numDims == 3)), Teuchos::Exceptions::InvalidParameter,
				  std::endl << "Error!  Point charges are only supported for TET4 and HEX8 meshes in 3D currently." << std::endl);

    //! function pointer to quantum electron density member function
    bool (QCAD::PoissonSource<EvalT,Traits>::*point_test_fn) (const MeshScalarT*, const MeshScalarT*, int);
    if(numDims == 3 && numNodes == 4)      point_test_fn = &QCAD::PoissonSource<EvalT,Traits>::pointIsInTetrahedron; //NOTE: this test should be replaced with cell type test
    else if(numDims == 3 && numNodes == 8) point_test_fn = &QCAD::PoissonSource<EvalT,Traits>::pointIsInHexahedron;  //NOTE: this test should be replaced with cell type test
    else if(numDims == 2)                  point_test_fn = &QCAD::PoissonSource<EvalT,Traits>::pointIsInPolygon;
    else                                   point_test_fn = NULL;
    
    //std::cout << "DEBUG: Looking for point charges in ws " << workset.wsIndex << " - scanned " 
    //	<< numWorksetsScannedForPtCharges << " worksets so far" << std::endl;
    MeshScalarT* cellVertices = new MeshScalarT[numNodes*numDims];
    for (std::size_t cell=0; cell < workset.numCells; ++cell) {
	for( std::size_t node=0; node<numNodes; ++node ) {
	  for( std::size_t k=0; k<numDims; ++k )
	    cellVertices[node*numDims+k] = coordVecAtVertices(cell,node,k);
	}

	for( std::size_t i=0; i < pointCharges.size(); ++i) {
	  if( (this->*point_test_fn)(cellVertices, pointCharges[i].position, numNodes) ) {
	    std::cout << "DEBUG: FOUND POINT CHARGE in ws " << workset.wsIndex << ", cell " << cell << std::endl;
	    //std::cout << "DEBUG: CELL " << cell << "VERTICES = " << std::endl;
	    for(std::size_t v=0; v<numNodes; v++) {
	      std::cout << " (  ";
	      for(std::size_t d=0; d<numDims; d++) std::cout << cellVertices[v*numDims] << "  ";
	      std::cout << ")" << std::endl;
	    }
	      
	    pointCharges[i].iWorkset = workset.wsIndex;
	    pointCharges[i].iCell = cell;
	  }
	}
    }
    delete [] cellVertices;

    assert(workset.wsIndex == numWorksetsScannedForPtCharges); //equality should always hold in if stmt above
    numWorksetsScannedForPtCharges++;
  }
  
  //! add point charge contributions
  for( std::size_t i=0; i < pointCharges.size(); ++i) {
    if( pointCharges[i].iWorkset != (int)workset.wsIndex ) continue; //skips if iWorkset == -1 (not found)
    
    MeshScalarT cellVol = 0.0;
    std::size_t cell = pointCharges[i].iCell; // iCell should be valid here since iWorkset is
    for (std::size_t qp=0; qp < numQPs; ++qp)
	cellVol += weights(cell,qp);
    
    ScalarT qpChargeDen = pointCharges[i].charge / (cellVol*pow(X0,3)); // 3 was (int)numDims but I think this is wrong (Suzey?); // [cm^-3] value of qps so that integrated charge is correct
    //std::cout << "DEBUG: ADDING POINT CHARGE (den=" << qpChargeDen << ", was " << chargeDensity(cell,0) << ") to ws "
    //		<< workset.wsIndex << ", cell " << cell << std::endl;
    for (std::size_t qp=0; qp < numQPs; ++qp) {
	ScalarT scaleFactor = 1.0; //TODO: get appropriate scale factor from meshRegions (later?)
	poissonSource(cell, qp) += 1.0/Lambda2 * scaleFactor * qpChargeDen;
	chargeDensity(cell, qp) += qpChargeDen;    
    }
  }
}


// **********************************************************************

template<typename EvalT, typename Traits>
bool QCAD::PoissonSource<EvalT,Traits>::
pointIsInTetrahedron(const MeshScalarT* cellVertices, const MeshScalarT* position, int nVertices)
{
  // Assumes cellVertices contains 4 3D points (length 12)
  //  and position is one 3D point (length 3).
  // Returns true if position lies within the tetrahedron defined by the 4 points, false otherwise.
  
  MeshScalarT v1[4], v2[4], v3[4], v4[4], p[4];
  for(int i=0; i<3; i++) {
    v1[i] = cellVertices[i]; 
    v2[i] = cellVertices[3+i];
    v3[i] = cellVertices[6+i];
    v4[i] = cellVertices[9+i];
    p[i] = position[i];
  }
  v1[3] = v2[3] = v3[3] = v4[3] = p[3] = 1; //last entry in each "4D position" == 1

  const MeshScalarT *mx[4], *refMx[4]; 
  MeshScalarT refDet, det;

  refMx[0] = mx[0] = v1; refMx[1] = mx[1] = v2; 
  refMx[2] = mx[2] = v3; refMx[3] = mx[3] = v4;
  refDet = determinant(refMx, 4);

  for(int i=0; i < 4; i++) {
    mx[i] = p; det = determinant(mx, 4); mx[i] = refMx[i];
    if( (det < 0 && refDet > 0) || (det > 0 && refDet < 0) )
      return false;
  }
  
  return true;
}


// **********************************************************************


template<typename EvalT, typename Traits>
bool QCAD::PoissonSource<EvalT,Traits>::
pointIsInHexahedron(const MeshScalarT* cellVertices, const MeshScalarT* position, int nVertices)
{
  // Assumes cellVertices contains 8 3D points (length 24) in the order specified by shards::Hexahedron 
  //  and position is one 3D point (length 3).
  // Returns true if position lies within the hexahedron defined by the 8 points, false otherwise.
  
  /* From shards::Hexadedron (vertex ordering)
         7                    6
           o------------------o
          /|                 /|
         / |                / |
        /  |               /  |
       /   |              /   |
      /    |             /    |
     /     |            /     |
  4 /      |         5 /      |
   o------------------o       |
   |       |          |       |
   |     3 o----------|-------o 2
   |      /           |      /
   |     /            |     /
   |    /             |    /
   |   /              |   /
   |  /               |  /
   | /                | /
   |/                 |/
   o------------------o
  0                    1

  */
  
  // Perform the following tests:
  // 1) is P on same side of 0123 as 4?
  // 2) is P on same side of 4567 as 0?
  // 3) is P on same side of 0154 as 2?
  // 4) is P on same side of 2376 as 0?
  // 5) is P on same side of 0374 as 2?
  // 6) is P on same side of 1265 as 0?
  // Note: we assume faces are planar and so only need 3 of 4 vertices to determine plane.

  const MeshScalarT *v0 = cellVertices;
  const MeshScalarT *v1 = cellVertices + 3;
  const MeshScalarT *v2 = cellVertices + 6;
  const MeshScalarT *v3 = cellVertices + 9;
  const MeshScalarT *v4 = cellVertices + 12;
  const MeshScalarT *v5 = cellVertices + 15;
  const MeshScalarT *v6 = cellVertices + 18;
  const MeshScalarT *v7 = cellVertices + 21;

  if(!sameSideOfPlane( v0, v1, v2, v4, position)) return false;
  if(!sameSideOfPlane( v4, v5, v6, v0, position)) return false;
  if(!sameSideOfPlane( v0, v1, v5, v2, position)) return false;
  if(!sameSideOfPlane( v2, v3, v7, v0, position)) return false;
  if(!sameSideOfPlane( v0, v3, v7, v2, position)) return false;
  if(!sameSideOfPlane( v1, v2, v6, v0, position)) return false;

  return true;
}


// **********************************************************************
template<typename EvalT, typename Traits>
bool QCAD::PoissonSource<EvalT,Traits>::
pointIsInPolygon(const MeshScalarT* cellVertices, const MeshScalarT* position, int nVertices)
{
  // Assumes cellVertices contains nVertices 2D points in some order travelling around the polygon,
  //  and position is one 2D point (length 2).
  // Returns true if position lies within the polygon, false otherwise.
  //  Algorithm = ray trace along positive x-axis

  bool c = false;
  int n = nVertices;
  MeshScalarT x=position[0], y=position[1];
  const int X=0,Y=1;

  for (int i = 0, j = n-1; i < n; j = i++) {
    const MeshScalarT* pi = &cellVertices[2*i]; // 2 == dimension
    const MeshScalarT* pj = &cellVertices[2*j]; // 2 == dimension
    if ((((pi[Y] <= y) && (y < pj[Y])) ||
	 ((pj[Y] <= y) && (y < pi[Y]))) &&
	(x < (pj[X] - pi[X]) * (y - pi[Y]) / (pj[Y] - pi[Y]) + pi[X]))
      c = !c;
  }
  return c;
}


// **********************************************************************
template<typename EvalT, typename Traits>
typename QCAD::PoissonSource<EvalT,Traits>::MeshScalarT
QCAD::PoissonSource<EvalT,Traits>::
determinant(const MeshScalarT** mx, int N)
{
  // Returns the determinant of an NxN matrix mx
  // mx is an array of arrays: mx[i][j] gives matrix entry in row i, column j
  MeshScalarT det = 0, term;

  //Loop over all permutations and keep track of sign (Dijkstra's algorithm)
  int t, sign = 0;
  int* inds = new int[N];
  for(int i=0; i<N; i++) inds[i] = i;

  while(true) {
    //inds holds indices of permutation and sign % 2 is sign
    term = 1;
    for(int i=0; i<N; i++)
      term = term * mx[i][inds[i]];
    det += ((sign % 2) ? -1 : 1) * term;

    int i = N - 1;
    while (i > 0 && inds[i-1] >= inds[i]) 
      i = i-1;

    if(i == 0) break; //we're done

    int j = N;
    while (inds[j-1] <= inds[i-1]) 
      j = j-1;
  
    t = inds[i-1]; inds[i-1] = inds[j-1]; inds[j-1] = t;  // swap values at positions (i-1) and (j-1)
    sign++;

    i++; j = N;
    while (i < j) {
      t = inds[i-1]; inds[i-1] = inds[j-1]; inds[j-1] = t;  // swap values at positions (i-1) and (j-1)
      sign++;
      i++;
      j--;
    }
  }

  delete [] inds;
  return det;
}


// **********************************************************************
template<typename EvalT, typename Traits>
bool QCAD::PoissonSource<EvalT,Traits>::
  sameSideOfPlane(const MeshScalarT* plane0, const MeshScalarT* plane1, const MeshScalarT* plane2, 
		  const MeshScalarT* ptA, const MeshScalarT* ptB)
{
  // 3D only : assumes all arguments are length 3 arrays
  MeshScalarT detA, detB;

  const MeshScalarT *mx[3];  // row mx[0] defines the point in question, rows mx[1] and mx[2] define plane
  MeshScalarT row0[3], row1[3], row2[3]; mx[0] = row0; mx[1] = row1; mx[2] = row2;

  for(int i=0; i<3; i++) {
    row0[i] = ptA[i] - plane0[i]; // use plane0 as reference position (subtract off of others)
    row1[i] = plane1[i] - plane0[i];
    row2[i] = plane2[i] - plane0[i];
  }
  detA = determinant(mx, 3);

  for(int i=0; i<3; i++)
    row0[i] = ptB[i] - plane0[i]; // replace row 0 with ptB-plane0

  detB = determinant(mx, 3);

  if( (detA < 0 && detB > 0) || (detA > 0 && detB < 0) )
    return false;
  return true;
}

//! ----------------- Point charge functions ---------------------

template<typename EvalT, typename Traits>
void QCAD::PoissonSource<EvalT,Traits>::source_cloudcharges(typename Traits::EvalData workset)
{
  //! add point charge contributions
  for (std::size_t cell=0; cell < workset.numCells; ++cell) {
    for( std::size_t i=0; i < cloudCharges.size(); ++i) {
      ScalarT cutoff2 = cloudCharges[i].cutoff * cloudCharges[i].cutoff;
      ScalarT width2  = cloudCharges[i].width  * cloudCharges[i].width;

      for (std::size_t qp=0; qp < numQPs; ++qp) {
	ScalarT distance2 =        (cloudCharges[i].position[0] - coordVec(cell,qp,0)) * (cloudCharges[i].position[0] - coordVec(cell,qp,0));
	if(numDims>1) distance2 += (cloudCharges[i].position[1] - coordVec(cell,qp,1)) * (cloudCharges[i].position[1] - coordVec(cell,qp,1));
	if(numDims>2) distance2 += (cloudCharges[i].position[2] - coordVec(cell,qp,2)) * (cloudCharges[i].position[2] - coordVec(cell,qp,2));

        if (distance2 <= cutoff2) {
	  poissonSource(cell, qp) += cloudCharges[i].amplitude * exp(-distance2/(2.0*width2));
        }
      }
    }
  }
}

//! ----------------- Miscellaneous helper functions ---------------------


// **********************************************************************
template<typename EvalT,typename Traits>
inline typename QCAD::PoissonSource<EvalT,Traits>::ScalarT
QCAD::PoissonSource<EvalT,Traits>::computeFDIntMinusOneHalf(const ScalarT x)
{
   // Use the approximate -1/2 FD integral by P. Van Halen and D. L. Pulfrey, 
   // "Accurate, short series approximations to Fermi-Dirac integrals of order 
   // -/2, 1/2, 1, 3/2, 2, 5/2, 3, and 7/2" J. Appl. Phys. 57, 5271 (1985) 
   // and its Erratum in J. Appl. Phys. 59, 2264 (1986). The approximation
   // has error < 1e-5 in the entire x range.  
   
   ScalarT fdInt; 
   double a1, a2, a3, a4, a5, a6, a7; 
   if (x <= 0.)  // eqn.(4) in the reference
   {
     a1 = 0.999909;  // Table I in Erratum
     a2 = 0.706781;
     a3 = 0.572752;
     a4 = 0.466318;
     a5 = 0.324511;
     a6 = 0.152889;
     a7 = 0.033673;
     fdInt = a1*exp(x)-a2*exp(2.*x)+a3*exp(3.*x)-a4*exp(4.*x)+a5*exp(5.*x)-a6*exp(6.*x)+a7*exp(7.*x);
   }
   else if (x >= 5.)  // eqn.(6) in Erratum
   {
     a1 = 1.12837;  // Table II in Erratum
     a2 = -0.470698;
     a3 = -0.453108;
     a4 = -228.975;
     a5 = 8303.50;
     a6 = -118124;
     a7 = 632895;
     fdInt = sqrt(x)*(a1+ a2/pow(x,2.)+ a3/pow(x,4.)+ a4/pow(x,6.)+ a5/pow(x,8.)+ a6/pow(x,10.)+ a7/pow(x,12.));
   }
   else if ((x > 0.) && (x <= 2.5))  // eqn.(7) in Erratum
   {
     double a8, a9;
     a1 = 0.604856;  // Table III in Erratum
     a2 = 0.380080;
     a3 = 0.059320;
     a4 = -0.014526;
     a5 = -0.004222;
     a6 = 0.001335;
     a7 = 0.000291;
     a8 = -0.000159;
     a9 = 0.000018;
     fdInt = a1+ a2*x+ a3*pow(x,2.)+ a4*pow(x,3.)+ a5*pow(x,4.)+ a6*pow(x,5.)+ a7*pow(x,6.)+ a8*pow(x,7.)+ a9*pow(x,8.);
   }
   else  // 2.5<x<5, eqn.(7) in Erratum
   {
     double a8;
     a1 = 0.638086;  // Table III in Erratum
     a2 = 0.292266;
     a3 = 0.159486;
     a4 = -0.077691;
     a5 = 0.018650;
     a6 = -0.002736;
     a7 = 0.000249;
     a8 = -0.000013;
     fdInt = a1+ a2*x+ a3*pow(x,2.)+ a4*pow(x,3.)+ a5*pow(x,4.)+ a6*pow(x,5.)+ a7*pow(x,6.)+ a8*pow(x,7.);
   }
   
   return fdInt;
}


// *****************************************************************************
template<typename EvalT, typename Traits>
typename QCAD::PoissonSource<EvalT,Traits>::ScalarT
QCAD::PoissonSource<EvalT,Traits>::computeVxcLDA (const double & relPerm, 
      const double & effMass, const ScalarT& eDensity)
{
  // Compute the exchange-correlation potential energy within the Local Density
  // Approximation. Use the parameterized expression from: 
  // [1] L. Hedin and B. I. Lundqvist, J. Phys. C 4, 2064 (1971);
  // [2] F. Stern and S. Das Sarma, Phys. Rev. B 30, 840 (1984);
  // [3] Dragical Vasileska, "Solving the Effective Mass Schrodinger Equation 
  // in State-of-the-Art Devices", http://nanohub.org. 
  // Potential in returned in units of eV.
  
  // first 100.0 converts eps0 from [C/(V.cm)] to [C/(V.m)],
  // second 100.0 converts b from [m] to [cm]. 
  ScalarT b = (eps0*100.0)*hbar*hbar / (m0*eleQ*eleQ) * 100.0;  // [cm]
  b = b * (4.0*pi*relPerm/effMass); 
  
  double eVPerJ = 1.0/eleQ; 
  ScalarT Ry = eleQ*eleQ / (eps0*b) * eVPerJ / (8.*pi*relPerm);  // [eV]
  
  double alpha = pow(4.0/9.0/pi, 1./3.);
  ScalarT Vxc; 
  
  if (eDensity <= 1.0) 
    Vxc = 0.0; 
  else
  {
    ScalarT rs = pow(4.0*pi*eDensity*pow(b,3.0)/3.0, -1./3.);  // [unitless]
    ScalarT x = rs / 21.0; 
    Vxc = -2.0/(pi*alpha*rs) * Ry * (1.0+ 0.7734*x*log(1.+1.0/x)); // [eV]
  }
  
  return Vxc; 
}

// **********************************************************************
template<typename EvalT, typename Traits>
typename QCAD::PoissonSource<EvalT,Traits>::ScalarT 
QCAD::PoissonSource<EvalT, Traits>::getCellScaleFactor(std::size_t cell, const std::vector<bool>& bEBInRegion, ScalarT init_factor)
{
  ScalarT ret = init_factor;
  std::size_t nRegions = meshRegionList.size();
  if(nRegions > 0) {
    for(std::size_t i=0; i<nRegions; i++) { 
      if(!bEBInRegion[i] && meshRegionList[i]->cellIsInRegion(cell)) {
	ret *= meshRegionFactors[i];
      }
    }
  }
  return ret;
}


// **********************************************************************
template<typename EvalT, typename Traits>
typename QCAD::PoissonSource<EvalT,Traits>::ScalarT 
QCAD::PoissonSource<EvalT, Traits>::getReferencePotential(typename Traits::EvalData workset)
{
  //! Constant energy reference for heterogeneous structures
  ScalarT qPhiRef;

  std::string refMtrlName, category;
  refMtrlName = materialDB->getParam<std::string>("Reference Material");
  category = materialDB->getMaterialParam<std::string>(refMtrlName,"Category");
  if (category == "Semiconductor") {
 
    // Get quantities in desired energy (voltage) units, which we denote "[myV]"
 
    // Same qPhiRef needs to be used for the entire structure
    ScalarT temperature = temperatureField(0); //get shared temperature parameter from field
    double mdn = materialDB->getMaterialParam<double>(refMtrlName,"Electron DOS Effective Mass");
    double mdp = materialDB->getMaterialParam<double>(refMtrlName,"Hole DOS Effective Mass");
    double Chi = materialDB->getMaterialParam<double>(refMtrlName,"Electron Affinity") / energy_unit_in_eV; // in [myV]
    double Eg0 = materialDB->getMaterialParam<double>(refMtrlName,"Zero Temperature Band Gap") / energy_unit_in_eV; // in [myV]
    double alpha = materialDB->getMaterialParam<double>(refMtrlName,"Band Gap Alpha Coefficient"); // in [eV/K]
    double beta = materialDB->getMaterialParam<double>(refMtrlName,"Band Gap Beta Coefficient");  // in [K]
    ScalarT Eg = Eg0-(alpha*pow(temperature,2.0)/(beta+temperature)) / energy_unit_in_eV; // in [myV]
  
    ScalarT kbT = kbBoltz*temperature / energy_unit_in_eV; // in [myV] (desired voltage unit)
    ScalarT Eic = -Eg/2. + 3./4.*kbT*log(mdp/mdn);  // (Ei-Ec) in [myV]
    qPhiRef = Chi - Eic;  // (Evac-Ei) in [myV] where Evac = vacuum level
  }
  else if (category == "Insulator") {
    double Chi = materialDB->getMaterialParam<double>(refMtrlName,"Electron Affinity");
    qPhiRef = Chi / energy_unit_in_eV; // in [myV]
  }
  else if (category == "Metal") {
    double workFn = materialDB->getMaterialParam<double>(refMtrlName,"Work Function"); 
    qPhiRef = workFn / energy_unit_in_eV; // in [myV]
  }
  else {
    TEUCHOS_TEST_FOR_EXCEPTION (true, Teuchos::Exceptions::InvalidParameter, std::endl 
 			  << "Error!  Invalid category " << category 
 			  << " for reference material !" << std::endl);
  }

  return qPhiRef;
}


// **********************************************************************
