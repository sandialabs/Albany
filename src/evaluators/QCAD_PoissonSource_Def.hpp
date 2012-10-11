/********************************************************************\
*            Albany, Copyright (2010) Sandia Corporation             *
*                                                                    *
* Notice: This computer software was prepared by Sandia Corporation, *
* hereinafter the Contractor, under Contract DE-AC04-94AL85000 with  *
* the Department of Energy (DOE). All rights in the computer software*
* are reserved by DOE on behalf of the United States Government and  *
* the Contractor as provided in the Contract. You are authorized to  *
* use this computer software for Governmental purposes but it is not *
* to be released or distributed to the public. NEITHER THE GOVERNMENT*
* NOR THE CONTRACTOR MAKES ANY WARRANTY, EXPRESS OR IMPLIED, OR      *
* ASSUMES ANY LIABILITY FOR THE USE OF THIS SOFTWARE. This notice    *
* including this sentence must appear on any copies of this software.*
*    Questions to Andy Salinger, agsalin@sandia.gov                  *
\********************************************************************/


#include <fstream>
#include "Teuchos_TestForException.hpp"
#include "Phalanx_DataLayout.hpp"
#include "Sacado_ParameterRegistration.hpp"

template<typename EvalT, typename Traits>
QCAD::PoissonSource<EvalT, Traits>::
PoissonSource(Teuchos::ParameterList& p,
                 const Teuchos::RCP<Albany::Layouts>& dl) :
  coordVec(p.get<std::string>("Coordinate Vector Name"), dl->qp_gradient),
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
  approxQuanEDen("Approx Quantum EDensity",dl->qp_scalar)
{
  // Material database
  materialDB = p.get< Teuchos::RCP<QCAD::MaterialDatabase> >("MaterialDB");

  Teuchos::ParameterList* psList = p.get<Teuchos::ParameterList*>("Parameter List");

  Teuchos::RCP<const Teuchos::ParameterList> reflist = 
      this->getValidPoissonSourceParameters();
  psList->validateParameters(*reflist,0);

  std::vector<PHX::DataLayout::size_type> dims;
  dl->qp_gradient->dimensions(dims);
  numQPs  = dims[1];
  numDims = dims[2];

  // get values from the input .xml and use default values if not provided
  factor = psList->get("Factor", 1.0);
  device = psList->get("Device", "defaultdevice");
  nonQuantumRegionSource = psList->get("Non Quantum Region Source", "semiclassical");
  quantumRegionSource    = psList->get("Quantum Region Source", "semiclassical"); 
  imagPartOfCoulombSrc   = psList->get<bool>("Imaginary Part Of Coulomb Source", false); 
  carrierStatistics = psList->get("Carrier Statistics", "Boltzmann Statistics");
  incompIonization = psList->get("Incomplete Ionization", "False");
  bUsePredictorCorrector = psList->get<bool>("Use predictor-corrector method",false);
  bIncludeVxc = psList->get<bool>("Include exchange-correlation potential",false);

  // find element blocks and voltages applied on them
  std::string preName = "DBC on NS "; 
  std::string postName = " for DOF Phi";
  std::size_t preLen = preName.length();
  std::size_t postLen = postName.length();  

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
      mapDBCValue[ebName] = dbcValue;
      std::cout << "ebName = " << ebName << ", value = " << mapDBCValue[ebName] << std::endl;  
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

  if(quantumRegionSource == "schrodinger" || 
     quantumRegionSource == "coulomb" || 
     quantumRegionSource == "ci") {
    std::string evecFieldRoot = p.get<string>("Eigenvector field name root");
    nEigenvectors = psList->get<int>("Eigenvectors from States");

    eigenvector_Re.resize(nEigenvectors);
    eigenvector_Im.resize(nEigenvectors);

    char buf[200];
    for (int k = 0; k < nEigenvectors; ++k) {
      sprintf(buf, "%s_Re%d", evecFieldRoot.c_str(), k);
      PHX::MDField<ScalarT,Cell,QuadPoint> fr(buf,dl->qp_scalar);
      eigenvector_Re[k] = fr; this->addDependentField(eigenvector_Re[k]);

      sprintf(buf, "%s_Im%d", evecFieldRoot.c_str(), k);
      PHX::MDField<ScalarT,Cell,QuadPoint> fi(buf,dl->qp_scalar);
      eigenvector_Im[k] = fi; this->addDependentField(eigenvector_Im[k]);
    }
  }
  else {
    nEigenvectors = 0;
  }

  // Add factor as a Sacado-ized parameter
  Teuchos::RCP<ParamLib> paramLib =
      p.get< Teuchos::RCP<ParamLib> >("Parameter Library", Teuchos::null);
  new Sacado::ParameterRegistration<EvalT, SPL_Traits>(
      "Poisson Source Factor", this, paramLib);

  // Add parameters from material database as Sacado params
  std::vector<string> dopingParamNames = materialDB->getAllMatchingParams<std::string>("Doping Parameter Name");
  std::vector<string> chargeParamNames = materialDB->getAllMatchingParams<std::string>("Charge Parameter Name");
  
  std::vector<string>::iterator s;
  for(s = dopingParamNames.begin(); s != dopingParamNames.end(); s++) {
    if( psList->isParameter(*s) ) {
      materialParams[*s] = psList->get<double>(*s);
      new Sacado::ParameterRegistration<EvalT, SPL_Traits>(*s, this, paramLib);
    }
  }
  for(s = chargeParamNames.begin(); s != chargeParamNames.end(); s++) {
    if( psList->isParameter(*s) ) {
      materialParams[*s] = psList->get<double>(*s);
      new Sacado::ParameterRegistration<EvalT, SPL_Traits>(*s, this, paramLib);
    }
  }

  if(quantumRegionSource == "coulomb") {
    //Add Sacado parameters to set indices of eigenvectors to be multipled together
    new Sacado::ParameterRegistration<EvalT, SPL_Traits>("Source Eigenvector 1", this, paramLib);
    new Sacado::ParameterRegistration<EvalT, SPL_Traits>("Source Eigenvector 2", this, paramLib);
  }

  this->addDependentField(potential);
  this->addDependentField(coordVec);
  this->addDependentField(temperatureField);

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
  
  this->setName("Poisson Source"+PHX::TypeString<EvalT>::value);
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

  for (int k = 0; k < nEigenvectors; ++k) {
    this->utils.setFieldData(eigenvector_Re[k],fm);
    this->utils.setFieldData(eigenvector_Im[k],fm);
  }
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
  else TEUCHOS_TEST_FOR_EXCEPT(true); return factor; //dummy so all control paths return
}

// **********************************************************************
template<typename EvalT,typename Traits>
Teuchos::RCP<const Teuchos::ParameterList>
QCAD::PoissonSource<EvalT,Traits>::getValidPoissonSourceParameters() const
{
  Teuchos::RCP<Teuchos::ParameterList> validPL =
       rcp(new Teuchos::ParameterList("Valid Poisson Problem Params"));;

  validPL->set<double>("Factor", 1.0, "Constant multiplier in source term");
  validPL->set<string>("Device", "defaultdevice", "Switch between different device models");
  validPL->set<string>("Non Quantum Region Source", "semiclassical", "Source type for non-quantum regions");
  validPL->set<string>("Quantum Region Source", "semiclassical", "Source type for quantum regions");
  validPL->set<bool>("Imaginary Part Of Coulomb Source",false,"Whether to use imag or real part of coulomb quantum region source");
  //validPL->set<double>("Donor Doping", 1e14, "Doping for nsilicon element blocks [cm^-3]");
  validPL->set<double>("Acceptor Doping", 1e14, "Doping for psilicon element blocks [cm^-3]");
  validPL->set<string>("Carrier Statistics", "Boltzmann Statistics", "Carrier statistics");
  validPL->set<string>("Incomplete Ionization", "False", "Partial ionization of dopants");
  //validPL->set<double>("Donor Activation Energy", 0.045, "Donor activation energy [eV]");
  validPL->set<double>("Acceptor Activation Energy", 0.045, "Acceptor activation energy [eV]");
  validPL->set<int>("Eigenvectors from States", 0, "Number of eigenvectors to take from eigendata information");
  validPL->set<bool>("Use predictor-corrector method",false, "Enable use of predictor-corrector method for S-P iterations");
  validPL->set<bool>("Include exchange-correlation potential",false, "Include the exchange correlation term in the output potential state");
  
  validPL->set<double>("Oxide Width", 0., "Oxide width for 1D MOSCapacitor device");
  validPL->set<double>("Silicon Width", 0., "Silicon width for 1D MOSCapacitor device");
  
  std::vector<string> dopingParamNames = materialDB->getAllMatchingParams<std::string>("Doping Parameter Name");
  std::vector<string> chargeParamNames = materialDB->getAllMatchingParams<std::string>("Charge Parameter Name");
  std::vector<string>::iterator s;
  for(s = dopingParamNames.begin(); s != dopingParamNames.end(); s++)
    validPL->set<double>( *s, 0.0, "Doping Parameter [cm^-3]");
  for(s = chargeParamNames.begin(); s != chargeParamNames.end(); s++)
    validPL->set<double>( *s, 0.0, "Charge Parameter [cm^-3]");
  
  return validPL;
}


// *****************************************************************************
template<typename EvalT, typename Traits>
void QCAD::PoissonSource<EvalT, Traits>::
evaluateFields_elementblocks(typename Traits::EvalData workset)
{
  ScalarT temperature = temperatureField(0); //get shared temperature parameter from field

  // Scaling factors
  double X0 = length_unit_in_m/1e-2; // length scaling to get to [cm] (structure dimension in [um])
  ScalarT V0 = kbBoltz*temperature/1.0; // kb*T/q in [V]
  ScalarT Lambda2 = eps0/(eleQ*X0*X0); // derived scaling factor
  
  //! Constant energy reference for heterogeneous structures
  ScalarT qPhiRef;
  {
    std::string refMtrlName, category;
    refMtrlName = materialDB->getParam<std::string>("Reference Material");
    category = materialDB->getMaterialParam<std::string>(refMtrlName,"Category");
    if (category == "Semiconductor") {
        
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
    else if (category == "Insulator") {
      double Chi = materialDB->getMaterialParam<double>(refMtrlName,"Electron Affinity");
      qPhiRef = Chi;
    }
    else if (category == "Metal") {
      double workFn = materialDB->getMaterialParam<double>(refMtrlName,"Work Function");
      qPhiRef = workFn;
    }
    else {
      TEUCHOS_TEST_FOR_EXCEPTION (true, Teuchos::Exceptions::InvalidParameter, std::endl 
			  << "Error!  Invalid category " << category 
			  << " for reference material !" << std::endl);
    }
  }  

  string matrlCategory = materialDB->getElementBlockParam<string>(workset.EBName,"Category");

  //! function pointer to quantum electron density member function
  ScalarT (QCAD::PoissonSource<EvalT,Traits>::*quantum_edensity_fn) 
    (typename Traits::EvalData, std::size_t, std::size_t, const ScalarT, const bool, const double);

  if(quantumRegionSource == "schrodinger")
    quantum_edensity_fn = &QCAD::PoissonSource<EvalT,Traits>::eDensityForPoissonSchrodinger;
  else if(quantumRegionSource == "ci")
    quantum_edensity_fn = &QCAD::PoissonSource<EvalT,Traits>::eDensityForPoissonCI;
  else quantum_edensity_fn = NULL;


  //***************************************************************************
  //! element block with "Semiconductor" material
  //***************************************************************************
  if(matrlCategory == "Semiconductor") 
  {
    //! temperature-independent material parameters
    double mdn = materialDB->getElementBlockParam<double>(workset.EBName,"Electron DOS Effective Mass");
    double mdp = materialDB->getElementBlockParam<double>(workset.EBName,"Hole DOS Effective Mass");
    double Tref = materialDB->getElementBlockParam<double>(workset.EBName,"Reference Temperature");
    
    double Chi = materialDB->getElementBlockParam<double>(workset.EBName,"Electron Affinity");
    double Eg0 = materialDB->getElementBlockParam<double>(workset.EBName,"Zero Temperature Band Gap");
    double alpha = materialDB->getElementBlockParam<double>(workset.EBName,"Band Gap Alpha Coefficient");
    double beta = materialDB->getElementBlockParam<double>(workset.EBName,"Band Gap Beta Coefficient");
    
    //! constant prefactor in calculating Nc and Nv in [cm-3]
    double NcvFactor = 2.0*pow((kbBoltz*eleQ*m0*Tref)/(2*pi*pow(hbar,2)),3./2.)*1e-6;
            // eleQ converts kbBoltz in [eV/K] to [J/K], 1e-6 converts [m-3] to [cm-3]
            
    //! strong temperature-dependent material parameters
    ScalarT Nc;  // conduction band effective DOS in [cm-3]
    ScalarT Nv;  // valence band effective DOS in [cm-3]
    ScalarT Eg;  // band gap at T [K] in [eV]
    ScalarT ni;  // intrinsic carrier concentration in [cm-3]
    
    Nc = NcvFactor*pow(mdn,1.5)*pow(temperature/Tref,1.5);  // in [cm-3]
    Nv = NcvFactor*pow(mdp,1.5)*pow(temperature/Tref,1.5); 
    Eg = Eg0-alpha*pow(temperature,2.0)/(beta+temperature); // in [eV]
    
    ScalarT kbT = kbBoltz*temperature;      // in [eV]
    ni = sqrt(Nc*Nv)*exp(-Eg/(2.0*kbT));    // in [cm-3]

    //! parameters for computing exchange-correlation potential
    const string& condBandMin = materialDB->getElementBlockParam<string>(workset.EBName,"Conduction Band Minimum");
    double ml = materialDB->getElementBlockParam<double>(workset.EBName,"Longitudinal Electron Effective Mass");
    double mt = materialDB->getElementBlockParam<double>(workset.EBName,"Transverse Electron Effective Mass");        
    if ((condBandMin == "Gamma Valley") && (abs(ml-mt) > 1e-10))
      TEUCHOS_TEST_FOR_EXCEPTION (true, std::logic_error, "Gamma Valley's longitudinal and "
        << "transverse electron effective mass must be equal ! "
        << "Please check the values in materials.xml" << std::endl);
    
    double invEffMass = (2.0/mt + 1.0/ml) / 3.0;
    double averagedEffMass = 1.0 / invEffMass; 
    double relPerm = materialDB->getElementBlockParam<double>(workset.EBName,"Permittivity");
    
    //! argument offset in calculating electron and hole density
    ScalarT eArgOffset = (-qPhiRef+Chi)/kbT;
    ScalarT hArgOffset = (qPhiRef-Chi-Eg)/kbT;
    
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

    //! obtain the fermi energy in a given element block
    double fermiE = 0.0;  // default, [eV]
    if (mapDBCValue.count(workset.EBName) > 0) 
    {
      fermiE = -1.0*mapDBCValue[workset.EBName]; 
      // std::cout << "EBName = " << workset.EBName << ", fermiE = " << fermiE << std::endl; 
    }

    //! get doping concentration and activation energy
    //** Note: doping profile unused currently
    string dopantType = materialDB->getElementBlockParam<string>(workset.EBName,"Dopant Type","None");
    string dopingProfile;
    ScalarT inArg, dopingConc;

    if(dopantType != "None") {
      double dopantActE;
      dopingProfile = materialDB->getElementBlockParam<string>(workset.EBName,"Doping Profile","Constant");
      dopantActE = materialDB->getElementBlockParam<double>(workset.EBName,"Dopant Activation Energy",0.045);
    
      if( materialDB->isElementBlockParam(workset.EBName, "Doping Value") ) 
        dopingConc = materialDB->getElementBlockParam<double>(workset.EBName,"Doping Value");
      else if( materialDB->isElementBlockParam(workset.EBName, "Doping Parameter Name") ) {
        double scl = materialDB->getElementBlockParam<double>(workset.EBName,"Doping Parameter Scaling", 1.0);
        dopingConc = materialParams[ materialDB->getElementBlockParam<string>(workset.EBName,"Doping Parameter Name") ] * scl;
      }
      else TEUCHOS_TEST_FOR_EXCEPTION (true, Teuchos::Exceptions::InvalidParameter,
        std::endl << "Error!  Unknown dopant concentration for " << workset.EBName << "!"<< std::endl);

      if(dopantType == "Donor") 
        inArg = eArgOffset + dopantActE/kbT + fermiE/kbT;
      else if(dopantType == "Acceptor") 
        inArg = hArgOffset + dopantActE/kbT - fermiE/kbT;
      else TEUCHOS_TEST_FOR_EXCEPTION (true, Teuchos::Exceptions::InvalidParameter,
	       std::endl << "Error!  Unknown dopant type " << dopantType << "!"<< std::endl);
    }
    else {
      dopingProfile = "Constant";
      dopingConc = 0.0;
      inArg = 0.0;
    }

    //! Schrodinger source for electrons
    if(materialDB->getElementBlockParam<bool>(workset.EBName,"quantum",false)) 
    {
      if(quantumRegionSource == "schrodinger" || quantumRegionSource == "ci") 
      {
        // retrieve Previous Poisson Potential
	      Albany::MDArray prevPhiArray = (*workset.stateArrayPtr)["Previous Poisson Potential"];

        // loop over cells and qps
        for (std::size_t cell=0; cell < workset.numCells; ++cell)
        {
          for (std::size_t qp=0; qp < numQPs; ++qp)
          {
            ScalarT approxEDensity = 0.0;
            ScalarT prevPhi = prevPhiArray(cell,qp); 
          
            if(bUsePredictorCorrector)
              // compute the approximate quantum electron density using predictor-corrector method
              approxEDensity = (this->*quantum_edensity_fn)(workset, cell, qp, prevPhi, true, fermiE);

            else  // otherwise, use the exact quantum density
              approxEDensity = (this->*quantum_edensity_fn)(workset, cell, qp, 0.0, false, fermiE);
	    
            // compute the exact quantum electron density
            ScalarT eDensity = (this->*quantum_edensity_fn)(workset, cell, qp, 0.0, false, fermiE);
          
            // obtain the scaled potential
            const ScalarT& unscaled_phi = potential(cell,qp); //[V]
            ScalarT phi = unscaled_phi / V0; 
           
            // compute the hole density treated as classical
            ScalarT hDensity = Nv*(this->*carrStat)(-phi + hArgOffset); 

            // obtain the ionized dopants
            ScalarT ionN  = 0.0;
            if (dopantType == "Donor")  // function takes care of sign
              ionN = (this->*ionDopant)(dopantType,phi + inArg)*dopingConc;
            else if (dopantType == "Acceptor")
              ionN = (this->*ionDopant)(dopantType,-phi + inArg)*dopingConc;
            else 
              ionN = 0.0;
              
            // the scaled full RHS
            ScalarT charge; 
            charge = 1.0/Lambda2*(hDensity- approxEDensity + ionN);
            poissonSource(cell, qp) = factor*charge;

            // output states
            chargeDensity(cell, qp) = hDensity -eDensity +ionN;
            electronDensity(cell, qp) = eDensity;
            holeDensity(cell, qp) = hDensity;
            electricPotential(cell, qp) = phi*V0;
            ionizedDopant(cell, qp) = ionN;
            approxQuanEDen(cell,qp) = approxEDensity;
            artCBDensity(cell, qp) = ( eDensity > 1e-6 ? eDensity : -Nc*(this->*carrStat)( -(phi+eArgOffset) ));

            if (bIncludeVxc)  // include Vxc
            {
              ScalarT Vxc = computeVxcLDA(relPerm, averagedEffMass, approxEDensity);
              conductionBand(cell, qp) = qPhiRef -Chi -phi*V0 +Vxc; // [eV]
              
              // Suzey: need to be discussed (April 06, 2012) ? 
              electricPotential(cell, qp) = phi*V0 + Vxc; //add xc correction to electric potential (used in CI delta_ij computation)
            }
            else  // not include Vxc
            {
              conductionBand(cell, qp) = qPhiRef -Chi -phi*V0; // [eV]
              electricPotential(cell, qp) = phi*V0;
            }
            valenceBand(cell, qp) = conductionBand(cell,qp)-Eg;
          }
        }  // end of loop over cells
      }  // end of if (schrodinger or ci)


      else if(quantumRegionSource == "coulomb") 
      {
        //RHS == evec[i] * evec[j]
        int i = (int)QCAD::EvaluatorTools<EvalT,Traits>::getDoubleValue( sourceEvecInds[0] );
        int j = (int)QCAD::EvaluatorTools<EvalT,Traits>::getDoubleValue( sourceEvecInds[1] );

        //int valleyDegeneracyFactor = materialDB->getElementBlockParam<int>(workset.EBName,"Number of conduction band min",2);
        // scale so electron density is in [cm^-3] (assume 3D? Suzey?) as expected of RHS of Poisson eqn
        ScalarT prefactor = 1.0/pow(X0,3.);
	
        // loop over cells and qps
        for (std::size_t cell=0; cell < workset.numCells; ++cell) 
        {
          for (std::size_t qp=0; qp < numQPs; ++qp) 
          {
            // obtain the scaled potential
            const ScalarT& unscaled_phi = potential(cell,qp); //[V]
            ScalarT phi = unscaled_phi / V0; 

            // the scaled full RHS   note: wavefunctions are assumed normalized.  Source term 
	    //  is conj(evec_j) * evec_i
            ScalarT charge;
	    if(imagPartOfCoulombSrc) 
	      charge = - prefactor * ( eigenvector_Re[i](cell,qp) * eigenvector_Im[j](cell,qp) - 
				       eigenvector_Im[i](cell,qp) * eigenvector_Re[j](cell,qp));
	    else
	      charge = - prefactor * ( eigenvector_Re[i](cell,qp) * eigenvector_Re[j](cell,qp) + 
				       eigenvector_Im[i](cell,qp) * eigenvector_Im[j](cell,qp));

            poissonSource(cell, qp) = factor * 1.0/Lambda2 * charge; //sign??

            chargeDensity(cell, qp) = charge;
            electronDensity(cell, qp) = charge;
            holeDensity(cell, qp) = 0.0;
            electricPotential(cell, qp) = phi*V0;

            //never include Vxc
	    conductionBand(cell, qp) = qPhiRef-Chi-phi*V0; // [eV]
            valenceBand(cell, qp) = conductionBand(cell,qp)-Eg;
          }
        }
      }  // end of else if (coulomb)
      
      else if(quantumRegionSource == "semiclassical") {
        goto jumpToSemiClRHS_semiconductor; //TODO later: functions instead of goto stmts
      }
      else if(quantumRegionSource == "none") {
        goto jumpToNoneRHS_semiconductor;
      }
      else TEUCHOS_TEST_FOR_EXCEPTION (true, Teuchos::Exceptions::InvalidParameter,
        std::endl << "Error!  Unknown quantum region source: " << quantumRegionSource << "!"<< std::endl);
    
    } // end of if quantum = true 


    //! calculate the classical charge (RHS) for Poisson equation
    else 
    {
      if(nonQuantumRegionSource == "semiclassical") 
      {
       jumpToSemiClRHS_semiconductor:
      
        for (std::size_t cell=0; cell < workset.numCells; ++cell)
        {
          for (std::size_t qp=0; qp < numQPs; ++qp)
          {
            const ScalarT& unscaled_phi = potential(cell,qp);  // [V]
            ScalarT phi = unscaled_phi / V0; 
          
            // obtain the ionized dopants
            ScalarT ionN;
            if (dopantType == "Donor")  // function takes care of sign
              ionN = (this->*ionDopant)(dopantType,phi + inArg)*dopingConc;
            else if (dopantType == "Acceptor")
              ionN = (this->*ionDopant)(dopantType,-phi + inArg)*dopingConc;
            else 
              ionN = 0.0; 

            // the scaled full RHS
            ScalarT charge, eDensity, hDensity; 
            eDensity = Nc*(this->*carrStat)(phi + eArgOffset + fermiE/kbT);
            hDensity = Nv*(this->*carrStat)(-phi + hArgOffset - fermiE/kbT);
            charge = 1.0/Lambda2 * (hDensity - eDensity + ionN);
            poissonSource(cell, qp) = factor*charge;
          
            // output states
            chargeDensity(cell, qp) = charge*Lambda2;
            electronDensity(cell, qp) = eDensity;
            holeDensity(cell, qp) = hDensity;
            electricPotential(cell, qp) = phi*V0;
            ionizedDopant(cell, qp) = ionN;
            conductionBand(cell, qp) = qPhiRef-Chi-phi*V0; // [eV]
            valenceBand(cell, qp) = conductionBand(cell,qp)-Eg;
            approxQuanEDen(cell,qp) = 0.0; 
            artCBDensity(cell, qp) = ( eDensity > 1e-6 ? eDensity 
				       : -Nc*(this->*carrStat)( -(phi+eArgOffset) ));
          }
        } 
      }  // end of if(nonQuantumRegionSource == "semiclassical") 
      
      else if(nonQuantumRegionSource == "none") 
      {
       jumpToNoneRHS_semiconductor:

        for (std::size_t cell=0; cell < workset.numCells; ++cell)
        {
          for (std::size_t qp=0; qp < numQPs; ++qp)
          {
            const ScalarT& unscaled_phi = potential(cell,qp);  // [V]
            ScalarT phi = unscaled_phi / V0; 

            // the scaled full RHS
            ScalarT charge, eDensity, hDensity; 
            eDensity = 0.0;
            hDensity = 0.0;
            charge = 0.0;
            poissonSource(cell, qp) = factor*charge;
          
            // output states
            chargeDensity(cell, qp) = charge*Lambda2;
            electronDensity(cell, qp) = eDensity;
            holeDensity(cell, qp) = hDensity;
            electricPotential(cell, qp) = phi*V0;
            ionizedDopant(cell, qp) = 0.0;
            conductionBand(cell, qp) = qPhiRef-Chi-phi*V0; // [eV]
            valenceBand(cell, qp) = conductionBand(cell,qp)-Eg;
            approxQuanEDen(cell,qp) = 0.0; 
            artCBDensity(cell, qp) = 0.0;
          }
        }
      }  // end of else if(nonQuantumRegionSource == "none")
      
      else TEUCHOS_TEST_FOR_EXCEPTION (true, Teuchos::Exceptions::InvalidParameter,
        std::endl << "Error!  Unknown non-quantum region source: " << nonQuantumRegionSource << "!"<< std::endl);
    
    } // end of if quantum = false

  } // end of if(matrlCategory == "Semiconductor") 


  //***************************************************************************
  //! element block with "Insulator" material
  //***************************************************************************
  else if(matrlCategory == "Insulator")
  {
    double Eg = materialDB->getElementBlockParam<double>(workset.EBName,"Band Gap",0.0);
    double Chi = materialDB->getElementBlockParam<double>(workset.EBName,"Electron Affinity",0.0);

    //! Fixed charge in insulator
    ScalarT fixedCharge; // [cm^-3]
    if( materialDB->isElementBlockParam(workset.EBName, "Charge Value") ) {
      fixedCharge = materialDB->getElementBlockParam<double>(workset.EBName,"Charge Value");
      //std::cout << "DEBUG: applying fixed charge " << fixedCharge << " to element block '" << workset.EBName << "'" << std::endl;
    }
    else if( materialDB->isElementBlockParam(workset.EBName, "Charge Parameter Name") ) { 
      double scl = materialDB->getElementBlockParam<double>(workset.EBName,"Charge Parameter Scaling", 1.0);
      fixedCharge = materialParams[ materialDB->getElementBlockParam<string>(workset.EBName,"Charge Parameter Name") ] * scl;
      //std::cout << "DEBUG: applying fixed charge " << fixedCharge << " to element block '" << workset.EBName << "' via param" << std::endl;
    }
    else fixedCharge = 0.0; 

    //! parameters for computing exchange-correlation potential
    double ml = materialDB->getElementBlockParam<double>(workset.EBName,"Longitudinal Electron Effective Mass");
    double mt = materialDB->getElementBlockParam<double>(workset.EBName,"Transverse Electron Effective Mass");        
    if (abs(ml-mt) > 1e-10) 
      TEUCHOS_TEST_FOR_EXCEPTION (true, std::logic_error, "Insulator's longitudinal and "
	       << "transverse electron effective mass must be equal ! "
	       << "Please check the values in materials.xml" << std::endl);

    double invEffMass = (2.0/mt + 1.0/ml) / 3.0;
    double averagedEffMass = 1.0 / invEffMass; 
    double relPerm = materialDB->getElementBlockParam<double>(workset.EBName,"Permittivity");

    //! Schrodinger source for electrons
    if(materialDB->getElementBlockParam<bool>(workset.EBName,"quantum",false)) 
    {
      if(quantumRegionSource == "schrodinger" || quantumRegionSource == "ci") 
      {
        // retrieve Previous Poisson Potential
        Albany::MDArray prevPhiArray = (*workset.stateArrayPtr)["Previous Poisson Potential"];

        // loop over cells and qps
        for (std::size_t cell=0; cell < workset.numCells; ++cell)
        {
          for (std::size_t qp=0; qp < numQPs; ++qp)
          {
            ScalarT approxEDensity = 0.0;
            ScalarT prevPhi = prevPhiArray(cell,qp);
          
            if (bUsePredictorCorrector) 
              // compute the approximate quantum electron density using predictor-corrector method
              approxEDensity = (this->*quantum_edensity_fn)(workset, cell, qp, prevPhi, true, 0.0);
            else
              approxEDensity = (this->*quantum_edensity_fn)(workset, cell, qp, 0.0, false, 0.0);

            // compute the exact quantum electron density
            ScalarT eDensity = (this->*quantum_edensity_fn)(workset, cell, qp, 0.0, false, 0.0);

            // obtain the scaled potential
            const ScalarT& unscaled_phi = potential(cell,qp); //[V]
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
            electricPotential(cell, qp) = phi*V0;
            ionizedDopant(cell, qp) = 0.0;
            approxQuanEDen(cell,qp) = approxEDensity;
            artCBDensity(cell, qp) = eDensity;
	  
            if (bIncludeVxc)  // include Vxc
            {
              ScalarT Vxc = computeVxcLDA(relPerm, averagedEffMass, approxEDensity);
              conductionBand(cell, qp) = qPhiRef-Chi-phi*V0 +Vxc; // [eV]
              
              // Suzey: need to be discussed (April 06, 2012) ? 
              electricPotential(cell, qp) = phi*V0 + Vxc; //add xc correction to electric potential (used in CI delta_ij computation)
            }
            else  // not include Vxc
            { 
              conductionBand(cell, qp) = qPhiRef-Chi-phi*V0; // [eV]
              electricPotential(cell, qp) = phi*V0;
            }
            valenceBand(cell, qp) = conductionBand(cell,qp)-Eg;
          }
        }  // end of loop over cells
      }  // end of if (schrodinger || ci) 
      
      else if(quantumRegionSource == "coulomb") 
      {
        //RHS == evec[i] * evec[j]
        int i = (int)QCAD::EvaluatorTools<EvalT,Traits>::getDoubleValue( sourceEvecInds[0] );
        int j = (int)QCAD::EvaluatorTools<EvalT,Traits>::getDoubleValue( sourceEvecInds[1] );

        //convert to cm^-3 and assume 3D
        ScalarT prefactor = 1.0/pow(X0,3.); //3D
	
        // loop over cells and qps
        for (std::size_t cell=0; cell < workset.numCells; ++cell) 
        {
          for (std::size_t qp=0; qp < numQPs; ++qp) 
          {
            // obtain the scaled potential
            const ScalarT& unscaled_phi = potential(cell,qp); //[V]
            ScalarT phi = unscaled_phi / V0; 

            // the scaled full RHS   note: wavefunctions are assumed normalized and **REAL** here 
            ScalarT charge = -prefactor * ( eigenvector_Re[i](cell,qp) * eigenvector_Re[j](cell,qp) );
            poissonSource(cell, qp) = factor * 1.0/Lambda2 * charge; //sign??

            chargeDensity(cell, qp) = charge;
            electronDensity(cell, qp) = charge;
            holeDensity(cell, qp) = 0.0;
            electricPotential(cell, qp) = phi*V0;

            //never include Vxc
	    conductionBand(cell, qp) = qPhiRef-Chi-phi*V0; // [eV]
            valenceBand(cell, qp) = conductionBand(cell,qp)-Eg;
          }
        }
      }  // end of else if (coulomb)
      
      else if(quantumRegionSource == "semiclassical") {
        goto jumpToSemiClRHS_insulator; 
      }
      else if(quantumRegionSource == "none") {
        goto jumpToNoneRHS_insulator;
      }
      else TEUCHOS_TEST_FOR_EXCEPTION (true, Teuchos::Exceptions::InvalidParameter,
        std::endl << "Error!  Unknown quantum region source: " << quantumRegionSource << "!"<< std::endl);

    }  // end of if quantum = true 
    
    
    else { // use semiclassical source

      if(nonQuantumRegionSource == "semiclassical") 
      {
       jumpToSemiClRHS_insulator:
      
        for (std::size_t cell=0; cell < workset.numCells; ++cell)
        {
          for (std::size_t qp=0; qp < numQPs; ++qp)
          {
            const ScalarT& unscaled_phi = potential(cell,qp);  //[V]
            ScalarT phi = unscaled_phi / V0;  //[unitless]
          
            // the scaled full RHS
            ScalarT charge; 
            charge = 1.0/Lambda2*fixedCharge;  // only fixed charge in an insulator
            poissonSource(cell, qp) = factor*charge;
	    
            chargeDensity(cell, qp) = fixedCharge; // fixed space charge in an insulator
            electronDensity(cell, qp) = 0.0;       // no electrons in an insulator
            holeDensity(cell, qp) = 0.0;           // no holes in an insulator
            electricPotential(cell, qp) = phi*V0;
            ionizedDopant(cell, qp) = 0.0;
            conductionBand(cell, qp) = qPhiRef-Chi-phi*V0; // [eV]
            valenceBand(cell, qp) = conductionBand(cell,qp)-Eg;
            approxQuanEDen(cell,qp) = 0.0;
            artCBDensity(cell, qp) = 0.0;
          }
        }
      }  // end of if(nonQuantumRegionSource == "semiclassical") 
      
      else if(nonQuantumRegionSource == "none") 
      {
       jumpToNoneRHS_insulator:

        for (std::size_t cell=0; cell < workset.numCells; ++cell)
        {
          for (std::size_t qp=0; qp < numQPs; ++qp)
          {
            const ScalarT& unscaled_phi = potential(cell,qp);  //[V]
            ScalarT phi = unscaled_phi / V0;  //[unitless]
          
            // the scaled full RHS
            ScalarT charge = 0.0;  // no charge in this RHS mode
            poissonSource(cell, qp) = factor*charge;
	    
            chargeDensity(cell, qp) = 0.0;         // no charge in this RHS mode
            electronDensity(cell, qp) = 0.0;       // no electrons in an insulator
            holeDensity(cell, qp) = 0.0;           // no holes in an insulator
            electricPotential(cell, qp) = phi*V0;
            ionizedDopant(cell, qp) = 0.0;
            conductionBand(cell, qp) = qPhiRef-Chi-phi*V0; // [eV]
            valenceBand(cell, qp) = conductionBand(cell,qp)-Eg;
            approxQuanEDen(cell,qp) = 0.0;
            artCBDensity(cell, qp) = 0.0;
          }
        }
      }  // end of else if(nonQuantumRegionSource == "none") 
      
      else TEUCHOS_TEST_FOR_EXCEPTION (true, Teuchos::Exceptions::InvalidParameter,
        std::endl << "Error!  Unknown non-quantum region source: " << nonQuantumRegionSource << "!"<< std::endl);
    
    }  // end of if quantum = false
  
  }  // end of else if(matrlCategory == "Insulator")


  //***************************************************************************
  //! element block with "Metal" material 
  //***************************************************************************
  else if(matrlCategory == "Metal")
  {
    // Assume all metals are non-quantum regions, and calculation of charge, etc is the same
    // for all cases -- always zero charge in metals.
    double workFunc = materialDB->getElementBlockParam<double>(workset.EBName,"Work Function");
    
    // The following assumes Metal is surrounded by Dirichlet BC
    for (std::size_t cell=0; cell < workset.numCells; ++cell)
    {
      for (std::size_t qp=0; qp < numQPs; ++qp)
      {
        const ScalarT& unscaled_phi = potential(cell,qp); //[V]
        ScalarT phi = unscaled_phi / V0; 
	
        // the scaled full RHS
        ScalarT charge; 
        charge = 0.0;  // no charge in metal bulk
        poissonSource(cell, qp) = factor*charge;
	
        // output states
        chargeDensity(cell, qp) = 0.0;    
        electronDensity(cell, qp) = 0.0;  
        holeDensity(cell, qp) = 0.0;      
        electricPotential(cell, qp) = phi*V0; 
        ionizedDopant(cell, qp) = 0.0;
        conductionBand(cell, qp) = qPhiRef-workFunc-phi*V0; // [eV]
        valenceBand(cell, qp) = conductionBand(cell,qp); // No band gap in metal
        approxQuanEDen(cell,qp) = 0.0;
        artCBDensity(cell, qp) = 0.0;
      }
    }
  }

  //! invalid material category
  else
  {
    TEUCHOS_TEST_FOR_EXCEPTION (true, Teuchos::Exceptions::InvalidParameter,
			std::endl << "Error!  Unknown material category " 
			<< matrlCategory << "!" << std::endl);
  } 
}


// **********************************************************************
template<typename EvalT, typename Traits>
void QCAD::PoissonSource<EvalT, Traits>::
evaluateFields_default(typename Traits::EvalData workset)
{
  MeshScalarT* coord;
  ScalarT charge;

  for (std::size_t cell=0; cell < workset.numCells; ++cell) 
  {
    for (std::size_t qp=0; qp < numQPs; ++qp) 
    {
      coord = &coordVec(cell,qp,0);
      const ScalarT& phi = potential(cell,qp);

      switch (numDims) {
      case 2:
        if (coord[1]<0.8) charge = (coord[1]*coord[1]);
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

  MeshScalarT* coord;
  
  for (std::size_t cell = 0; cell < workset.numCells; ++cell) 
  {
    for (std::size_t qp = 0; qp < numQPs; ++qp) 
    {
      coord = &coordVec(cell,qp,0);
      
     // Silicon region
     if ( (coord[0] > oxideWidth) && (coord[0] <= (oxideWidth + siliconWidth)) )
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
      ScalarT ni;  // intrinsic carrier concentration in [cm-3]
    
      Nc = NcvFactor*pow(mdn,1.5)*pow(temperature/Tref,1.5);  // in [cm-3]
      Nv = NcvFactor*pow(mdp,1.5)*pow(temperature/Tref,1.5); 
      Eg = Eg0-alpha*pow(temperature,2.0)/(beta+temperature); // in [eV]
    
      ScalarT kbT = kbBoltz*temperature;      // in [eV]
      ni = sqrt(Nc*Nv)*exp(-Eg/(2.0*kbT));    // in [cm-3]
    
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
        // retrieve Previous Poisson Potential
        Albany::MDArray prevPhiArray = (*workset.stateArrayPtr)["Previous Poisson Potential"];
        ScalarT prevPhi = prevPhiArray(cell,qp);
        ScalarT approxEDensity = 0.0;
        
        if(bUsePredictorCorrector)
          // compute the approximate quantum electron density using predictor-corrector method
          approxEDensity = eDensityForPoissonSchrodinger(workset, cell, qp, prevPhi, true, 0.0);

        else  // otherwise, use the exact quantum density
          approxEDensity = eDensityForPoissonSchrodinger(workset, cell, qp, 0.0, false, 0.0);

        // compute the exact quantum electron density
        ScalarT eDensity = eDensityForPoissonSchrodinger(workset, cell, qp, 0.0, false, 0.0);
          
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
        electricPotential(cell, qp) = phi*V0;
        ionizedDopant(cell, qp) = ionN;
        approxQuanEDen(cell,qp) = approxEDensity;
        artCBDensity(cell, qp) = ( eDensity > 1e-6 ? eDensity : -Nc*(this->*carrStat)( -(phi+eArgOffset) ));
        
        if (bIncludeVxc)  // include Vxc
        {
          ScalarT Vxc = computeVxcLDA(relPerm, averagedEffMass, approxEDensity);
          conductionBand(cell, qp) = qPhiRef-Chi-phi*V0 +Vxc; // [eV]
	  electricPotential(cell, qp) = phi*V0 + Vxc; //add xc correction to electric potential (used in CI delta_ij computation)
          // conductionBand(cell, qp) = qPhiRef -Chi -0.5*(phi*V0 +prevPhi) +Vxc; // [eV]
        }
        else { // not include Vxc
          conductionBand(cell, qp) = qPhiRef-Chi-phi*V0; // [eV]
	  electricPotential(cell, qp) = phi*V0;
          // conductionBand(cell, qp) = qPhiRef -Chi -0.5*(phi*V0 +prevPhi); // [eV]
	}
        
        valenceBand(cell, qp) = conductionBand(cell,qp)-Eg;

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
        electricPotential(cell, qp) = phi*V0;
        ionizedDopant(cell, qp) = ionN;
        conductionBand(cell, qp) = qPhiRef-Chi-phi*V0; // [eV]
        valenceBand(cell, qp) = conductionBand(cell,qp)-Eg;
        approxQuanEDen(cell,qp) = 0.0; 
        artCBDensity(cell, qp) = ( eDensity > 1e-6 ? eDensity 
				     : -Nc*(this->*carrStat)( -(phi+eArgOffset) ));
      }
      
     } // end of if ( (coord[0] > oxideWidth) ...)


     // Oxide region
     else if ((coord[0] >= 0) && (coord[0] <= oxideWidth))
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
        // retrieve Previous Poisson Potential
        Albany::MDArray prevPhiArray = (*workset.stateArrayPtr)["Previous Poisson Potential"];
        ScalarT prevPhi = prevPhiArray(cell,qp);
        ScalarT approxEDensity = 0.0;
        
        if (bUsePredictorCorrector) 
          // compute the approximate quantum electron density using predictor-corrector method
          approxEDensity = eDensityForPoissonSchrodinger(workset, cell, qp, prevPhi, true, 0.0);

        else
          approxEDensity = eDensityForPoissonSchrodinger(workset, cell, qp, 0.0, false, 0.0);

        // compute the exact quantum electron density
        ScalarT eDensity = eDensityForPoissonSchrodinger(workset, cell, qp, 0.0, false, 0.0);

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
	  electricPotential(cell, qp) = phi*V0 + Vxc; //add xc correction to electric potential (used in CI delta_ij computation)
          // conductionBand(cell, qp) = qPhiRef -Chi -0.5*(phi*V0 +prevPhi) +Vxc; // [eV]
        }
        else  { // not include Vxc
          conductionBand(cell, qp) = qPhiRef-Chi-phi*V0; // [eV]
	  electricPotential(cell, qp) = phi*V0;
          // conductionBand(cell, qp) = qPhiRef -Chi -0.5*(phi*V0 +prevPhi); // [eV]
	}
        
        valenceBand(cell, qp) = conductionBand(cell,qp)-Eg;
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
        electricPotential(cell, qp) = phi*V0;
        ionizedDopant(cell, qp) = 0.0;
        conductionBand(cell, qp) = qPhiRef-Chi-phi*V0; // [eV]
        valenceBand(cell, qp) = conductionBand(cell,qp)-Eg;
        approxQuanEDen(cell,qp) = 0.0;
        artCBDensity(cell, qp) = 0.0;
      }
     
     } // end of else if ((coord[0] >= 0) ...)
     
     else 
      TEUCHOS_TEST_FOR_EXCEPTION (true, Teuchos::Exceptions::InvalidParameter,
	       std::endl << "Error!  x-coord:" << coord[0] << "is outside the oxideWidth" << 
	       " + siliconWidth range: " << oxideWidth + siliconWidth << "!"<< std::endl);

    }  // end of loop over QPs
    
  }  // end of loop over cells
  
}


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
QCAD::PoissonSource<EvalT,Traits>::eDensityForPoissonSchrodinger 
  (typename Traits::EvalData workset, std::size_t cell, std::size_t qp, 
   const ScalarT prevPhi, const bool bUsePredCorr, const double Ef)
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
  ScalarT kbT = kbBoltz*temperature;  // in [eV]
  ScalarT eDensity = 0.0; 

  const std::vector<double>& neg_eigenvals = *(workset.eigenDataPtr->eigenvalueRe);
  std::vector<double> eigenvals( neg_eigenvals );
  for(unsigned int i=0; i<eigenvals.size(); ++i) eigenvals[i] *= -1; //apply minus sign (b/c of eigenval convention)

  // determine deltaPhi used in computing quantum electron density
  ScalarT deltaPhi = 0.0;  // 0 by default
  if (bUsePredCorr)  // true: apply the p-c method
  {
    const ScalarT& phi = potential(cell,qp); //[V]
    deltaPhi = (phi - prevPhi) / (kbT /1.0);  //[unitless]
  }
  else
    deltaPhi = 0.0;  // false: do not apply the p-c method
  
  // compute quantum electron density according to dimensionality
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
      ScalarT eDenPrefactor = valleyDegeneracyFactor*dos2D*kbT/X0;

      // loop over eigenvalues to compute electron density [cm^-3]
      for(int i = 0; i < nEigenvectors; i++) 
      {
        // note: wavefunctions are assumed normalized here 
        ScalarT wfSquared = ( eigenvector_Re[i](cell,qp)*eigenvector_Re[i](cell,qp) + 
 			      eigenvector_Im[i](cell,qp)*eigenvector_Im[i](cell,qp) );
 			  
        ScalarT tmpArg = (Ef-eigenvals[i])/kbT + deltaPhi;
        ScalarT logFunc; 
        if (tmpArg > MAX_EXPONENT)
          logFunc = tmpArg;  // exp(tmpArg) blows up for large positive tmpArg, leading to bad derivative
        else
          logFunc = log(1.0 + exp(tmpArg));
 			      
        eDensity += wfSquared*logFunc;
      }
      eDensity = eDenPrefactor*eDensity; // in [cm^-3]
      
      break; 
    }  // end of case 1 block


    case 2: // 2D wavefunction (2D confinement)
    {
      // mUnconfined = effective mass in the unconfined direction (z dir. when the 2D wavefunc. is in x-y plane)
      // For Delta2-band and assume SiO2/Si interface parallel to [100] plane, mUnconfined=0.19. 
      double mUnconfined = materialDB->getElementBlockParam<double>(workset.EBName,"Transverse Electron Effective Mass");
        
      // n1D below is a factor that is part of the line electron density 
      // in the unconfined dir. and includes spin degeneracy and in unit of [cm^-1]
      ScalarT n1D = sqrt(2.0*mUnconfined*m0*kbT/(pi*hbar*hbar*eVPerJ*cm2Perm2));
        
      // subband-independent prefactor in calculating electron density
      // X0^2 is used to scale wavefunc. squared from [um^-2] or [nm^-2] to [cm^-2]
      ScalarT eDenPrefactor = valleyDegeneracyFactor*n1D/pow(X0,2.);

      // loop over eigenvalues to compute electron density [cm^-3]
      for(int i=0; i < nEigenvectors; i++) 
      {
        // note: wavefunctions are assumed normalized here 
        ScalarT wfSquared = ( eigenvector_Re[i](cell,qp)*eigenvector_Re[i](cell,qp) + 
 			      eigenvector_Im[i](cell,qp)*eigenvector_Im[i](cell,qp) );
 			      
        ScalarT inArg = (Ef-eigenvals[i])/kbT + deltaPhi;
        eDensity += wfSquared * computeFDIntMinusOneHalf(inArg); 
      }
      eDensity = eDenPrefactor*eDensity; // in [cm^-3]

      break;
    }  // end of case 2 block    


    case 3: // 3D wavefunction (3D confinement)
    { 
      //degeneracy factor
      int spinDegeneracyFactor = 2;
      double degeneracyFactor = spinDegeneracyFactor * valleyDegeneracyFactor;
        
      // subband-independent prefactor in calculating electron density
      // X0^3 is used to scale wavefunc. squared from [um^-3] or [nm^-3] to [cm^-3]
      ScalarT eDenPrefactor = degeneracyFactor/pow(X0,3.);

      // loop over eigenvalues to compute electron density [cm^-3]
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

        // note: wavefunctions are assumed normalized here 
        ScalarT wfSquared = ( eigenvector_Re[i](cell,qp)*eigenvector_Re[i](cell,qp) + 
 			      eigenvector_Im[i](cell,qp)*eigenvector_Im[i](cell,qp) );				

        eDensity += wfSquared*fermiFactor; 
      }
      eDensity = eDenPrefactor*eDensity; // in [cm^-3]

      break;
    }  // end of case 3 block 
      
    default:
      TEUCHOS_TEST_FOR_EXCEPTION (true, Teuchos::Exceptions::InvalidParameter,
        std::endl << "Error!  Invalid number of dimensions " << numDims << "!"<< std::endl);
      break; 
      
  }  // end of switch (numDims) 

  return eDensity; 
}


//Similar as eDensityForPoissonSchrodinger but assume |wf|^2 values are given in eigenvector_Re[.] instead of eigenvector
// Note: sum of |wf|^2 values == N where N is the number of electrons in the wave function, not necessarily == 1
template<typename EvalT, typename Traits>
typename QCAD::PoissonSource<EvalT,Traits>::ScalarT
QCAD::PoissonSource<EvalT,Traits>::eDensityForPoissonCI 
  (typename Traits::EvalData workset, std::size_t cell, std::size_t qp, 
   const ScalarT prevPhi, const bool bUsePredCorr, const double Ef)
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

  ScalarT temperature = temperatureField(0); //get shared temperature parameter from field
  ScalarT kbT = kbBoltz*temperature;  // in [eV]
  ScalarT eDensity = 0.0; 
  //double Ef = 0.0;  //Fermi energy == 0

  const std::vector<double>& neg_eigenvals = *(workset.eigenDataPtr->eigenvalueRe);
  std::vector<double> eigenvals( neg_eigenvals );
  int nCIEvals = eigenvals.size(); //not necessarily == nEigenvectors, since CI could have not converged as many as requested
  int nEvals = std::min(nCIEvals, nEigenvectors); // the number of eigen-pairs to use (we don't gather more than nEigenvectors)

  //I don't believe CI eigenvalues are negated...
  //for(unsigned int i=0; i<nEvals; ++i) eigenvals[i] *= -1; //apply minus sign (b/c of eigenval convention)

  //Note: NO predictor corrector method used here yet -- need to understand what's going on better first
  
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
      ScalarT n1D = 1.0; //sqrt(2.0*mUnconfined*m0*kbT/(pi*hbar*hbar*eVPerJ*cm2Perm2));
        
      // subband-independent prefactor in calculating electron density
      // X0^2 is used to scale wavefunc. squared from [um^-2] or [nm^-2] to [cm^-2]
      ScalarT eDenPrefactor = valleyDegeneracyFactor*n1D/pow(X0,2.);

      
      // Get Z = sum( exp(-E_i/kT) )
      ScalarT Z = 0.0;
      for(int i=0; i < nEvals; i++) Z += exp(-eigenvals[i]/kbT);

      // loop over eigenvalues to compute electron density [cm^-3]
      for(int i=0; i < nEvals; i++) 
      {
        ScalarT wfSquared = ( eigenvector_Re[i](cell,qp) );
        ScalarT wfOcc = exp(-eigenvals[i]/kbT) / Z;
        eDensity += wfSquared * wfOcc;
      }
      eDensity = eDenPrefactor*eDensity; // in [cm^-3]

      break;
    }  // end of case 2 block    


    case 3: // 3D wavefunction (3D confinement)
    { 
      //degeneracy factor
      double degeneracyFactor = valleyDegeneracyFactor; //CI includes spin degeneracy
        
      // subband-independent prefactor in calculating electron density
      // X0^3 is used to scale wavefunc. squared from [um^-3] or [nm^-3] to [cm^-3]
      ScalarT eDenPrefactor = degeneracyFactor/pow(X0,3.);

      // Get Z = sum( exp(-E_i/kT) )
      //ScalarT Z = 0.0;
      //for(int i=0; i < nEvals; i++) Z += exp(-eigenvals[i]/kbT);

      // loop over eigenvalues to compute electron density [cm^-3]
      for(int i = 0; i < nEvals; i++) 
      {
        ScalarT wfSquared = ( eigenvector_Re[i](cell,qp) );
        ScalarT oneOverOcc = 0.0;  // get 1/(exp(-eigenvals[i]/kbT) / Z) as sum, then invert (avoids inf issues)
	for(int j=0; j < nEvals; j++) oneOverOcc += exp(-(eigenvals[j]-eigenvals[i])/kbT);

	ScalarT wfOcc = 0.0; //exp(-eigenvals[i]/kbT) / Z;
	if(!std::isinf(QCAD::EvaluatorTools<EvalT,Traits>::getDoubleValue(oneOverOcc)))
	  wfOcc = 1/oneOverOcc; //otherwise just leave as zero since denom is infinite

        eDensity += wfSquared * wfOcc;
      }
      eDensity = eDenPrefactor*eDensity; // in [cm^-3]

      break;
    }  // end of case 3 block 
      
    default:
      TEUCHOS_TEST_FOR_EXCEPTION (true, Teuchos::Exceptions::InvalidParameter,
        std::endl << "Error!  Invalid number of dimensions " << numDims << "!"<< std::endl);
      break; 
      
  }  // end of switch (numDims) 

  return eDensity; 
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

