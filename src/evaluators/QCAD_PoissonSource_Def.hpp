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
PoissonSource(Teuchos::ParameterList& p) :
  coordVec(p.get<std::string>("Coordinate Vector Name"),
     p.get<Teuchos::RCP<PHX::DataLayout> >("QP Vector Data Layout")),
  potential(p.get<std::string>("Variable Name"),
      p.get<Teuchos::RCP<PHX::DataLayout> >("QP Scalar Data Layout")),
  temperatureField(p.get<std::string>("Temperature Name"),
      p.get<Teuchos::RCP<PHX::DataLayout> >("Shared Param Data Layout")),
  poissonSource(p.get<std::string>("Source Name"),
      p.get<Teuchos::RCP<PHX::DataLayout> >("QP Scalar Data Layout")),
  chargeDensity("Charge Density",
      p.get<Teuchos::RCP<PHX::DataLayout> >("QP Scalar Data Layout")),
  electronDensity("Electron Density",
      p.get<Teuchos::RCP<PHX::DataLayout> >("QP Scalar Data Layout")),
  artCBDensity("Artificial Conduction Band Density",
      p.get<Teuchos::RCP<PHX::DataLayout> >("QP Scalar Data Layout")),
  holeDensity("Hole Density",
      p.get<Teuchos::RCP<PHX::DataLayout> >("QP Scalar Data Layout")),
  electricPotential("Electric Potential",
      p.get<Teuchos::RCP<PHX::DataLayout> >("QP Scalar Data Layout")),
  ionizedDopant("Ionized Dopant",
      p.get<Teuchos::RCP<PHX::DataLayout> >("QP Scalar Data Layout")),
  conductionBand("Conduction Band",
      p.get<Teuchos::RCP<PHX::DataLayout> >("QP Scalar Data Layout")),
  valenceBand("Valence Band",
      p.get<Teuchos::RCP<PHX::DataLayout> >("QP Scalar Data Layout")),
  approxQuantumEDensity("Approximate Quantum Electron Density",
      p.get<Teuchos::RCP<PHX::DataLayout> >("QP Scalar Data Layout"))
{
  // Material database
  materialDB = p.get< Teuchos::RCP<QCAD::MaterialDatabase> >("MaterialDB");

  Teuchos::ParameterList* psList = p.get<Teuchos::ParameterList*>("Parameter List");

  Teuchos::RCP<const Teuchos::ParameterList> reflist = 
      this->getValidPoissonSourceParameters();
  psList->validateParameters(*reflist,0);

  Teuchos::RCP<PHX::DataLayout> vector_dl =
      p.get< Teuchos::RCP<PHX::DataLayout> >("QP Vector Data Layout");
  std::vector<PHX::DataLayout::size_type> dims;
  vector_dl->dimensions(dims);
  numQPs  = dims[1];
  numDims = dims[2];

  // get values from the input .xml and use default values if not provided
  factor = psList->get("Factor", 1.0);
  device = psList->get("Device", "defaultdevice");
  carrierStatistics = psList->get("Carrier Statistics", "Boltzmann Statistics");
  incompIonization = psList->get("Incomplete Ionization", "False");
  //dopingDonor = psList->get("Donor Doping", 1e14);
  //dopingAcceptor = psList->get("Acceptor Doping", 1e14);
  //donorActE = psList->get("Donor Activation Energy", 0.040);
  //acceptorActE = psList->get("Acceptor Activation Energy", 0.045);
  
  // passed down from main list
  length_unit_in_m = p.get<double>("Length unit in m");
  bSchrodingerInQuantumRegions = p.get<bool>("Use Schrodinger source");
  bUsePredictorCorrector = p.get<bool>("Use predictor-corrector method");

  if(bSchrodingerInQuantumRegions) {
    std::string evecFieldRoot = p.get<string>("Eigenvector field name root");
    nEigenvectors = p.get<int>("Schrodinger eigenvectors");

    eigenvector_Re.resize(nEigenvectors);
    eigenvector_Im.resize(nEigenvectors);

    char buf[200];
    Teuchos::RCP<PHX::DataLayout> dl = 
      p.get<Teuchos::RCP<PHX::DataLayout> >("QP Scalar Data Layout");

    for (int k = 0; k < nEigenvectors; ++k) {
      sprintf(buf, "%s_Re%d", evecFieldRoot.c_str(), k);
      PHX::MDField<ScalarT,Cell,QuadPoint> fr(buf,dl);
      eigenvector_Re[k] = fr; this->addDependentField(eigenvector_Re[k]);

      sprintf(buf, "%s_Im%d", evecFieldRoot.c_str(), k);
      PHX::MDField<ScalarT,Cell,QuadPoint> fi(buf,dl);
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
      new Sacado::ParameterRegistration<EvalT, SPL_Traits>(*s, this, paramLib);
      materialParams[*s] = psList->get<double>(*s);
    }
  }
  for(s = chargeParamNames.begin(); s != chargeParamNames.end(); s++) {
    if( psList->isParameter(*s) ) {
      new Sacado::ParameterRegistration<EvalT, SPL_Traits>(*s, this, paramLib);
      materialParams[*s] = psList->get<double>(*s);
    }
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
  this->addEvaluatedField(approxQuantumEDensity);
  
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
  this->utils.setFieldData(approxQuantumEDensity,fm);

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
  else TEST_FOR_EXCEPT(true); return factor; //dummy so all control paths return
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
  //validPL->set<double>("Donor Doping", 1e14, "Doping for nsilicon element blocks [cm^-3]");
  //validPL->set<double>("Acceptor Doping", 1e14, "Doping for psilicon element blocks [cm^-3]");
  validPL->set<string>("Carrier Statistics", "Boltzmann Statistics", "Carrier statistics");
  validPL->set<string>("Incomplete Ionization", "False", "Partial ionization of dopants");
  //validPL->set<double>("Donor Activation Energy", 0.045, "Donor activation energy [eV]");
  //validPL->set<double>("Acceptor Activation Energy", 0.045, "Acceptor activation energy [eV]");
  
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
  ScalarT V0 = kbBoltz*temperature/1.0; // kb*T/q in [V], scaling for potential        
  ScalarT Lambda2C0 = V0*eps0/(eleQ*X0*X0); // derived scaling factor (unitless)
  
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
      TEST_FOR_EXCEPTION (true, Teuchos::Exceptions::InvalidParameter, std::endl 
			  << "Error!  Invalid category " << category 
			  << " for reference material !" << std::endl);
    }
  }  

  string matrlCategory = materialDB->getElementBlockParam<string>(workset.EBName,"Category");


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
    
    //! function pointer to carrier statistics member function
    ScalarT (QCAD::PoissonSource<EvalT,Traits>::*carrStat) (const ScalarT);
    
    if (carrierStatistics == "Boltzmann Statistics")
      carrStat = &QCAD::PoissonSource<EvalT,Traits>::computeMBStat;  

    else if (carrierStatistics == "Fermi-Dirac Statistics")
      carrStat = &QCAD::PoissonSource<EvalT,Traits>::computeFDIntOneHalf;

    else if (carrierStatistics == "0-K Fermi-Dirac Statistics")
      carrStat = &QCAD::PoissonSource<EvalT,Traits>::computeZeroKFDInt;

    else TEST_FOR_EXCEPTION (true, Teuchos::Exceptions::InvalidParameter,
        std::endl << "Error!  Unknown carrier statistics ! " << std::endl);

    //! function pointer to ionized dopants member function
    ScalarT (QCAD::PoissonSource<EvalT,Traits>::*ionDopant) (const std::string, const ScalarT&); 
    
    if (incompIonization == "False")
      ionDopant = &QCAD::PoissonSource<EvalT,Traits>::fullDopants; 
    
    else if (incompIonization == "True")
      ionDopant = &QCAD::PoissonSource<EvalT,Traits>::ionizedDopants;
    
    else TEST_FOR_EXCEPTION (true, Teuchos::Exceptions::InvalidParameter,
        std::endl << "Error!  Invalid incomplete ionization option ! " << std::endl);


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
      else if( materialDB->isElementBlockParam(workset.EBName, "Doping Parameter Name") ) 
        dopingConc = materialParams[ materialDB->getElementBlockParam<string>(workset.EBName,"Doping Parameter Name") ];
      else TEST_FOR_EXCEPTION (true, Teuchos::Exceptions::InvalidParameter,
        std::endl << "Error!  Unknown dopant concentration for " << workset.EBName << "!"<< std::endl);

      if(dopantType == "Donor") 
        inArg = eArgOffset + dopantActE/kbT;
      else if(dopantType == "Acceptor") 
        inArg = hArgOffset + dopantActE/kbT;
      else TEST_FOR_EXCEPTION (true, Teuchos::Exceptions::InvalidParameter,
	       std::endl << "Error!  Unknown dopant type " << dopantType << "!"<< std::endl);
    }
    else {
      dopingProfile = "Constant";
      dopingConc = 0.0;
      inArg = 0.0;
    }


    //! Schrodinger source for electrons
    if(bSchrodingerInQuantumRegions && 
      materialDB->getElementBlockParam<bool>(workset.EBName,"quantum",false)) 
    {
      // retrieve Previous Poisson Potential
      Albany::MDArray prevPhiArray = (*workset.stateArrayPtr)["Previous Poisson Potential"];

      // loop over cells and qps
      for (std::size_t cell=0; cell < workset.numCells; ++cell)
      {
        for (std::size_t qp=0; qp < numQPs; ++qp)
        {
          ScalarT approxEDensity = 0.0;
          if(bUsePredictorCorrector)
          {
            ScalarT prevPhi = prevPhiArray(cell,qp);
            
            // compute the approximate quantum electron density using predictor-corrector method
            approxEDensity = eDensityForPoissonSchrond(workset, cell, qp, prevPhi, true);
          }
          else  // otherwise, use the exact quantum density
            approxEDensity = eDensityForPoissonSchrond(workset, cell, qp, 0.0, false);

          // compute the exact quantum electron density
          ScalarT eDensity = eDensityForPoissonSchrond(workset, cell, qp, 0.0, false);
          
          // obtain the scaled potential
          const ScalarT& phi = potential(cell,qp);
           
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
          charge = 1.0/Lambda2C0*(hDensity- approxEDensity + ionN);
          poissonSource(cell, qp) = factor*charge;

          // output states
          chargeDensity(cell, qp) = charge*Lambda2C0;
          electronDensity(cell, qp) = eDensity;
          holeDensity(cell, qp) = hDensity;
          electricPotential(cell, qp) = phi*V0;
          ionizedDopant(cell, qp) = ionN;
          conductionBand(cell, qp) = qPhiRef-Chi-phi*V0; // [eV]
          valenceBand(cell, qp) = conductionBand(cell,qp)-Eg;
          approxQuantumEDensity(cell,qp) = approxEDensity;
	  artCBDensity(cell, qp) = ( eDensity > 1e-6 ? eDensity : -Nc*(this->*carrStat)( -(phi+eArgOffset) ));
        }
      }  // end of loop over cells
    }

    //! calculate the classical charge (RHS) for Poisson equation
    else
    {
      for (std::size_t cell=0; cell < workset.numCells; ++cell)
      {
        for (std::size_t qp=0; qp < numQPs; ++qp)
        {
          const ScalarT& phi = potential(cell,qp);
          
          // obtain the ionized dopants
          ScalarT ionN;
          if (dopantType == "Donor")  // function takes care of sign
            ionN = (this->*ionDopant)(dopantType,phi+inArg)*dopingConc;
          else if (dopantType == "Acceptor")
            ionN = (this->*ionDopant)(dopantType,-phi+inArg)*dopingConc;
          else 
            ionN = 0.0; 

          // the scaled full RHS
          ScalarT charge; 
          charge = 1.0/Lambda2C0*(Nv*(this->*carrStat)(-phi+hArgOffset)- Nc*(this->*carrStat)(phi+eArgOffset) + ionN);
          poissonSource(cell, qp) = factor*charge;
          
          // output states
          chargeDensity(cell, qp) = charge*Lambda2C0;
          electronDensity(cell, qp) = Nc*(this->*carrStat)(phi+eArgOffset);
          holeDensity(cell, qp) = Nv*(this->*carrStat)(-phi+hArgOffset);
          electricPotential(cell, qp) = phi*V0;
          ionizedDopant(cell, qp) = ionN;
          conductionBand(cell, qp) = qPhiRef-Chi-phi*V0; // [eV]
          valenceBand(cell, qp) = conductionBand(cell,qp)-Eg;
          approxQuantumEDensity(cell,qp) = 0.0; 
	  artCBDensity(cell, qp) = ( electronDensity(cell, qp) > 1e-6 ? electronDensity(cell, qp) 
				     : -Nc*(this->*carrStat)( -(phi+eArgOffset) ));

        }
      }
    }

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
    if( materialDB->isElementBlockParam(workset.EBName, "Charge Value") ) 
      fixedCharge = materialDB->getElementBlockParam<double>(workset.EBName,"Charge Value");
    else if( materialDB->isElementBlockParam(workset.EBName, "Charge Parameter Name") ) 
      fixedCharge  = materialParams[ materialDB->getElementBlockParam<string>(workset.EBName,"Charge Parameter Name") ];
    else fixedCharge = 0.0; 


    //! Schrodinger source for electrons
    if(bSchrodingerInQuantumRegions && 
      materialDB->getElementBlockParam<bool>(workset.EBName,"quantum",false)) 
    {
      // retrieve Previous Poisson Potential
      Albany::MDArray prevPhiArray = (*workset.stateArrayPtr)["Previous Poisson Potential"];

      // loop over cells and qps
      for (std::size_t cell=0; cell < workset.numCells; ++cell)
      {
        for (std::size_t qp=0; qp < numQPs; ++qp)
        {
          ScalarT approxEDensity = 0.0;
          if (bUsePredictorCorrector) 
          {
            ScalarT prevPhi = prevPhiArray(cell,qp);
            
            // compute the approximate quantum electron density using predictor-corrector method
            approxEDensity = eDensityForPoissonSchrond(workset, cell, qp, prevPhi, true);
          }
          else
            approxEDensity = eDensityForPoissonSchrond(workset, cell, qp, 0.0, false);

          // compute the exact quantum electron density
          ScalarT eDensity = eDensityForPoissonSchrond(workset, cell, qp, 0.0, false);

          // obtain the scaled potential
          const ScalarT& phi = potential(cell,qp);

          //(No other classical density in insulator)
              
          // the scaled full RHS
          ScalarT charge;
          charge = 1.0/Lambda2C0*(-approxEDensity + fixedCharge);
          poissonSource(cell, qp) = factor*charge;

          // output states
          chargeDensity(cell, qp) = -eDensity + fixedCharge; 
          electronDensity(cell, qp) = eDensity;  // quantum electrons in an insulator
          holeDensity(cell, qp) = 0.0;           // no holes in an insulator
          electricPotential(cell, qp) = phi*V0;
          ionizedDopant(cell, qp) = 0.0;
          conductionBand(cell, qp) = qPhiRef-Chi-phi*V0; // [eV]
          valenceBand(cell, qp) = conductionBand(cell,qp)-Eg;
          approxQuantumEDensity(cell,qp) = approxEDensity;
	  artCBDensity(cell, qp) = eDensity;

        }
      }  // end of loop over cells
    }
    
    else { // use semiclassical source

      for (std::size_t cell=0; cell < workset.numCells; ++cell)
      {
        for (std::size_t qp=0; qp < numQPs; ++qp)
        {
          const ScalarT& phi = potential(cell,qp);
          
          // the scaled full RHS
          ScalarT charge; 
          charge = 1.0/Lambda2C0*fixedCharge;  // only fixed charge in an insulator
          poissonSource(cell, qp) = factor*charge;
	  
          chargeDensity(cell, qp) = fixedCharge; // fixed space charge in an insulator
          electronDensity(cell, qp) = 0.0;       // no electrons in an insulator
          holeDensity(cell, qp) = 0.0;           // no holes in an insulator
          electricPotential(cell, qp) = phi*V0;
          ionizedDopant(cell, qp) = 0.0;
          conductionBand(cell, qp) = qPhiRef-Chi-phi*V0; // [eV]
          valenceBand(cell, qp) = conductionBand(cell,qp)-Eg;
          approxQuantumEDensity(cell,qp) = 0.0;
	  artCBDensity(cell, qp) = 0.0;
        }
      }
    }
  }  // end of else if(matrlCategory == "Insulator")


  //***************************************************************************
  //! element block with "Metal" material
  //***************************************************************************
  else if(matrlCategory == "Metal")
  {
    double workFunc = materialDB->getElementBlockParam<double>(workset.EBName,"Work Function");
    
    // The following assumes Metal is surrounded by Dirichlet BC
    for (std::size_t cell=0; cell < workset.numCells; ++cell)
    {
      for (std::size_t qp=0; qp < numQPs; ++qp)
      {
        const ScalarT& phi = potential(cell,qp);
        
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
        approxQuantumEDensity(cell,qp) = 0.0;
	artCBDensity(cell, qp) = 0.0;
      }
    }
  }

  //! invalid material category
  else
  {
    TEST_FOR_EXCEPTION (true, Teuchos::Exceptions::InvalidParameter,
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
      default: TEST_FOR_EXCEPT(true);
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
      approxQuantumEDensity(cell,qp) = 0.0;
      artCBDensity(cell, qp) = 0.0;
    }
  }
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
    TEST_FOR_EXCEPTION (true, Teuchos::Exceptions::InvalidParameter,
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
    ionDopants = 1.0 / (1. + 2.*exp(x));  
  else if (dopType == "Acceptor")
    ionDopants = -1.0 / (1. + 4.*exp(x));
  else if (dopType == "None")
    ionDopants = 0.0;
  else
  {
    ionDopants = 0.0;
    TEST_FOR_EXCEPTION (true, Teuchos::Exceptions::InvalidParameter,
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
QCAD::PoissonSource<EvalT,Traits>::eDensityForPoissonSchrond 
(typename Traits::EvalData workset, std::size_t cell, std::size_t qp, const ScalarT prevPhi, const bool bUsePredCorr)
{
  // Use the predictor-corrector method proposed by A. Trellakis, A. T. Galick,  
  // and U. Ravaioli, "Iteration scheme for the solution of the two-dimensional  
  // Schrodinger-Poisson equations in quantum structures", J. Appl. Phys. 81, 7880 (1997). 
  // If bUsePredCorr = true, use the predictor-corrector approach.
  // If bUsePredCorr = false, calculate the exact quantum electron density. 
  
  // unit conversion factor
  double eVPerJ = 1.0/eleQ; 
  double cm2Perm2 = 1.0e4; 
  
  // For Delta2-band in Silicon, valley degeneracy factor = 2
  int valleyDegeneracyFactor = materialDB->getElementBlockParam<int>(workset.EBName,"Num of conduction band min",2);

  // Scaling factors
  double X0 = length_unit_in_m/1e-2; // length scaling to get to [cm] (structure dimension in [um])

  ScalarT temperature = temperatureField(0); //get shared temperature parameter from field
  ScalarT kbT = kbBoltz*temperature;  // in [eV]
  ScalarT eDensity = 0.0; 
  double Ef = 0.0;  //Fermi energy == 0

  const std::vector<double>& neg_eigenvals = *(workset.eigenDataPtr->eigenvalueRe);
  std::vector<double> eigenvals( neg_eigenvals );
  for(unsigned int i=0; i<eigenvals.size(); ++i) eigenvals[i] *= -1; //apply minus sign (b/c of eigenval convention)
  
  // determine deltaPhi used in computing quantum electron density
  ScalarT deltaPhi = 0.0;  // 0 by default
  
  // if (abs(prevPhi) > 0.0) // for non-zero prevPhi, apply the predictor-corrector method
  // This is a bad condition since prevPhi can be 0 for given (cell,qp) even when bUsePredCorr = true
  
  if (bUsePredCorr)  // true: apply the p-c method
  {
    const ScalarT& phi = potential(cell,qp);
    deltaPhi = phi - prevPhi; 
  }
  else
    deltaPhi = 0.0;  // false: do not apply the p-c method
  
  // compute quantum electron density according to dimensionality
  switch (numDims)
  {
    case 1: // 1D wavefunction (1D confinement)
    {  
      // m11 = in-plane effective mass (x-y plane when the 1D wavefunc. is along z)
      // For Delta2-band (or valley), m11 is the transverse effective mass (0.19). 
      
      double m11 = materialDB->getElementBlockParam<double>(workset.EBName,"Electron Effective Mass Y");
        
      // 2D density of states in [#/(eV.cm^2)] where 2D is the unconfined x-y plane
      // dos2D below includes the spin degeneracy of 2
      double dos2D = m11*m0/(pi*hbar*hbar*eVPerJ*cm2Perm2); 
        
      // subband-independent prefactor in calculating electron density
      // X0 is used to scale wavefunc. squared from [um^-1] or [nm^-1] to [cm^-1]
      ScalarT eDenPrefactor = valleyDegeneracyFactor*dos2D*kbT/X0;

      // loop over eigenvalues to compute electron density [cm^-3]
      for(int i = 0; i < nEigenvectors; i++) 
      {
        // note: wavefunctions are assumed normalized here 
        // (need to normalize them in the Schrodinger solver)
        ScalarT wfSquared = ( eigenvector_Re[i](cell,qp)*eigenvector_Re[i](cell,qp) + 
 			      eigenvector_Im[i](cell,qp)*eigenvector_Im[i](cell,qp) );
        eDensity += wfSquared*log(1. + exp((Ef-eigenvals[i])/kbT + deltaPhi) );
      }
      eDensity = eDenPrefactor*eDensity; // in [cm^-3]
      
      break; 
    }  // end of case 1 block


    case 2: // 2D wavefunction (2D confinement)
    {
      // mUnconfined = effective mass in the unconfined direction (x dir. when the 2D wavefunc. is in y-z plane)
      // For Delta2-band and assume SiO2/Si interface parallel to [100] plane, mUnconfined=0.19. 
      double mUnconfined = materialDB->getElementBlockParam<double>(workset.EBName,"Electron Effective Mass X");
        
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
        // (need to normalize them in the Schrodinger solver)
        ScalarT wfSquared = ( eigenvector_Re[i](cell,qp)*eigenvector_Re[i](cell,qp) + 
 			      eigenvector_Im[i](cell,qp)*eigenvector_Im[i](cell,qp) );
        ScalarT inArg = (Ef-eigenvals[i])/kbT + deltaPhi;
        eDensity += wfSquared*computeFDIntMinusOneHalf(inArg); 
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
        // Fermi-Dirac distribution
        ScalarT fermiFactor = 1.0/( exp((eigenvals[i]-Ef)/kbT + deltaPhi) + 1.0 );
              
        // note: wavefunctions are assumed normalized here 
        // (need to normalize them in the Schrodinger solver)
        ScalarT wfSquared = ( eigenvector_Re[i](cell,qp)*eigenvector_Re[i](cell,qp) + 
 			      eigenvector_Im[i](cell,qp)*eigenvector_Im[i](cell,qp) );
        eDensity += wfSquared*fermiFactor; 
      }
      eDensity = eDenPrefactor*eDensity; // in [cm^-3]

      break;
    }  // end of case 3 block 
      
    default:
      TEST_FOR_EXCEPTION (true, Teuchos::Exceptions::InvalidParameter,
        std::endl << "Error!  Invalid number of dimensions " << numDims << "!"<< std::endl);
      break; 
      
  }  // end of switch (numDims) 

  return eDensity; 
}


// **********************************************************************

//TODO: remove after new version is tested
/*template<typename EvalT,typename Traits>
std::vector<double>
QCAD::PoissonSource<EvalT,Traits>::ReadEigenvaluesFromFile(int numberToRead)
{
  std::vector<double> eigenvals;

  //Open eigenvalue filename and read into eigenvals vector
  std::ifstream evalData;
  evalData.open(eigenValueFilename.c_str());
  TEST_FOR_EXCEPTION(!evalData.is_open(), Teuchos::Exceptions::InvalidParameter,
		     std::endl << "Error! Cannot open eigenvalue filename  " 
		     << eigenValueFilename << std::endl);

  eigenvals.resize(numberToRead);

  const double TOL = 1e-6;
  char buf[100];
  int index;
  double RePart, ImPart;

  evalData.getline(buf,100); //skip header
  while ( !evalData.eof() ) {
    evalData >> index >> RePart >> ImPart;
    if(fabs(ImPart) > TOL)
      std::cout << "WARNING: eigenvalue " << index << " has Im Part: " 
		<< RePart << " + " << ImPart << "i" << std::endl;
    if(index < numberToRead) eigenvals[index] = -RePart; //negative b/c of convention
    //std::cout << "DEBUG eval(" << index << ") = " << RePart << std::endl;
  }
  evalData.close();
  return eigenvals;
}
*/
