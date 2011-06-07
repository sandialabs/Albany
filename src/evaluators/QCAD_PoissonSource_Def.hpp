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
  poissonSource(p.get<std::string>("Source Name"),
      p.get<Teuchos::RCP<PHX::DataLayout> >("QP Scalar Data Layout")),
  chargeDensity("Charge Density",
      p.get<Teuchos::RCP<PHX::DataLayout> >("QP Scalar Data Layout")),
  electronDensity("Electron Density",
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
      p.get<Teuchos::RCP<PHX::DataLayout> >("QP Scalar Data Layout"))
{
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
  dopingDonor = psList->get("Donor Doping", 1e14);
  dopingAcceptor = psList->get("Acceptor Doping", 1e14);
  donorActE = psList->get("Donor Activation Energy", 0.040);
  acceptorActE = psList->get("Acceptor Activation Energy", 0.045);

  // Material database
  materialDB = p.get< Teuchos::RCP<QCAD::MaterialDatabase> >("MaterialDB");
  
  // passed down from main list
  temperature = p.get<double>("Temperature");
  length_unit_in_m = p.get<double>("Length unit in m");
  bSchrodingerInQuantumRegions = p.get<bool>("Use Schrodinger source");

  if(bSchrodingerInQuantumRegions) {
    eigenValueFilename = p.get<string>("Eigenvalues file");
    nEigenvectors = p.get<int>("Schrodinger eigenvectors");
    evecStateRoot = p.get<string>("Eigenvector state name root");
  }
  else {
    nEigenvectors = 0;
    eigenValueFilename = "";
    evecStateRoot = "";
  }

  // Scaling factors
  X0 = length_unit_in_m/1e-2; // length scaling to get to [cm] (structure dimension in [um])
  V0 = kbBoltz*temperature/1.0; // kb*T/q in [V], scaling for potential

  // Add factor  and temperature as a Sacado-ized parameters
  Teuchos::RCP<ParamLib> paramLib =
      p.get< Teuchos::RCP<ParamLib> >("Parameter Library", Teuchos::null);
  new Sacado::ParameterRegistration<EvalT, SPL_Traits>(
      "Poisson Source Factor", this, paramLib);
  new Sacado::ParameterRegistration<EvalT, SPL_Traits>(
      "Lattice Temperature", this, paramLib);

  this->addDependentField(potential);
  this->addDependentField(coordVec);

  this->addEvaluatedField(poissonSource);
  this->addEvaluatedField(chargeDensity);
  this->addEvaluatedField(electronDensity);
  this->addEvaluatedField(holeDensity);
  this->addEvaluatedField(electricPotential);
  this->addEvaluatedField(ionizedDopant);
  this->addEvaluatedField(conductionBand);
  this->addEvaluatedField(valenceBand);
  
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

  this->utils.setFieldData(chargeDensity,fm);
  this->utils.setFieldData(electronDensity,fm);
  this->utils.setFieldData(holeDensity,fm);
  this->utils.setFieldData(electricPotential,fm);

  this->utils.setFieldData(ionizedDopant,fm);
  this->utils.setFieldData(conductionBand,fm);
  this->utils.setFieldData(valenceBand,fm);
}

// **********************************************************************
template<typename EvalT, typename Traits>
void QCAD::PoissonSource<EvalT, Traits>::
evaluateFields(typename Traits::EvalData workset)
{
  //! implement RHS of the scaled Poisson eqn in a pmos capacitor 
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
  else if(n == "Lattice Temperature") return temperature;
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
  validPL->set<double>("Donor Doping", 1e14, "Doping for nsilicon element blocks [cm^-3]");
  validPL->set<double>("Acceptor Doping", 1e14, "Doping for psilicon element blocks [cm^-3]");
  validPL->set<string>("Carrier Statistics", "Boltzmann Statistics", "Carrier statistics");
  validPL->set<string>("Incomplete Ionization", "False", "Partial ionization of dopants");
  validPL->set<double>("Donor Activation Energy", 0.045, "Donor activation energy [eV]");
  validPL->set<double>("Acceptor Activation Energy", 0.045, "Acceptor activation energy [eV]");
  
  return validPL;
}


// *****************************************************************************
template<typename EvalT, typename Traits>
void QCAD::PoissonSource<EvalT, Traits>::
evaluateFields_elementblocks(typename Traits::EvalData workset)
{
  string matrlCategory = materialDB->getElementBlockParam<string>(workset.EBName,"Category");

  //! Constant energy reference for heterogeneous structures
  ScalarT qPhiRef;
  string refMtrlName = "Silicon"; //hardcoded reference material name - change this later
  {
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


  //! Schrodinger source
  if(bSchrodingerInQuantumRegions && 
     materialDB->getElementBlockParam<bool>(workset.EBName,"quantum",false)) {

    //read eigenvalues from file
    std::vector<double> eigenvals = ReadEigenvaluesFromFile(nEigenvectors);
    
    cout << "DEBUG: Using Schrodinger source for element block " << workset.EBName << endl;
    ScalarT scalingFctr = V0*eps0*X0/eleQ; // scaling factor for charge density with dimensions L^-d
    ScalarT kbT = kbBoltz*temperature;      // in [eV]
    char buf[100];

    //Degeneracy factor
    int spinDegeneracyFactor = 2;
    int valleyDegeneracyFactor = materialDB->getElementBlockParam<int>(workset.EBName,"Num of conduction band min",1);
    double degeneracyFactor = spinDegeneracyFactor * valleyDegeneracyFactor;

    //TODO: Suzey: add 1D and 2D cases where we need integral over k-states - I just include order of magnitude estimate here
    double dimFactor = 1.0, pi = 3.141592, aSi = 5e-6; // (HACK just for now)
    switch (numDims) {
    case 1: dimFactor = pow(2*pi/aSi,2); break;
    case 2: dimFactor = pow(2*pi/aSi,1); break;
    }

    //! Zero out poisson source field -- there's probably a function call that does this that I don't know about
    for (std::size_t cell=0; cell < workset.numCells; ++cell)
      for (std::size_t qp=0; qp < numQPs; ++qp)
	poissonSource(cell, qp) = 0.0; 

    //! Compute schrodinger source
    for(int i=0; i<nEigenvectors; i++) {

      double Ef = 0.0;  //Fermi energy == 0
      ScalarT fermiFactor = 1.0/( exp( (eigenvals[i]-Ef)/kbT) + 1.0 );
      cout << "DEBUG: Eigenvector " << i << " (eval=" << eigenvals[i] << "eV) with weight " << fermiFactor << endl;

      Albany::StateVariables& newState = *workset.newState;
      sprintf(buf,"%s_Re%d", evecStateRoot.c_str(), i);
      Intrepid::FieldContainer<RealType>& wfRe = *newState[buf];
      sprintf(buf,"%s_Im%d", evecStateRoot.c_str(), i);
      Intrepid::FieldContainer<RealType>& wfIm = *newState[buf];

      for (std::size_t cell=0; cell < workset.numCells; ++cell) {
	for (std::size_t qp=0; qp < numQPs; ++qp) {

	  //charge contribution from ith wavefunction in scaled units
	  ScalarT quantumCharge = (pow(wfRe(cell,qp),2) + pow(wfIm(cell,qp),2))*fermiFactor; // in [L^-d]

	  // the scaled full RHS
	  quantumCharge = -1/scalingFctr * degeneracyFactor * dimFactor * quantumCharge; // in scaled units
	  poissonSource(cell, qp) += factor*quantumCharge;
	}
      }
    }

    //! output states
    double Chi = materialDB->getElementBlockParam<double>(workset.EBName,"Electron Affinity");
    ScalarT Eg;

    if(matrlCategory == "Semiconductor") {
      double Eg0 = materialDB->getElementBlockParam<double>(workset.EBName,"Zero Temperature Band Gap");
      double alpha = materialDB->getElementBlockParam<double>(workset.EBName,"Band Gap Alpha Coefficient");
      double beta = materialDB->getElementBlockParam<double>(workset.EBName,"Band Gap Beta Coefficient");
      Eg = Eg0-alpha*pow(temperature,2.0)/(beta+temperature); // in [eV]
    }
    else {
      Eg = materialDB->getElementBlockParam<double>(workset.EBName,"Band Gap",0.0);
    }

    for (std::size_t cell=0; cell < workset.numCells; ++cell) {
      for (std::size_t qp=0; qp < numQPs; ++qp) {
	const ScalarT& phi = potential(cell,qp);
	ScalarT charge = poissonSource(cell, qp); 
	
	// output states
	chargeDensity(cell, qp) = charge*Lambda2*C0;
	electronDensity(cell, qp) = charge*Lambda2*C0; //all electrons ?
	holeDensity(cell, qp) = 0.0;
	electricPotential(cell, qp) = phi*V0;
        conductionBand(cell, qp) = qPhiRef-Chi-phi*V0; // [eV]
        valenceBand(cell, qp) = conductionBand(cell,qp)-Eg;
      }
    }

  }


  //***************************************************************************
  //! element block with "Semiconductor" material
  //***************************************************************************
  else if(matrlCategory == "Semiconductor") 
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
    double NcvFactor = 2.0*pow((kbBoltz*1.602e-19*m0*Tref)/(2*pi*pow(hbar,2)),3./2.)*1e-6;
            // 1.602e-19 converts kbBoltz in [eV/K] to [J/K], 1e-6 converts [m-3] to [cm-3]
    
    //! strong temperature-dependent material parameters
    ScalarT Nc;  // conduction band effective DOS in [cm-3]
    ScalarT Nv;  // valence band effective DOS in [cm-3]
    ScalarT Eg;  // band gap at T [K] in [eV]
    ScalarT ni;  // intrinsic carrier concentration in [cm-3]
    ScalarT Eic; // intrinsic Fermi level - conduction band edge in [eV]
    ScalarT Evi; // valence band edge - intrinsic Fermi level in [eV]
    ScalarT WFintSC;  // semiconductor intrinsic workfunction in [eV]
    
    Nc = NcvFactor*pow(mdn,1.5)*pow(temperature/Tref,1.5);  // in [cm-3]
    Nv = NcvFactor*pow(mdp,1.5)*pow(temperature/Tref,1.5); 
    Eg = Eg0-alpha*pow(temperature,2.0)/(beta+temperature); // in [eV]
    
    ScalarT kbT = kbBoltz*temperature;      // in [eV]
    ni = sqrt(Nc*Nv)*exp(-Eg/(2.0*kbT));    // in [cm-3]
    Eic = -Eg/2. + 3./4.*kbT*log(mdp/mdn);  // (Ei-Ec) in [eV]
    Evi = -Eg/2. - 3./4.*kbT*log(mdp/mdn);  // (Ev-Ei) in [eV]
    WFintSC = Chi - Eic;  // (Evac-Ei) in [eV] where Evac = vacuum level

    //! material parameter dependent scaling factor 
    C0 = (Nc > Nv) ? Nc : Nv;  // scaling for conc. [cm^-3]
    Lambda2 = V0*eps0/(eleQ*X0*X0*C0); // derived scaling factor (unitless)
    
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
     //Later: get doping activation energy and concentration from parameter named in material db
    string dopantType = materialDB->getElementBlockParam<string>(workset.EBName,"dopantType","None");
    string dopingProfile = materialDB->getElementBlockParam<string>(workset.EBName,"dopingProfile","Constant");

    double dopantActE, dopingConc;
    if(dopantType == "Donor") {
      dopingConc = dopingDonor;
      dopantActE = donorActE;
    }
    else if(dopantType == "Acceptor") {
      dopingConc = dopingAcceptor;
      dopantActE = acceptorActE;
    }
    else if(dopantType == "None") {
      dopingConc = dopantActE = 0.0;
    }
    else {
      TEST_FOR_EXCEPTION (true, Teuchos::Exceptions::InvalidParameter,
       std::endl << "Error!  Unknown dopant type " << dopantType << "!"<< std::endl);
    }


    for (std::size_t cell=0; cell < workset.numCells; ++cell)
    {
      for (std::size_t qp=0; qp < numQPs; ++qp)
      {
        const ScalarT& phi = potential(cell,qp);
          
        // obtain the ionized donors
        ScalarT ionN, inArg;
        inArg = phi + Eic/kbT + dopantActE/kbT;  //Suzey: is this correct for n- and p-type?
        ionN = (this->*ionDopant)(dopantType,inArg)*dopingConc;
          
        // the scaled full RHS
        ScalarT charge; 
        charge = 1.0/Lambda2*(Nv*(this->*carrStat)(-phi+Evi/kbT)- Nc*(this->*carrStat)(phi+Eic/kbT) + ionN)/C0;
        poissonSource(cell, qp) = factor*charge;
          
        // output states
        chargeDensity(cell, qp) = charge*Lambda2*C0;
        electronDensity(cell, qp) = Nc*(this->*carrStat)(phi+Eic/kbT);
        holeDensity(cell, qp) = Nv*(this->*carrStat)(-phi+Evi/kbT);
        electricPotential(cell, qp) = phi*V0;
	ionizedDopant(cell, qp) = ionN;
        conductionBand(cell, qp) = qPhiRef-Chi-phi*V0; // [eV]
        valenceBand(cell, qp) = conductionBand(cell,qp)-Eg;
      }
    }

  }
    
  //***************************************************************************
  //! element block with "Insulator" material
  //***************************************************************************
  else if(matrlCategory == "Insulator")
  {
    double Eg = materialDB->getElementBlockParam<double>(workset.EBName,"Band Gap",0.0);
    double Chi = materialDB->getElementBlockParam<double>(workset.EBName,"Electron Affinity",0.0);

    for (std::size_t cell=0; cell < workset.numCells; ++cell)
    {
      for (std::size_t qp=0; qp < numQPs; ++qp)
      {
        const ScalarT& phi = potential(cell,qp);
        ScalarT charge; 
        charge = 0.0;  // no charge in an insulator
        poissonSource(cell, qp) = factor*charge;
        
        chargeDensity(cell, qp) = 0.0;    // no space charge in an insulator
        electronDensity(cell, qp) = 0.0;  // no electrons in an insulator
        holeDensity(cell, qp) = 0.0;      // no holes in an insulator
        electricPotential(cell, qp) = phi*V0;
        ionizedDopant(cell, qp) = 0.0;
        conductionBand(cell, qp) = qPhiRef-Chi-phi*V0; // [eV]
        valenceBand(cell, qp) = conductionBand(cell,qp)-Eg;
      }
    }
  }

  //***************************************************************************
  //! element block with "Metal" material
  //***************************************************************************
  else if(matrlCategory == "Metal")
  {
    double Eg = materialDB->getElementBlockParam<double>(workset.EBName,"Band Gap",0.0);
    double Chi = materialDB->getElementBlockParam<double>(workset.EBName,"Electron Affinity",0.0);

    // polysilicon needs special considertion and the following is temporary
    for (std::size_t cell=0; cell < workset.numCells; ++cell)
    {
      for (std::size_t qp=0; qp < numQPs; ++qp)
      {
        const ScalarT& phi = potential(cell,qp);
        ScalarT charge; 
        charge = 0.0;  // no charge in metal bulk
        poissonSource(cell, qp) = factor*charge;
        
        // output states
        chargeDensity(cell, qp) = 0.0;    
        electronDensity(cell, qp) = 0.0;  
        holeDensity(cell, qp) = 0.0;      
        electricPotential(cell, qp) = phi*V0; 
        ionizedDopant(cell, qp) = 0.0;
        conductionBand(cell, qp) = qPhiRef-Chi-phi*V0; // [eV]
        valenceBand(cell, qp) = conductionBand(cell,qp)-Eg;
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

      // scale even default device since Poisson Dirichlet evaluator always scales DBCs
      poissonSource(cell, qp) = factor*charge / V0; 
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
   // Use the approximate 1/2 FD integral D. Bednarczyk and J. Bednarczyk, 
   // "The approximation of the Fermi-Dirac integral F_{1/2}(x),"
   // Physics Letters A, vol.64, no.4, pp.409-410, 1978. The approximation 
   // has error < 0.4% in the entire x range.  
   
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
    TEST_FOR_EXCEPT(fabs(ImPart) > TOL);
    if(index < numberToRead) eigenvals[index] = -RePart; //negative b/c of convention
    //std::cout << "DEBUG eval(" << index << ") = " << RePart << std::endl;
  }
  evalData.close();
  return eigenvals;
}
