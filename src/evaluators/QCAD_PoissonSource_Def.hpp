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
  ionizedDonor("Ionized Donor",
      p.get<Teuchos::RCP<PHX::DataLayout> >("QP Scalar Data Layout")),
  ionizedAcceptor("Ionized Acceptor",
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
  
  // passed down from main list
  temperature = p.get<double>("Temperature");
  length_unit_in_m = p.get<double>("Length unit in m");

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
  this->addEvaluatedField(ionizedDonor);
  this->addEvaluatedField(ionizedAcceptor);
  
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

  this->utils.setFieldData(ionizedDonor,fm);
  this->utils.setFieldData(ionizedAcceptor,fm);
}

// **********************************************************************
template<typename EvalT, typename Traits>
void QCAD::PoissonSource<EvalT, Traits>::
evaluateFields(typename Traits::EvalData workset)
{
  //! implement RHS of the scaled Poisson equation in pn diode 
  if (device == "pndiode") evaluateFields_pndiode(workset);
    
  //! implement RHS of the scaled Poisson eqn in a pmos capacitor 
  else if (device == "pmoscap") evaluateFields_pmoscap(workset);

  //! implement RHS of the scaled Poisson eqn in a pmos capacitor 
  else if (device == "elementblocks") evaluateFields_elementblocks(workset);

  //! otherwise, run the /examples/Poisson2D device  
  else evaluateFields_default(workset);

  //! testing/debugging workset element block names
  /*if (workset.EBName == "silicon") cout << "Poisson Source: in  silicon  material" << endl;
    else if (workset.EBName == "sio2") cout << "Poisson Source: in  sio2  material" << endl;
    else  cout << "Poisson Source: in unknown material. HELP." << endl;  */
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
evaluateFields_pndiode(typename Traits::EvalData workset)
{
  C0 = 1.45e10;  // scaling for conc. [cm^-3] (Silicon intrinsic conc. at 300 K)
  Lambda2 = V0*eps0/(eleQ*X0*X0*C0); // derived scaling factor (unitless)

  // define doping concentration for pndiode
  double acceptorCnst = 1e16;	// constant acceptor doping in [cm^-3]
  double donorCnst = 1e16;      // constant donor doping in [cm^-3]
  
  const MeshScalarT* coord;
  ScalarT charge;

  for (std::size_t cell=0; cell < workset.numCells; ++cell) 
  {
    for (std::size_t qp=0; qp < numQPs; ++qp) 
    {
      coord = &coordVec(cell,qp,0);
      const ScalarT& phi = potential(cell,qp);

      switch (numDims) {
      case 2:
        // assign doping profile 
        if (coord[0] <= 0.5)      // acceptor doping for x/X0<=0.5 
          charge = -acceptorCnst/C0;  // normalized by C0
        else                      // donor doping for x/X0>0.5
          charge = donorCnst/C0; 
        
        // define the full RHS (scaled)
        charge = 1.0/Lambda2*(charge+exp(-phi)-exp(phi));
        
        // compute quantities that are output to .exo file
        chargeDensity(cell, qp) = charge*Lambda2*C0;  // space charge density [cm-3]
        electronDensity(cell, qp) = exp(phi)*C0;     // electron density [cm-3]
        holeDensity(cell, qp) = exp(-phi)*C0;          // hole density [cm-3]
        electricPotential(cell, qp) = phi*V0;         // electric potential [V]

        break;
      default: TEST_FOR_EXCEPT(true);
      }  // end of switch(numDims)

      // returns a scaled 'charge',
      // not the actual charge density in [cm-3]
      poissonSource(cell, qp) = factor*charge;
    }
  }

  fillOutputState(workset);
}

// **********************************************************************
template<typename EvalT, typename Traits>
void QCAD::PoissonSource<EvalT, Traits>::
evaluateFields_pmoscap(typename Traits::EvalData workset)
{
  C0 = 1.45e10;  // scaling for conc. [cm^-3] (Silicon intrinsic conc. at 300 K)
  Lambda2 = V0*eps0/(eleQ*X0*X0*C0); // derived scaling factor (unitless)

  // define substrate acceptor doping in [cm^-3]
  double acceptorCnst = 1e14;  

  const MeshScalarT* coord;
  ScalarT charge;

  for (std::size_t cell=0; cell < workset.numCells; ++cell) 
  {
    for (std::size_t qp=0; qp < numQPs; ++qp) 
    {
      coord = &coordVec(cell,qp,0);
      const ScalarT& phi = potential(cell,qp);
  
      switch (numDims) {
      case 2:
        // consider the Silicon region (y/X0 > 0)
        if (coord[1] > 0.0)  
        {
          charge = -acceptorCnst/C0;  // acceptor doping in Silicon
          charge = 1.0/Lambda2*(charge+exp(-phi)-exp(phi));  // the full RHS
          
          // compute quantities that are output to .exo file
          chargeDensity(cell, qp) = charge*Lambda2*C0;  // space charge density [cm-3]
          electronDensity(cell, qp) = exp(phi)*C0;     // electron density [cm-3]
          holeDensity(cell, qp) = exp(-phi)*C0;          // hole density [cm-3]
        }  
        
        // consider the SiO2 region (y/X0 < 0)
        else
        {
          charge = 0.0;  // no charge in SiO2 (solve the Lapalace equation)
          chargeDensity(cell, qp) = 0.0;      // no space charge in SiO2
          electronDensity(cell, qp) = 0.0;    // no electrons in SiO2
          holeDensity(cell, qp) = 0.0;        // no holes in SiO2
        }
        electricPotential(cell, qp) = phi*V0;	// electric potential [V]
          
        break;
      default: TEST_FOR_EXCEPT(true);
      }

      // returns a scaled 'charge',
      // not the actual charge density in [cm-3]
      poissonSource(cell, qp) = factor*charge;
    }
  }
  fillOutputState(workset);
}

// **********************************************************************
template<typename EvalT, typename Traits>
void QCAD::PoissonSource<EvalT, Traits>::
evaluateFields_elementblocks(typename Traits::EvalData workset)
{
  // Later:
  //mass = materialManager.GetMaterialParameter(workset.EBname, "mass")
  //Eg = materialManager.GetMaterialParameter(workset.EBname, "Eg")

  //! temperature-independent material parameters
  double ml;  // longitudinal electron effective mass in [m0]
  double mt;  // traverse electron effective mass in [m0]
  double mdn; // electron DOS effective mass (valley deg. included) in [m0]
  double mhh; // heavy hole effective mass in [m0]
  double mlh; // light hole effective mass in [m0]
  double mdp; // hole DOS effective mass (including hh and lh) in [m0]

  double Tref;  // reference temperature [K] in computing Nc and Nv
  double NcvFactor;  // constant prefactor in calculating Nc and Nv in [cm-3]
  
  double Chi;   // electron affinity in [eV]
  double Eg0;   // band gap at 0 K in [eV]
  double alpha; // temperature coefficient for band gap in [eV/K]
  double beta;  // temperature coefficient for band gap in [K]
  	
  //! strong temperature-dependent material parameters
  ScalarT Nc;  // conduction band effective DOS in [cm-3]
  ScalarT Nv;  // valence band effective DOS in [cm-3]
  ScalarT Eg;  // band gap at T [K] in [eV]
  ScalarT ni;  // intrinsic carrier concentration in [cm-3]
  ScalarT Eic; // intrinsic Fermi level - conduction band edge in [eV]
  ScalarT Evi; // valence band edge - intrinsic Fermi level in [eV]
  ScalarT WFintSC;  // semiconductor intrinsic workfunction in [eV]

  //! assign Silicon parameters for the time being
  ml = 0.98;  // all effective masses are in unit of [m0]
  mt = 0.19;
  mdn = pow(6,2./3.)*pow(pow(mt,2)*ml,1./3.); // bulk Si has 6 cond. minimum
  mhh = 0.56;
  mlh = 0.16;
  mdp = pow(pow(mhh,1.5)+pow(mlh,1.5),2./3.);

  Tref = 300.; // 300 K is often used as reference temperature
  NcvFactor = 2.0*pow((kbBoltz*1.602e-19*m0*Tref)/(2*pi*pow(hbar,2)),3./2.)*1e-6;
  // 1.602e-19 converts kbBoltz in [eV/K] to [J/K], 1e-6 converts [m-3] to [cm-3]
  
  Chi = 4.05;      // in [eV]
  Eg0 = 1.1455;    // in [eV]
  alpha = 4.73e-4; // in [eV/K]
  beta = 636.;     // in [K]

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
  
  else
  {
    TEST_FOR_EXCEPTION (true, Teuchos::Exceptions::InvalidParameter,
      std::endl << "Error!  Unknown carrier statistics ! " << std::endl);
  } 

  //! function pointer to ionized dopants member function
  ScalarT (QCAD::PoissonSource<EvalT,Traits>::*ionDopant) (const std::string, const ScalarT&); 
  
  if (incompIonization == "False")
    ionDopant = &QCAD::PoissonSource<EvalT,Traits>::fullDopants; 
  
  else if (incompIonization == "True")
    ionDopant = &QCAD::PoissonSource<EvalT,Traits>::ionizedDopants;
  
  else
  {
    TEST_FOR_EXCEPTION (true, Teuchos::Exceptions::InvalidParameter,
      std::endl << "Error!  Invalid incomplete ionization option ! " << std::endl);
  } 
    

  //***************************************************************************
  //! element block "silicon" 
  //***************************************************************************
  if (workset.EBName == "silicon")
  {
    for (std::size_t cell=0; cell < workset.numCells; ++cell)
    {
      for (std::size_t qp=0; qp < numQPs; ++qp)
      {
        const ScalarT& phi = potential(cell,qp);
          
        // the scaled full RHS
        ScalarT charge; 
        charge = 1.0/Lambda2*(Nv*(this->*carrStat)(-phi+Evi/kbT)- Nc*(this->*carrStat)(phi+Eic/kbT))/C0;
        poissonSource(cell, qp) = factor*charge;
          
        // output states
        chargeDensity(cell, qp) = charge*Lambda2*C0;
        electronDensity(cell, qp) = Nc*(this->*carrStat)(phi+Eic/kbT);
        holeDensity(cell, qp) = Nv*(this->*carrStat)(-phi+Evi/kbT);
        electricPotential(cell, qp) = phi*V0;
      }
    }      
  }  

  //***************************************************************************
  //! element block "nsilicon" 
  //***************************************************************************
  else if (workset.EBName == "nsilicon")
  {
    for (std::size_t cell=0; cell < workset.numCells; ++cell)
    {
      for (std::size_t qp=0; qp < numQPs; ++qp)
      {
        const ScalarT& phi = potential(cell,qp);
          
        // obtain the ionized donors
        ScalarT ionNd, inArg;
        inArg = phi + Eic/kbT + donorActE/kbT;
        ionNd = (this->*ionDopant)("Donor",inArg)*dopingDonor;
          
        // the scaled full RHS
        ScalarT charge; 
        charge = 1.0/Lambda2*(Nv*(this->*carrStat)(-phi+Evi/kbT)- Nc*(this->*carrStat)(phi+Eic/kbT)+ ionNd)/C0;
        poissonSource(cell, qp) = factor*charge;
          
        // output states
        chargeDensity(cell, qp) = charge*Lambda2*C0;
        electronDensity(cell, qp) = Nc*(this->*carrStat)(phi+Eic/kbT);
        holeDensity(cell, qp) = Nv*(this->*carrStat)(-phi+Evi/kbT);
        electricPotential(cell, qp) = phi*V0;
        ionizedDonor(cell, qp) = ionNd;
      }
    }
  }  
  
  //***************************************************************************
  //! element block "psilicon" 
  //***************************************************************************
  else if (workset.EBName == "psilicon")
  {
    for (std::size_t cell=0; cell < workset.numCells; ++cell)
    {
      for (std::size_t qp=0; qp < numQPs; ++qp)
      {
        const ScalarT& phi = potential(cell,qp);
          
        // obtain the ionized acceptors
        ScalarT ionNa, inArg;
        inArg = -phi + Evi/kbT + acceptorActE/kbT;
        ionNa = (this->*ionDopant)("Acceptor",inArg)*dopingAcceptor;
          
        // the scaled full RHS
        ScalarT charge; 
        charge = 1.0/Lambda2*(Nv*(this->*carrStat)(-phi+Evi/kbT)- Nc*(this->*carrStat)(phi+Eic/kbT)- ionNa)/C0;
        poissonSource(cell, qp) = factor*charge;
          
        // output states
        chargeDensity(cell, qp) = charge*Lambda2*C0;
        electronDensity(cell, qp) = Nc*(this->*carrStat)(phi+Eic/kbT);
        holeDensity(cell, qp) = Nv*(this->*carrStat)(-phi+Evi/kbT);
        electricPotential(cell, qp) = phi*V0;
        ionizedAcceptor(cell, qp) = ionNa;
      }
    }
  }      

  //***************************************************************************
  //! element block "sio2" 
  //***************************************************************************
  else if (workset.EBName == "sio2")
  {
    for (std::size_t cell=0; cell < workset.numCells; ++cell)
    {
      for (std::size_t qp=0; qp < numQPs; ++qp)
      {
        const ScalarT& phi = potential(cell,qp);
        ScalarT charge; 
        charge = 0.0;  // no charge in SiO2
        poissonSource(cell, qp) = factor*charge;
        
        chargeDensity(cell, qp) = 0.0;    // no space charge in SiO2  
        electronDensity(cell, qp) = 0.0;  // no electrons in SiO2  
        holeDensity(cell, qp) = 0.0;      // no holes in SiO2
        electricPotential(cell, qp) = phi*V0;  
      }
    }
  }

  //***************************************************************************
  //! element block "poly" 
  //***************************************************************************
  else if (workset.EBName == "poly")
  {
    for (std::size_t cell=0; cell < workset.numCells; ++cell)
    {
      for (std::size_t qp=0; qp < numQPs; ++qp)
      {
        const ScalarT& phi = potential(cell,qp);
        ScalarT charge; 
        charge = 0.0;  // no charge in poly treated as conductor
        poissonSource(cell, qp) = factor*charge;
        
        chargeDensity(cell, qp) = 0.0;    
        electronDensity(cell, qp) = 0.0;  
        holeDensity(cell, qp) = 0.0;      
        electricPotential(cell, qp) = phi*V0;  
      }
    }
  }

  //! invalid element block name 
  else
  {
    TEST_FOR_EXCEPTION (true, Teuchos::Exceptions::InvalidParameter,
      std::endl << "Error!  Unknown element block name " << 
      workset.EBName << "!" << std::endl);
  }  

  fillOutputState(workset);
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
        chargeDensity(cell, qp) = charge;	// default device has NO scaling
        break;
      default: TEST_FOR_EXCEPT(true);
      }

      poissonSource(cell, qp) = factor*charge;
    }
  }

  fillOutputState(workset);
}


// **********************************************************************

// ANDY: remove this function and references above when new state output framwork is added (egn)
template<typename EvalT, typename Traits>
void QCAD::PoissonSource<EvalT, Traits>::
fillOutputState(typename Traits::EvalData workset)
{
  // STATE OUTPUT
  Intrepid::FieldContainer<RealType>& CDState = *((*workset.newState)["SpaceChargeDensity"]);
  Intrepid::FieldContainer<RealType>& eDensityState = *((*workset.newState)["ElectronDensity"]);
  Intrepid::FieldContainer<RealType>& hDensityState = *((*workset.newState)["HoleDensity"]);
  Intrepid::FieldContainer<RealType>& ePotentialState = *((*workset.newState)["ElectricPotential"]);
  Intrepid::FieldContainer<RealType>& ionNdState = *((*workset.newState)["IonizedDonor"]);
  Intrepid::FieldContainer<RealType>& ionNaState = *((*workset.newState)["IonizedAcceptor"]);

  for (std::size_t cell=0; cell < workset.numCells; ++cell) 
  {
    for (std::size_t qp=0; qp < numQPs; ++qp) 
    {
      // STATE OUTPUT: Save off real part into saved state vector
      const ScalarT& CD = chargeDensity(cell,qp); 	
      CDState(cell,qp) = Sacado::ScalarValue<ScalarT>::eval(CD);

      const ScalarT& eDensity = electronDensity(cell,qp); 
      eDensityState(cell,qp) = Sacado::ScalarValue<ScalarT>::eval(eDensity);
      
      const ScalarT& hDensity = holeDensity(cell,qp); 
      hDensityState(cell,qp) = Sacado::ScalarValue<ScalarT>::eval(hDensity);
      
      const ScalarT& ePotential = electricPotential(cell,qp); 
      ePotentialState(cell,qp) = Sacado::ScalarValue<ScalarT>::eval(ePotential);

      const ScalarT& ionNd = ionizedDonor(cell,qp); 
      ionNdState(cell,qp) = Sacado::ScalarValue<ScalarT>::eval(ionNd);

      const ScalarT& ionNa = ionizedAcceptor(cell,qp); 
      ionNaState(cell,qp) = Sacado::ScalarValue<ScalarT>::eval(ionNa);
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
  return 1.0;  // fully ionized (create function to use function pointer)
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
    ionDopants = 1.0 / (1. + 4.*exp(x));
  else
  {
    ionDopants = 0.0;
    TEST_FOR_EXCEPTION (true, Teuchos::Exceptions::InvalidParameter,
      std::endl << "Error!  Unknown dopant type! " << std::endl);
  }
   
  return ionDopants; 
}
