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

  factor = psList->get("Factor", 1.0);
  device = psList->get("Device", "defaultdevice");
  carrierStatistics = psList->get("Carrier Statistics", "Boltzmann");
  incompIonization = psList->get("Incomplete Ionization", "False");
  dopingDonor = psList->get("Donor Doping", 1e14);
  dopingAcceptor = psList->get("Acceptor Doping", 1e14);

  //passed down from main list
  temperature = p.get<double>("Temperature");
  length_unit_in_m = p.get<double>("Length unit in m");

  //Scaling factors
  X0 = length_unit_in_m / 1e-2; // length scaling to get to [cm] (so 1e-4 for mesh in [um])
  C0 = 1.45e10;  // Scaling for conc. [cm^-3] (Silicon intrinsic concentration at 300 K)
  V0 = kbBoltz * temperature / 1.0; // kb*T/q in [V], scaling for potential
  Lambda2 = V0 * eps0/(eleQ*X0*X0*C0); // derived scaling factor that appears in the scaled Poisson equation,in [1]


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
}

// **********************************************************************
template<typename EvalT, typename Traits>
void QCAD::PoissonSource<EvalT, Traits>::
evaluateFields(typename Traits::EvalData workset)
{
  /***** implement RHS of the scaled Poisson equation in pn diode *****/
  if (device == "pndiode") evaluateFields_pndiode(workset);
    
  /***** implement RHS of the scaled Poisson eqn in a pmos capacitor *****/
  else if (device == "pmoscap") evaluateFields_pmoscap(workset);

  /***** implement RHS of the scaled Poisson eqn in a pmos capacitor *****/
  else if (device == "elementblocks") evaluateFields_elementblocks(workset);

  /***** otherwise, run the /examples/Poisson2D device  *****/
  else evaluateFields_default(workset);

  //Testing/debugging workset element block names
  /*if (workset.EBName == "silicon") cout << "Posson Source: in  silicon  material" << endl;
    else if (workset.EBName == "sio2") cout << "Posson Source: in  sio2  material" << endl;
    else  cout << "Posson Source: in unknown material. HELP." << endl;  */
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
  validPL->set<string>("Carrier Statistics", "Boltzmann", "Carrier Statistics");
  validPL->set<string>("Incomplete Ionization", "False", "Partial Ionization of dopants");

  return validPL;
}


// *****************************************************************************
template<typename EvalT, typename Traits>
void QCAD::PoissonSource<EvalT, Traits>::
evaluateFields_pndiode(typename Traits::EvalData workset)
{
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
        electronDensity(cell, qp) = exp(-phi)*C0;     // electron density [cm-3]
        holeDensity(cell, qp) = exp(phi)*C0;          // hole density [cm-3]
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
          electronDensity(cell, qp) = exp(-phi)*C0;     // electron density [cm-3]
          holeDensity(cell, qp) = exp(phi)*C0;          // hole density [cm-3]
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

  // Material parameters
  double mass, Eg, Nv, Nc;
  mass = 1;
  Eg = 2; 

  ScalarT charge;

  for (std::size_t cell=0; cell < workset.numCells; ++cell) 
  {
    for (std::size_t qp=0; qp < numQPs; ++qp) 
    {
      const ScalarT& phi = potential(cell,qp);
      
      if(workset.EBName == "silicon") {
        charge = 1.0/Lambda2*(exp(-phi)-exp(phi)); // full RHS (scaled)
        chargeDensity(cell, qp) = charge*Lambda2*C0;  // space charge density [cm-3]
        electronDensity(cell, qp) = exp(-phi)*C0;     // electron density [cm-3]
        holeDensity(cell, qp) = exp(phi)*C0;          // hole density [cm-3]
      }

      else if(workset.EBName == "silicon.ntype") {
	charge = dopingDonor/C0; // normalized by C0
        charge = 1.0/Lambda2*(charge + exp(-phi)-exp(phi)); // full RHS (scaled)
        chargeDensity(cell, qp) = charge*Lambda2*C0;  // space charge density [cm-3]
        electronDensity(cell, qp) = exp(-phi)*C0;     // electron density [cm-3]
        holeDensity(cell, qp) = exp(phi)*C0;          // hole density [cm-3]
      }

      else if(workset.EBName == "silicon.ptype") {
	charge = -dopingAcceptor/C0;  // normalized by C0
        charge = 1.0/Lambda2*(charge + exp(-phi)-exp(phi)); // full RHS (scaled)
        chargeDensity(cell, qp) = charge*Lambda2*C0;  // space charge density [cm-3]
        electronDensity(cell, qp) = exp(-phi)*C0;     // electron density [cm-3]
        holeDensity(cell, qp) = exp(phi)*C0;          // hole density [cm-3]
      }
	
      else if (workset.EBName == "sio2") {
	charge = 0.0;  // no charge in SiO2 (solve the Lapalace equation)
	chargeDensity(cell, qp) = 0.0;      // no space charge in SiO2
	electronDensity(cell, qp) = 0.0;    // no electrons in SiO2
	holeDensity(cell, qp) = 0.0;        // no holes in SiO2
      }

      else if (workset.EBName == "polysilicon") {
	charge = 0.0;  // no charge in poly - treat as conductor so all at boundaries
	chargeDensity(cell, qp) = 0.0;    
	electronDensity(cell, qp) = 0.0;
	holeDensity(cell, qp) = 0.0;
      }

      else {
	//ANDY?? better exception class?
	TEST_FOR_EXCEPTION(true, Teuchos::Exceptions::InvalidParameter,
                          std::endl << "Error!  Unknown element block name "
			   << workset.EBName << "!" << std::endl);
      }
      
      // returns a scaled 'charge',
      // not the actual charge density in [cm-3]
      electricPotential(cell, qp) = phi*V0;	// electric potential [V]
      poissonSource(cell, qp) = factor*charge;
    }
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
    }
  }
}
