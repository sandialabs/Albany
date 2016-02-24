//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#include "Teuchos_TestForException.hpp"
#include "Phalanx_DataLayout.hpp"
#include "Sacado_ParameterRegistration.hpp"
#include "Albany_AbstractDiscretization.hpp"


namespace QCAD {

template<typename EvalT,typename Traits>
PoissonDirichlet<EvalT, Traits>::
PoissonDirichlet(Teuchos::ParameterList& p) :
  PHAL::Dirichlet<EvalT,Traits>(p)
{
  // get parameters from ParameterList
  user_value = p.get<RealType>("Dirichlet Value");
  materialDB = p.get< Teuchos::RCP<QCAD::MaterialDatabase> >("MaterialDB");
  energy_unit_in_eV = p.get<double>("Energy unit in eV");

  Teuchos::ParameterList* psList = p.get<Teuchos::ParameterList*>("Poisson Source Parameter List");

  carrierStatistics = psList->get("Carrier Statistics", "Boltzmann Statistics");
  incompIonization = psList->get("Incomplete Ionization", "False");
  //dopingDonor = psList->get("Donor Doping", 1e14);
  //dopingAcceptor = psList->get("Acceptor Doping", 1e14);
  //donorActE = psList->get("Donor Activation Energy", 0.040);
  //acceptorActE = psList->get("Acceptor Activation Energy", 0.045);
 
  temperature = p.get<double>("Temperature"); //To be replaced by SharedParameter evaluator access

  // obtain material or eb name for a given nodeset 
  // std::string nodeSetName = PHAL::DirichletBase<EvalT,Traits>::nodeSetID;
  nodeSetName = p.get<std::string>("Node Set ID");

  contactType = "None";  // default
  sbHeight = 0.0; 
  material = materialDB->getNodeSetParam<std::string>(nodeSetName,"material","");
  if (material.length() > 0) contactType = "Contact On Insulator";   // must be here

  if (material.length() == 0) 
  {
    ebName = materialDB->getNodeSetParam<std::string>(nodeSetName,"elementBlock","");
    if (ebName.length() > 0)
    {
      if (p.isParameter("Schottky Barrier Height"))
      {
        sbHeight = p.get<double>("Schottky Barrier Height");  // [eV]
        contactType = "Schottky Contact";
      }
      else
        contactType = "Ohmic Contact"; 

      material = materialDB->getElementBlockParam<std::string>(ebName,"material","");
    }
    
    // if ebName.length() == 0, contactType = "None" and material.length() == 0
  }

  // private scaling parameter (note: kbT is used in calculating qPhiRef below)
  const double kbBoltz = 8.617343e-05; //[eV/K]
  kbT = kbBoltz*temperature;
  V0 = kbBoltz*temperature/1.0; // kb*T/q in [V], scaling for potential

  // obtain qPhiRef (energy reference for heterogeneous structures)
  if ( (material.length() > 0) || (ebName.length() > 0) )
  {
    std::string refMtrlName, category;
    refMtrlName = materialDB->getParam<std::string>("Reference Material");
    category = materialDB->getMaterialParam<std::string>(refMtrlName,"Category");
    if (category == "Semiconductor") {
      double mdn = materialDB->getMaterialParam<double>(refMtrlName,"Electron DOS Effective Mass");
      double mdp = materialDB->getMaterialParam<double>(refMtrlName,"Hole DOS Effective Mass");
      double Chi = materialDB->getMaterialParam<double>(refMtrlName,"Electron Affinity");
      double Eg0 = materialDB->getMaterialParam<double>(refMtrlName,"Zero Temperature Band Gap");
      double alpha = materialDB->getMaterialParam<double>(refMtrlName,"Band Gap Alpha Coefficient");
      double beta = materialDB->getMaterialParam<double>(refMtrlName,"Band Gap Beta Coefficient");
      
      ScalarT Eg = Eg0 - alpha * std::pow(temperature,2.0) / (beta+temperature); // in [eV]
      ScalarT Eic = -Eg/2. + 3./4.*kbT*std::log(mdp/mdn);  // (Ei-Ec) in [eV]
      qPhiRef = Chi - Eic;  // qPhiRef is basically the difference between vacuum energy level and intrinsic Fermi level
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
}


// ****************************************************************************
template<typename EvalT,typename Traits>
void PoissonDirichlet<EvalT, Traits>::
evaluateFields(typename Traits::EvalData dirichletWorkset)
{

  // Universal constants
  const double kbBoltz = 8.617343e-05;  // [eV/K]
  const double eleQ = 1.602176487e-19;  // [C]
  const double m0 = 9.10938215e-31;     // [kg]
  const double hbar = 1.054571628e-34;  // [J.s]
  const double pi = 3.141592654;

  ScalarT bcValue = 0.0; 

  if (material.length() > 0)  
  {
    std::string category = materialDB->getMaterialParam<std::string>(material,"Category");

    //! Metal contact on insulator
    if ((category == "Metal") && (contactType == "Contact On Insulator"))
    {  
      double metalWorkFunc = materialDB->getMaterialParam<double>(material,"Work Function");
      ScalarT offsetDueToWorkFunc = (metalWorkFunc - qPhiRef) / 1.0;  // 1.0 converts from [eV] to [V]
      bcValue = (user_value - offsetDueToWorkFunc) / energy_unit_in_eV;  //[meV]
    }
  
    //! Ohmic contact on semiconductor (charge neutrality and equilibrium ) 
    else if ((category == "Semiconductor") && (contactType == "Ohmic Contact"))
    {
      // Temperature-independent material parameters
      double mdn, mdp, Tref, Eg0, alpha, beta, Chi;
      if (ebName.length() > 0)  
      {
        mdn = materialDB->getElementBlockParam<double>(ebName,"Electron DOS Effective Mass");
        mdp = materialDB->getElementBlockParam<double>(ebName,"Hole DOS Effective Mass");
        Tref = materialDB->getElementBlockParam<double>(ebName,"Reference Temperature");
	
        Eg0 = materialDB->getElementBlockParam<double>(ebName,"Zero Temperature Band Gap");
        alpha = materialDB->getElementBlockParam<double>(ebName,"Band Gap Alpha Coefficient");
        beta = materialDB->getElementBlockParam<double>(ebName,"Band Gap Beta Coefficient");
        Chi = materialDB->getElementBlockParam<double>(ebName,"Electron Affinity");
      }
      else 
      {
        TEUCHOS_TEST_FOR_EXCEPTION(true, std::logic_error, std::endl 
	          << "Error! elementBlock must exist for Ohmic Contact nodeset of " << nodeSetName << std::endl);
      }

      // Constant prefactor in calculating Nc and Nv in [cm-3]
      double NcvFactor = 2.0 * std::pow((kbBoltz*eleQ*m0*Tref) / (2*pi*std::pow(hbar,2)),3./2.)*1.e-6;
          // eleQ converts kbBoltz in [eV/K] to [J/K], 1e-6 converts [m-3] to [cm-3]
    
      // Strong temperature-dependent material parameters
      ScalarT Nc;  // conduction band effective DOS in [cm-3]
      ScalarT Nv;  // valence band effective DOS in [cm-3]
      ScalarT Eg;  // band gap at T [K] in [eV]
    
      Nc = NcvFactor * std::pow(mdn,1.5) * std::pow(temperature/Tref,1.5);  // in [cm-3]
      Nv = NcvFactor * std::pow(mdp,1.5) * std::pow(temperature/Tref,1.5); 
      Eg = Eg0 - alpha * std::pow(temperature,2.0)/(beta+temperature); // in [eV]
    
      ScalarT builtinPotential = 0.0; 
      std::string dopantType;
      dopantType = materialDB->getElementBlockParam<std::string>(ebName,"Dopant Type");
 
      // Intrinsic semiconductor (no doping)
      if (dopantType == "None")  
      {
        // apply charge neutrality (p=n) and MB statistics
        builtinPotential = (qPhiRef-Chi-0.5*Eg)/1.0 + 0.5*kbT*std::log(Nv/Nc)/1.0;
        bcValue = (user_value + builtinPotential) / energy_unit_in_eV;  // [myV]
      }
    
      else // Extrinsic semiconductor (doped)
      {
        double dopingConc, dopantActE;
        dopingConc = materialDB->getElementBlockParam<double>(ebName,"Doping Value");
        dopantActE = materialDB->getElementBlockParam<double>(ebName,"Dopant Activation Energy", 0.045);
 
        if ((carrierStatistics=="Boltzmann Statistics") && (incompIonization=="False"))
          builtinPotential = potentialForMBComplIon(Nc,Nv,Eg,Chi,dopantType,dopingConc);
    
        else if ((carrierStatistics=="Boltzmann Statistics") && (incompIonization=="True"))
          builtinPotential = potentialForMBIncomplIon(Nc,Nv,Eg,Chi,dopantType,dopingConc,dopantActE);

        else if ((carrierStatistics=="Fermi-Dirac Statistics") && (incompIonization=="False"))
          builtinPotential = potentialForFDComplIon(Nc,Nv,Eg,Chi,dopantType,dopingConc);
	
        else if ((carrierStatistics=="0-K Fermi-Dirac Statistics") && (incompIonization=="False"))
          builtinPotential = potentialForZeroKFDComplIon(Nc,Nv,Eg,Chi,dopantType,dopingConc);
    
        // For cases of FD and 0-K FD with incompIonization==True, one needs to 
        // numerically solve a non-trivial equation. Since when incompIonization==True
        // is enabled, the MB almost always holds, so use potentialForMBIncomplIon.
        else  
          builtinPotential = potentialForMBIncomplIon(Nc,Nv,Eg,Chi,dopantType,dopingConc,dopantActE);
        
        bcValue = (user_value + builtinPotential) / energy_unit_in_eV;  // [myV]
        
      }
    }  // end of else if (contactType == "Ohmic Contact")

    //! Schottky contact on semiconductor  
    else if ((category == "Semiconductor") && (contactType == "Schottky Contact"))
    {
      // Temperature-independent material parameters
      double Tref, Eg0, alpha, beta, Chi;
      if (ebName.length() > 0)  
      {
        Tref = materialDB->getElementBlockParam<double>(ebName,"Reference Temperature");
        Eg0 = materialDB->getElementBlockParam<double>(ebName,"Zero Temperature Band Gap");
        alpha = materialDB->getElementBlockParam<double>(ebName,"Band Gap Alpha Coefficient");
        beta = materialDB->getElementBlockParam<double>(ebName,"Band Gap Beta Coefficient");
        Chi = materialDB->getElementBlockParam<double>(ebName,"Electron Affinity");
      }
      else 
        TEUCHOS_TEST_FOR_EXCEPTION(true, std::logic_error, std::endl 
	          << "Error! elementBlock must exist for Schottky Contact nodeset of " << nodeSetName << std::endl);

      // Temperature-dependent material parameters
      ScalarT Eg;  // band gap at T [K] in [eV]
      Eg = Eg0 - alpha * std::pow(temperature,2.0) / (beta+temperature); // in [eV]
    
      ScalarT builtinPotential = 0.0; 
      std::string dopantType = materialDB->getElementBlockParam<std::string>(ebName,"Dopant Type");
      builtinPotential = potentialForSchottkyBarrier(Eg, Chi, dopantType);

      bcValue = (user_value + builtinPotential) / energy_unit_in_eV; 
    }
    
  } // end of if (material.length() > 0)
    
  //! Otherwise, just use the user_value converted into the units of the solution. 
  //    (DBCs are always specified in volts by the user)
  else
    bcValue = user_value / energy_unit_in_eV;

  //! Register bcValue 
  PHAL::DirichletBase<EvalT,Traits>::value = bcValue;
    
  //! Call base class evaluateFields, which sets relevant nodes using value member
  PHAL::Dirichlet<EvalT,Traits>::evaluateFields(dirichletWorkset);

  // std::string nodeSetName = PHAL::DirichletBase<EvalT,Traits>::nodeSetID;
  // std::cout << "nodeSetName = " << nodeSetName << ", bcValue = " << bcValue << std::endl;   

}


// *****************************************************************************
template<typename EvalT,typename Traits>
inline typename QCAD::PoissonDirichlet<EvalT,Traits>::ScalarT
QCAD::PoissonDirichlet<EvalT,Traits>::inverseFDIntOneHalf(const ScalarT x)
{
  // Use the approximate inverse of 1/2 FD integral by N. G. Nillsson, 
  // "An Accurate Approximation of the Generalized Einstein Relation for 
  // Degenerate Semiconductors," physica status solidi (a) 19, K75 (1973).
  
  const double pi = 3.1415926536;
  const double a = 0.24;
  const double b = 1.08; 

  ScalarT nu = pow(3./4.*sqrt(pi)*x, 2./3.);
  ScalarT invFDInt = 0.0; 
  
  if (x > 0.)
    invFDInt = log(x)/(1.-pow(x,2.)) + nu/(1.+pow(a+b*nu,-2.));
  else
    TEUCHOS_TEST_FOR_EXCEPTION (true, Teuchos::Exceptions::InvalidParameter, std::endl 
      << "Error! x in inverseFDIntOneHalf(x) must be greater than 0 " << std::endl);

  return invFDInt;
}


// *****************************************************************************
template<typename EvalT,typename Traits>
typename QCAD::PoissonDirichlet<EvalT,Traits>::ScalarT
QCAD::PoissonDirichlet<EvalT,Traits>::potentialForMBComplIon(const ScalarT &Nc, 
   const ScalarT &Nv, const ScalarT &Eg, const double &Chi, const std::string &dopType, const double &dopingConc)
{
  ScalarT Cn = Nc*exp((-qPhiRef+Chi)/kbT); 
  ScalarT Cp = Nv*exp((qPhiRef-Chi-Eg)/kbT);
  ScalarT builtinPotential = 0.0; 
  
  // for high-T, include n and p in charge neutrality: p=n+Na or p+Nd=n
  if ((Cn > 1.e-5) && (Cp > 1.e-5))
  {
    double signedDopingConc;
    if(dopType == "Donor") 
      signedDopingConc = dopingConc;
    else if(dopType == "Acceptor") 
      signedDopingConc = -dopingConc;
    else 
    {
      TEUCHOS_TEST_FOR_EXCEPTION (true, Teuchos::Exceptions::InvalidParameter, std::endl 
        << "Error!  Unknown dopant type " << dopType << "!"<< std::endl);
    }
    ScalarT tmp1 = signedDopingConc/(2.0*Cn);
    ScalarT tmp2 = tmp1 + sqrt(pow(tmp1,2.0) + Cp/Cn);
    if (tmp2 <= 0.0)
    {
      if(dopType == "Donor") 
        builtinPotential = (qPhiRef-Chi)/1.0 + V0*log(dopingConc/Nc);  
      else if(dopType == "Acceptor") 
        builtinPotential = (qPhiRef-Chi-Eg)/1.0 - V0*log(dopingConc/Nv);  
    }
    else
      builtinPotential = V0*log(tmp2); 
  }
  
  // for low-T (where Cn=0, Cp=0), consider only p=Na or n=Nd
  else
  {
    if(dopType == "Donor") 
      builtinPotential = (qPhiRef-Chi)/1.0 + V0*log(dopingConc/Nc);  
    else if(dopType == "Acceptor") 
      builtinPotential = (qPhiRef-Chi-Eg)/1.0 - V0*log(dopingConc/Nv);  
    else 
    {
      TEUCHOS_TEST_FOR_EXCEPTION (true, Teuchos::Exceptions::InvalidParameter, std::endl 
        << "Error!  Unknown dopant type " << dopType << "!"<< std::endl);
    }
  }
  
  return builtinPotential;
}


// *****************************************************************************
template<typename EvalT,typename Traits>
typename QCAD::PoissonDirichlet<EvalT,Traits>::ScalarT
QCAD::PoissonDirichlet<EvalT,Traits>::potentialForMBIncomplIon(const ScalarT &Nc, 
      const ScalarT &Nv, const ScalarT &Eg, const double &Chi, const std::string &dopType, 
      const double &dopingConc, const double &dopantActE )
{
  // maximum allowed exponent in an exponential function (unitless)
  const double MAX_EXPONENT = 100.0; 
  ScalarT builtinPotential;
  
  // assume n = Nd+ to have an analytical expression (neglect p)  
  if(dopType == "Donor") 
  {
    ScalarT offset = (-dopantActE +qPhiRef -Chi) /1.0;
    if (dopantActE/kbT > MAX_EXPONENT)  // exp(dopantActE/kbT) -> +infinity for very small T
    {
      builtinPotential = offset +V0 *(0.5*log(dopingConc/(2.*Nc)) + dopantActE/(2.*kbT) );
    }
    else
    { 
      ScalarT tmp = -1./4.+1./4.*sqrt(1.+8.*dopingConc/Nc*exp(dopantActE/kbT));
      builtinPotential = offset + V0*log(tmp);
    }  
  }
  
  // assume p = Na- to have an analytical expression (neglect n)
  else if(dopType == "Acceptor") 
  {
    ScalarT offset = (dopantActE +qPhiRef -Chi -Eg) /1.0; 
    if (dopantActE/kbT > MAX_EXPONENT)  // exp(dopantActE/kbT) -> +infinity for very small T
    {
      builtinPotential = offset -V0 *(0.5*log(dopingConc/(4.*Nv)) + dopantActE/(2.*kbT) );
    }
    else
    {
      ScalarT tmp = -1./8.+1./8.*sqrt(1.+16.*dopingConc/Nv*exp(dopantActE/kbT));
      builtinPotential = offset - V0*log(tmp);
    }
  }

  else 
  {
    TEUCHOS_TEST_FOR_EXCEPTION (true, Teuchos::Exceptions::InvalidParameter, std::endl 
      << "Error!  Unknown dopant type " << dopType << "!"<< std::endl);
  }

  return builtinPotential;
}


// *****************************************************************************
template<typename EvalT,typename Traits>
typename QCAD::PoissonDirichlet<EvalT,Traits>::ScalarT
QCAD::PoissonDirichlet<EvalT,Traits>::potentialForFDComplIon(const ScalarT &Nc, 
    const ScalarT &Nv, const ScalarT &Eg, const double &Chi, const std::string &dopType, const double &dopingConc)
{
  ScalarT builtinPotential;
    
  // assume n = Nd to have an analytical expression (neglect p)
  if(dopType == "Donor") 
  {
    ScalarT invFDInt = inverseFDIntOneHalf(dopingConc/Nc);
    builtinPotential = (qPhiRef-Chi)/1.0 + V0*invFDInt;
  }
  
  // assume p = Na to have an analytical expression (neglect n)
  else if(dopType == "Acceptor") 
  {
    ScalarT invFDInt = inverseFDIntOneHalf(dopingConc/Nv);
    builtinPotential = (qPhiRef-Chi-Eg)/1.0 - V0*invFDInt;
  }
  else {
    TEUCHOS_TEST_FOR_EXCEPTION (true, Teuchos::Exceptions::InvalidParameter, std::endl 
      << "Error!  Unknown dopant type " << dopType << "!"<< std::endl);
  }
    
  return builtinPotential;
}


// *****************************************************************************
template<typename EvalT,typename Traits>
typename QCAD::PoissonDirichlet<EvalT,Traits>::ScalarT
QCAD::PoissonDirichlet<EvalT,Traits>::potentialForZeroKFDComplIon(const ScalarT &Nc, 
    const ScalarT &Nv, const ScalarT &Eg, const double &Chi, const std::string &dopType, const double &dopingConc)
{
  const double pi = 3.1415926536;

  ScalarT builtinPotential;
    
  // assume n = Nd to have an analytical expression (neglect p)
  if(dopType == "Donor") 
  {
    if (dopingConc < Nc)  // Fermi level is below conduction band (due to doping)
      builtinPotential = potentialForFDComplIon(Nc,Nv,Eg,Chi,dopType,dopingConc);
    else  // Fermi level is in conduction band (degenerate)
    { 
      ScalarT invFDInt = pow(3./4.*sqrt(pi)*(dopingConc/Nc),2./3.);
      builtinPotential = (qPhiRef-Chi)/1.0+ V0*invFDInt;
    }
  }
  
  // assume p = Na to have an analytical expression (neglect n)
  else if(dopType == "Acceptor") 
  {
    if (dopingConc < Nv)  // Fermi level is above valence band (due to doping) 
      builtinPotential = potentialForFDComplIon(Nc,Nv,Eg,Chi,dopType,dopingConc); 
    else  // Fermi level is in valence band (degenerate)
    {  
      ScalarT invFDInt = pow(3./4.*sqrt(pi)*(dopingConc/Nv),2./3.);
      builtinPotential = (qPhiRef-Chi-Eg)/1.0- V0*invFDInt;
    }
  }
  
  else 
  {
    TEUCHOS_TEST_FOR_EXCEPTION (true, Teuchos::Exceptions::InvalidParameter, std::endl 
      << "Error!  Unknown dopant type " << dopType << "!"<< std::endl);
  }
    
  return builtinPotential;
}


// *****************************************************************************
template<typename EvalT,typename Traits>
typename QCAD::PoissonDirichlet<EvalT,Traits>::ScalarT
QCAD::PoissonDirichlet<EvalT,Traits>::potentialForSchottkyBarrier(
    const ScalarT &Eg, const double &Chi, const std::string &dopType)
{
  ScalarT builtinPotential;
  
  if ( (dopType == "Donor") || (dopType == "None"))  // SB on n-type SC
    builtinPotential = -(sbHeight + Chi - qPhiRef)/1.0;  // 1.0 converts [eV] to [V]
  
  else if (dopType == "Acceptor")  // SB on p-type SC
    builtinPotential = -(-sbHeight + Chi + Eg - qPhiRef)/1.0;  // in [V]

  else 
    TEUCHOS_TEST_FOR_EXCEPTION (true, Teuchos::Exceptions::InvalidParameter, std::endl 
      << "Error!  Unknown dopant type " << dopType << "!"<< std::endl);
  
  return builtinPotential;
}

}  //end of QCAD namespace
