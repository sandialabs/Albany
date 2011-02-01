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

  // Add factor as a Sacado-ized parameter
  Teuchos::RCP<ParamLib> paramLib =
      p.get< Teuchos::RCP<ParamLib> >("Parameter Library", Teuchos::null);
  new Sacado::ParameterRegistration<EvalT, SPL_Traits>(
      "Poisson Source Factor", this, paramLib);

  this->addDependentField(potential);
  this->addDependentField(coordVec);

  this->addEvaluatedField(poissonSource);
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
}

// **********************************************************************
template<typename EvalT, typename Traits>
void QCAD::PoissonSource<EvalT, Traits>::
evaluateFields(typename Traits::EvalData workset)
{
  // STATE OUTPUT
  Intrepid::FieldContainer<RealType>& CDState = *((*workset.newState)["ChargeDistribution"]);
  
  for (std::size_t cell=0; cell < workset.numCells; ++cell) 
  {
    for (std::size_t qp=0; qp < numQPs; ++qp) 
    {
      const ScalarT& CD = chargeDistribution(numDims, &coordVec(cell,qp,0), potential(cell,qp));
      poissonSource(cell,qp) = factor * CD;
      // STATE OUTPUT: Save off real part into saved state vector
      CDState(cell,qp) = Sacado::ScalarValue<ScalarT>::eval(CD);
    }
  }
}

// **********************************************************************
template<typename EvalT,typename Traits>
typename QCAD::PoissonSource<EvalT,Traits>::ScalarT& 
QCAD::PoissonSource<EvalT,Traits>::getValue(const std::string &n)
{
  return factor;
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

  return validPL;
}

// **********************************************************************
template<typename EvalT,typename Traits>
typename QCAD::PoissonSource<EvalT,Traits>::ScalarT
QCAD::PoissonSource<EvalT,Traits>::chargeDistribution(
    const int numDim, const MeshScalarT* coord, const ScalarT& phi) const
{
  ScalarT charge;

  /***** define universal constants as double constants *****/
  // Boltzmann constant in [eV/K]
  const double kbBoltz = 8.617343e-05;       
   
  // vacuum permittivity in [C/(V.cm)]
  const double eps0 = 8.854187817e-12*0.01;  
   
  // electron elemental charge in [C]
  const double eleQ = 1.602e-19;             

  // define lattice temperature in [K]
  const double lattTemp = 300.0; 

  /***** define scaling parameters *****/
  // length scaling [cm]: the input and output mesh is assumed in [um] 
  const double X0 = 1e-4;  
  
  // kb*T/q in [V], scaling for potential                    
  const double V0 = kbBoltz*lattTemp/1.0; 
  
  // Silicon intrinsic concentration at 300 K in [cm^-3],scaling for conc.
  const double C0 = 1.45e10;   

  // derived scaling factor that appears in the scaled Poisson equation,in [1]
  const double Lambda2 = V0*eps0/(eleQ*X0*X0*C0);  
  
  /***** implement RHS of the equilibrium Poisson equation in pn diode *****/
  if (device == "pndiode") 
  {
    // define doping concentration
    double acceptor = 1e16; // constant acceptor doping in [cm^-3]
    double donor = 1e16;     // constant donor doping in [cm^-3]
    switch (numDim) 
    {
       case 2:
        // assign doping profile 
        if (coord[0] <= 0.5)      // acceptor doping for x/X0<=0.5 
          charge = -acceptor/C0;   // normalized by C0
        else                      // donor doping for x/X0>0.5
          charge = donor/C0; 
        
        // define the full RHS 
        charge = 1.0/Lambda2*(charge+exp(-phi)-exp(phi));

        break;
      case 1:
      case 3:
       default:
        TEST_FOR_EXCEPT(true);
    }  // end of switch(numDim)
  }    // end of if (device="pndidoe")
  
  /***** implement RHS of the equilibrium Poisson eqn in a pmos capacitor *****/
  else if (device == "pmoscap") 
  {
    // define substrate acceptor doping in [cm^-3]
    double acceptor = 1e14;  
    switch (numDim)
    {
      case 2:
        // consider the Silicon region (y/X0 > 0)
        if (coord[1] > 0.0)  
        {
          charge = -acceptor/C0;  // acceptor doping in Silicon
          charge = 1.0/Lambda2*(charge+exp(-phi)-exp(phi));  // the full RHS
        }  
        
        // consider the SiO2 region (y/X0 < 0)
        else
          charge = 0.0;  // no charge in SiO2 (solve the Lapalace equation)
          
        break;
      case 1:
      case 3:
      default: 
        TEST_FOR_EXCEPT(true);
    }
  }    // end of else if (device=="pmoscap")
  
  /***** otherwise, run the /examples/Poisson2D device  *****/
  else 
  {
    switch (numDim) 
    {
      case 2:
        if (coord[1]<0.8) charge = (coord[1]*coord[1]);
        else charge = 3.0;
        charge *= (1.0 + exp(-phi));
        break;
       case 1:
       case 3:
       default:
         TEST_FOR_EXCEPT(true);
    }
  }
  
  return charge;
}
// *****************************************************************************
