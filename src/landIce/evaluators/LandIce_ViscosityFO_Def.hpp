//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#include "Teuchos_TestForException.hpp"
#include "Teuchos_VerboseObject.hpp"
#include "Phalanx_DataLayout.hpp"

#include "Albany_Layouts.hpp"
#include "PHAL_AlbanyTraits.hpp"
#include "LandIce_ViscosityFO.hpp"
#include "LandIce_ViscosityFO.hpp"

//uncomment the following line if you want debug output to be printed to screen
//#define OUTPUT_TO_SCREEN

namespace LandIce {

//**********************************************************************
template<typename EvalT, typename Traits, typename VelT, typename TemprT>
ViscosityFO<EvalT, Traits, VelT, TemprT>::
ViscosityFO(const Teuchos::ParameterList& p,
            const Teuchos::RCP<Albany::Layouts>& dl) :
  pi (3.1415926535897932385),
  actenh (1.39e5),        // [J mol-1]
  actenl (6.0e4),         // [J mol-1]
  gascon (8.314),         // [J mol-1 K-1]
  switchingT (263.15),    // [K]
  arrmlh (1.733e3),       // [Pa-3 s-1]
  arrmll (3.613e-13),     // [Pa-3 s-1]
  scyr   (3.1536e7),      // [s y-1]
  k4scyr (3.1536e19),     // [s y-1]
  arrmh (k4scyr*arrmlh),  // [Pa-3 yr-1]
  arrml (k4scyr*arrmll),  // [Pa-3 yr-1]
  A(1.0),
  n(3.0),
  Ugrad (p.get<std::string> ("Velocity Gradient QP Variable Name"), dl->qp_vecgradient),
  homotopyParam(p.get<std::string>("Continuation Parameter Name"), dl->shared_param),
  mu    (p.get<std::string> ("Viscosity QP Variable Name"), dl->qp_scalar),
  flowRate_type(UNIFORM)
{
  Teuchos::ParameterList* visc_list =
   p.get<Teuchos::ParameterList*>("Parameter List");

  std::string viscType = visc_list->get("Type", "Constant");

  extractStrainRateSq = visc_list->get("Extract Strain Rate Sq", false);
  useStiffeningFactor = visc_list->get("Use Stiffening Factor", false);

  std::string flowRateType;
  if(visc_list->isParameter("Flow Rate Type"))
    flowRateType = visc_list->get<std::string>("Flow Rate Type");
  else
    flowRateType = "Uniform";

  stereographicMapList = p.get<Teuchos::ParameterList*>("Stereographic Map");
  useStereographicMap = stereographicMapList->get("Use Stereographic Map", false);
  useP0Temp = p.get<bool>("Use P0 Temperature");

  if(useStereographicMap)
    U = decltype(U)(p.get<std::string> ("Velocity QP Variable Name"), dl->qp_vector);

  if(useStiffeningFactor)
    stiffeningFactor = decltype(stiffeningFactor)(p.get<std::string> ("Stiffening Factor QP Name"), dl->qp_scalar);

  A = visc_list->get("Glen's Law A", 1.0);
  n = visc_list->get("Glen's Law n", 3.0);

  coordVec = decltype(coordVec)(
            p.get<std::string>("Coordinate Vector Variable Name"),dl->qp_gradient);

  Teuchos::RCP<Teuchos::FancyOStream> out(Teuchos::VerboseObjectBase::getDefaultOStream());
  if (viscType == "Constant"){
#ifdef OUTPUT_TO_SCREEN
    *out << "Constant viscosity!" << std::endl;
#endif
    visc_type = CONSTANT;
  } else if (viscType == "ExpTrig") {
#ifdef OUTPUT_TO_SCREEN
   *out << "Exp trig viscosity!" << std::endl;
#endif
    visc_type = EXPTRIG;
  } else if (viscType == "Glen's Law"){
    const auto knp1 = pow(1000,n+1);
    // Turn A's units from [Pa^-n s^-1] to [k^-1 kPa^-n yr^-1]
    // This makes the final units of viscosity kPa k yr, which makes [mu*Ugrad] = kPa
    A *= knp1*scyr;
    visc_type = GLENSLAW;
#ifdef OUTPUT_TO_SCREEN
    *out << "Glen's law viscosity!" << std::endl;
#endif
    if (useStereographicMap) {
      this->addDependentField(U);
      this->addDependentField(coordVec);
    }

    if (flowRateType == "Uniform") {
      flowRate_type = UNIFORM;
#ifdef OUTPUT_TO_SCREEN
      *out << "Uniform Flow Rate A: " << A << " [k^-1 kPa^-" << n << " yr^-1]\n";
#endif
    } else if (flowRateType == "From File") {
      flowRate_type = FROMFILE;
      flowFactorA=decltype(flowFactorA)(p.get<std::string> ("Ice Softness Variable Name"), dl->cell_scalar2);
      this->addDependentField(flowFactorA);
#ifdef OUTPUT_TO_SCREEN
      *out << "Flow Rate read in from file (exodus or ascii).\n"
           << "  NOTE: A units must be [k^-1 kPa^-n yr^-1]!\n";
#endif
    } else if (flowRateType == "From CISM") {
      flowRate_type = FROMCISM;
      flowFactorA=decltype(flowFactorA)(p.get<std::string> ("Ice Softness Variable Name"), dl->cell_scalar2);
      this->addDependentField(flowFactorA);
#ifdef OUTPUT_TO_SCREEN
      *out << "Flow Rate passed in from CISM.\n"
           << "  NOTE: A units must be [k^-1 kPa^-n yr^-1]!\n";
#endif
    } else if (flowRateType == "Temperature Based") {
      flowRate_type = TEMPERATUREBASED;
      if(useP0Temp) {
        temperature = decltype(temperature)(p.get<std::string> ("Temperature Variable Name"), dl->cell_scalar2);
      } else {        
        temperature = decltype(temperature)(p.get<std::string> ("Temperature Variable Name"), dl->qp_scalar);
      }
      this->addDependentField(temperature);
#ifdef OUTPUT_TO_SCREEN
      *out << "Flow Rate computed using temperature field." << std::endl;
#endif
    } else {
      TEUCHOS_TEST_FOR_EXCEPTION(true, Teuchos::Exceptions::InvalidParameter,
        std::endl << "Error in LandIce::ViscosityFO:  \"" << flowRateType << "\" is not a valid parameter for Flow Rate Type" << std::endl);
    }
#ifdef OUTPUT_TO_SCREEN
    *out << "n: " << n << std::endl;
#endif
  }

  this->addDependentField(Ugrad);
  this->addDependentField(homotopyParam);
  if (visc_type == EXPTRIG) this->addDependentField(coordVec);
  if(useStiffeningFactor)
    this->addDependentField(stiffeningFactor);
  this->addEvaluatedField(mu);

  if (extractStrainRateSq) {
    epsilonSq = decltype(epsilonSq)(p.get<std::string> ("EpsilonSq QP Variable Name"), dl->qp_scalar),
    this->addEvaluatedField(epsilonSq);
  }

  std::vector<PHX::DataLayout::size_type> dims;
  dl->qp_gradient->dimensions(dims);
  numQPs  = dims[1];
  numDims = dims[2];
  numCells = dims[0] ;

  performContinuousHomotopy = visc_list->get("Continuous Homotopy With Constant Initial Viscosity", false);
  expCoeff = performContinuousHomotopy ? visc_list->get<double>("Coefficient For Continuous Homotopy") : 0.0;

  //dummy initialization
  R=R2=x_0=y_0=0;

  this->setName("ViscosityFO"+PHX::print<EvalT>());
}

//**********************************************************************
template<typename EvalT, typename Traits, typename VelT, typename TemprT>
void ViscosityFO<EvalT, Traits, VelT, TemprT>::
postRegistrationSetup(typename Traits::SetupData d,
                      PHX::FieldManager<Traits>& fm)
{
  d.fill_field_dependencies(this->dependentFields(),this->evaluatedFields());
  if (d.memoizer_active()) memoizer.enable_memoizer();
}

//**********************************************************************
template<typename EvalT,typename Traits,typename VelT, typename TemprT>
KOKKOS_INLINE_FUNCTION
TemprT ViscosityFO<EvalT,Traits,VelT,TemprT>::flowRate (const TemprT& T) const {
  return (T < switchingT) ? arrml / exp (actenl / gascon / ((T > TemprT(150)) ? T : TemprT(150))) :
    arrmh / exp (actenh / gascon / T);
}

//**********************************************************************
//Kokkos functors
template<typename EvalT, typename Traits, typename VelT, typename TemprT>
KOKKOS_INLINE_FUNCTION
void ViscosityFO<EvalT, Traits, VelT, TemprT>::operator () (const ViscosityFO_CONSTANT_Tag&, const int& cell) const{
  for (unsigned int qp=0; qp < numQPs; ++qp)
          mu(cell,qp) = 1.0;
}

template<typename EvalT, typename Traits, typename VelT, typename TemprT>
KOKKOS_INLINE_FUNCTION
void ViscosityFO<EvalT, Traits, VelT, TemprT>::operator () (const ViscosityFO_EXPTRIG_Tag&, const int& cell) const
{
  double a = 1.0;
  for (unsigned int qp=0; qp < numQPs; ++qp)
  {
    MeshScalarT x = coordVec(cell,qp,0);
    MeshScalarT y2pi = 2.0*pi*coordVec(cell,qp,1);
    MeshScalarT muargt = (a*a + 4.0*pi*pi - 2.0*pi*a)*sin(y2pi)*sin(y2pi) + 1.0/4.0*(2.0*pi+a)*(2.0*pi+a)*cos(y2pi)*cos(y2pi);
    muargt = sqrt(muargt)*exp(a*x);
    mu(cell,qp) = 1.0/2.0*pow(A, -1.0/n)*pow(muargt, 1.0/n - 1.0);
  }
}

template<typename EvalT, typename Traits, typename VelT, typename TemprT>
KOKKOS_INLINE_FUNCTION
void ViscosityFO<EvalT, Traits, VelT, TemprT>::glenslaw (const int& cell) const
{
  double power = 0.5*(1.0/n - 1.0);
  ScalarT scale = performContinuousHomotopy ? std::pow(1.0-homotopyParam(0),expCoeff) : ScalarT(0);
  ScalarT ff = pow(10.0, -10.0*homotopyParam(0));
  ScalarT epsilonEqpSq = 0.0;
  if(useStereographicMap) {
    if (extractStrainRateSq) {
      //evaluate non-linear viscosity, given by Glen's law, at quadrature points
      for (unsigned int qp=0; qp < numQPs; ++qp) {
        MeshScalarT x = coordVec(cell,qp,0)-x_0;
        MeshScalarT y = coordVec(cell,qp,1)-y_0;
        MeshScalarT h = 4.0*R2/(4.0*R2 + x*x + y*y);
        MeshScalarT invh_x = x/2.0/R2;
        MeshScalarT invh_y = y/2.0/R2;

        OutputScalarT eps00 = Ugrad(cell,qp,0,0)/h-invh_y*U(cell,qp,1); //epsilon_xx
        OutputScalarT eps01 = (Ugrad(cell,qp,0,1)/h+invh_x*U(cell,qp,0)+Ugrad(cell,qp,1,0)/h+invh_y*U(cell,qp,1))/2.0; //epsilon_xy
        OutputScalarT eps02 = Ugrad(cell,qp,0,2)/2.0; //epsilon_xz
        OutputScalarT eps11 = Ugrad(cell,qp,1,1)/h-invh_x*U(cell,qp,0); //epsilon_yy
        OutputScalarT eps12 = Ugrad(cell,qp,1,2)/2.0; //epsilon_yz

        epsilonEqpSq = eps00*eps00 + eps11*eps11 + eps00*eps11 + eps01*eps01 + eps02*eps02 + eps12*eps12;
        epsilonSq(cell,qp) = epsilonEqpSq;
        epsilonEqpSq += ff; //add regularization "fudge factor"
        mu(cell,qp) *= pow(epsilonEqpSq,  power); //non-linear viscosity, given by Glen's law
      }
    } else {
      for (unsigned int qp=0; qp < numQPs; ++qp) {
        MeshScalarT x = coordVec(cell,qp,0)-x_0;
        MeshScalarT y = coordVec(cell,qp,1)-y_0;
        MeshScalarT h = 4.0*R2/(4.0*R2 + x*x + y*y);
        MeshScalarT invh_x = x/2.0/R2;
        MeshScalarT invh_y = y/2.0/R2;

        OutputScalarT eps00 = Ugrad(cell,qp,0,0)/h-invh_y*U(cell,qp,1); //epsilon_xx
        OutputScalarT eps01 = (Ugrad(cell,qp,0,1)/h+invh_x*U(cell,qp,0)+Ugrad(cell,qp,1,0)/h+invh_y*U(cell,qp,1))/2.0; //epsilon_xy
        OutputScalarT eps02 = Ugrad(cell,qp,0,2)/2.0; //epsilon_xz
        OutputScalarT eps11 = Ugrad(cell,qp,1,1)/h-invh_x*U(cell,qp,0); //epsilon_yy
        OutputScalarT eps12 = Ugrad(cell,qp,1,2)/2.0; //epsilon_yz

        epsilonEqpSq = eps00*eps00 + eps11*eps11 + eps00*eps11 + eps01*eps01 + eps02*eps02 + eps12*eps12;
        epsilonEqpSq += ff; //add regularization "fudge factor"
        mu(cell,qp) *= pow(epsilonEqpSq,  power); //non-linear viscosity, given by Glen's law
      }
    }
  } else {
    if (extractStrainRateSq) {
      for (unsigned int qp=0; qp < numQPs; ++qp) {
        //evaluate non-linear viscosity, given by Glen's law, at quadrature points
        typename PHAL::Ref<const VelT>::type u00 = Ugrad(cell,qp,0,0); //epsilon_xx
        typename PHAL::Ref<const VelT>::type u11 = Ugrad(cell,qp,1,1); //epsilon_yy
        epsilonEqpSq = u00*u00 + u11*u11 + u00*u11; //epsilon_xx^2 + epsilon_yy^2 + epsilon_xx*epsilon_yy
        epsilonEqpSq += 0.25*(Ugrad(cell,qp,0,1) + Ugrad(cell,qp,1,0))*(Ugrad(cell,qp,0,1) + Ugrad(cell,qp,1,0)); //+0.25*epsilon_xy^2

        for (unsigned int dim = 2; dim < numDims; ++dim) //3D case
          epsilonEqpSq += 0.25*(Ugrad(cell,qp,0,dim)*Ugrad(cell,qp,0,dim) + Ugrad(cell,qp,1,dim)*Ugrad(cell,qp,1,dim) ); // + 0.25*epsilon_xz^2 + 0.25*epsilon_yz^2

        epsilonSq(cell,qp) = epsilonEqpSq;
        epsilonEqpSq += ff; //add regularization "fudge factor"
        mu(cell,qp) *= (scale + (1.0-scale)*pow(epsilonEqpSq,  power)); //non-linear viscosity, given by Glen's law
      }
    } else {
      for (unsigned int qp=0; qp < numQPs; ++qp) {
        //evaluate non-linear viscosity, given by Glen's law, at quadrature points
        typename PHAL::Ref<const VelT>::type u00 = Ugrad(cell,qp,0,0); //epsilon_xx
        typename PHAL::Ref<const VelT>::type u11 = Ugrad(cell,qp,1,1); //epsilon_yy
        epsilonEqpSq = u00*u00 + u11*u11 + u00*u11; //epsilon_xx^2 + epsilon_yy^2 + epsilon_xx*epsilon_yy
        epsilonEqpSq += 0.25*(Ugrad(cell,qp,0,1) + Ugrad(cell,qp,1,0))*(Ugrad(cell,qp,0,1) + Ugrad(cell,qp,1,0)); //+0.25*epsilon_xy^2

        for (unsigned int dim = 2; dim < numDims; ++dim) //3D case
          epsilonEqpSq += 0.25*(Ugrad(cell,qp,0,dim)*Ugrad(cell,qp,0,dim) + Ugrad(cell,qp,1,dim)*Ugrad(cell,qp,1,dim) ); // + 0.25*epsilon_xz^2 + 0.25*epsilon_yz^2

        epsilonEqpSq += ff; //add regularization "fudge factor"
        mu(cell,qp) *= pow(epsilonEqpSq,  power); //non-linear viscosity, given by Glen's law
      }
    }
  }
  if(useStiffeningFactor)
    for (unsigned int qp=0; qp < numQPs; ++qp)
      mu(cell,qp) *= std::exp(stiffeningFactor(cell,qp));
}

template<typename EvalT, typename Traits, typename VelT, typename TemprT>
KOKKOS_INLINE_FUNCTION
void ViscosityFO<EvalT, Traits, VelT, TemprT>::operator () (const ViscosityFO_GLENSLAW_UNIFORM_Tag&, const int& cell) const{
  RealType flowFactor= 1.0/2.0*pow(A, -1.0/n); 
  //We start setting mu=flowFactor, then we multiply it by other terms in glenslaw function
  for (unsigned int qp=0; qp < numQPs; ++qp)
    mu(cell,qp) = flowFactor;
  glenslaw(cell);
}

template<typename EvalT, typename Traits, typename VelT, typename TemprT>
KOKKOS_INLINE_FUNCTION
void ViscosityFO<EvalT, Traits, VelT, TemprT>::operator () (const ViscosityFO_GLENSLAW_TEMPERATUREBASED_Tag&, const int& cell) const
{
  TemprT flowFactor = 1.0/2.0*pow(flowRate(temperature(cell)), -1.0/n);
  //We start setting mu=flowFactor, then we multiply it by other terms in glenslaw function
  for (unsigned int qp=0; qp < numQPs; ++qp)
    mu(cell,qp) = flowFactor;
  glenslaw(cell);
}

template<typename EvalT, typename Traits, typename VelT, typename TemprT>
KOKKOS_INLINE_FUNCTION
void ViscosityFO<EvalT, Traits, VelT, TemprT>::operator () (const ViscosityFO_GLENSLAW_TEMPERATUREBASEDQP_Tag&, const int& cell) const
{
  //We start setting mu=flowFactor, then we multiply it by other terms in glenslaw function
  for(int qp=0; qp< numQPs; ++qp)
    mu(cell,qp) = 1.0/2.0*pow(flowRate(temperature(cell,qp)), -1.0/n);
  glenslaw(cell);
}

template<typename EvalT, typename Traits, typename VelT, typename TemprT>
KOKKOS_INLINE_FUNCTION
void ViscosityFO<EvalT, Traits, VelT, TemprT>::operator () (const ViscosityFO_GLENSLAW_FROMFILE_Tag&, const int& cell) const
{
  RealType flowFactor = 1.0/2.0*pow(flowFactorA(cell), -1.0/n);
  //We start setting mu=flowFactor, then we multiply it by other terms in glenslaw function
  for (unsigned int qp=0; qp < numQPs; ++qp)
    mu(cell,qp) = flowFactor;
  glenslaw(cell);
}

//**********************************************************************
template<typename EvalT, typename Traits, typename VelT, typename TemprT>
void ViscosityFO<EvalT, Traits, VelT, TemprT>::
evaluateFields(typename Traits::EvalData workset)
{
#ifdef OUTPUT_TO_SCREEN
  static ScalarT printedH = -9999;
  if (workset.wsIndex==0 && printedH != homotopyParam(0)) {
    std::cout << "[ViscosityFO] homotopyParam: " << homotopyParam(0) << "\n";
    printedH = homotopyParam(0);
  }
#endif

  if (memoizer.have_saved_data(workset,this->evaluatedFields()))
    return;

  switch (visc_type) {
    case CONSTANT:
      Kokkos::parallel_for(ViscosityFO_CONSTANT_Policy(0,workset.numCells),*this);
      break;
    case EXPTRIG:
      Kokkos::parallel_for(ViscosityFO_EXPTRIG_Policy(0,workset.numCells),*this);
      break;
    case GLENSLAW:
      if(useStereographicMap) {
        R = stereographicMapList->get<double>("Earth Radius", 6371);
        x_0 = stereographicMapList->get<double>("X_0", 0);//-136);
        y_0 = stereographicMapList->get<double>("Y_0", 0);//-2040);
        R2 = std::pow(R,2);
      }

      switch (flowRate_type) {
        case UNIFORM:
          Kokkos::parallel_for(ViscosityFO_GLENSLAW_UNIFORM_Policy(0,workset.numCells),*this);
          break;
        case TEMPERATUREBASED:
          if(useP0Temp)
            Kokkos::parallel_for(ViscosityFO_GLENSLAW_TEMPERATUREBASED_Policy(0,workset.numCells),*this);
          else
            Kokkos::parallel_for(ViscosityFO_GLENSLAW_TEMPERATUREBASED_QP_Policy(0,workset.numCells),*this);
          break;
        case FROMFILE:
        case FROMCISM:
          Kokkos::parallel_for(ViscosityFO_GLENSLAW_FROMFILE_Policy(0,workset.numCells),*this);
        break;
      }
      break;
    default:
      TEUCHOS_TEST_FOR_EXCEPTION (true, std::logic_error, "Error! Unexpected value for 'visc_type'.\n");
  }
}

} // Namespace LandIce
