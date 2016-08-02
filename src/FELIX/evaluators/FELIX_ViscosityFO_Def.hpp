//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#include "Teuchos_TestForException.hpp"
#include "Teuchos_VerboseObject.hpp"
#include "Phalanx_DataLayout.hpp"

#include "Intrepid2_FunctionSpaceTools.hpp"
#include "Albany_Layouts.hpp"

//uncomment the following line if you want debug output to be printed to screen
//#define OUTPUT_TO_SCREEN

namespace FELIX {

const double pi = 3.1415926535897932385;
const double actenh = 1.39e5;     //[J mol-1]
const double actenl = 6.0e4;      //[J mol-1]
const double gascon = 8.314;      //[J mol-1 K-1]
const double switchingT = 263.15; // [K]

#ifdef USE_CISM_FLOW_PARAMETERS
  const double arrmlh = 1.733e3;    // [Pa-3 s-1]
  const double arrmll = 3.613e-13;  // [Pa-3 s-1]
  const double k4scyr = 3.1536e19;  // [s y-1]
  const double arrmh = k4scyr*arrmlh;  // [Pa-3 yr-1]
  const double arrml = k4scyr*arrmll;  // [Pa-3 yr-1]
#else
  const double arrmh = 6.26e22;        // [Pa-3 yr-1]
  const double arrml = 1.3e7;          // [Pa-3 yr-1]
#endif


namespace {
template<typename ParamScalarT>
KOKKOS_INLINE_FUNCTION
ParamScalarT flowRate (const ParamScalarT& T) {
  return (T < switchingT) ? arrml / exp (actenl / gascon / T) : arrmh / exp (actenh / gascon / T);
}
}

//**********************************************************************
template<typename EvalT, typename Traits, typename VelT, typename TemprT>
ViscosityFO<EvalT, Traits, VelT, TemprT>::
ViscosityFO(const Teuchos::ParameterList& p,
            const Teuchos::RCP<Albany::Layouts>& dl) :
  Ugrad (p.get<std::string> ("Velocity Gradient QP Variable Name"), dl->qp_vecgradient),
  mu    (p.get<std::string> ("Viscosity QP Variable Name"), dl->qp_scalar),
  epsilonSq (p.get<std::string> ("EpsilonSq QP Variable Name"), dl->qp_scalar),
  temperature(p.get<std::string> ("Temperature Variable Name"), dl->cell_scalar2),
  flowFactorA(p.get<std::string> ("Flow Factor Variable Name"), dl->cell_scalar2),
  homotopyParam("Glen's Law Homotopy Parameter", dl->shared_param),
  A(1.0),
  n(3.0),
  flowRate_type(UNIFORM)
{
  Teuchos::ParameterList* visc_list =
   p.get<Teuchos::ParameterList*>("Parameter List");

  std::string viscType = visc_list->get("Type", "Constant");

  extractStrainRateSq = visc_list->get("Extract Strain Rate Sq", false);

  std::string flowRateType;
  if(visc_list->isParameter("Flow Rate Type"))
    flowRateType = visc_list->get<std::string>("Flow Rate Type");
  else
    flowRateType = "Uniform";

  stereographicMapList = p.get<Teuchos::ParameterList*>("Stereographic Map");
  useStereographicMap = stereographicMapList->get("Use Stereographic Map", false);

  if(useStereographicMap)
    U = PHX::MDField<VelT,Cell,QuadPoint,VecDim>(p.get<std::string> ("Velocity QP Variable Name"), dl->qp_vector);


  A = visc_list->get("Glen's Law A", 1.0);
  n = visc_list->get("Glen's Law n", 3.0);

  coordVec = PHX::MDField<MeshScalarT,Cell,QuadPoint,Dim>(
            p.get<std::string>("Coordinate Vector Variable Name"),dl->qp_gradient);

  Teuchos::RCP<Teuchos::FancyOStream> out(Teuchos::VerboseObjectBase::getDefaultOStream());
  if (viscType == "Constant"){
#ifdef OUTPUT_TO_SCREEN
    *out << "Constant viscosity!" << std::endl;
#endif
    visc_type = CONSTANT;
  }
  else if (viscType == "ExpTrig") {
#ifdef OUTPUT_TO_SCREEN
   *out << "Exp trig viscosity!" << std::endl;
#endif
    visc_type = EXPTRIG;
  }
  //mu for x-z form of FO Stokes equations
  else if (viscType == "Glen's Law X-Z"){
    visc_type = GLENSLAW_XZ;
#ifdef OUTPUT_TO_SCREEN
    *out << "Glen's law x-z viscosity!" << std::endl;
#endif
    flowRate_type = UNIFORM;
#ifdef OUTPUT_TO_SCREEN
      *out << "Uniform Flow Rate A: " << A << std::endl;
#endif
  }
  else if (viscType == "Glen's Law"){
    visc_type = GLENSLAW;
#ifdef OUTPUT_TO_SCREEN
    *out << "Glen's law viscosity!" << std::endl;
#endif
    if (useStereographicMap) {
      this->addDependentField(U);
      this->addDependentField(coordVec);
    }

    if (flowRateType == "Uniform")
    {
      flowRate_type = UNIFORM;
#ifdef OUTPUT_TO_SCREEN
      *out << "Uniform Flow Rate A: " << A << std::endl;
#endif
    }
    else if (flowRateType == "From File")
    {
      flowRate_type = FROMFILE;
      this->addDependentField(flowFactorA);
#ifdef OUTPUT_TO_SCREEN
      *out << "Flow Rate read in from file (exodus or ascii)." << std::endl;
#endif
    }
    else if (flowRateType == "From CISM")
    {
      flowRate_type = FROMCISM;
      this->addDependentField(flowFactorA);
#ifdef OUTPUT_TO_SCREEN
      *out << "Flow Rate passed in from CISM." << std::endl;
#endif
    }
    else if (flowRateType == "Temperature Based")
    {
      flowRate_type = TEMPERATUREBASED;
      this->addDependentField(temperature);
#ifdef OUTPUT_TO_SCREEN
      *out << "Flow Rate computed using temperature field." << std::endl;
#endif
    }
    else {
      TEUCHOS_TEST_FOR_EXCEPTION(true, Teuchos::Exceptions::InvalidParameter,
        std::endl << "Error in FELIX::ViscosityFO:  \"" << flowRateType << "\" is not a valid parameter for Flow Rate Type" << std::endl);
    }
#ifdef OUTPUT_TO_SCREEN
    *out << "n: " << n << std::endl;
#endif
  }

  this->addDependentField(Ugrad);
  this->addDependentField(homotopyParam);
  if (visc_type == EXPTRIG) this->addDependentField(coordVec);
  this->addEvaluatedField(mu);

  if (extractStrainRateSq)
    this->addEvaluatedField(epsilonSq);

  std::vector<PHX::DataLayout::size_type> dims;
  dl->qp_gradient->dimensions(dims);
  numQPs  = dims[1];
  numDims = dims[2];
  numCells = dims[0] ;

  this->setName("ViscosityFO");
}

//**********************************************************************
template<typename EvalT, typename Traits, typename VelT, typename TemprT>
void ViscosityFO<EvalT, Traits, VelT, TemprT>::
postRegistrationSetup(typename Traits::SetupData d,
                      PHX::FieldManager<Traits>& fm)
{
  this->utils.setFieldData(Ugrad,fm);
  this->utils.setFieldData(mu,fm);
  if (extractStrainRateSq)
    this->utils.setFieldData(epsilonSq,fm);

  if (visc_type == EXPTRIG) this->utils.setFieldData(coordVec,fm);
  if (flowRate_type == TEMPERATUREBASED)
    this->utils.setFieldData(temperature,fm);
  if (flowRate_type == FROMFILE || flowRate_type == FROMCISM)
    this->utils.setFieldData(flowFactorA,fm);
  if (useStereographicMap) {
    this->utils.setFieldData(U, fm);
    this->utils.setFieldData(coordVec,fm);
  }
  this->utils.setFieldData(homotopyParam, fm);
}

//**********************************************************************
//Kokkos functors
#ifdef ALBANY_KOKKOS_UNDER_DEVELOPMENT
template<typename EvalT, typename Traits, typename VelT, typename TemprT>
KOKKOS_INLINE_FUNCTION
void ViscosityFO<EvalT, Traits, VelT, TemprT>::operator () (const ViscosityFO_CONSTANT_Tag& tag, const int& cell) const{
  for (int qp=0; qp < numQPs; ++qp)
          mu(cell,qp) = 1.0;
}

template<typename EvalT, typename Traits, typename VelT, typename TemprT>
KOKKOS_INLINE_FUNCTION
void ViscosityFO<EvalT, Traits, VelT, TemprT>::operator () (const ViscosityFO_EXPTRIG_Tag& tag, const int& cell) const
{
  double a = 1.0;
  for (int qp=0; qp < numQPs; ++qp)
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
void ViscosityFO<EvalT, Traits, VelT, TemprT>::glenslaw (const TemprT &flowFactorVec, const int& cell) const
{
  double power = 0.5*(1.0/n - 1.0);
  double a = 1.0;
  if (homotopyParam(0) == 0.0)
  {
    //set constant viscosity
    for (int qp=0; qp < numQPs; ++qp)
    {
      mu(cell,qp) = flowFactorVec;
    }
  }
  else
  {
    ScalarT ff = pow(10.0, -10.0*homotopyParam(0));
    ScalarT epsilonEqpSq = 0.0;
    if(useStereographicMap)
    {
      if (extractStrainRateSq)
      {
        //evaluate non-linear viscosity, given by Glen's law, at quadrature points
      for (int qp=0; qp < numQPs; ++qp)
      {
        MeshScalarT x = coordVec(cell,qp,0)-x_0;
        MeshScalarT y = coordVec(cell,qp,1)-y_0;
        MeshScalarT h = 4.0*R2/(4.0*R2 + x*x + y*y);
        MeshScalarT invh_x = x/2.0/R2;
        MeshScalarT invh_y = y/2.0/R2;

        VelT eps00 = Ugrad(cell,qp,0,0)/h-invh_y*U(cell,qp,1); //epsilon_xx
        VelT eps01 = (Ugrad(cell,qp,0,1)/h+invh_x*U(cell,qp,0)+Ugrad(cell,qp,1,0)/h+invh_y*U(cell,qp,1))/2.0; //epsilon_xy
        VelT eps02 = Ugrad(cell,qp,0,2)/2.0; //epsilon_xz
        VelT eps11 = Ugrad(cell,qp,1,1)/h-invh_x*U(cell,qp,0); //epsilon_yy
        VelT eps12 = Ugrad(cell,qp,1,2)/2.0; //epsilon_yz

        epsilonEqpSq = eps00*eps00 + eps11*eps11 + eps00*eps11 + eps01*eps01 + eps02*eps02 + eps12*eps12;
        epsilonSq(cell,qp) = epsilonEqpSq;
        epsilonEqpSq += ff; //add regularization "fudge factor"
        mu(cell,qp) = flowFactorVec*pow(epsilonEqpSq,  power); //non-linear viscosity, given by Glen's law
      }
      }
      else
      {
      for (int qp=0; qp < numQPs; ++qp)
      {
        MeshScalarT x = coordVec(cell,qp,0)-x_0;
        MeshScalarT y = coordVec(cell,qp,1)-y_0;
        MeshScalarT h = 4.0*R2/(4.0*R2 + x*x + y*y);
        MeshScalarT invh_x = x/2.0/R2;
        MeshScalarT invh_y = y/2.0/R2;

        VelT eps00 = Ugrad(cell,qp,0,0)/h-invh_y*U(cell,qp,1); //epsilon_xx
        VelT eps01 = (Ugrad(cell,qp,0,1)/h+invh_x*U(cell,qp,0)+Ugrad(cell,qp,1,0)/h+invh_y*U(cell,qp,1))/2.0; //epsilon_xy
        VelT eps02 = Ugrad(cell,qp,0,2)/2.0; //epsilon_xz
        VelT eps11 = Ugrad(cell,qp,1,1)/h-invh_x*U(cell,qp,0); //epsilon_yy
        VelT eps12 = Ugrad(cell,qp,1,2)/2.0; //epsilon_yz

        epsilonEqpSq = eps00*eps00 + eps11*eps11 + eps00*eps11 + eps01*eps01 + eps02*eps02 + eps12*eps12;
        epsilonEqpSq += ff; //add regularization "fudge factor"
        mu(cell,qp) = flowFactorVec*pow(epsilonEqpSq,  power); //non-linear viscosity, given by Glen's law
      }
      }
    }
    else
    {
      if (extractStrainRateSq)
      {
        for (int qp=0; qp < numQPs; ++qp)
        {
        //evaluate non-linear viscosity, given by Glen's law, at quadrature points
        typename PHAL::Ref<VelT>::type u00 = Ugrad(cell,qp,0,0); //epsilon_xx
        typename PHAL::Ref<VelT>::type u11 = Ugrad(cell,qp,1,1); //epsilon_yy
        epsilonEqpSq = u00*u00 + u11*u11 + u00*u11; //epsilon_xx^2 + epsilon_yy^2 + epsilon_xx*epsilon_yy
        epsilonEqpSq += 0.25*(Ugrad(cell,qp,0,1) + Ugrad(cell,qp,1,0))*(Ugrad(cell,qp,0,1) + Ugrad(cell,qp,1,0)); //+0.25*epsilon_xy^2

        for (int dim = 2; dim < numDims; ++dim) //3D case
          epsilonEqpSq += 0.25*(Ugrad(cell,qp,0,dim)*Ugrad(cell,qp,0,dim) + Ugrad(cell,qp,1,dim)*Ugrad(cell,qp,1,dim) ); // + 0.25*epsilon_xz^2 + 0.25*epsilon_yz^2

        epsilonSq(cell,qp) = epsilonEqpSq;
        epsilonEqpSq += ff; //add regularization "fudge factor"
        mu(cell,qp) = flowFactorVec*pow(epsilonEqpSq,  power); //non-linear viscosity, given by Glen's law
        }
      }
      else
      {
        for (int qp=0; qp < numQPs; ++qp)
        {
        //evaluate non-linear viscosity, given by Glen's law, at quadrature points
        typename PHAL::Ref<VelT>::type u00 = Ugrad(cell,qp,0,0); //epsilon_xx
        typename PHAL::Ref<VelT>::type u11 = Ugrad(cell,qp,1,1); //epsilon_yy
        epsilonEqpSq = u00*u00 + u11*u11 + u00*u11; //epsilon_xx^2 + epsilon_yy^2 + epsilon_xx*epsilon_yy
        epsilonEqpSq += 0.25*(Ugrad(cell,qp,0,1) + Ugrad(cell,qp,1,0))*(Ugrad(cell,qp,0,1) + Ugrad(cell,qp,1,0)); //+0.25*epsilon_xy^2

        for (int dim = 2; dim < numDims; ++dim) //3D case
          epsilonEqpSq += 0.25*(Ugrad(cell,qp,0,dim)*Ugrad(cell,qp,0,dim) + Ugrad(cell,qp,1,dim)*Ugrad(cell,qp,1,dim) ); // + 0.25*epsilon_xz^2 + 0.25*epsilon_yz^2

        epsilonEqpSq += ff; //add regularization "fudge factor"
        mu(cell,qp) = flowFactorVec*pow(epsilonEqpSq,  power); //non-linear viscosity, given by Glen's law
        }
      }
    }
  }
}

template<typename EvalT, typename Traits, typename VelT, typename TemprT>
KOKKOS_INLINE_FUNCTION
void ViscosityFO<EvalT, Traits, VelT, TemprT>::operator () (const ViscosityFO_GLENSLAW_UNIFORM_Tag& tag, const int& cell) const{
  TemprT flowFactorVec;
  flowFactorVec = 1.0/2.0*pow(A, -1.0/n);
  glenslaw(flowFactorVec,cell);
}

template<typename EvalT, typename Traits, typename VelT, typename TemprT>
KOKKOS_INLINE_FUNCTION
void ViscosityFO<EvalT, Traits, VelT, TemprT>::operator () (const ViscosityFO_GLENSLAW_TEMPERATUREBASED_Tag& tag, const int& cell) const
{
  TemprT flowFactorVec;
  flowFactorVec =1.0/2.0*pow(flowRate<TemprT>(temperature(cell)), -1.0/n);
  glenslaw(flowFactorVec,cell);
}

template<typename EvalT, typename Traits, typename VelT, typename TemprT>
KOKKOS_INLINE_FUNCTION
void ViscosityFO<EvalT, Traits, VelT, TemprT>::operator () (const ViscosityFO_GLENSLAW_FROMFILE_Tag& tag, const int& cell) const
{
   TemprT flowFactorVec;
   flowFactorVec =1.0/2.0*pow(flowFactorA(cell), -1.0/n);
   glenslaw(flowFactorVec,cell);
}


template<typename EvalT, typename Traits, typename VelT, typename TemprT>
KOKKOS_INLINE_FUNCTION
void ViscosityFO<EvalT, Traits, VelT, TemprT>::glenslaw_xz (const TemprT &flowFactorVec, const int& cell) const
{
  double power = 0.5*(1.0/n - 1.0);
  double a = 1.0;
  if (homotopyParam(0) == 0.0)
  {
    //set constant viscosity
    for (int qp=0; qp < numQPs; ++qp)
    {
      mu(cell,qp) = flowFactorVec;
    }
  }
  else
  {
    ScalarT ff = pow(10.0, -10.0*homotopyParam(0));
    ScalarT epsilonEqpSq = 0.0;
    if (extractStrainRateSq)
    {
      for (int qp=0; qp < numQPs; ++qp)
      {
        typename PHAL::Ref<VelT>::type u00 = Ugrad(cell,qp,0,0); //epsilon_xx
        epsilonEqpSq = u00*u00; //epsilon_xx^2
        epsilonEqpSq += 0.25*(Ugrad(cell,qp,0,0) + Ugrad(cell,qp,0,1))*(Ugrad(cell,qp,0,0) + Ugrad(cell,qp,0,1)); //+0.25*epsilon_xz^2
        epsilonSq(cell,qp) = epsilonEqpSq;
        epsilonEqpSq += ff; //add regularization "fudge factor"
        mu(cell,qp) = flowFactorVec*pow(epsilonEqpSq,  power); //non-linear viscosity, given by Glen's law
      }
    }
    else
    {
      for (int qp=0; qp < numQPs; ++qp)
      {
        typename PHAL::Ref<VelT>::type u00 = Ugrad(cell,qp,0,0); //epsilon_xx
        epsilonEqpSq = u00*u00; //epsilon_xx^2
        epsilonEqpSq += 0.25*(Ugrad(cell,qp,0,0) + Ugrad(cell,qp,0,1))*(Ugrad(cell,qp,0,0) + Ugrad(cell,qp,0,1)); //+0.25*epsilon_xz^2
        epsilonEqpSq += ff; //add regularization "fudge factor"
        mu(cell,qp) = flowFactorVec*pow(epsilonEqpSq,  power); //non-linear viscosity, given by Glen's law
      }
    }
  }
}

template<typename EvalT, typename Traits, typename VelT, typename TemprT>
KOKKOS_INLINE_FUNCTION
void ViscosityFO<EvalT, Traits, VelT, TemprT>::operator () (const ViscosityFO_GLENSLAW_XZ_UNIFORM_Tag& tag, const int& cell) const
{
  TemprT flowFactorVec;
  flowFactorVec = 1.0/2.0*pow(A, -1.0/n);
  glenslaw_xz(flowFactorVec,cell);
}

template<typename EvalT, typename Traits, typename VelT, typename TemprT>
KOKKOS_INLINE_FUNCTION
void ViscosityFO<EvalT, Traits, VelT, TemprT>::operator () (const ViscosityFO_GLENSLAW_XZ_TEMPERATUREBASED_Tag& tag, const int& cell) const
{
  TemprT flowFactorVec;
  flowFactorVec =1.0/2.0*pow(flowRate<TemprT>(temperature(cell)), -1.0/n);
  glenslaw_xz(flowFactorVec,cell);
}

template<typename EvalT, typename Traits, typename VelT, typename TemprT>
KOKKOS_INLINE_FUNCTION
void ViscosityFO<EvalT, Traits, VelT, TemprT>::operator () (const ViscosityFO_GLENSLAW_XZ_FROMFILE_Tag& tag, const int& cell) const
{
   TemprT flowFactorVec;
   flowFactorVec =1.0/2.0*pow(flowFactorA(cell), -1.0/n);
   glenslaw_xz(flowFactorVec,cell);
}

template<typename EvalT, typename Traits, typename VelT, typename TemprT>
KOKKOS_INLINE_FUNCTION
void ViscosityFO<EvalT, Traits, VelT, TemprT>::operator () (const ViscosityFO_GLENSLAW_XZ_FROMCISM_Tag& tag, const int& cell) const
{

}

#endif
//**********************************************************************
template<typename EvalT, typename Traits, typename VelT, typename TemprT>
void ViscosityFO<EvalT, Traits, VelT, TemprT>::
evaluateFields(typename Traits::EvalData workset)
{

//std::cout << "before viscosity coord vec" << coordVec(0,0,0) << "   " <<coordVec(1,1,1) << "   " <<coordVec(2,2,2) << "   " <<coordVec(3,3,3) << "   " <<std::endl;
#ifndef ALBANY_KOKKOS_UNDER_DEVELOPMENT
  double a = 1.0;
  switch (visc_type)
  {
    case CONSTANT:
      for (std::size_t cell=0; cell < workset.numCells; ++cell)
      {
        for (std::size_t qp=0; qp < numQPs; ++qp)
          mu(cell,qp) = 1.0;
      }
      break;
    case EXPTRIG:
      for (std::size_t cell=0; cell < workset.numCells; ++cell)
      {
        for (std::size_t qp=0; qp < numQPs; ++qp)
        {
          MeshScalarT x = coordVec(cell,qp,0);
          MeshScalarT y2pi = 2.0*pi*coordVec(cell,qp,1);
          MeshScalarT muargt = (a*a + 4.0*pi*pi - 2.0*pi*a)*sin(y2pi)*sin(y2pi) + 1.0/4.0*(2.0*pi+a)*(2.0*pi+a)*cos(y2pi)*cos(y2pi);
          muargt = sqrt(muargt)*exp(a*x);
          mu(cell,qp) = 1.0/2.0*pow(A, -1.0/n)*pow(muargt, 1.0/n - 1.0);
        }
      }
      break;
    case GLENSLAW:
    case GLENSLAW_XZ:
      std::vector<TemprT> flowFactorVec; //create vector of the flow factor A at each cell
      flowFactorVec.resize(workset.numCells);
      switch (flowRate_type)
      {
        case UNIFORM:
          for (std::size_t cell=0; cell < workset.numCells; ++cell)
          {
            flowFactorVec[cell] = 1.0/2.0*pow(A, -1.0/n);
          }
          break;
        case TEMPERATUREBASED:
          for (std::size_t cell=0; cell < workset.numCells; ++cell)
          {
            //evaluate non-linear viscosity, given by Glen's law, at quadrature points
            temperature(cell) = std::max(temperature(cell), 240.0); //Albany::ADValue(temperature(cell))
            //std::cout << temperature(cell) << std::endl;
            flowFactorVec[cell] = 1.0/2.0*pow(flowRate<TemprT>(temperature(cell)), -1.0/n);
          }
          break;
        case FROMFILE:
        case FROMCISM:
          for (std::size_t cell=0; cell < workset.numCells; ++cell)
            flowFactorVec[cell] = 1.0/2.0*pow(flowFactorA(cell), -1.0/n);
          break;
      }
      double power = 0.5*(1.0/n - 1.0);
      if (homotopyParam(0) == 0.0)
      {
        //set constant viscosity
        for (std::size_t cell=0; cell < workset.numCells; ++cell)
        {
          for (std::size_t qp=0; qp < numQPs; ++qp)
          {
            mu(cell,qp) = flowFactorVec[cell];
          }
        }
      }
      else
      {
        //set Glen's law viscosity with regularization specified by homotopyParam
        ScalarT ff = pow(10.0, -10.0*homotopyParam(0));
        ScalarT epsilonEqpSq = 0.0; //used to define the viscosity in non-linear Stokes
        if (visc_type == GLENSLAW)
        {
          if(useStereographicMap)
          {
            double R = stereographicMapList->get<double>("Earth Radius", 6371);
            double x_0 = stereographicMapList->get<double>("X_0", 0);//-136);
            double y_0 = stereographicMapList->get<double>("Y_0", 0);//-2040);
            double R2 = std::pow(R,2);
            if(extractStrainRateSq)
            {
        for (std::size_t cell=0; cell < workset.numCells; ++cell)
        {
          //evaluate non-linear viscosity, given by Glen's law, at quadrature points
          for (std::size_t qp=0; qp < numQPs; ++qp)
          {
          MeshScalarT x = coordVec(cell,qp,0)-x_0;
          MeshScalarT y = coordVec(cell,qp,1)-y_0;
          MeshScalarT h = 4.0*R2/(4.0*R2 + x*x + y*y);
          MeshScalarT invh_x = x/2.0/R2;
          MeshScalarT invh_y = y/2.0/R2;

          VelT eps00 = Ugrad(cell,qp,0,0)/h-invh_y*U(cell,qp,1); //epsilon_xx
          VelT eps01 = (Ugrad(cell,qp,0,1)/h+invh_x*U(cell,qp,0)+Ugrad(cell,qp,1,0)/h+invh_y*U(cell,qp,1))/2.0; //epsilon_xy
          VelT eps02 = Ugrad(cell,qp,0,2)/2.0; //epsilon_xz
          VelT eps11 = Ugrad(cell,qp,1,1)/h-invh_x*U(cell,qp,0); //epsilon_yy
          VelT eps12 = Ugrad(cell,qp,1,2)/2.0; //epsilon_yz

          epsilonEqpSq = eps00*eps00 + eps11*eps11 + eps00*eps11 + eps01*eps01 + eps02*eps02 + eps12*eps12;
          epsilonSq(cell,qp) = epsilonEqpSq;
          epsilonEqpSq += ff; //add regularization "fudge factor"
          mu(cell,qp) = flowFactorVec[cell]*pow(epsilonEqpSq,  power); //non-linear viscosity, given by Glen's law
          }
        }
            }
            else
            {
          for (std::size_t cell=0; cell < workset.numCells; ++cell)
          {
            //evaluate non-linear viscosity, given by Glen's law, at quadrature points
            for (std::size_t qp=0; qp < numQPs; ++qp)
            {
            MeshScalarT x = coordVec(cell,qp,0)-x_0;
            MeshScalarT y = coordVec(cell,qp,1)-y_0;
            MeshScalarT h = 4.0*R2/(4.0*R2 + x*x + y*y);
            MeshScalarT invh_x = x/2.0/R2;
            MeshScalarT invh_y = y/2.0/R2;

            VelT eps00 = Ugrad(cell,qp,0,0)/h-invh_y*U(cell,qp,1); //epsilon_xx
            VelT eps01 = (Ugrad(cell,qp,0,1)/h+invh_x*U(cell,qp,0)+Ugrad(cell,qp,1,0)/h+invh_y*U(cell,qp,1))/2.0; //epsilon_xy
            VelT eps02 = Ugrad(cell,qp,0,2)/2.0; //epsilon_xz
            VelT eps11 = Ugrad(cell,qp,1,1)/h-invh_x*U(cell,qp,0); //epsilon_yy
            VelT eps12 = Ugrad(cell,qp,1,2)/2.0; //epsilon_yz

            epsilonEqpSq = eps00*eps00 + eps11*eps11 + eps00*eps11 + eps01*eps01 + eps02*eps02 + eps12*eps12;
            epsilonEqpSq += ff; //add regularization "fudge factor"
            mu(cell,qp) = flowFactorVec[cell]*pow(epsilonEqpSq,  power); //non-linear viscosity, given by Glen's law
            }
          }
            }
          }
          else
          {
              if(extractStrainRateSq)
              {
                for (std::size_t cell=0; cell < workset.numCells; ++cell)
                {
                  for (std::size_t qp=0; qp < numQPs; ++qp)
            {
                    //evaluate non-linear viscosity, given by Glen's law, at quadrature points
              typename PHAL::Ref<VelT>::type u00 = Ugrad(cell,qp,0,0); //epsilon_xx
              typename PHAL::Ref<VelT>::type u11 = Ugrad(cell,qp,1,1); //epsilon_yy
              epsilonEqpSq = u00*u00 + u11*u11 + u00*u11; //epsilon_xx^2 + epsilon_yy^2 + epsilon_xx*epsilon_yy
              epsilonEqpSq += 0.25*(Ugrad(cell,qp,0,1) + Ugrad(cell,qp,1,0))*(Ugrad(cell,qp,0,1) + Ugrad(cell,qp,1,0)); //+0.25*epsilon_xy^2

              for (int dim = 2; dim < numDims; ++dim) //3D case
                epsilonEqpSq += 0.25*(Ugrad(cell,qp,0,dim)*Ugrad(cell,qp,0,dim) + Ugrad(cell,qp,1,dim)*Ugrad(cell,qp,1,dim) ); // + 0.25*epsilon_xz^2 + 0.25*epsilon_yz^2

              epsilonSq(cell,qp) = epsilonEqpSq;
              epsilonEqpSq += ff; //add regularization "fudge factor"
              mu(cell,qp) = flowFactorVec[cell]*pow(epsilonEqpSq,  power); //non-linear viscosity, given by Glen's law
            }
           }
              }
              else
              {
                for (std::size_t cell=0; cell < workset.numCells; ++cell)
                {
                  for (std::size_t qp=0; qp < numQPs; ++qp)
            {
                    //evaluate non-linear viscosity, given by Glen's law, at quadrature points
              typename PHAL::Ref<VelT>::type u00 = Ugrad(cell,qp,0,0); //epsilon_xx
              typename PHAL::Ref<VelT>::type u11 = Ugrad(cell,qp,1,1); //epsilon_yy
              epsilonEqpSq = u00*u00 + u11*u11 + u00*u11; //epsilon_xx^2 + epsilon_yy^2 + epsilon_xx*epsilon_yy
              epsilonEqpSq += 0.25*(Ugrad(cell,qp,0,1) + Ugrad(cell,qp,1,0))*(Ugrad(cell,qp,0,1) + Ugrad(cell,qp,1,0)); //+0.25*epsilon_xy^2

              for (int dim = 2; dim < numDims; ++dim) //3D case
                epsilonEqpSq += 0.25*(Ugrad(cell,qp,0,dim)*Ugrad(cell,qp,0,dim) + Ugrad(cell,qp,1,dim)*Ugrad(cell,qp,1,dim) ); // + 0.25*epsilon_xz^2 + 0.25*epsilon_yz^2

              epsilonEqpSq += ff; //add regularization "fudge factor"
              mu(cell,qp) = flowFactorVec[cell]*pow(epsilonEqpSq,  power); //non-linear viscosity, given by Glen's law
            }
           }
              }
          }
        } //endif visc_type == GLENSLAW
        else
        { //XZ FO Stokes equations -- treat 2nd dimension as z
          if(extractStrainRateSq)
          {
            for (std::size_t cell=0; cell < workset.numCells; ++cell)
        {
              for (std::size_t qp=0; qp < numQPs; ++qp)
          {
            typename PHAL::Ref<VelT>::type u00 = Ugrad(cell,qp,0,0); //epsilon_xx
            epsilonEqpSq = u00*u00; //epsilon_xx^2
            epsilonEqpSq += 0.25*Ugrad(cell,qp,0,1)*Ugrad(cell,qp,0,1); //+0.25*epsilon_xz^2
            epsilonSq(cell,qp) = epsilonEqpSq;
            epsilonEqpSq += ff; //add regularization "fudge factor"
            mu(cell,qp) = flowFactorVec[cell]*pow(epsilonEqpSq,  power); //non-linear viscosity, given by Glen's law
          }
        }
          }
          else
          {
            for (std::size_t cell=0; cell < workset.numCells; ++cell)
        {
              for (std::size_t qp=0; qp < numQPs; ++qp)
          {
            typename PHAL::Ref<VelT>::type u00 = Ugrad(cell,qp,0,0); //epsilon_xx
            epsilonEqpSq = u00*u00; //epsilon_xx^2
            epsilonEqpSq += 0.25*Ugrad(cell,qp,0,1)*Ugrad(cell,qp,0,1); //+0.25*epsilon_xz^2
            epsilonEqpSq += ff; //add regularization "fudge factor"
            mu(cell,qp) = flowFactorVec[cell]*pow(epsilonEqpSq,  power); //non-linear viscosity, given by Glen's law
          }
        }
          }
        }
      } //endif Glen's law viscosity with regularization specified by homotopyParam
      break;
  }
#else
  switch (visc_type)
  {
    case CONSTANT:
      Kokkos::parallel_for(ViscosityFO_CONSTANT_Policy(0,workset.numCells),*this);
      break;
    case EXPTRIG:
      Kokkos::parallel_for(ViscosityFO_EXPTRIG_Policy(0,workset.numCells),*this);
      break;
    case GLENSLAW:
      if(useStereographicMap)
      {
        R = stereographicMapList->get<double>("Earth Radius", 6371);
        x_0 = stereographicMapList->get<double>("X_0", 0);//-136);
        y_0 = stereographicMapList->get<double>("Y_0", 0);//-2040);
        R2 = std::pow(R,2);
      }

      switch (flowRate_type)
      {
        case UNIFORM:
          Kokkos::parallel_for(ViscosityFO_GLENSLAW_UNIFORM_Policy(0,workset.numCells),*this);
          break;
        case TEMPERATUREBASED:
          Kokkos::parallel_for(ViscosityFO_GLENSLAW_TEMPERATUREBASED_Policy(0,workset.numCells),*this);
          break;
        case FROMFILE:
        case FROMCISM:
          Kokkos::parallel_for(ViscosityFO_GLENSLAW_FROMFILE_Policy(0,workset.numCells),*this);
        break;
      }
      break;
    case GLENSLAW_XZ:
      switch (flowRate_type)
      {
        case UNIFORM:
          Kokkos::parallel_for(ViscosityFO_GLENSLAW_XZ_UNIFORM_Policy(0,workset.numCells),*this);
          break;
        case TEMPERATUREBASED:
          Kokkos::parallel_for(ViscosityFO_GLENSLAW_XZ_TEMPERATUREBASED_Policy(0,workset.numCells),*this);
          break;
        case FROMFILE:
        case FROMCISM:
          Kokkos::parallel_for(ViscosityFO_GLENSLAW_XZ_FROMFILE_Policy(0,workset.numCells),*this);
          break;
      }
      break;
    default:
      TEUCHOS_TEST_FOR_EXCEPTION (true, std::logic_error, "Error! Unexpected value for 'visc_type'.\n");
  }
#endif
}

} // Namespace FELIX
