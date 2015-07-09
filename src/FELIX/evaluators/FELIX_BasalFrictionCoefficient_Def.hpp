//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#include "Teuchos_TestForException.hpp"
#include "Teuchos_VerboseObject.hpp"
#include "Phalanx_DataLayout.hpp"
#include "Intrepid_FunctionSpaceTools.hpp"
#include "Albany_Layouts.hpp"

//uncomment the following line if you want debug output to be printed to screen
#define OUTPUT_TO_SCREEN

namespace FELIX
{

template<typename EvalT, typename Traits>
BasalFrictionCoefficient<EvalT, Traits>::BasalFrictionCoefficient (const Teuchos::ParameterList& p,
                                                                   const Teuchos::RCP<Albany::Layouts>& dl) :
  u_grid      (NULL),
  u_grid_h    (NULL),
  beta_coeffs (NULL),
  beta        (p.get<std::string> ("FELIX Basal Friction Coefficient Name"), dl->node_scalar),
  beta_type   (FROM_FILE)
{
#ifdef OUTPUT_TO_SCREEN
  Teuchos::RCP<Teuchos::FancyOStream> output(Teuchos::VerboseObjectBase::getDefaultOStream());
#endif

  Teuchos::ParameterList* beta_list = p.get<Teuchos::ParameterList*>("Parameter List");
  Teuchos::ParameterList* physics = p.get<Teuchos::ParameterList*>("Physical Parameters");

  std::string betaType = (beta_list->isParameter("Type") ? beta_list->get<std::string>("Type") : "From File");

  if (betaType == "From File")
  {
#ifdef OUTPUT_TO_SCREEN
    *output << "Constant beta, loaded from file.\n";
#endif
    beta_type = FROM_FILE;

    beta_given = PHX::MDField<ScalarT,Cell,Node>(p.get<std::string> ("Given Beta Variable Name"), dl->node_scalar);
    this->addDependentField (beta_given);
  }
  else if (betaType == "Power Law")
  {
    beta_type = POWER_LAW;

    mu    = beta_list->get<double>("Coulomb Friction Coefficient");
    power = beta_list->get("Power Exponent",1.0);
    TEUCHOS_TEST_FOR_EXCEPTION(power<-1.0, Teuchos::Exceptions::InvalidParameter,
        std::endl << "Error in FELIX::BasalFrictionCoefficient: \"Power Exponent\" must be greater than (or equal to) -1.\n");

#ifdef OUTPUT_TO_SCREEN
    *output << "Velocity-dependent beta (power law):\n\n"
            << "      beta = mu * N * |u|^p \n\n"
            << "  with N being the effective pressure, |u| the sliding velocity, and\n"
            << "    - mu (Coulomb Friction Coefficient): " << mu << "\n"
            << "    - p  (Power Exponent): " << power << "\n";
#endif

    N      = PHX::MDField<ScalarT,Cell,Node>(p.get<std::string> ("Effective Pressure Variable Name"), dl->node_scalar);
    u_norm = PHX::MDField<ScalarT,Cell,Node>(p.get<std::string> ("Velocity Norm Variable Name"), dl->node_scalar);
    this->addDependentField (N);
    this->addDependentField (u_norm);
  }
  else if (betaType == "Regularized Coulomb")
  {
    beta_type = REGULARIZED_COULOMB;

    mu    = beta_list->get<double>("Coulomb Friction Coefficient");
    power = beta_list->get("Power Exponent",1.0);
    L = beta_list->get("Regularization Parameter",1e-4);
#ifdef OUTPUT_TO_SCREEN
    *output << "Velocity-dependent beta (regularized coulomb law):\n\n"
            << "      beta = mu * N * |u|^{p-1} / [|u| + L*N^(1/p)]^p\n\n"
            << "  with N being the effective pressure, |u| the sliding velocity, and\n"
            << "    - mu (Coulomb Friction Coefficient): " << mu << "\n"
            << "    - L  (Regularization Parameter): " << L << "\n"
            << "    - p  (Power Exponent): " << power << "\n";
#endif

    N      = PHX::MDField<ScalarT,Cell,Node>(p.get<std::string> ("Effective Pressure Variable Name"), dl->node_scalar);
    u_norm = PHX::MDField<ScalarT,Cell,Node>(p.get<std::string> ("Velocity Norm Variable Name"), dl->node_scalar);
    this->addDependentField (N);
    this->addDependentField (u_norm);
  }
  else if (betaType == "Piecewise Linear")
  {
    beta_type = PIECEWISE_LINEAR;

    Teuchos::ParameterList& pw_params = beta_list->sublist("Piecewise Linear Parameters");

    Teuchos::Array<double> coeffs = pw_params.get<Teuchos::Array<double> >("Values");
    TEUCHOS_TEST_FOR_EXCEPTION(coeffs.size()<1, Teuchos::Exceptions::InvalidParameter,
        std::endl << "Error in FELIX::BasalFrictionCoefficient: \"Values\" must be at least of size 1.\n");

    nb_pts = coeffs.size();
    beta_coeffs = new double[nb_pts+1];
    for (int i(0); i<nb_pts; ++i)
    {
        beta_coeffs[i] = coeffs[i];
    }
#ifdef OUTPUT_TO_SCREEN
    *output << "Velocity-dependent beta (piecewise linear FE):\n\n"
            << "      beta = mu * N * [sum_i c_i*phi_i(|u|)] \n\n"
            << "  with N being the effective pressure, |u| the sliding velocity, and\n"
            << "    - mu  (Coulomb Friction Coefficient): " << mu << "\n"
            << "    - c_i (Values): [" << beta_coeffs[0];
    for (int i(1); i<nb_pts; ++i)
        *output << " " << beta_coeffs[i];
    *output << "]\n";
#endif

    u_grid      = new double[nb_pts+1];
    u_grid_h    = new double[nb_pts];
    if (pw_params.get("Uniform Grid",true))
    {
        double u_max = pw_params.get("Max Velocity",1.0);
        double du = u_max/(nb_pts-1);
        for (int i(0); i<nb_pts; ++i)
            u_grid[i] = i*du;

        for (int i(0); i<nb_pts-1; ++i)
            u_grid_h[i] = du;

        u_grid[nb_pts] = u_max;
        u_grid_h[nb_pts-1] = 1;
#ifdef OUTPUT_TO_SCREEN
        *output << "    - Uniform grid in [0," << u_max << "]\n";
#endif
    }
    else
    {
        Teuchos::Array<double> pts = pw_params.get<Teuchos::Array<double> >("Grid");
        TEUCHOS_TEST_FOR_EXCEPTION(pts.size()!=nb_pts, Teuchos::Exceptions::InvalidParameter,
            std::endl << "Error in FELIX::BasalFrictionCoefficient: \"Grid\" and \"Values\" must be of the same size.\n");

        for (int i(0); i<nb_pts; ++i)
            u_grid[i] = pts[i];
        for (int i(0); i<nb_pts-1; ++i)
            u_grid_h[i] = u_grid[i+1]-u_grid[i];

        u_grid[nb_pts] = u_grid[nb_pts-1];
        u_grid_h[nb_pts-1] = 1;

#ifdef OUTPUT_TO_SCREEN
        *output << "    - User-defined grid: [" << u_grid[0];
        for (int i(1); i<nb_pts; ++i)
            *output << " " << u_grid[i];
        *output << "]\n";
#endif
    }

    u_norm  = PHX::MDField<ScalarT,Cell,Node>(p.get<std::string> ("Velocity Norm Variable Name"), dl->node_scalar);
    this->addDependentField (u_norm);
  }
  else
  {
    TEUCHOS_TEST_FOR_EXCEPTION(true, Teuchos::Exceptions::InvalidParameter,
        std::endl << "Error in FELIX::BasalFrictionCoefficient:  \"" << betaType << "\" is not a valid parameter for Beta Type" << std::endl);
  }

  std::vector<PHX::DataLayout::size_type> dims;
  dl->node_vector->dimensions(dims);
  numCells = dims[0];
  numNodes = dims[1];
  numDims  = dims[2];

  this->addEvaluatedField(beta);

  this->setName("BasalFrictionCoefficient");
}

template<typename EvalT, typename Traits>
BasalFrictionCoefficient<EvalT, Traits>::~BasalFrictionCoefficient()
{
    delete[] u_grid;
    delete[] u_grid_h;
    delete[] beta_coeffs;
}

//**********************************************************************
template<typename EvalT, typename Traits>
void BasalFrictionCoefficient<EvalT, Traits>::
postRegistrationSetup (typename Traits::SetupData d,
                       PHX::FieldManager<Traits>& fm)
{
  switch (beta_type)
  {
    case FROM_FILE:
        this->utils.setFieldData(beta_given,fm);
        break;
    case POWER_LAW:
    case REGULARIZED_COULOMB:
        this->utils.setFieldData(N,fm);
    case PIECEWISE_LINEAR:
        this->utils.setFieldData(u_norm,fm);
  }

  this->utils.setFieldData(beta,fm);
}

template<typename EvalT,typename Traits>
void BasalFrictionCoefficient<EvalT,Traits>::setHomotopyParamPtr(ScalarT* h)
{
    homotopyParam = h;
}

//**********************************************************************
//Kokkos functor
template<typename EvalT, typename Traits>
KOKKOS_INLINE_FUNCTION
void BasalFrictionCoefficient<EvalT, Traits>::operator () (const int i) const
{
    switch (beta_type)
    {
        case FROM_FILE:
            for (int node=0; node < numNodes; ++node)
                beta(i,node) = beta_given(i,node);

            break;

        case POWER_LAW:
            for (int node=0; node < numNodes; ++node)
            {
                beta(i,node) = mu * N(i,node) * std::pow(u_norm(i,node), power);
            }
            break;

        case REGULARIZED_COULOMB:
        {
            ScalarT ff = 0;
            if (homotopyParam!=0 && *homotopyParam!=0)
                ff = pow(10.0, -10.0*(*homotopyParam));

            for (int node=0; node < numNodes; ++node)
            {
                beta(i,node) = mu * N(i,node) * std::pow (u_norm(i,node), power-1)
                             / std::pow(u_norm(i,node) + L*std::pow(N(i,node),1./power), power);
            }
            break;
        }
        case PIECEWISE_LINEAR:
        {
            ptrdiff_t where;
            ScalarT xi;
            for (int node=0; node < numNodes; ++node)
            {
                where = std::lower_bound(u_grid,u_grid+nb_pts,u_norm(i,node)) - u_grid;
                xi = (u_norm(i,node) - u_grid[where]) / u_grid_h[where];

                beta(i,node) = (1-xi)*beta_coeffs[where] + xi*beta_coeffs[where+1];
            }
            break;
        }

    }
}

//**********************************************************************
template<typename EvalT, typename Traits>
void BasalFrictionCoefficient<EvalT, Traits>::evaluateFields (typename Traits::EvalData workset)
{
#ifndef ALBANY_KOKKOS_UNDER_DEVELOPMENT
    switch (beta_type)
    {
        case FROM_FILE:
            for (int cell=0; cell<workset.numCells; ++cell)
                for (int node=0; node < numNodes; ++node)
                    beta(cell,node) = beta_given(cell,node);

            break;

        case POWER_LAW:
            for (int cell=0; cell<workset.numCells; ++cell)
            {
                for (int node=0; node < numNodes; ++node)
                {
                    beta(cell,node) = mu * N(cell,node) * std::pow (u_norm(cell,node), power);
                }
            }
            break;

        case REGULARIZED_COULOMB:
        {
            ScalarT ff = 0;
            if (homotopyParam!=0 && *homotopyParam!=0)
                ff = pow(10.0, -10.0*(*homotopyParam));

            for (int cell=0; cell<workset.numCells; ++cell)
            {
                for (int node=0; node < numNodes; ++node)
                {
                    beta(cell,node) = mu * N(cell,node) * std::pow (u_norm(cell,node), power-1)
                                 / std::pow( u_norm(cell,node) + L*std::pow(N(cell,node),1./power), power);
                }
            }
            break;
        }
        case PIECEWISE_LINEAR:
        {
            ptrdiff_t where;
            ScalarT xi;
            for (int cell=0; cell<workset.numCells; ++cell)
            {
                for (int node=0; node < numNodes; ++node)
                {
                    where = std::lower_bound(u_grid,u_grid+nb_pts,getScalarTValue(u_norm(cell,node))) - u_grid;
                    xi = (u_norm(cell,node) - u_grid[where]) / u_grid_h[where];
                    beta(cell,node) = (1-xi)*beta_coeffs[where] + xi*beta_coeffs[where+1];
                }
            }
            break;
        }
    }
#else
  Kokkos::parallel_for (workset.numCells, *this);
#endif
}

template<>
double BasalFrictionCoefficient<PHAL::AlbanyTraits::Residual,PHAL::AlbanyTraits>::getScalarTValue(const ScalarT& s)
{
    return s;
}

template<typename EvalT, typename Traits>
double BasalFrictionCoefficient<EvalT, Traits>::getScalarTValue(const ScalarT& s)
{
    return s.val();
}

} // Namespace FELIX
