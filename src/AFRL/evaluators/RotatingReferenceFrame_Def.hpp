//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#include <fstream>
#include <sstream>
#include "Teuchos_TestForException.hpp"
#include "Phalanx_DataLayout.hpp"
#include "Sacado.hpp"
#include "Sacado_ParameterRegistration.hpp"
#include "Albany_Utils.hpp"

namespace AFRL {

template<typename EvalT, typename Traits>
RotatingReferenceFrame<EvalT, Traits>::
RotatingReferenceFrame(Teuchos::ParameterList& p,
                       const Teuchos::RCP<Albany::Layouts>& dl) :
  coordinates (p.get<std::string> ("Current Coordinates Name"),
               dl->vertices_vector),
  density (p.get<std::string> ("Density Name"),
           dl->cell_scalar2),
  weights (p.get<std::string>("Weights Name"),
           dl->qp_scalar),
  force (p.get<std::string> ("Force Name"),
         dl->node_vector)
{
  Teuchos::ParameterList* cond_list =
    p.get<Teuchos::ParameterList*>("Parameter List");

  Teuchos::RCP<const Teuchos::ParameterList> reflist =
    this->getValidRotatingReferenceFrameParameters();

  // Check the parameters contained in the input file. Do not check the defaults
  // set programmatically
  cond_list->validateParameters(*reflist, 0,
    Teuchos::VALIDATE_USED_ENABLED, Teuchos::VALIDATE_DEFAULTS_DISABLED);

  Teuchos::RCP<PHX::DataLayout> vector_dl =
    p.get< Teuchos::RCP<PHX::DataLayout> >("QP Vector Data Layout");
  std::vector<PHX::DataLayout::size_type> dims;
  vector_dl->dimensions(dims);
  this->numQPs  = dims[1];

  this->axisOrigin[0] = cond_list->get("axis origin x", 0.);
  this->axisOrigin[1] = cond_list->get("axis origin y", 0.);
  this->axisOrigin[2] = cond_list->get("axis origin z", 0.);
  this->axisDirection[0] = cond_list->get("axis direction x", 0.);
  this->axisDirection[1] = cond_list->get("axis direction y", 0.);
  this->axisDirection[2] = cond_list->get("axis direction z", 1.);
  this->angularFrequency = cond_list->get("angular frequency", 0.);

  // Ensure that axisDirection is normalized
  double len = 0.;
  for (int i = 0; i < 3; i++)
  {
    len += this->axisDirection[i] * this->axisDirection[i];
  }
  len = sqrt(len);
  for (int i = 0; i < 3; i++)
  {
    this->axisDirection[i] /= len;
  }

  this->addDependentField(coordinates);
  this->addDependentField(density);
  this->addDependentField(weights);
  this->addEvaluatedField(force);

  this->setName("Rotating Reference Frame");
}

template<typename EvalT, typename Traits>
RotatingReferenceFrame<EvalT, Traits>::
~RotatingReferenceFrame()
{
}

// **********************************************************************
template<typename EvalT, typename Traits>
void RotatingReferenceFrame<EvalT, Traits>::
postRegistrationSetup(typename Traits::SetupData d,
                      PHX::FieldManager<Traits>& fm)
{
  this->utils.setFieldData(coordinates, fm);
  this->utils.setFieldData(density, fm);
  this->utils.setFieldData(weights, fm);
  this->utils.setFieldData(force, fm);
}

// **********************************************************************
namespace
{
  template <class ScalarT>
  double val(const ScalarT& v)
  {
    return Sacado::ScalarValue<Sacado::Fad::DFad<ScalarT> >::eval(v);
  }
}

template<typename EvalT, typename Traits>
void RotatingReferenceFrame<EvalT, Traits>::
evaluateFields(typename Traits::EvalData workset)
{
  double omega2 = this->angularFrequency * this->angularFrequency;
  double xyz[3], len2, dot, r, cellVol, m, fMag, fDir[3];
  for (int cell = 0; cell < workset.numCells; ++cell)
  {
    // Determine the cell's distance from the axis of rotation
    len2 = dot = 0.;
    for (std::size_t i = 0; i < 3; i++)
    {
      xyz[i] = fDir[i] = this->coordinates(cell, 0, i) - this->axisOrigin[i];
      dot += xyz[i] * this->axisDirection[i];
      len2 += xyz[i] * xyz[i];
    }
    r = std::sqrt(len2 - dot*dot);

    // Determine the direction of force due to centripedal acceleration
    len2 = 0.;
    for (std::size_t i = 0; i < 3; i++)
    {
      fDir[i] -= this->axisDirection[i] * dot;
      len2 += fDir[i] * fDir[i];
    }
    double lenReciprocal = 1./sqrt(len2);
    for (std::size_t i = 0; i < 3; i++)
    {
      fDir[i] *= lenReciprocal;
    }

    // Determine the cell's mass
    cellVol = 0.;
    for (std::size_t qp=0; qp < this->numQPs; ++qp)
    {
      cellVol += weights(cell,qp);
    }
    m = val(this->density(cell)) * cellVol;

    // Determine the force due to centripedal acceleration
    fMag = m * omega2 * r;

    for (std::size_t i = 0; i < 3; i++)
    {
      this->force(cell, 0, i) = fDir[i] * fMag;
    }
  }
}

// **********************************************************************
template<typename EvalT,typename Traits>
Teuchos::RCP<const Teuchos::ParameterList>
RotatingReferenceFrame<EvalT,Traits>::getValidRotatingReferenceFrameParameters() const
{
  Teuchos::RCP<Teuchos::ParameterList> validPL =
    rcp(new Teuchos::ParameterList("Valid Rotating Reference Frame Params"));;

  validPL->set<double>("axis origin x", 0.0, "Axis origin x coordinate");
  validPL->set<double>("axis origin y", 0.0, "Axis origin y coordinate");
  validPL->set<double>("axis origin z", 0.0, "Axis origin z coordinate");

  validPL->set<double>("axis direction x", 0.0, "Axis direction x coordinate");
  validPL->set<double>("axis direction y", 0.0, "Axis direction y coordinate");
  validPL->set<double>("axis direction z", 0.0, "Axis direction z coordinate");

  validPL->set<double>("angular frequency", 0.0, "Angular frequency (rad/s)");

  return validPL;
}

// **********************************************************************
// **********************************************************************
}
