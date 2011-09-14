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


#ifndef QCAD_SCHRODINGERRESID_HPP
#define QCAD_SCHRODINGERRESID_HPP

#include "Phalanx_ConfigDefs.hpp"
#include "Phalanx_Evaluator_WithBaseImpl.hpp"
#include "Phalanx_Evaluator_Derived.hpp"
#include "Phalanx_MDField.hpp"

#include "QCAD_MaterialDatabase.hpp"

namespace QCAD {
/** \brief Finite Element Interpolation Evaluator

    This evaluator interpolates nodal DOF values to quad points.

*/

template<typename EvalT, typename Traits>
class SchrodingerResid : public PHX::EvaluatorWithBaseImpl<Traits>,
		    public PHX::EvaluatorDerived<EvalT, Traits>  {

public:

  SchrodingerResid(const Teuchos::ParameterList& p);

  void postRegistrationSetup(typename Traits::SetupData d,
                      PHX::FieldManager<Traits>& vm);

  void evaluateFields(typename Traits::EvalData d);

private:

  typedef typename EvalT::ScalarT ScalarT;
  typedef typename EvalT::MeshScalarT MeshScalarT;

  //! Helper function to compute inverse effective mass (possible position dependent)
  ScalarT getInvEffMass(const std::string& EBName, const int numDim, const MeshScalarT* coord);

  //! Reference parameter list generator to check xml input file
  Teuchos::RCP<const Teuchos::ParameterList>
      getValidMaterialParameters() const;

  // Input:
  std::size_t numQPs;
  std::size_t numDims;

  PHX::MDField<MeshScalarT,Cell,Node,QuadPoint> wBF;
  PHX::MDField<ScalarT,Cell,QuadPoint> psi;
  PHX::MDField<ScalarT,Cell,QuadPoint> psiDot;
  PHX::MDField<MeshScalarT,Cell,Node,QuadPoint,Dim> wGradBF;
  PHX::MDField<ScalarT,Cell,QuadPoint,Dim> psiGrad;
  PHX::MDField<ScalarT,Cell,QuadPoint> V;
  PHX::MDField<MeshScalarT,Cell,QuadPoint,Dim> coordVec;
  PHX::MDField<ScalarT,Cell,QuadPoint> invEffMass; //unused, really just intermediate

  bool enableTransient;
  bool havePotential;
  bool haveMaterial;
  bool bOnlyInQuantumBlocks;

  // Output:
  PHX::MDField<ScalarT,Cell,Node> psiResidual;

  // Intermediate workspace
  Intrepid::FieldContainer<ScalarT> psiGradWithMass;
  Intrepid::FieldContainer<ScalarT> psiV;

  //! units
  double energy_unit_in_eV, length_unit_in_m;
  
  //! Material database
  Teuchos::RCP<QCAD::MaterialDatabase> materialDB;
  
  //! parameters for Finite Wall potential
  std::string potentialType;
  double barrEffMass; // in [m0]
  double barrWidth;   // in length_unit_in_m
  double wellEffMass;
  double wellWidth; 

};

}

#endif
