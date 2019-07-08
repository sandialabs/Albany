//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#ifndef ALBANY_NULL_SPACE_UTILS_HPP
#define ALBANY_NULL_SPACE_UTILS_HPP

#include "Albany_ThyraTypes.hpp"

namespace Albany {

// Forward declaration of a helper class, used to hide Tpetra/Epetra details
struct TraitsImplBase;

class RigidBodyModes {
public:
  //! Construct RBM object.
  RigidBodyModes(int numPDEs);

  //! Update the number of PDEs present.
  void setNumPDEs(int numPDEs_) { numPDEs = numPDEs_; }

  //! Set sizes of nullspace etc.
  void setParameters(const int numPDEs, const int numElasticityDim,
                     const int numScalar, const int nullSpaceDim, const bool setNonElastRBM = false);

  //! Set Piro solver parameter list.
  void setPiroPL(const Teuchos::RCP<Teuchos::ParameterList>& piroParams);

  //! Update the parameter list.
  void updatePL(const Teuchos::RCP<Teuchos::ParameterList>& mlParams);

  //! Is ML used on this problem?
  bool isMLUsed() const { return mlUsed; }

  //! Is MueLu used on this problem?
  bool isMueLuUsed() const { return mueLuUsed; }

  //! Pass coordinates and, if numElasticityDim > 0, the null space to ML or
  //! MueLu. The data accessed through getCoordArrays must have been
  //! set. soln_map must be set only if using MueLu and numElasticityDim >
  //! 0. Both maps are nonoverlapping.
  void setCoordinatesAndNullspace(
    const Teuchos::RCP<Thyra_MultiVector> &coordMV,
    const Teuchos::RCP<const Thyra_VectorSpace>& soln_vs = Teuchos::null);

  //! Pass only the coordinates.
  void setCoordinates(const Teuchos::RCP<Thyra_MultiVector> &coordMV);

private:
  int numPDEs, numElasticityDim, numScalar, nullSpaceDim;
  bool mlUsed, mueLuUsed, setNonElastRBM;

  Teuchos::RCP<Teuchos::ParameterList> plist;

  Teuchos::RCP<Thyra_MultiVector> coordMV;

  Teuchos::RCP<TraitsImplBase> traits;
};

} // namespace Albany

#endif // ALBANY_NULL_SPACE_UTILS_HPP
