//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#ifndef ALBANY_NULLSPACEUTILS_HPP
#define ALBANY_NULLSPACEUTILS_HPP

#include "Albany_DataTypes.hpp"

namespace Albany {

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

  //! Resize object as mesh changes. Parameters must already have been set.
  void resize(const int numSpaceDim, const LO numNodes);

  //! Access the arrays to store the coordinates.
  void getCoordArrays(double*& x, double*& y, double*& z);

  //! Access the arrays to store the coordinates -- same as x, y and z but
  //! concatenated.
  double* getCoordArray();

  //! Is ML used on this problem?
  bool isMLUsed() const { return mlUsed; }

  //! Is MueLu used on this problem?
  bool isMueLuUsed() const { return mueLuUsed; }

  //! Pass coordinates and, if numElasticityDim > 0, the null space to ML or
  //! MueLu. The data accessed through getCoordArrays must have been
  //! set. soln_map must be set only if using MueLu and numElasticityDim >
  //! 0. Both maps are nonoverlapping.
  void setCoordinatesAndNullspace(
    const Teuchos::RCP<const Tpetra_Map>& node_map,
    const Teuchos::RCP<const Tpetra_Map>& soln_map = Teuchos::null);

  //! Pass only the coordinates.
  void setCoordinates(const Teuchos::RCP<const Tpetra_Map>& node_map);

private:
  int numPDEs, numElasticityDim, numScalar, nullSpaceDim, numSpaceDim;
  bool mlUsed, mueLuUsed, setNonElastRBM;

  Teuchos::RCP<Teuchos::ParameterList> plist;

  std::vector<double> xyz, rr;
};

} // namespace Albany

#endif /* ALBANY_NULLSPACEUTILS_HPP */
