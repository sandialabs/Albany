//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#ifndef ALBANY_NULLSPACEUTILS_HPP
#define ALBANY_NULLSPACEUTILS_HPP

#include "Albany_DataTypes.hpp"

namespace Albany {

struct Tpetra_NullSpace_Traits {

  typedef Tpetra_MultiVector base_array_type;
  typedef Teuchos::RCP<base_array_type> array_type;
  typedef base_array_type::global_ordinal_type GO_type;
  typedef base_array_type::local_ordinal_type LO_type;
  const int Ndof;
  const int NscalarDof;
  const int NSdim;
  const LO_type vec_leng;
  array_type Array;

  Tpetra_NullSpace_Traits(const int ndof, const int nscalardof, const int nsdim,
     const LO_type veclen, array_type &array)
   : Ndof(ndof), NscalarDof(nscalardof), NSdim(nsdim), vec_leng(veclen), Array(array) {}

  void zero(){
      Array->putScalar(0.0);
  }

  double &ArrObj(const LO_type DOF, const int i, const int j){
     Teuchos::ArrayRCP<ST> rdata = Array->getDataNonConst(j);
     return rdata[DOF + i];
  }

};

struct Epetra_NullSpace_Traits {

  typedef std::vector<ST> array_type;
  const int Ndof;
  const int NscalarDof;
  const int NSdim;
  const array_type::size_type vec_leng;
  array_type& Array;

  Epetra_NullSpace_Traits(const int ndof, const int nscalardof, const int nsdim, const array_type::size_type veclen,
      array_type &array)
   : Ndof(ndof), NscalarDof(nscalardof), NSdim(nsdim), vec_leng(veclen), Array(array) {}

  void zero(){
    for (array_type::size_type i = 0; i < vec_leng*(NSdim + NscalarDof); i++)
       Array[i] = 0.0;
  }

  double &ArrObj(const array_type::size_type DOF, const int i, const int j){
     return Array[DOF + i + j * vec_leng];
  }

};

class RigidBodyModes {
public:
  typedef Tpetra_MultiVector::global_ordinal_type GO_type;
  typedef Tpetra_MultiVector::local_ordinal_type LO_type;

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
    const Teuchos::RCP<Tpetra_MultiVector> &coordMV,
    const Teuchos::RCP<const Tpetra_Map>& soln_map = Teuchos::null);

  //! Pass only the coordinates.
  void setCoordinates(const Teuchos::RCP<Tpetra_MultiVector> &coordMV);

private:
  int numPDEs, numElasticityDim, numScalar, nullSpaceDim;
  bool mlUsed, mueLuUsed, setNonElastRBM;

  Teuchos::RCP<Teuchos::ParameterList> plist;

  Teuchos::RCP<Tpetra_MultiVector> coordMV;

  Tpetra_NullSpace_Traits::array_type trr;
  Epetra_NullSpace_Traits::array_type err;

};

} // namespace Albany

#endif /* ALBANY_NULLSPACEUTILS_HPP */
