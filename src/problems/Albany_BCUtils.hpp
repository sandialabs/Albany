//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#ifndef ALBANY_BCUTILS_HPP
#define ALBANY_BCUTILS_HPP

#include <vector>
#include <string>

#include "Teuchos_RCP.hpp"
#include "Teuchos_ParameterList.hpp"

#include "Albany_DataTypes.hpp"
#include "PHAL_AlbanyTraits.hpp"
#include "Phalanx.hpp"
#include "PHAL_FactoryTraits.hpp"

#include "QCAD_MaterialDatabase.hpp"


namespace Albany {

/*!
 * \brief Generic Functions to help define BC Field Manager
 */

//! Traits classes used for BCUtils
struct DirichletTraits {

  enum { type = PHAL::DirichletFactoryTraits<PHAL::AlbanyTraits>::id_dirichlet };
  enum { typeTd = PHAL::DirichletFactoryTraits<PHAL::AlbanyTraits>::id_timedep_bc };
  enum { typeKf = PHAL::DirichletFactoryTraits<PHAL::AlbanyTraits>::id_kfield_bc };
  enum { typeTo = PHAL::DirichletFactoryTraits<PHAL::AlbanyTraits>::id_torsion_bc };
  enum { typeSw = PHAL::DirichletFactoryTraits<PHAL::AlbanyTraits>::id_schwarz_bc };
  enum { typeDa = PHAL::DirichletFactoryTraits<PHAL::AlbanyTraits>::id_dirichlet_aggregator };
  enum { typeFb = PHAL::DirichletFactoryTraits<PHAL::AlbanyTraits>::id_dirichlet_coordinate_function };

  static const std::string bcParamsPl;

  typedef PHAL::DirichletFactoryTraits<PHAL::AlbanyTraits> factory_type;

  static Teuchos::RCP<const Teuchos::ParameterList>
  getValidBCParameters(
    const std::vector<std::string>& nodeSetIDs,
    const std::vector<std::string>& bcNames);

  static std::string
  constructBCName(const std::string ns, const std::string dof);

  static std::string
  constructTimeDepBCName(const std::string ns, const std::string dof);

};

struct NeumannTraits {

  enum { type = PHAL::NeumannFactoryTraits<PHAL::AlbanyTraits>::id_neumann };
  enum { typeNa = PHAL::NeumannFactoryTraits<PHAL::AlbanyTraits>::id_neumann_aggregator };
  enum { typeGCV = PHAL::NeumannFactoryTraits<PHAL::AlbanyTraits>::id_gather_coord_vector };
  enum { typeGS = PHAL::NeumannFactoryTraits<PHAL::AlbanyTraits>::id_gather_solution };
  enum { typeTd = PHAL::NeumannFactoryTraits<PHAL::AlbanyTraits>::id_timedep_bc };
  enum { typeSF = PHAL::NeumannFactoryTraits<PHAL::AlbanyTraits>::id_load_stateField };

  static const std::string bcParamsPl;

  typedef PHAL::NeumannFactoryTraits<PHAL::AlbanyTraits> factory_type;

  static Teuchos::RCP<const Teuchos::ParameterList>
  getValidBCParameters(
    const std::vector<std::string>& sideSetIDs,
    const std::vector<std::string>& bcNames,
    const std::vector<std::string>& conditions);

  static std::string
  constructBCName(const std::string ns, const std::string dof,
                  const std::string condition);

  static std::string
  constructTimeDepBCName(const std::string ns,
                         const std::string dof, const std::string condition);

};

template<typename BCTraits>

class BCUtils {

  public:

    BCUtils() {}

    //! Type of traits class being used
    typedef BCTraits traits_type;

    //! Function to check if the Neumann/Dirichlet BC section of input file is present
    bool haveBCSpecified(const Teuchos::RCP<Teuchos::ParameterList>& params) const {

      // If the BC sublist is not in the input file,
      // side/node sets can be contained in the Exodus file but are not defined in the problem statement.
      // This is OK, just return

      return params->isSublist(traits_type::bcParamsPl);

    }

    //! Specific implementation for Dirichlet BC Evaluator below

    Teuchos::RCP<PHX::FieldManager<PHAL::AlbanyTraits> >
    constructBCEvaluators(
      const std::vector<std::string>& nodeSetIDs,
      const std::vector<std::string>& bcNames,
      Teuchos::RCP<Teuchos::ParameterList> params,
      Teuchos::RCP<ParamLib> paramLib,
      const int numEqn = 0);

    //! Specific implementation for Dirichlet BC Evaluator below

    Teuchos::RCP<PHX::FieldManager<PHAL::AlbanyTraits> >
    constructBCEvaluators(
      const Teuchos::RCP<Albany::MeshSpecsStruct>& meshSpecs,
      const std::vector<std::string>& bcNames,
      const Teuchos::ArrayRCP<std::string>& dof_names,
      bool isVectorField,
      int offsetToFirstDOF,
      const std::vector<std::string>& conditions,
      const Teuchos::Array<Teuchos::Array<int> >& offsets,
      const Teuchos::RCP<Albany::Layouts>& dl,
      Teuchos::RCP<Teuchos::ParameterList> params,
      Teuchos::RCP<ParamLib> paramLib,
      const Teuchos::RCP<QCAD::MaterialDatabase>& materialDB = Teuchos::null);

  private:

    //! Generic implementation of Field Manager construction function
    Teuchos::RCP<PHX::FieldManager<PHAL::AlbanyTraits> >
    buildFieldManager(const std::map<std::string, Teuchos::RCP<Teuchos::ParameterList> >& evals_to_build,
                      std::string& allBC, Teuchos::RCP<PHX::DataLayout>& dummy);

};

//! Specific implementation for Dirichlet BC Evaluator

template<>
Teuchos::RCP<PHX::FieldManager<PHAL::AlbanyTraits> >
BCUtils<DirichletTraits>::constructBCEvaluators(
  const std::vector<std::string>& nodeSetIDs,
  const std::vector<std::string>& bcNames,
  Teuchos::RCP<Teuchos::ParameterList> params,
  Teuchos::RCP<ParamLib> paramLib,
  const int numEqn);

//! Specific implementation for Dirichlet BC Evaluator

template<>
Teuchos::RCP<PHX::FieldManager<PHAL::AlbanyTraits> >
BCUtils<NeumannTraits>::constructBCEvaluators(
  const Teuchos::RCP<Albany::MeshSpecsStruct>& meshSpecs,
  const std::vector<std::string>& bcNames,
  const Teuchos::ArrayRCP<std::string>& dof_names,
  bool isVectorField,
  int offsetToFirstDOF,
  const std::vector<std::string>& conditions,
  const Teuchos::Array<Teuchos::Array<int> >& offsets,
  const Teuchos::RCP<Albany::Layouts>& dl,
  Teuchos::RCP<Teuchos::ParameterList> params,
  Teuchos::RCP<ParamLib> paramLib,
  const Teuchos::RCP<QCAD::MaterialDatabase>& materialDB);

}

// Define macro for explicit template instantiation
#define BCUTILS_INSTANTIATE_TEMPLATE_CLASS_DIRICHLET(name) \
  template class name<Albany::DirichletTraits>;
#define BCUTILS_INSTANTIATE_TEMPLATE_CLASS_NEUMANN(name) \
  template class name<Albany::NeumannTraits>;

#define BCUTILS_INSTANTIATE_TEMPLATE_CLASS(name)     \
  BCUTILS_INSTANTIATE_TEMPLATE_CLASS_DIRICHLET(name)   \
  BCUTILS_INSTANTIATE_TEMPLATE_CLASS_NEUMANN(name)

#endif
