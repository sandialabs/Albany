//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
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

#include "Albany_MaterialDatabase.hpp"

namespace Albany {

/*!
 * \brief Generic Functions to help define BC Field Manager
 */

//! Traits classes used for BCUtils
struct DirichletTraits {

  enum { type = PHAL::DirichletFactoryTraits<PHAL::AlbanyTraits>::id_dirichlet };
  enum { typeTd = PHAL::DirichletFactoryTraits<PHAL::AlbanyTraits>::id_timedep_bc };
  enum { typeKf = PHAL::DirichletFactoryTraits<PHAL::AlbanyTraits>::id_kfield_bc };
  enum { typeEq = PHAL::DirichletFactoryTraits<PHAL::AlbanyTraits>::id_eq_concentration_bc };
  enum { typeTo = PHAL::DirichletFactoryTraits<PHAL::AlbanyTraits>::id_torsion_bc };
  enum { typeSw = PHAL::DirichletFactoryTraits<PHAL::AlbanyTraits>::id_schwarz_bc };
  enum { typePd = PHAL::DirichletFactoryTraits<PHAL::AlbanyTraits>::id_pd_neigh_fit_bc };
  enum { typeDa = PHAL::DirichletFactoryTraits<PHAL::AlbanyTraits>::id_dirichlet_aggregator };
  enum { typeFb = PHAL::DirichletFactoryTraits<PHAL::AlbanyTraits>::id_dirichlet_coordinate_function };
  enum { typeF  = PHAL::DirichletFactoryTraits<PHAL::AlbanyTraits>::id_dirichlet_field };
  enum { typeON = PHAL::DirichletFactoryTraits<PHAL::AlbanyTraits>::id_dirichlet_off_nodeset };

  static const std::string bcParamsPl;

  typedef PHAL::DirichletFactoryTraits<PHAL::AlbanyTraits> factory_type;

  static Teuchos::RCP<const Teuchos::ParameterList>
  getValidBCParameters(
    const std::vector<std::string>& nodeSetIDs,
    const std::vector<std::string>& bcNames);

  static std::string
  constructBCName(const std::string& ns, const std::string& dof);

  static std::string
  constructBCNameField(const std::string& ns, const std::string& dof);

  static std::string
  constructTimeDepBCName(const std::string& ns, const std::string& dof);

  static std::string
  constructBCNameOffNodeSet(const std::string& ns, const std::string& dof);

  static std::string
  constructPressureDepBCName(const std::string& ns, const std::string& dof);
};

struct NeumannTraits {

  enum { type = PHAL::NeumannFactoryTraits<PHAL::AlbanyTraits>::id_neumann };
  enum { typeNa = PHAL::NeumannFactoryTraits<PHAL::AlbanyTraits>::id_neumann_aggregator };
  enum { typeGCV = PHAL::NeumannFactoryTraits<PHAL::AlbanyTraits>::id_gather_coord_vector };
  enum { typeGS = PHAL::NeumannFactoryTraits<PHAL::AlbanyTraits>::id_gather_solution };
  enum { typeTd = PHAL::NeumannFactoryTraits<PHAL::AlbanyTraits>::id_timedep_bc };
  enum { typeSF = PHAL::NeumannFactoryTraits<PHAL::AlbanyTraits>::id_load_stateField };
  enum { typeSNP = PHAL::NeumannFactoryTraits<PHAL::AlbanyTraits>::id_GatherScalarNodalParameter };

  static const std::string bcParamsPl;

  typedef PHAL::NeumannFactoryTraits<PHAL::AlbanyTraits> factory_type;

  static Teuchos::RCP<const Teuchos::ParameterList>
  getValidBCParameters(
    const std::vector<std::string>& sideSetIDs,
    const std::vector<std::string>& bcNames,
    const std::vector<std::string>& conditions);

  static std::string
  constructBCName(const std::string& ns, const std::string& dof,
                  const std::string& condition);

  static std::string
  constructTimeDepBCName(const std::string& ns,
                         const std::string& dof, const std::string& condition);

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

    Teuchos::Array<Teuchos::Array<int>> getOffsets() const {
      return offsets_; 
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
      const Teuchos::RCP<Albany::MaterialDatabase>& materialDB = Teuchos::null);

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
      const std::vector<Teuchos::RCP<PHX::Evaluator<PHAL::AlbanyTraits> > >& extra_evaluators,
      const Teuchos::RCP<Albany::MaterialDatabase>& materialDB = Teuchos::null);

  private:

    //! Builds the list
    void buildEvaluatorsList(
      std::map<std::string, Teuchos::RCP<Teuchos::ParameterList> >& evaluatorss_to_build,
      const std::vector<std::string>& nodeSetIDs,
      const std::vector<std::string>& bcNames,
      Teuchos::RCP<Teuchos::ParameterList> params,
      Teuchos::RCP<ParamLib> paramLib,
      const int numEqn);

    //! Creates the list of evaluators (together with their parameter lists) to build
    void buildEvaluatorsList (
      std::map<std::string, Teuchos::RCP<Teuchos::ParameterList> >& evaluators_to_build,
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
      const Teuchos::RCP<Albany::MaterialDatabase>& materialDB = Teuchos::null);

    //! Generic implementation of Field Manager construction function
    Teuchos::RCP<PHX::FieldManager<PHAL::AlbanyTraits> >
    buildFieldManager(
        const Teuchos::RCP< std::vector< Teuchos::RCP<PHX::Evaluator_TemplateManager<PHAL::AlbanyTraits> > > > evaluators,
        std::string& allBC, Teuchos::RCP<PHX::DataLayout>& dummy);

    protected: 
  
    Teuchos::Array<Teuchos::Array<int> > offsets_; 
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
  const Teuchos::RCP<Albany::MaterialDatabase>& materialDB);

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
  const std::vector<Teuchos::RCP<PHX::Evaluator<PHAL::AlbanyTraits> > >& extra_evaluators,
  const Teuchos::RCP<Albany::MaterialDatabase>& materialDB);

template<>
void  BCUtils<DirichletTraits>::buildEvaluatorsList (
  std::map<std::string, Teuchos::RCP<Teuchos::ParameterList> >& evaluators_to_build,
  const std::vector<std::string>& nodeSetIDs,
  const std::vector<std::string>& bcNames,
  Teuchos::RCP<Teuchos::ParameterList> params,
  Teuchos::RCP<ParamLib> paramLib,
  int numEqn);

template<>
void  BCUtils<NeumannTraits>::buildEvaluatorsList (
  std::map<std::string, Teuchos::RCP<Teuchos::ParameterList> >& evaluators_to_build,
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
  const Teuchos::RCP<Albany::MaterialDatabase>& materialDB);

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
