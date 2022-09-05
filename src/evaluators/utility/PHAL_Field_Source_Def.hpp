//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#include "PHAL_SharedParameter.hpp"
#include "Albany_Utils.hpp"
#include "Albany_StringUtils.hpp"

#include "Sacado_ParameterAccessor.hpp"
#include "Sacado_ParameterRegistration.hpp"
#include "Teuchos_VerboseObject.hpp"
#include "Teuchos_Array.hpp"
#include "Teuchos_TestForException.hpp"

#include "Phalanx_DataLayout_MDALayout.hpp"

#include <cmath>
#include <sstream>
#include <iostream>
#include <fstream>
#include <algorithm>

namespace PHAL
{

    namespace Source_Functions
    {

        enum class ParamEnum
        {
            Amplitude = 0,
            Radius = 1
        };

        template <typename EvalT, typename Traits>
        inline bool Gaussian<EvalT, Traits>::check_for_existance(Teuchos::ParameterList &source_list)
        {
            std::string g("Gaussian");
            bool exists = source_list.getEntryPtr("Type");
            if (exists)
                exists = g == source_list.get("Type", g);
            return exists;
        }

        template <typename EvalT, typename Traits>
        typename Gaussian<EvalT, Traits>::ScalarT &
        Gaussian<EvalT, Traits>::getValue(const std::string &n)
        {
            if (n == util::strint("Amplitude", m_num))
                return m_amplitude;
            else if (n == util::strint("Radius", m_num))
                return m_radius;
            TEUCHOS_TEST_FOR_EXCEPTION(true, Teuchos::Exceptions::InvalidParameter,
                                       std::endl
                                           << "Error! Logic error in getting parameter " << n
                                           << " in Gaussian::getValue()" << std::endl);
        }

        template <typename EvalT, typename Traits>
        inline Gaussian<EvalT, Traits>::Gaussian(Teuchos::ParameterList &source_list,
                                                 Teuchos::ParameterList &scalarParam_list,
                                                 std::size_t num,
                                                 PHX::FieldManager<Traits> &fm,
                                                 const Teuchos::RCP<Albany::Layouts> &dl) : mdf_amplitude(source_list.get<std::string>(util::strint("Gaussian: Amplitude", num)), dl->shared_param),
                                                                                            mdf_radius(source_list.get<std::string>(util::strint("Gaussian: Radius", num)), dl->shared_param)
        {
            Teuchos::ParameterList &paramList = source_list.sublist("Spatial", true);
            m_amplitude = paramList.get("Amplitude", 1.0);
            m_radius = paramList.get("Radius", 1.0);
            // sigma = 1.0/(sqrt(2.0)*m_radius);
            m_centroid = source_list.get(util::strint("Center", num), m_centroid);
            m_num = num;

            Teuchos::RCP<ParamLib> paramLib =
                source_list.get<Teuchos::RCP<ParamLib>>("Parameter Library", Teuchos::null);

            Teuchos::RCP<Albany::ScalarParameterAccessors<EvalT>> accessors =
                source_list.get<Teuchos::RCP<Albany::ScalarParameterAccessors<EvalT>>>("Accessors");
            { //Shared parameter for sensitivity analysis: amplitude
                const std::string param_name = source_list.get<std::string>(util::strint("Gaussian: Amplitude", num));
                // Check if param_name has already been registered in fm or not:
                bool already_registered = false;
                const auto dag = fm.template getDagManager<EvalT>();
                const auto dag_nodes = dag.getDagNodes();
                const std::string shared_param_name = "Shared Parameter " + param_name + PHX::print<EvalT>();
                for (std::size_t n = 0; n < dag_nodes.size(); ++n)
                {
                    if (shared_param_name.compare(dag_nodes[n].get()->getName()) == 0)
                    {
                        already_registered = true;
                        break;
                    }
                }
                if (!already_registered)
                {
                    Teuchos::RCP<Teuchos::ParameterList> p = Teuchos::rcp(new Teuchos::ParameterList(util::strint("Gaussian: Amplitude", num)));
                    p->set<Teuchos::RCP<ParamLib>>("Parameter Library", paramLib);
                    p->set<std::string>("Parameter Name", param_name);
                    p->set<Teuchos::RCP<Albany::ScalarParameterAccessors<EvalT>>>("Accessors", accessors);
                    p->set<const Teuchos::ParameterList*>("Parameters List", &scalarParam_list);
                    p->set<double>("Default Nominal Value", paramList.get("Amplitude", 1.0));

                    Teuchos::RCP<PHAL::SharedParameter<EvalT, Traits>> ptr_amplitude;
                    ptr_amplitude = Teuchos::rcp(new PHAL::SharedParameter<EvalT, Traits>(*p, dl));
                    fm.template registerEvaluator<EvalT>(ptr_amplitude);
                }
            }
            { //Shared parameter for sensitivity analysis: radius
                const std::string param_name = source_list.get<std::string>(util::strint("Gaussian: Radius", num));
                // Check if param_name has already been registered in fm or not:
                bool already_registered = false;
                const auto dag = fm.template getDagManager<EvalT>();
                const auto dag_nodes = dag.getDagNodes();
                const std::string shared_param_name = "Shared Parameter " + param_name + PHX::print<EvalT>();
                for (std::size_t n = 0; n < dag_nodes.size(); ++n)
                {
                    if (shared_param_name.compare(dag_nodes[n].get()->getName()) == 0)
                    {
                        already_registered = true;
                        break;
                    }
                }
                if (!already_registered)
                {
                    Teuchos::RCP<Teuchos::ParameterList> p = Teuchos::rcp(new Teuchos::ParameterList(util::strint("Gaussian: Radius", num)));
                    p->set<Teuchos::RCP<ParamLib>>("Parameter Library", paramLib);
                    p->set<std::string>("Parameter Name", param_name);
                    p->set<Teuchos::RCP<Albany::ScalarParameterAccessors<EvalT>>>("Accessors", accessors);
                    p->set<const Teuchos::ParameterList*>("Parameters List", &scalarParam_list);
                    p->set<double>("Default Nominal Value", paramList.get("Radius", 1.0));

                    Teuchos::RCP<PHAL::SharedParameter<EvalT, Traits>> ptr_radius;
                    ptr_radius = Teuchos::rcp(new PHAL::SharedParameter<EvalT, Traits>(*p, dl));
                    fm.template registerEvaluator<EvalT>(ptr_radius);
                }
            }

            this->addNonConstDependentField(mdf_amplitude);
            this->addNonConstDependentField(mdf_radius);

            const PHX::Tag<ScalarT> fieldTag(source_list.get<std::string>(util::strint("Gaussian: Field", num)), dl->dummy);

            this->addEvaluatedField(fieldTag);

            this->setName(util::strint("Gaussian", num));
        }

        template <typename EvalT, typename Traits>
        typename EvalT::ScalarT Gaussian<EvalT, Traits>::
            evaluateFields(const std::vector<typename EvalT::MeshScalarT> &coords)
        {
            ScalarT exponent = 0;
            const double pi = 3.1415926535897932385;

            m_amplitude = mdf_amplitude(0);
            m_radius = mdf_radius(0);

            ScalarT sigma_pi = 1.0 / (m_radius * std::sqrt(2 * pi));
            const std::size_t nsd = coords.size();
            for (std::size_t i = 0; i < nsd; ++i)
            {
                exponent += std::pow(m_centroid[i] - coords[i], 2);
            }
            exponent /= (2.0 * std::pow(m_radius, 2));
            ScalarT x(0.0);
            if (nsd == 1)
                x = m_amplitude * sigma_pi * std::exp(-exponent);
            else if (nsd == 2)
                x = m_amplitude * std::pow(sigma_pi, 2) * std::exp(-exponent);
            else if (nsd == 3)
                x = m_amplitude * std::pow(sigma_pi, 3) * std::exp(-exponent);
            return x;
        }

    } // namespace Source_Functions
} // namespace PHAL
