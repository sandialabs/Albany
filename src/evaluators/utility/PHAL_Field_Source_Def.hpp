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

            this->addDependentField(mdf_amplitude);
            this->addDependentField(mdf_radius);

            const PHX::Tag<ScalarT> fieldTag(source_list.get<std::string>(util::strint("Gaussian: Field", num)), dl->dummy);

            this->addEvaluatedField(fieldTag);

            this->setName(util::strint("Gaussian", num));
        }

        template <typename EvalT, typename Traits>
        void Gaussian<EvalT, Traits>::
            addToSource(const PHX::MDField<const MeshScalarT, Cell, Point, Dim> &coords,
                        const PHX::MDField<ScalarT, Cell, Point> &source,
                        const int numCells, const RealType scaling)
        {
            const double sqrt_2pi = std::sqrt(2 * 3.1415926535897932385);

            const int nsd = coords.extent_int(2);
            TEUCHOS_TEST_FOR_EXCEPTION(nsd < 1 || nsd > 3, std::logic_error,
                                       "Error! Gaussian source only supports 1, 2 or 3 spatial dimensions.\n");
            TEUCHOS_TEST_FOR_EXCEPTION(static_cast<int>(m_centroid.size()) < nsd, std::logic_error,
                                       "Error! Gaussian source center has fewer than " << nsd << " components.\n");

            double c[3] = {0.0, 0.0, 0.0};
            for (int i = 0; i < nsd; ++i)
                c[i] = m_centroid[i];
            const double c0 = c[0], c1 = c[1], c2 = c[2];

            const auto x = coords.get_static_view();
            const auto f = source.get_static_view();
            // The amplitude/radius parameters live in device memory: read them in the kernel
            const auto amplitude = mdf_amplitude.get_view();
            const auto radius = mdf_radius.get_view();
            const int numQPs = x.extent_int(1);
            // Compute the Gaussian coefficients once per call, on device (where the
            // amplitude/radius parameters live): coeffs(0)=factor, coeffs(1)=2*radius^2.
            // NOTE: doing this per thread in the main kernel raises register pressure on GPUs.
            if (m_coeffs.size() == 0)
                m_coeffs = Sacado::createDynRankView(source.get_view(), "Gaussian coefficients", 2);
            const auto coeffs = m_coeffs;
            Kokkos::parallel_for("Gaussian Source: coefficients",
                                 Kokkos::RangePolicy<PHX::Device::execution_space>(0, 1),
                                 KOKKOS_LAMBDA(const int) {
                const ScalarT amp = amplitude(0);
                const ScalarT rad = radius(0);
                const ScalarT sigma_pi = 1.0 / (rad * sqrt_2pi);
                coeffs(1) = 2.0 * std::pow(rad, 2);
                if (nsd == 1)
                    coeffs(0) = scaling * amp * sigma_pi;
                else if (nsd == 2)
                    coeffs(0) = scaling * amp * std::pow(sigma_pi, 2);
                else
                    coeffs(0) = scaling * amp * std::pow(sigma_pi, 3);
            });

            Kokkos::parallel_for("Gaussian Source",
                                 Kokkos::MDRangePolicy<PHX::Device::execution_space, Kokkos::Rank<2>>({0, 0}, {numCells, numQPs}),
                                 KOKKOS_LAMBDA(const int cell, const int qp) {
                // NOTE: no runtime-indexed local array (it may end up in scratch memory on GPUs)
                auto cc = [&](const int i) -> double { return i == 0 ? c0 : (i == 1 ? c1 : c2); };
                MeshScalarT dist2 = 0;
                for (int i = 0; i < nsd; ++i)
                    dist2 += (cc(i) - x(cell, qp, i)) * (cc(i) - x(cell, qp, i));
                f(cell, qp) += coeffs(0) * std::exp(-dist2 / coeffs(1));
            });
        }

    } // namespace Source_Functions
} // namespace PHAL
