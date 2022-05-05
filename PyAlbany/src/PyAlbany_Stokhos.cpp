#include "PyAlbany_Stokhos.hpp"

using namespace PyAlbany;

KLExpansion::KLExpansion(int _ndim) : ndim(_ndim) {
#ifdef ALBANY_STOKHOS    
    domain_upper = Teuchos::Array<double>(ndim, 1.0);
    domain_lower = Teuchos::Array<double>(ndim, 0.0);
    correlation_lengths = Teuchos::Array<double>(ndim, 1.0);
    num_KL_per_dim = Teuchos::Array<int>(ndim, 1);

    eps = 1e-6;
    tol = 1e-10;
    max_it = 100;
    num_KL_terms = 1;
#endif
}

void KLExpansion::setUpperBound(int i, double _domain_upper_i) {
#ifdef ALBANY_STOKHOS 
    domain_upper[i] = _domain_upper_i;
#endif
}
void KLExpansion::setLowerBound(int i, double _domain_lower_i) {
#ifdef ALBANY_STOKHOS 
    domain_lower[i] = _domain_lower_i;
#endif
}
void KLExpansion::setCorrelationLength(int i, double _correlation_length_i) {
#ifdef ALBANY_STOKHOS 
    correlation_lengths[i] = _correlation_length_i;
#endif
}
void KLExpansion::setNumberOfKLTerm(int i, double _num_KL_terms_i) {
#ifdef ALBANY_STOKHOS 
    num_KL_per_dim[i] = _num_KL_terms_i;
#endif
}

void KLExpansion::createModes() {
#ifdef ALBANY_STOKHOS 
    Teuchos::ParameterList solverParams;
    solverParams.set("Number of KL Terms", num_KL_terms);
    solverParams.set("Mean", RST::zero());
    solverParams.set("Standard Deviation", RST::one());
    solverParams.set("Bound Perturbation Size", eps);
    solverParams.set("Nonlinear Solver Tolerance", tol);
    solverParams.set("Maximum Nonlinear Solver Iterations", max_it);

    solverParams.set("Domain Upper Bounds", domain_upper);
    solverParams.set("Domain Lower Bounds", domain_lower);
    solverParams.set("Correlation Lengths", correlation_lengths);
    solverParams.set("Number of KL Terms per dimension", num_KL_per_dim);

    randomField = RandomFieldType(solverParams);
#endif
}

void KLExpansion::getModes(double* phi, int n_nodes, int n_modes, double* x, int n_nodes_x) {
#ifdef ALBANY_STOKHOS 
    Kokkos::View<double *, Kokkos::LayoutLeft, PyTrilinosVector::node_type::device_type> weights("w", n_modes);
    for (std::size_t i_mode = 0; i_mode<n_modes; ++ i_mode) {
        weights(i_mode) = 0;
    }

    for (std::size_t i_node = 0; i_node<n_nodes; ++ i_node) {
        const double point[1] = {x[i_node]};
        for (std::size_t i_mode = 0; i_mode<n_modes; ++ i_mode) {
            weights(i_mode) = 1.;
            phi[i_node * n_modes + i_mode] = randomField.evaluate(point, weights);
            weights(i_mode) = 0.;
        }
    }
#endif
}

void KLExpansion::getModes(double* phi, int n_nodes, int n_modes, double* x, int n_nodes_x, double* y, int n_nodes_y) {
#ifdef ALBANY_STOKHOS 
    Kokkos::View<double *, Kokkos::LayoutLeft, PyTrilinosVector::node_type::device_type> weights("w", n_modes);
    for (std::size_t i_mode = 0; i_mode<n_modes; ++ i_mode) {
        weights(i_mode) = 0;
    }

    for (std::size_t i_node = 0; i_node<n_nodes; ++ i_node) {
        const double point[2] = {x[i_node], y[i_node]};
        for (std::size_t i_mode = 0; i_mode<n_modes; ++ i_mode) {
            weights(i_mode) = 1.;
            phi[i_node * n_modes + i_mode] = randomField.evaluate(point, weights);
            weights(i_mode) = 0.;
        }
    }
#endif
}

void KLExpansion::getModes(double* phi, int n_nodes, int n_modes, double* x, int n_nodes_x, double* y, int n_nodes_y, double* z, int n_nodes_z) {
#ifdef ALBANY_STOKHOS 
    Kokkos::View<double *, Kokkos::LayoutLeft, PyTrilinosVector::node_type::device_type> weights("w", n_modes);
    for (std::size_t i_mode = 0; i_mode<n_modes; ++ i_mode) {
        weights(i_mode) = 0;
    }

    for (std::size_t i_node = 0; i_node<n_nodes; ++ i_node) {
        const double point[3] = {x[i_node], y[i_node], z[i_node]};
        for (std::size_t i_mode = 0; i_mode<n_modes; ++ i_mode) {
            weights(i_mode) = 1.;
            phi[i_node * n_modes + i_mode] = randomField.evaluate(point, weights);
            weights(i_mode) = 0.;
        }
    }
#endif
}

void KLExpansion::getEigenValues(double* eigenvalues, int n_modes) {
#ifdef ALBANY_STOKHOS 
    for (std::size_t i_mode = 0; i_mode<n_modes; ++ i_mode) {
        eigenvalues[i_mode] = randomField.eigenvalue(i_mode);
    }
#endif
}
