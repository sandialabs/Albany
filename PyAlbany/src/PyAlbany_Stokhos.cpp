#include "PyAlbany_Stokhos.hpp"

using namespace PyAlbany;

KLExpansion::KLExpansion(int _ndim) : ndim(_ndim) {
    domain_upper = Teuchos::Array<double>(ndim, 1.0);
    domain_lower = Teuchos::Array<double>(ndim, 0.0);
    correlation_lengths = Teuchos::Array<double>(ndim, 1.0);
    num_KL_per_dim = Teuchos::Array<int>(ndim, 1);

    eps = 1e-6;
    tol = 1e-10;
    max_it = 100;
    num_KL_terms = 1;
}

void KLExpansion::setUpperBound(int i, double _domain_upper_i) {
    domain_upper[i] = _domain_upper_i;
}
void KLExpansion::setLowerBound(int i, double _domain_lower_i) {
    domain_lower[i] = _domain_lower_i;
}
void KLExpansion::setCorrelationLength(int i, double _correlation_length_i) {
    correlation_lengths[i] = _correlation_length_i;
}
void KLExpansion::setNumberOfKLTerm(int i, double _num_KL_terms_i) {
    num_KL_per_dim[i] = _num_KL_terms_i;
}

void KLExpansion::createModes() {
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
}

void KLExpansion::getModes(double* phi, int n_nodes, int n_modes, double* x, int n_nodes_x) {
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
}

void KLExpansion::getModes(double* phi, int n_nodes, int n_modes, double* x, int n_nodes_x, double* y, int n_nodes_y) {
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
}

void KLExpansion::getModes(double* phi, int n_nodes, int n_modes, double* x, int n_nodes_x, double* y, int n_nodes_y, double* z, int n_nodes_z) {
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
}

void KLExpansion::getEigenValues(double* eigenvalues, int n_modes) {
    for (std::size_t i_mode = 0; i_mode<n_modes; ++ i_mode) {
        eigenvalues[i_mode] = randomField.eigenvalue(i_mode);
    }
}
