//
// Created by justin on 2/17/21.
//
#include "../include/matrix_free_gmg.h"
namespace SAND {
    using namespace dealii;


    template <int dim, int fe_degree, typename number>
    ElasticityOperator<dim, fe_degree, number>::ElasticityOperator()
      : MatrixFreeOperators::Base<dim,
                                  LinearAlgebra::distributed::Vector<number>>()
    {}

    template <int dim, int fe_degree, typename number>
    void ElasticityOperator<dim, fe_degree, number>::clear()
    {
      MatrixFreeOperators::Base<dim, LinearAlgebra::distributed::Vector<number>>::
        clear();
    }

    template <int dim, int fe_degree, typename number>
    void ElasticityOperator<dim, fe_degree, number>::local_apply(
      const MatrixFree<dim, number> &                   data,
      LinearAlgebra::distributed::Vector<number> &      dst,
      const LinearAlgebra::distributed::Vector<number> &src,
      const std::pair<unsigned int, unsigned int> &     cell_range) const
    {
      FEEvaluation<dim, fe_degree, fe_degree + 1, 1, number> phi(data);
      for (unsigned int cell = cell_range.first; cell < cell_range.second; ++cell)
        {
          phi.reinit(cell);
          phi.read_dof_values(src);

          phi.evaluate(EvaluationFlags::gradients);

          for (unsigned int q = 0; q < phi.n_q_points; ++q)
          {

            phi.submit_divergence(coefficient_rho_p_lambda(cell, q) * phi.get_gradient(q), q);
            phi.submit_symmetric_gradient(coefficient_2_rho_p_mu(cell, q) * phi.get_gradient(q), q);

          }
          phi.integrate(EvaluationFlags::gradients);
          phi.distribute_local_to_global(dst);
        }
    }

}
template class SAND::ElasticityOperator<2>;
template class SAND::ElasticityOperator<3>;
