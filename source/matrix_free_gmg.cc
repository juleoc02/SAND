//
// Created by justin on 2/17/21.
//
#include "../include/matrix_free_gmg.h"
#include "../include/input_information.h"
#include "../include/parameters_and_components.h"
namespace SAND {
    using namespace dealii;


    template <int dim, int fe_degree, typename number>
    ElasticityOperator<dim, fe_degree, number>::ElasticityOperator(DoFHandler<dim> &big_dof_handler_in, BlockVector<double> &state_in)
      : MatrixFreeOperators::Base<dim,
                                  LinearAlgebra::distributed::Vector<number>>(),
        big_dof_handler(big_dof_handler_in),
        displacement_dof_handler(big_dof_handler.get_triangulation()),
        fe_displacement(FE_Q<dim>(1) ^ dim),
        state(state_in)
    {
        displacement_dof_handler.distribute_dofs(fe_displacement);

    }

    template <int dim, int fe_degree, typename number>
    void ElasticityOperator<dim, fe_degree, number>::clear()
    {
      MatrixFreeOperators::Base<dim, LinearAlgebra::distributed::Vector<number>>::clear();
    }

    template <int dim, int fe_degree, typename number>
    double ElasticityOperator<dim, fe_degree, number>::coefficient_rho_p_lambda(const  auto &cell_iter)
    {
        std::vector<unsigned int> i(cell_iter->get_fe().n_dofs_per_cell());
        cell_iter->get_dof_indices(i);
        return std::pow(state.block(SolutionBlocks::density)[i[cell_iter->get_fe().component_to_system_index(0, 0)]],Input::density_penalty_exponent)*Input::material_lambda;
    }

    template <int dim, int fe_degree, typename number>
    double ElasticityOperator<dim, fe_degree, number>::coefficient_2_rho_p_mu(const auto &cell_iter)
    {
        std::vector<unsigned int> i(cell_iter->get_fe().n_dofs_per_cell());
        cell_iter->get_dof_indices(i);
        return std::pow(state.block(SolutionBlocks::density)[i[cell_iter->get_fe().component_to_system_index(0, 0)]],Input::density_penalty_exponent)*2*Input::material_mu;

    }


    template <int dim, int fe_degree, typename number>
    void ElasticityOperator<dim, fe_degree, number>::local_apply(
      const MatrixFree<dim, number> &                   data,
      LinearAlgebra::distributed::Vector<number> &      dst,
      const LinearAlgebra::distributed::Vector<number> &src,
      const std::pair<unsigned int, unsigned int> &     cell_range)
    {
      FEEvaluation<dim, 1, 2, dim, double> phi(data);
      auto big_dof_iter = big_dof_handler.begin_active();
      auto displacement_dof_iter = displacement_dof_handler.begin_active();
      for (const auto &cell_iter: triangulation.active_cell_iterators())
      {
          big_dof_iter.copy_from(cell_iter);
          displacement_dof_iter.copy_from(cell_iter);
          phi.reinit(displacement_dof_iter);
          phi.read_dof_values(src);

          phi.evaluate(EvaluationFlags::gradients);

          for (unsigned int q = 0; q < phi.n_q_points; ++q)
          {

            phi.submit_divergence(coefficient_rho_p_lambda(big_dof_iter) * phi.get_divergence(q), q);
            phi.submit_symmetric_gradient(coefficient_2_rho_p_mu(big_dof_iter) * phi.get_symmetric_gradient(q), q);

          }

          phi.integrate(EvaluationFlags::gradients);
          phi.distribute_local_to_global(dst);
      }
    }
}
template class SAND::ElasticityOperator<2,1,double>;
template class SAND::ElasticityOperator<3,1,double>;
