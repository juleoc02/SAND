#ifndef MATRIX_FREE_GMG_H
#define MATRIX_FREE_GMG_H

#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/function.h>
#include <deal.II/base/timer.h>
#include <deal.II/lac/affine_constraints.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/la_parallel_vector.h>
#include <deal.II/lac/precondition.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_system.h>
#include <deal.II/grid/tria.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/multigrid/multigrid.h>
#include <deal.II/multigrid/mg_transfer_matrix_free.h>
#include <deal.II/multigrid/mg_tools.h>
#include <deal.II/multigrid/mg_coarse.h>
#include <deal.II/multigrid/mg_smoother.h>
#include <deal.II/multigrid/mg_matrix.h>
#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/vector_tools.h>
#include <deal.II/matrix_free/matrix_free.h>
#include <deal.II/matrix_free/operators.h>
#include <deal.II/matrix_free/fe_evaluation.h>



namespace SAND
{
    using namespace dealii;

    template <int dim, typename number>
    struct OperatorCellData
    {
       Table<2, VectorizedArray<number>> viscosity;

       std::size_t
       memory_consumption() const;
    };

    template <int dim, int fe_degree, typename number>
    class ElasticityOperator
      : public MatrixFreeOperators::
          Base<dim, LinearAlgebra::distributed::Vector<number>>
    {
    public:

      using value_type = number;

      ElasticityOperator(DoFHandler<dim> &big_dof_handler_in, BlockVector<double> &state_in);

      void clear() override;

    private:

      double coefficient_rho_p_lambda(const auto &cell_iter, unsigned int q);
      double coefficient_2_rho_p_mu(const auto &cell_iter, unsigned int q);
      const BlockVector<double> &state;


      const DoFHandler<dim> &big_dof_handler;
      DoFHandler<dim> displacement_dof_handler;
      FESystem<dim> fe_displacement;
//      virtual void apply_add(
//        LinearAlgebra::distributed::Vector<number> &      dst,
//        const LinearAlgebra::distributed::Vector<number> &src) const override;

      void
      local_apply(const MatrixFree<dim, number> &                   data,
                  LinearAlgebra::distributed::Vector<number> &      dst,
                  const LinearAlgebra::distributed::Vector<number> &src,
                  const std::pair<unsigned int, unsigned int> &cell_range);

    };

}




#endif // MATRIX_FREE_GMG_H
