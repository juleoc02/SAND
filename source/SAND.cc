#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/function.h>
#include <deal.II/base/tensor.h>

#include <deal.II/lac/block_vector.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/block_sparse_matrix.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/precondition.h>
#include <deal.II/lac/linear_operator.h>
#include <deal.II/lac/packaged_operation.h>

#include <deal.II/grid/tria.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_refinement.h>
#include <deal.II/grid/tria_accessor.h>
#include <deal.II/grid/tria_iterator.h>

#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_accessor.h>
#include <deal.II/dofs/dof_renumbering.h>
#include <deal.II/dofs/dof_tools.h>

#include <deal.II/fe/fe_values.h>

#include <deal.II/numerics/vector_tools.h>
#include <deal.II/numerics/matrix_tools.h>
#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/error_estimator.h>

#include <deal.II/fe/fe_system.h>

#include <deal.II/fe/fe_q.h>

#include <fstream>
#include <iostream>

namespace SAND
{
  using namespace dealii;

  template <int dim>
    class SANDTopOpt
    {
      public:
        SANDTopOpt ();
        void
        run ();
      private:
        void
        setup_block_system ();
        void
        create_triangulation ();
        BlockSparsityPattern sparsity_pattern;
        BlockSparseMatrix<double> system_matrix;
        BlockVector<double> solution;
        BlockVector<double> system_rhs;
        Triangulation<dim> triangulation;
        DoFHandler<dim> dof_handler;
        FESystem<dim> fe;

    };

  template <int dim>
    SANDTopOpt<dim>::SANDTopOpt ()
  : dof_handler(triangulation)
  , fe(FE_Q<dim>(1), dim)
    {

    }

  template <int dim>
    void
    SANDTopOpt<dim>::run ()
    {
      create_triangulation();
      setup_block_system();
      //assemble bilinear form matrix - put in block
      //assemble hessian of lagrangian
      //calculate gradient for block RHS
      //calculate hessian
      //
    }

  template <int dim>
    void
    SANDTopOpt<dim>::create_triangulation ()
    {
      /*Make a square*/
      Triangulation<dim> triangulation_temp;
      Point<dim> point_1, point_2;
      point_1 (0) = 0;
      point_1 (1) = 0;
      point_2 (0) = 1;
      point_2 (1) = 1;
      GridGenerator::hyper_rectangle (triangulation, point_1, point_2);

      /*make 5 more squares*/
      for (int n = 1; n < 6; n++)
        {
          triangulation_temp.clear ();
          point_1 (0) = n;
          point_2 (0) = n + 1;
          GridGenerator::hyper_rectangle (triangulation_temp, point_1, point_2);
          /*glue squares together*/
          GridGenerator::merge_triangulations (triangulation_temp,
              triangulation, triangulation);
        }
      triangulation.refine_global (3);
    }

  template <int dim>
    void
    SANDTopOpt<dim>::setup_block_system ()
    {
      dof_handler.distribute_dofs (fe);
      const unsigned int n_u = dof_handler.n_dofs(), n_p = triangulation.n_active_cells();



      /*Setup 4-by-4 block matrix. top 2-by-2 is the hessian of the lagrangian system (with log barriers for component-wise inequality constraints)*/

      BlockDynamicSparsityPattern dsp (4, 4);
      dsp.block (0, 0).reinit (n_p, n_p);
      dsp.block (1, 0).reinit (n_u, n_p);
      dsp.block (2, 0).reinit (n_u, n_p);
      dsp.block (3, 0).reinit (1, n_p);
      dsp.block (0, 1).reinit (n_p, n_u);
      dsp.block (1, 1).reinit (n_u, n_u);
      dsp.block (2, 1).reinit (n_u, n_u);
      dsp.block (3, 1).reinit (1, n_u);
      dsp.block (0, 2).reinit (n_p, n_u);
      dsp.block (1, 2).reinit (n_u, n_u);
      dsp.block (2, 2).reinit (n_u, n_u);
      dsp.block (3, 2).reinit (1, n_u);
      dsp.block (0, 3).reinit (n_p, 1);
      dsp.block (1, 3).reinit (n_u, 1);
      dsp.block (2, 3).reinit (n_u, 1);
      dsp.block (3, 3).reinit (1, 1);

      /*hessian of lagrangian wrt density (block 0,0) only has entries on the diagonal*/
      for (unsigned int i=0; i<n_p; i++)
        {
          dsp.block(0, 0).add(i,i);
        }

      /*hessian of first density then displacement is a bit more complicated...*/


      /*Create sparsity pattern for elasticity constraint - DoFTools does this for me */
      DoFTools::make_sparsity_pattern (dof_handler, dsp.block (2,1));
      DoFTools::make_sparsity_pattern (dof_handler, dsp.block (1,2));

      /* */



      for (unsigned int i=0; i<4; i++)
        {
          for(unsigned int j=0; j<4; j++)
            {
              SparsityPattern sparsity;
              sparsity.copy_from(dsp.block(i,j));
              std::ofstream out("sparsity_pattern_" + std::to_string(i) + "_" + std::to_string(j) + ".svg");
              sparsity.print_svg(out);
            }
        }


      system_matrix.reinit (sparsity_pattern);

      solution.reinit (4);
      solution.block (0).reinit (n_p);
      solution.block (1).reinit (n_u);
      solution.block (2).reinit (n_u);
      solution.block (3).reinit (1);
      solution.collect_sizes ();



      system_rhs.reinit (4);
      system_rhs.block (0).reinit (n_p);
      system_rhs.block (1).reinit (n_u);
      system_rhs.block (2).reinit (n_u);
      system_rhs.block (3).reinit (1);
      system_rhs.collect_sizes ();
    }

} // namespace SAND

int
main ()
{
  try
    {
      SAND::SANDTopOpt<2> elastic_problem_2d;
      elastic_problem_2d.run ();
    }
  catch (std::exception &exc)
    {
      std::cerr << std::endl << std::endl
                << "----------------------------------------------------"
                << std::endl;
      std::cerr << "Exception on processing: " << std::endl << exc.what ()
                << std::endl << "Aborting!" << std::endl
                << "----------------------------------------------------"
                << std::endl;

      return 1;
    }
  catch (...)
    {
      std::cerr << std::endl << std::endl
                << "----------------------------------------------------"
                << std::endl;
      std::cerr << "Unknown exception!" << std::endl << "Aborting!" << std::endl
                << "----------------------------------------------------"
                << std::endl;
      return 1;
    }

  return 0;
}
