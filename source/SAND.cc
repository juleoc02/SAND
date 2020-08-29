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
#include <deal.II/lac/sparse_direct.h>
#include <deal.II/lac/solver_gmres.h>

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
        assemble_block_system ();
        void
        create_triangulation ();
        void
        make_initial_values ();
        void
        set_bcids ();
        void
        solve();
        void
        add_barriers(double barrier_size);
        BlockSparsityPattern sparsity_pattern;
        BlockSparseMatrix<double> system_matrix;
        BlockVector<double> solution;
        BlockVector<double> system_rhs;
        Triangulation<dim> triangulation;
        DoFHandler<dim> dof_handler;
        AffineConstraints<double> hanging_node_constraints;
        FESystem<dim> fe;

        Vector<double> density, fe_rhs, lambda_1, cell_measure,
            displacement_sol;
        double density_ratio, volume_max, lambda_2;
        unsigned int density_penalty;

        std::map<types::global_dof_index, double> boundary_values;

    };

  template <int dim>
    SANDTopOpt<dim>::SANDTopOpt ()
        :
        dof_handler (triangulation),
        fe (FE_Q<dim> (1), dim)
    {

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
    SANDTopOpt<dim>::make_initial_values ()
    {
      density_ratio = .5;
      density_penalty = 3;
      lambda_2 = 0;
      dof_handler.distribute_dofs (fe);
      cell_measure.reinit (triangulation.n_active_cells ());
      density.reinit (triangulation.n_active_cells ());
      /*displacement vector initialized as 0s*/
      displacement_sol.reinit (dof_handler.n_dofs ());
      /*rhs of fe system initialized to size of number of dofs*/
      fe_rhs.reinit (dof_handler.n_dofs ());
      /*lambda vector initialized to all 0s*/
      lambda_1.reinit (dof_handler.n_dofs ());
      /*densities set to average density*/
      for (const auto &cell : dof_handler.active_cell_iterators ())
        {
          unsigned int i = cell->active_cell_index ();
          density[i] = density_ratio;
          cell_measure[i] = cell->measure ();
        }
      volume_max = cell_measure * density;
    }

  template <int dim>
    void
    SANDTopOpt<dim>::set_bcids ()
    {
      for (const auto &cell : dof_handler.active_cell_iterators ())
        {
          for (unsigned int face_number = 0;
              face_number < GeometryInfo<dim>::faces_per_cell; ++face_number)
            {
              if (cell->face (face_number)->at_boundary ())
                {
                  const auto center = cell->face (face_number)->center ();
                  if (std::fabs (center (1) - 0) < 1e-12)
                    {
                      /*Boundary ID of 2 is the 0 neumann, so no external force*/
                      cell->face (face_number)->set_boundary_id (2);

                      for (unsigned int vertex_number = 0;
                          vertex_number < GeometryInfo<dim>::vertices_per_cell;
                          ++vertex_number)
                        {
                          const auto center = cell->vertex (vertex_number);
                          /*Find bottom left corner*/
                          if (std::fabs (center (0) - 0) < 1e-12 && std::fabs (
                                                                        center (
                                                                            1)
                                                                        - 0)
                                                                    < 1e-12)
                            {

                              const unsigned int x_displacement =
                                  cell->vertex_dof_index (vertex_number, 0);
                              const unsigned int y_displacement =
                                  cell->vertex_dof_index (vertex_number, 1);
                              /*set bottom left BC*/
                              boundary_values[x_displacement] = 0;
                              boundary_values[y_displacement] = 0;
                            }
                          /*Find bottom right corner*/
                          if (std::fabs (center (0) - 6) < 1e-12 && std::fabs (
                                                                        center (
                                                                            1)
                                                                        - 0)
                                                                    < 1e-12)
                            {
                              types::global_dof_index y_displacement =
                                  cell->vertex_dof_index (vertex_number, 1);
                              /*set bottom right BC*/
                              boundary_values[y_displacement] = 0;
                            }
                        }
                    }

                  if (std::fabs (center (1) - 1) < 1e-12)
                    {
                      /*Find top middle*/
                      if ((std::fabs (center (0) - 3) < .1 + 1e-12))
                        {
                          /*downward force is boundary id of 1*/
                          cell->face (face_number)->set_boundary_id (1);
                        }
                      else
                        {
                          cell->face (face_number)->set_boundary_id (2);
                        }
                    }

                  if (std::fabs (center (0) - 0) < 1e-12)
                    {
                      cell->face (face_number)->set_boundary_id (2);
                    }
                  if (std::fabs (center (0) - 6) < 1e-12)
                    {
                      cell->face (face_number)->set_boundary_id (2);
                    }
                }
            }
        }
    }

  template <int dim>
    void
    SANDTopOpt<dim>::setup_block_system ()
    {
      const unsigned int n_u = dof_handler.n_dofs (), n_p =
          triangulation.n_active_cells ();

      const unsigned int dofs_per_cell = fe.dofs_per_cell;
      std::vector<types::global_dof_index> local_dof_indices (dofs_per_cell);

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
      for (unsigned int i = 0; i < n_p; i++)
        {
          dsp.block (0, 0).add (i, i);
        }

      /*hessian of first density then displacement is a bit more complicated...*/
      /*iterate through cells*/
      for (const auto &cell : dof_handler.active_cell_iterators ())
        {

          /* find dofs corresponding to current cell*/
          unsigned int i = cell->active_cell_index ();
          cell->get_dof_indices (local_dof_indices);
          /*add those indices to blocks of sparsity pattern*/
          for (unsigned int j = 0; j < dofs_per_cell; j++)
            {
              dsp.block (1, 0).add (local_dof_indices[j], i);
              dsp.block (0, 1).add (i, local_dof_indices[j]);
            }
        }

      /*Create sparsity pattern for elasticity constraint - DoFTools does this for me */
      DoFTools::make_sparsity_pattern (dof_handler, dsp.block (2, 1));
      DoFTools::make_sparsity_pattern (dof_handler, dsp.block (1, 2));

      /*Create sparsity pattern for volume constraint part of matrix*/
      /*it's full... is this bad?*/

      for (const auto &cell : dof_handler.active_cell_iterators ())
        {
          unsigned int i = cell->active_cell_index ();
          dsp.block (3, 0).add (0, i);
          dsp.block (0, 3).add (i, 0);
        }

      sparsity_pattern.copy_from (dsp);

      for (unsigned int i = 0; i < 4; i++)
        {
          for (unsigned int j = 0; j < 4; j++)
            {
              std::ofstream out (
                  "sparsity_pattern_" + std::to_string (i) + "_"
                  + std::to_string (j) + ".svg");
              sparsity_pattern.block (i, j).print_svg (out);
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

  template <int dim>
    void
    SANDTopOpt<dim>::assemble_block_system ()
    {
      QGauss<dim> quadrature_formula (fe.degree + 1);
      QGauss<dim - 1> face_quadrature_formula (fe.degree + 1);
      FEValues<dim> fe_values (fe, quadrature_formula,
          update_values | update_gradients | update_quadrature_points
          | update_JxW_values);
      FEFaceValues<dim> fe_face_values (fe, face_quadrature_formula,
          update_values | update_quadrature_points | update_normal_vectors
          | update_JxW_values);

      const unsigned int dofs_per_cell = fe.dofs_per_cell;
      const unsigned int n_q_points = quadrature_formula.size ();
      const unsigned int n_face_q_points = face_quadrature_formula.size ();

      FullMatrix<double> cell_matrix (dofs_per_cell, dofs_per_cell);
      Vector<double> cell_rhs (dofs_per_cell);

      std::vector<types::global_dof_index> local_dof_indices (dofs_per_cell);

      std::vector<double> lambda_values (n_q_points);
      std::vector<double> mu_values (n_q_points);

      Functions::ConstantFunction<dim> lambda (1.), mu (1.);
      std::vector<Tensor<1, dim>> rhs_values (n_q_points);
      double penalized_density, grad_value, laplace_density,
          laplace_density_displacement;
      const FEValuesExtractors::Vector displacements (0);
      const FEValuesExtractors::Scalar just_x (0);
      const FEValuesExtractors::Scalar just_y (1);
      ComponentMask just_x_mask = fe.component_mask (just_x);
      ComponentMask just_y_mask = fe.component_mask (just_y);

      for (const auto &cell : dof_handler.active_cell_iterators ())
        {
          cell_matrix = 0;
          cell_rhs = 0;

          fe_values.reinit (cell);

          lambda.value_list (fe_values.get_quadrature_points (), lambda_values);
          mu.value_list (fe_values.get_quadrature_points (), mu_values);

          for (unsigned int q_point = 0; q_point < n_q_points; ++q_point)
            for (unsigned int i = 0; i < dofs_per_cell; ++i)
              {
                const SymmetricTensor<2, dim> phi_i_symmgrad =
                    fe_values[displacements].symmetric_gradient (i, q_point);
                const double phi_i_div = fe_values[displacements].divergence (i,
                    q_point);

                for (unsigned int j = 0; j < dofs_per_cell; ++j)
                  {
                    const SymmetricTensor<2, dim> phi_j_symmgrad =
                        fe_values[displacements].symmetric_gradient (j,
                            q_point);
                    const double phi_j_div =
                        fe_values[displacements].divergence (j, q_point);

                    cell_matrix (i, j) +=
                        (phi_i_div * phi_j_div * lambda_values[q_point] + 2
                            * mu_values[q_point]
                            * (phi_i_symmgrad * phi_j_symmgrad))
                        * fe_values.JxW (q_point);
                  }
              }

          for (unsigned int i = 0; i < dofs_per_cell; ++i)
            {
              const unsigned int component_i =
                  fe.system_to_component_index (i).first;

              for (unsigned int q_point = 0; q_point < n_q_points; ++q_point)
                cell_rhs (i) += fe_values.shape_value (i, q_point)
                    * rhs_values[q_point][component_i]
                    * fe_values.JxW (q_point);
            }
          Tensor<1, dim> traction;
          traction.clear ();
          traction[1] = -1;
          for (unsigned int face_number = 0;
              face_number < GeometryInfo<dim>::faces_per_cell; ++face_number)
            {
              if (cell->face (face_number)->at_boundary () && cell->face (
                                                                  face_number)->boundary_id ()
                                                              == 1)
                {
                  fe_face_values.reinit (cell, face_number);

                  for (unsigned int face_q_point = 0;
                      face_q_point < n_face_q_points; ++face_q_point)
                    {
                      for (unsigned int i = 0; i < dofs_per_cell; ++i)
                        {
                          cell_rhs (i) += traction
                              * fe_face_values[displacements].value (i,
                                  face_q_point)
                              * fe_face_values.JxW (face_q_point);
                        }
                    }
                }
            }
          penalized_density = pow (density[cell->active_cell_index ()],
              density_penalty);
          cell->get_dof_indices (local_dof_indices);

          for (unsigned int i = 0; i < dofs_per_cell; ++i)
            {
              for (unsigned int j = 0; j < dofs_per_cell; ++j)
                {
                  /*assemble FE system parts for block (2,1) and (1,2)*/
                  system_matrix.block (2, 1).add (local_dof_indices[i],
                      local_dof_indices[j],
                      penalized_density * cell_matrix (i, j));
                  system_matrix.block (1, 2).add (local_dof_indices[i],
                      local_dof_indices[j],
                      penalized_density * cell_matrix (i, j));

                  /*assemble FE RHS*/
                  system_rhs.block (1) (local_dof_indices[i]) += cell_rhs (i);

                }
            }

          /*assemble grad Au for blocks (2,0) and (0,2)*/

          for (unsigned int i = 0; i < dofs_per_cell; ++i)
            {
              grad_value = 0;

              for (unsigned int j = 0; j < dofs_per_cell; ++j)
                {

                  grad_value = grad_value
                      + cell_matrix (i, j) * solution[local_dof_indices[j]];

                }
              grad_value = grad_value
                  * density_penalty
                  * pow (density[cell->active_cell_index ()],
                      density_penalty - 1);
              system_matrix.block (2, 0).add (local_dof_indices[i],
                  cell->active_cell_index (), grad_value);

            }

          /*assemble hessian pf laplace wrt density, for blocks (0,0)*/

          laplace_density = 0;
          for (unsigned int i = 0; i < dofs_per_cell; ++i)
            {
              for (unsigned int j = 0; j < dofs_per_cell; ++j)
                {
                  laplace_density = laplace_density
                      + lambda_1[local_dof_indices[i]] * cell_matrix (i, j)
                        * solution[local_dof_indices[j]];

                }
            }
          laplace_density = laplace_density
              * density_penalty * (density_penalty - 1)
              * pow (density[cell->active_cell_index ()], density_penalty - 2);
          system_matrix.block (0, 0).add (cell->active_cell_index (),
              cell->active_cell_index (), laplace_density);

          /*assemble hessian pf laplace wrt both density, displacement for blocks (1,0) and (0,1)*/

          for (unsigned int i = 0; i < dofs_per_cell; ++i)
            {
              laplace_density_displacement = 0;
              for (unsigned int j = 0; j < dofs_per_cell; ++j)
                {
                  laplace_density_displacement = laplace_density_displacement
                      + cell_matrix (i, j) * lambda_1[local_dof_indices[j]];

                }

              laplace_density_displacement = laplace_density_displacement
                  * density_penalty
                  * pow (density[cell->active_cell_index ()],
                      density_penalty - 1);

              system_matrix.block (1, 0).add (local_dof_indices[i],
                  cell->active_cell_index (), laplace_density_displacement);
              system_matrix.block (0, 1).add (cell->active_cell_index (),
                  local_dof_indices[i], laplace_density_displacement);

            }

          /*assemble volume constraint part for cell_measure*/

          system_matrix.block (3, 0).add (0, cell->active_cell_index (),
              cell_measure[cell->active_cell_index ()]);
          system_matrix.block (0, 3).add (cell->active_cell_index (), 0,
              cell_measure[cell->active_cell_index ()]);

          /*assemble block 3 of rhs, max_volume - total_volume*/

          Vector<double> delta_f;
          delta_f.reinit (dof_handler.n_dofs ());

          system_rhs.block (3)[0] = volume_max - cell_measure * density;

          system_matrix.block (2, 1).vmult (delta_f, displacement_sol);
          for (unsigned int i = 0; i < dof_handler.n_dofs (); i++)
            {
              system_rhs.block (2)[i] = system_rhs.block (1)[i] - delta_f[i];
            }

        }

      /*Currently have no hanging nodes, so not a big deal, but eventually need to make sure this actually works...*/
      hanging_node_constraints.clear ();
      DoFTools::make_hanging_node_constraints (dof_handler,
          hanging_node_constraints);
      hanging_node_constraints.close ();
      hanging_node_constraints.condense (system_matrix.block (2, 1));
      hanging_node_constraints.condense (system_matrix.block (1, 2));
      hanging_node_constraints.condense (system_rhs.block (2));
      hanging_node_constraints.condense (system_rhs.block (1));

      /*This feels weird...*/
      MatrixTools::apply_boundary_values (boundary_values,
          system_matrix.block (1, 2), solution.block (1), system_rhs.block (1));
      MatrixTools::apply_boundary_values (boundary_values,
          system_matrix.block (2, 1), solution.block (2), system_rhs.block (2));

      /*assemble laplacian by density,dof*/

      /*assemble RHS*/
    }


  /*adds log barriers for density constraints*/
  template <int dim>
       void
       SANDTopOpt<dim>::add_barriers (double barrier_size)
       {
          for (const auto &cell : dof_handler.active_cell_iterators ())
            {
              unsigned int i = cell->active_cell_index();
              system_matrix.block(0,0).add(i,i,barrier_size/(density[i]*density[i]));
              system_matrix.block(0,0).add(i,i,barrier_size/((1-density[i])*(1-density[i])));
              system_rhs.block(0)[i] = system_rhs.block(0)[i] - (barrier_size/density[i]) + (barrier_size/(1-density[i]));
            }
       }

  template <int dim>
     void
     SANDTopOpt<dim>::solve ()
     {
            /*
            SparseDirectUMFPACK A_direct;
            A_direct.initialize(system_matrix);
            A_direct.vmult(solution, system_rhs);

            hanging_node_constraints.distribute(solution);
            */
            /*
            const unsigned int max_iters       = 200;
            const double       solve_tolerance = 1e-8 * system_rhs.l2_norm();
            SolverControl      solver_control(max_iters, solve_tolerance, true, true);
            solver_control.enable_history_data();
            SolverGMRES<Vector<double>> solver(
            solver_control, SolverGMRES<Vector<double>>::AdditionalData(50, true));
            solver.solve(system_matrix, solution, system_rhs, preconditioner);
             */
     }

  template <int dim>
    void
    SANDTopOpt<dim>::run ()
    {
      double barrier_size = 1;

      create_triangulation ();
      make_initial_values ();
      setup_block_system ();
      assemble_block_system ();
      add_barriers (barrier_size);
      solve();
      //
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
