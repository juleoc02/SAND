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
#include <deal.II/lac/affine_constraints.h>

#include <deal.II/grid/tria.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_refinement.h>

#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_renumbering.h>
#include <deal.II/dofs/dof_tools.h>

#include <deal.II/fe/fe_values.h>
#include <deal.II/fe/fe_dgq.h>
#include <deal.II/fe/fe_system.h>
#include <deal.II/fe/fe_q.h>

#include <deal.II/numerics/vector_tools.h>
#include <deal.II/numerics/matrix_tools.h>
#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/error_estimator.h>

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
        assemble_block_system (double barrier_size);
        void
        create_triangulation ();
        void
        set_bcids ();
        void
        solve ();
        void
        update_step ();
        void
        output (int j);


        BlockSparsityPattern sparsity_pattern;
        BlockSparseMatrix<double> system_matrix;
        BlockVector<double> linear_solution;
        BlockVector<double> system_rhs;
        BlockVector<double> nonlinear_solution;
        Triangulation<dim> triangulation;
        DoFHandler<dim> dof_handler;
        AffineConstraints<double> constraints;
        FESystem<dim> fe;
        double density_ratio;
        int density_penalty_exponent;

        std::map<types::global_dof_index, double> boundary_values;

    };

  template <int dim>
    SANDTopOpt<dim>::SANDTopOpt ()
        :
        dof_handler (triangulation),
        /*fe should have 1 FE_DGQ<dim>(0) element for density, dim FE_Q finite elements for displacement, another dim FE_Q elements for the lagrange multiplier on the FE constraint, and 2 more FE_DGQ<dim>(0) elements for the upper and lower bound constraints */
        fe (FE_DGQ < dim > (0) ^ 1,
            (FESystem < dim > (FE_Q < dim > (1) ^ dim)) ^ 2,
            FE_DGQ < dim > (0) ^ 4)
    {

    }

  template <int dim>
    void
    SANDTopOpt<dim>::create_triangulation ()
    {
      /*Make a square*/
      Triangulation < dim > triangulation_temp;
      Point<dim> point_1, point_2;
      point_1 (0) = 0;
      point_1 (1) = 0;
      point_2 (0) = 1;
      point_2 (1) = 1;
      GridGenerator::hyper_rectangle (triangulation, point_1, point_2);

      /*make 5 more squares*/
      for (int n = 1; n < 2; n++)
        {
          triangulation_temp.clear ();
          point_1 (0) = n;
          point_2 (0) = n + 1;
          GridGenerator::hyper_rectangle (triangulation_temp, point_1, point_2);
          /*glue squares together*/
          GridGenerator::merge_triangulations (triangulation_temp,
              triangulation, triangulation);
        }
      triangulation.refine_global (0);

      /*Set BCIDs   */
      for (const auto &cell : triangulation.active_cell_iterators ())
        {
          for (unsigned int face_number = 0;
              face_number < GeometryInfo < dim > ::faces_per_cell;
              ++face_number)
            {
              if (cell->face (face_number)->at_boundary ())
                {
                  const auto center = cell->face (face_number)->center ();
                  if (std::fabs (center (1) - 0) < 1e-12)
                    {
                      /*Boundary ID of 2 is the 0 neumann, so no external force*/
                      cell->face (face_number)->set_boundary_id (2);
                    }
                  if (std::fabs (center (1) - 1) < 1e-12)
                    {
                      /*Find top middle*/
                      if ((std::fabs (center (0) - 1) < 1))
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
                  if (std::fabs (center (0) - 2) < 1e-12)
                    {
                      cell->face (face_number)->set_boundary_id (2);
                    }
                }
            }
        }

      dof_handler.distribute_dofs (fe);

      DoFRenumbering::component_wise (dof_handler);

    }



  template <int dim>
    void
    SANDTopOpt<dim>::set_bcids ()
    {
      for (const auto &cell : dof_handler.active_cell_iterators ())
        {
          for (unsigned int face_number = 0;
              face_number < GeometryInfo < dim > ::faces_per_cell;
              ++face_number)
            {
              if (cell->face (face_number)->at_boundary ())
                {
                  const auto center = cell->face (face_number)->center ();
                  if (std::fabs (center (1) - 0) < 1e-12)
                    {

                      for (unsigned int vertex_number = 0;
                          vertex_number < GeometryInfo < dim > ::vertices_per_cell;
                          ++vertex_number)
                        {
                          const auto vert = cell->vertex (vertex_number);
                          /*Find bottom left corner*/
                          if (std::fabs (vert (0) - 0) < 1e-12 && std::fabs (
                                                vert (1)- 0) < 1e-12)
                            {

                              const unsigned int x_displacement =
                                  cell->vertex_dof_index (vertex_number, 0);
                              const unsigned int y_displacement =
                                  cell->vertex_dof_index (vertex_number, 1);
                              const unsigned int x_displacement_multiplier =
                                  cell->vertex_dof_index (vertex_number, 2);
                              const unsigned int y_displacement_multiplier =
                                  cell->vertex_dof_index (vertex_number, 3);
                              /*set bottom left BC*/
                              boundary_values[x_displacement] = 0;
                              boundary_values[y_displacement] = 0;
                              boundary_values[x_displacement_multiplier] = 0;
                              boundary_values[y_displacement_multiplier] = 0;
                            }
                          /*Find bottom right corner*/
                          if (std::fabs (vert (0) - 2) < 1e-12 && std::fabs (
                                                                        vert (
                                                                            1)
                                                                        - 0)
                                                                    < 1e-12)
                            {
                              const unsigned int y_displacement =
                                  cell->vertex_dof_index (vertex_number, 1);
                              const unsigned int y_displacement_multiplier =
                                  cell->vertex_dof_index (vertex_number, 3);
                                const unsigned int x_displacement =
                                        cell->vertex_dof_index (vertex_number, 0);
                                const unsigned int x_displacement_multiplier =
                                        cell->vertex_dof_index (vertex_number, 2);
                              /*set bottom left BC*/
                              boundary_values[y_displacement] = 0;
                              boundary_values[y_displacement_multiplier] = 0;
                              boundary_values[x_displacement] = 0;
                              boundary_values[x_displacement_multiplier] = 0;
                            }
                        }
                    }
                }
            }
        }
    }

  template <int dim>
    void
    SANDTopOpt<dim>::setup_block_system ()
    {
      const FEValuesExtractors::Scalar densities (0);
      const FEValuesExtractors::Vector displacements (1);
      const FEValuesExtractors::Vector displacement_multipliers (1 + dim);

      //MAKE n_u and n_P*****************************************************************

      /*Setup 7 by 7 block matrix*/

      std::vector<unsigned int> block_component (7, 2);
      block_component[0] = 0;
      block_component[1] = 1;
      const std::vector<types::global_dof_index> dofs_per_block =
          DoFTools::count_dofs_per_fe_block (dof_handler, block_component);

      const unsigned int n_p = dofs_per_block[0];
      const unsigned int n_u = dofs_per_block[1];
      std::cout << "n_p:  " << n_p << "   n_u   " << n_u << std::endl;

      BlockDynamicSparsityPattern dsp (7, 7);
      //first column has size n_p - hessian of lagrangian wrt p
      dsp.block (0, 0).reinit (n_p, n_p);
      dsp.block (1, 0).reinit (n_u, n_p);
      dsp.block (2, 0).reinit (n_u, n_p);
      dsp.block (3, 0).reinit (n_p, n_p);
      dsp.block (4, 0).reinit (n_p, n_p);
      dsp.block (5, 0).reinit (n_p, n_p);
      dsp.block (6, 0).reinit (n_p, n_p);
      //second column has size n_u - hessian of lagrangian wrt u
      dsp.block (0, 1).reinit (n_p, n_u);
      dsp.block (1, 1).reinit (n_u, n_u);
      dsp.block (2, 1).reinit (n_u, n_u);
      dsp.block (3, 1).reinit (n_p, n_u);
      dsp.block (4, 1).reinit (n_p, n_u);
      dsp.block (5, 1).reinit (n_p, n_u);
      dsp.block (6, 1).reinit (n_p, n_u);
      //third column has size n_p - density constraint slack variable
      dsp.block (0, 2).reinit (n_p, n_u);
      dsp.block (1, 2).reinit (n_u, n_u);
      dsp.block (2, 2).reinit (n_u, n_u);
      dsp.block (3, 2).reinit (n_p, n_u);
      dsp.block (4, 2).reinit (n_p, n_u);
      dsp.block (5, 2).reinit (n_p, n_u);
      dsp.block (6, 2).reinit (n_p, n_u);
      //fourth column has size n_p - density constraint slack variable
      dsp.block (0, 3).reinit (n_p, n_p);
      dsp.block (1, 3).reinit (n_u, n_p);
      dsp.block (2, 3).reinit (n_u, n_p);
      dsp.block (3, 3).reinit (n_p, n_p);
      dsp.block (4, 3).reinit (n_p, n_p);
      dsp.block (5, 3).reinit (n_p, n_p);
      dsp.block (6, 3).reinit (n_p, n_p);
      //fifth column has size n_u - FE constraint
      dsp.block (0, 4).reinit (n_p, n_p);
      dsp.block (1, 4).reinit (n_u, n_p);
      dsp.block (2, 4).reinit (n_u, n_p);
      dsp.block (3, 4).reinit (n_p, n_p);
      dsp.block (4, 4).reinit (n_p, n_p);
      dsp.block (5, 4).reinit (n_p, n_p);
      dsp.block (6, 4).reinit (n_p, n_p);
      //sixth column has size n_p - slack variable
      dsp.block (0, 5).reinit (n_p, n_p);
      dsp.block (1, 5).reinit (n_u, n_p);
      dsp.block (2, 5).reinit (n_u, n_p);
      dsp.block (3, 5).reinit (n_p, n_p);
      dsp.block (4, 5).reinit (n_p, n_p);
      dsp.block (5, 5).reinit (n_p, n_p);
      dsp.block (6, 5).reinit (n_p, n_p);
      //seventh column has size n_p - slack variable

      dsp.block (0, 6).reinit (n_p, n_p);
      dsp.block (1, 6).reinit (n_u, n_p);
      dsp.block (2, 6).reinit (n_u, n_p);
      dsp.block (3, 6).reinit (n_p, n_p);
      dsp.block (4, 6).reinit (n_p, n_p);
      dsp.block (5, 6).reinit (n_p, n_p);
      dsp.block (6, 6).reinit (n_p, n_p);

      dsp.collect_sizes ();

      Table < 2, DoFTools::Coupling > coupling (2 * dim + 5, 2 * dim + 5);

      coupling[0][0] = DoFTools::always;

      for (int i = 0; i < dim; i++)
        {
          coupling[0][1 + i] = DoFTools::always;
          coupling[1 + i][0] = DoFTools::always;
        }

      for (int i = 0; i < dim; i++)
        {
          coupling[0][1 + dim + i] = DoFTools::always;
          coupling[1 + dim + i][0] = DoFTools::always;
        }

      coupling[0][1 + 2 * dim] = DoFTools::none;
      coupling[0][1 + 2 * dim + 1] = DoFTools::none;
      coupling[0][1 + 2 * dim + 2] = DoFTools::none;
      coupling[0][1 + 2 * dim + 3] = DoFTools::none;
      coupling[1 + 2 * dim][0] = DoFTools::none;
      coupling[1 + 2 * dim + 1][0] = DoFTools::none;
      coupling[1 + 2 * dim + 2][0] = DoFTools::none;
      coupling[1 + 2 * dim + 3][0] = DoFTools::none;

      for (int i = 0; i < dim; i++)
        {
          for (int k = 0; k < dim; k++)
            {
              coupling[1 + i][1 + k] = DoFTools::always;
            }
          for (int k = 0; k < dim; k++)
            {
              coupling[1 + i][1 + dim + k] = DoFTools::always;
              coupling[1 + dim + k][1 + i] = DoFTools::always;
            }
          for (int k = 0; k < 4; k++)
            {
              coupling[1 + i][1 + 2 * dim + k] = DoFTools::none;
              coupling[1 + 2 * dim + k][1 + i] = DoFTools::none;
            }
        }

      for (int i = 0; i < dim + 4; i++)
        {
          for (int k = 0; k < dim + 4; k++)
            {
              coupling[1 + dim + i][1 + dim + k] = DoFTools::none;
              coupling[1 + dim + k][1 + dim + i] = DoFTools::none;
            }
        }

      constraints.clear ();

      ComponentMask density_mask = fe.component_mask (densities);

      IndexSet density_dofs = DoFTools::extract_dofs (dof_handler,
          density_mask);

//      const unsigned int first_density_dof = *density_dofs.begin ();
//      constraints.add_line (first_density_dof);
//      for (unsigned int i = 1;
//          i < density_dofs.n_elements (); ++i)
//        {
//          constraints.add_entry (first_density_dof,
//              density_dofs.nth_index_in_set (i), -1);
//        }


        const unsigned int last_density_dof = density_dofs.nth_index_in_set(density_dofs.n_elements ()-1);
        constraints.add_line (last_density_dof);
        for (unsigned int i = 1;
             i < density_dofs.n_elements (); ++i)
        {
            constraints.add_entry (last_density_dof,
                                   density_dofs.nth_index_in_set (i-1), -1);
        }


//      constraints.set_inhomogeneity (first_density_dof, 0);
//
//        VectorTools::interpolate_boundary_values(dof_handler,
//                                                 0,
//                                                 ZeroFunction<dim>(2*dim+5),
//                                                 constraints,
//                                                 fe.component_mask(displacements));
//        VectorTools::interpolate_boundary_values(dof_handler,
//                                                 0,
//                                                 ZeroFunction<dim>(2*dim+5),
//                                                 constraints,
//                                                 fe.component_mask(displacement_multipliers));
      constraints.close ();

//      DoFTools::make_sparsity_pattern (dof_handler, coupling, dsp, constraints,
//          false);
//changed it to below - works now?



      DoFTools::make_sparsity_pattern (dof_handler,dsp);
      constraints.condense(dsp);
      sparsity_pattern.copy_from (dsp);

      std::ofstream out ("sparsity.plt");
      sparsity_pattern.print_gnuplot (out);

      system_matrix.reinit (sparsity_pattern);

      linear_solution.reinit (7);
      nonlinear_solution.reinit (7);
      //first solution block - density
      nonlinear_solution.block (0).reinit (n_p);
      linear_solution.block (0).reinit (n_p);
      //second solution block - displacement
      nonlinear_solution.block (1).reinit (n_u);
      linear_solution.block (1).reinit (n_u);
      //third solution block - displacement multiplier
      nonlinear_solution.block (2).reinit (n_u);
      linear_solution.block (2).reinit (n_u);
      //second solution block - lower slack
      nonlinear_solution.block (3).reinit (n_p);
      linear_solution.block (3).reinit (n_p);
      //second solution block - lower slack multiplier
      nonlinear_solution.block (4).reinit (n_p);
      linear_solution.block (4).reinit (n_p);
      //second solution block - upper slack and delta
      nonlinear_solution.block (5).reinit (n_p);
      linear_solution.block (5).reinit (n_p);
      //second solution block - upper slack multiplier
      nonlinear_solution.block (6).reinit (n_p);
      linear_solution.block (6).reinit (n_p);

      linear_solution.collect_sizes ();
      nonlinear_solution.collect_sizes ();

      system_rhs.reinit (7);
      system_rhs.block (0).reinit (n_p);
      system_rhs.block (1).reinit (n_u);
      system_rhs.block (2).reinit (n_u);
      system_rhs.block (3).reinit (n_p);
      system_rhs.block (4).reinit (n_p);
      system_rhs.block (5).reinit (n_p);
      system_rhs.block (6).reinit (n_p);
      system_rhs.collect_sizes ();

      density_ratio = .5;
      density_penalty_exponent = 3;

      for (unsigned int i = 0; i < n_u; i++)
        {
          nonlinear_solution.block (1)[i] = 0;
          nonlinear_solution.block (2)[i] = 0;
        }
      for (unsigned int i = 0; i < n_p; i++)
        {
          nonlinear_solution.block (0)[i] = density_ratio;
          nonlinear_solution.block (3)[i] = density_ratio;
          nonlinear_solution.block (4)[i] = .5;
          nonlinear_solution.block (5)[i] = 1 - density_ratio;
          nonlinear_solution.block (6)[i] = .5;
        }

    }
  template <int dim>
    void
    SANDTopOpt<dim>::assemble_block_system (double barrier_size)
    {
      const FEValuesExtractors::Scalar densities (0);
      const FEValuesExtractors::Vector displacements (1);
      const FEValuesExtractors::Vector displacement_multipliers (1 + dim);
      const FEValuesExtractors::Scalar density_lower_slacks (1 + 2 * dim);
      const FEValuesExtractors::Scalar density_lower_slack_multipliers (
          2 + 2 * dim);
      const FEValuesExtractors::Scalar density_upper_slacks (3 + 2 * dim);
      const FEValuesExtractors::Scalar density_upper_slack_multipliers (
          4 + 2 * dim);

      /*Remove any values from old iterations*/
      system_matrix.reinit (sparsity_pattern);
      linear_solution = 0;
      system_rhs = 0;

      QGauss < dim > quadrature_formula (fe.degree + 1);
      QGauss < dim - 1 > face_quadrature_formula (fe.degree + 1);
      FEValues < dim > fe_values (fe, quadrature_formula,
          update_values | update_gradients | update_quadrature_points
          | update_JxW_values);
      FEFaceValues < dim > fe_face_values (fe, face_quadrature_formula,
          update_values | update_quadrature_points | update_normal_vectors
          | update_JxW_values);

      const unsigned int dofs_per_cell = fe.dofs_per_cell;
      const unsigned int n_q_points = quadrature_formula.size ();
      const unsigned int n_face_q_points = face_quadrature_formula.size ();

      FullMatrix<double> cell_matrix (dofs_per_cell, dofs_per_cell);
      Vector<double> cell_rhs (dofs_per_cell);
      FullMatrix<double> full_density_cell_matrix (dofs_per_cell,
          dofs_per_cell);
      FullMatrix<double> full_density_cell_matrix_for_Au (dofs_per_cell,
          dofs_per_cell);

      std::vector < types::global_dof_index > local_dof_indices (dofs_per_cell);

      std::vector<double> lambda_values (n_q_points);
      std::vector<double> mu_values (n_q_points);

      Functions::ConstantFunction<dim> lambda (1.), mu (1.);
      std::vector<Tensor<1, dim>> rhs_values (n_q_points);

      for (const auto &cell : dof_handler.active_cell_iterators ())
        {
          cell_matrix = 0;
          full_density_cell_matrix = 0;
          full_density_cell_matrix_for_Au = 0;
          cell_rhs = 0;

          cell->get_dof_indices (local_dof_indices);

          fe_values.reinit (cell);

          lambda.value_list (fe_values.get_quadrature_points (), lambda_values);
          mu.value_list (fe_values.get_quadrature_points (), mu_values);

          std::vector<double> old_density_values (n_q_points);
          std::vector<Tensor<1, dim>> old_displacement_values (n_q_points);
          std::vector<double> old_displacement_divs (n_q_points);
          std::vector<SymmetricTensor<2, dim>> old_displacement_symmgrads (
              n_q_points);
          std::vector<Tensor<1, dim>> old_displacement_multiplier_values (
              n_q_points);
          std::vector<double> old_displacement_multiplier_divs (n_q_points);
          std::vector<SymmetricTensor<2, dim>> old_displacement_multiplier_symmgrads (
              n_q_points);
          std::vector<double> old_lower_slack_multiplier_values (n_q_points);
          std::vector<double> old_upper_slack_multiplier_values (n_q_points);
          std::vector<double> old_lower_slack_values (n_q_points);
          std::vector<double> old_upper_slack_values (n_q_points);

          fe_values[densities].get_function_values (nonlinear_solution,
              old_density_values);
          fe_values[displacements].get_function_values (nonlinear_solution,
              old_displacement_values);
          fe_values[displacements].get_function_divergences (nonlinear_solution,
              old_displacement_divs);
          fe_values[displacements].get_function_symmetric_gradients (
              nonlinear_solution, old_displacement_symmgrads);
          fe_values[displacement_multipliers].get_function_values (
              nonlinear_solution, old_displacement_multiplier_values);
          fe_values[displacement_multipliers].get_function_divergences (
              nonlinear_solution, old_displacement_multiplier_divs);
          fe_values[displacement_multipliers].get_function_symmetric_gradients (
              nonlinear_solution, old_displacement_multiplier_symmgrads);
          fe_values[density_lower_slacks].get_function_values (
              nonlinear_solution, old_lower_slack_values);
          fe_values[density_lower_slack_multipliers].get_function_values (
              nonlinear_solution, old_lower_slack_multiplier_values);
          fe_values[density_upper_slacks].get_function_values (
              nonlinear_solution, old_upper_slack_values);
          fe_values[density_upper_slack_multipliers].get_function_values (
              nonlinear_solution, old_upper_slack_multiplier_values);

          Tensor < 1, dim > traction;
          traction[1] = -1;

          for (unsigned int q_point = 0; q_point < n_q_points; ++q_point)
            {

              for (unsigned int i = 0; i < dofs_per_cell; ++i)
                {
                  const SymmetricTensor<2, dim> displacement_phi_i_symmgrad =
                      fe_values[displacements].symmetric_gradient (i, q_point);
                  const double displacement_phi_i_div =
                      fe_values[displacements].divergence (i, q_point);

                  const SymmetricTensor<2, dim> displacement_multiplier_phi_i_symmgrad =
                                       fe_values[displacement_multipliers].symmetric_gradient (i,
                                           q_point);
                 const double displacement_multiplier_phi_i_div =
                     fe_values[displacement_multipliers].divergence (i,
                         q_point);


                  const double density_phi_i = fe_values[densities].value (i,
                      q_point);

                  const double lower_slack_multiplier_phi_i =
                      fe_values[density_lower_slack_multipliers].value (i,
                          q_point);

                  const double lower_slack_phi_i =
                      fe_values[density_lower_slacks].value (i, q_point);

                  const double upper_slack_phi_i =
                      fe_values[density_upper_slacks].value (i, q_point);

                  const double upper_slack_multiplier_phi_i =
                      fe_values[density_upper_slack_multipliers].value (i,
                          q_point);


                  for (unsigned int j = 0; j < dofs_per_cell; ++j)
                    {
                      const SymmetricTensor<2, dim> displacement_phi_j_symmgrad =
                          fe_values[displacements].symmetric_gradient (j,
                              q_point);
                      const double displacement_phi_j_div =
                          fe_values[displacements].divergence (j, q_point);

                      const SymmetricTensor<2, dim> displacement_multiplier_phi_j_symmgrad =
                          fe_values[displacement_multipliers].symmetric_gradient (
                              j, q_point);
                      const double displacement_multiplier_phi_j_div =
                          fe_values[displacement_multipliers].divergence (j,
                              q_point);

                      const double density_phi_j = fe_values[densities].value (
                          j, q_point);

                      const double lower_slack_phi_j =
                          fe_values[density_lower_slacks].value (j, q_point);

                      const double upper_slack_phi_j =
                          fe_values[density_upper_slacks].value (j, q_point);

                      const double lower_slack_multiplier_phi_j =
                          fe_values[density_lower_slack_multipliers].value (j,
                              q_point);

                      const double upper_slack_multiplier_phi_j =
                          fe_values[density_upper_slack_multipliers].value (j,
                              q_point);

                      //Equation 0
                      cell_matrix (i, j) +=
                          fe_values.JxW (q_point) *
                          (-1* density_phi_i * lower_slack_multiplier_phi_j

                           + density_phi_i * upper_slack_multiplier_phi_j

                           + density_penalty_exponent * (density_penalty_exponent
                               - 1)
                             * std::pow (
                                 old_density_values[q_point],
                                 density_penalty_exponent - 2)
                             * density_phi_i
                             * density_phi_j
                             * (old_displacement_multiplier_divs[q_point] * old_displacement_divs[q_point]
                                * lambda_values[q_point]
                                + 2 * mu_values[q_point]
                                  * (old_displacement_symmgrads[q_point] * old_displacement_multiplier_symmgrads[q_point]))

                           + density_penalty_exponent * std::pow (
                                 old_density_values[q_point],
                                 density_penalty_exponent - 1)
                             * density_phi_i
                             * (displacement_multiplier_phi_j_div * old_displacement_divs[q_point]
                                * lambda_values[q_point]
                                + 2 * mu_values[q_point]
                                  * (old_displacement_symmgrads[q_point] * displacement_multiplier_phi_j_symmgrad))

                           + density_penalty_exponent * std::pow (
                                 old_density_values[q_point],
                                 density_penalty_exponent - 1)
                             * density_phi_i
                             * (displacement_phi_j_div * old_displacement_multiplier_divs[q_point]
                                * lambda_values[q_point]
                                + 2 * mu_values[q_point]
                                  * (old_displacement_multiplier_symmgrads[q_point] * displacement_phi_j_symmgrad)));

                      //Equation 1

                      cell_matrix (i, j) +=
                          fe_values.JxW (q_point) * (
                          density_penalty_exponent * std::pow (
                              old_density_values[q_point],
                              density_penalty_exponent - 1)
                          * density_phi_j
                          * (old_displacement_multiplier_divs[q_point] * displacement_phi_i_div
                             * lambda_values[q_point]
                             + 2 * mu_values[q_point]
                               * (old_displacement_multiplier_symmgrads[q_point] * displacement_phi_i_symmgrad))

                          + std::pow (old_density_values[q_point],
                                density_penalty_exponent)
                            * (displacement_multiplier_phi_j_div * displacement_phi_i_div
                               * lambda_values[q_point]
                               + 2 * mu_values[q_point]
                                 * (displacement_multiplier_phi_j_symmgrad * displacement_phi_i_symmgrad))

                          );

                      //Equation 2 - Primal Feasibility

                      cell_matrix (i, j) +=
                          fe_values.JxW (q_point) * (

                          density_penalty_exponent * std::pow (
                              old_density_values[q_point],
                              density_penalty_exponent - 1)
                          * density_phi_j
                          * (old_displacement_divs[q_point] * displacement_multiplier_phi_i_div
                             * lambda_values[q_point]
                             + 2 * mu_values[q_point]
                               * (old_displacement_symmgrads[q_point] * displacement_multiplier_phi_i_symmgrad))

                          + std::pow (old_density_values[q_point],
                                density_penalty_exponent)
                            * (displacement_phi_j_div * displacement_multiplier_phi_i_div
                               * lambda_values[q_point]
                               + 2 * mu_values[q_point]
                                 * (displacement_phi_j_symmgrad * displacement_multiplier_phi_i_symmgrad)));

                      //Equation 3 - more primal feasibility
                      cell_matrix (i, j) +=
                          fe_values.JxW (q_point) * lower_slack_multiplier_phi_i*
                                  (density_phi_j - lower_slack_phi_j);

                      //Equation 4 - more primal feasibility
                      cell_matrix (i, j) +=
                          fe_values.JxW (q_point) * upper_slack_multiplier_phi_i * (
                              -1 * density_phi_j - upper_slack_phi_j);

                      //Equation 5 - complementary slackness
                      cell_matrix (i, j) += fe_values.JxW (q_point)
                          * (lower_slack_phi_i * lower_slack_multiplier_phi_j * old_lower_slack_values[q_point]
                             + lower_slack_phi_i * lower_slack_phi_j * old_lower_slack_multiplier_values[q_point]);

                      //Equation 6 - complementary slackness
                      cell_matrix (i, j) += fe_values.JxW (q_point)
                          * (upper_slack_phi_i * upper_slack_multiplier_phi_j
                             * old_upper_slack_values[q_point]

                             + upper_slack_phi_i * upper_slack_phi_j
                               * old_upper_slack_multiplier_values[q_point]);

                    }

                  //rhs eqn 0
                  cell_rhs(i) +=
                          fe_values.JxW(q_point) *(
                                  -1 *density_penalty_exponent* std::pow(old_density_values[q_point],density_penalty_exponent - 1)* density_phi_i
                                      * (old_displacement_multiplier_divs[q_point] * old_displacement_divs[q_point]
                                      * lambda_values[q_point]
                                      + 2 * mu_values[q_point] * (old_displacement_symmgrads[q_point]
                                      * old_displacement_multiplier_symmgrads[q_point]))
                                  + -1 * density_phi_i * old_upper_slack_multiplier_values[q_point]
                                + density_phi_i * old_lower_slack_multiplier_values[q_point]);

                  //rhs eqn 1 - boundary terms counted later
                  cell_rhs (i) +=
                      fe_values.JxW (q_point)* (
                        -1 * std::pow (old_density_values[q_point], density_penalty_exponent)
                        * (old_displacement_multiplier_divs[q_point] * displacement_phi_i_div
                             * lambda_values[q_point]
                           + 2* mu_values[q_point] * (old_displacement_multiplier_symmgrads[q_point]
                             * displacement_phi_i_symmgrad))
                      ) ;

                  //rhs eqn 2 - boundary terms counted later
                  cell_rhs (i) +=
                      fe_values.JxW (q_point)*(
                        -1 * std::pow (old_density_values[q_point],
                            density_penalty_exponent)
                        * (old_displacement_divs[q_point] * displacement_multiplier_phi_i_div
                           * lambda_values[q_point]
                           + 2*mu_values[q_point] * (old_displacement_symmgrads[q_point]
                               * displacement_multiplier_phi_i_symmgrad))
                      );

                  //rhs eqn 3
                  cell_rhs (i) +=
                      -1 * lower_slack_multiplier_phi_i
                      * (old_density_values[q_point] - old_lower_slack_values[q_point])
                      * fe_values.JxW (q_point);

                  //rhs eqn 4
                  cell_rhs (i) +=
                      -1 * upper_slack_multiplier_phi_i
                      * (1 - old_density_values[q_point]
                         - old_upper_slack_values[q_point])
                      * fe_values.JxW (q_point);

                  //rhs eqn 5
                  cell_rhs (i) +=
                      -1 * lower_slack_phi_i
                      * (old_lower_slack_values[q_point] * old_lower_slack_multiplier_values[q_point] - barrier_size)
                      * fe_values.JxW (q_point);

                  //rhs block 6
                  cell_rhs (i) +=
                      -1 * upper_slack_phi_i
                      * (old_upper_slack_values[q_point] * old_upper_slack_multiplier_values[q_point] - barrier_size)
                      * fe_values.JxW (q_point);

                }

            }
          for (unsigned int face_number = 0;
              face_number < GeometryInfo < dim > ::faces_per_cell;
              ++face_number)
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
                          cell_rhs (i) += -1
                              * traction
                              * fe_face_values[displacements].value (i,
                                  face_q_point)
                              * fe_face_values.JxW (face_q_point);

                          cell_rhs (i) += traction
                              * fe_face_values[displacement_multipliers].value (
                                  i, face_q_point)
                              * fe_face_values.JxW (face_q_point);
                        }
                    }
                }
            }

          MatrixTools::local_apply_boundary_values (boundary_values, local_dof_indices,
                     cell_matrix, cell_rhs, true);

          constraints.distribute_local_to_global(
               cell_matrix, cell_rhs, local_dof_indices, system_matrix, system_rhs);



        }
    }

  template <int dim>
    void
    SANDTopOpt<dim>::solve ()
    {
      constraints.condense(system_matrix);
      constraints.condense(system_rhs);

      SparseDirectUMFPACK A_direct;
      A_direct.initialize (system_matrix);
      A_direct.vmult (linear_solution, system_rhs);

      constraints.distribute(linear_solution);

    }

  template <int dim>
    void
    SANDTopOpt<dim>::update_step ()
    {


      double fraction_to_boundary = .995;

      double step_size_s_low = 0;
      double step_size_z_low = 0;
      double step_size_s_high = 1;
      double step_size_z_high = 1;
      double step_size_s, step_size_z;
      bool accept_s, accept_z;

      for (unsigned int k = 0; k < 50; k++)
        {
          step_size_s = (step_size_s_low + step_size_s_high) / 2;
          step_size_z = (step_size_z_low + step_size_z_high) / 2;

          BlockVector<double> nonlinear_solution_test_s =
              (fraction_to_boundary * nonlinear_solution) + (step_size_s
                  * linear_solution);

          BlockVector<double> nonlinear_solution_test_z =
              (fraction_to_boundary * nonlinear_solution) + (step_size_z
                  * linear_solution);

          accept_s = (nonlinear_solution_test_s.block (3).is_non_negative ())
              && (nonlinear_solution_test_s.block (5).is_non_negative ());
          accept_z = (nonlinear_solution_test_z.block (4).is_non_negative ())
              && (nonlinear_solution_test_z.block (6).is_non_negative ());

          if (accept_s)
            {
              step_size_s_low = step_size_s;
            }
          else
            {
              step_size_s_high = step_size_s;
            }
          if (accept_z)
            {
              step_size_z_low = step_size_z;
            }
          else
            {
              step_size_z_high = step_size_z;
            }
        }
      std::cout << step_size_s_low << "    " << step_size_z_low << std::endl;

      nonlinear_solution.block (0) = nonlinear_solution.block (0)
          + step_size_s_low * linear_solution.block (0);
      nonlinear_solution.block (1) = nonlinear_solution.block (1)
          + step_size_s_low * linear_solution.block (1);
      nonlinear_solution.block (2) = nonlinear_solution.block (2)
          + step_size_z_low * linear_solution.block (2);
      nonlinear_solution.block (3) = nonlinear_solution.block (3)
          + step_size_s_low * linear_solution.block (3);
      nonlinear_solution.block (4) = nonlinear_solution.block (4)
          + step_size_z_low * linear_solution.block (4);
      nonlinear_solution.block (5) = nonlinear_solution.block (5)
          + step_size_s_low * linear_solution.block (5);
      nonlinear_solution.block (6) = nonlinear_solution.block (6)
          + step_size_z_low * linear_solution.block (6);
    }

  template <int dim>
    void
    SANDTopOpt<dim>::output (int j)
    {
      std::vector < std::string > solution_names (1, "density");
      std::vector < DataComponentInterpretation::DataComponentInterpretation > data_component_interpretation (
                1, DataComponentInterpretation::component_is_scalar);
      for (unsigned int i=0; i<dim; i++)
        {
          solution_names.emplace_back("displacement");
          data_component_interpretation.push_back(
            DataComponentInterpretation::component_is_part_of_vector);
        }
      for (unsigned int i=0; i<dim; i++)
        {
          solution_names.emplace_back("displacement_multiplier");
          data_component_interpretation.push_back(
            DataComponentInterpretation::component_is_part_of_vector);
        }
      solution_names.emplace_back("low_slack");
      data_component_interpretation.push_back(
        DataComponentInterpretation::component_is_scalar);
      solution_names.emplace_back("low_slack_multiplier");
      data_component_interpretation.push_back(
        DataComponentInterpretation::component_is_scalar);
      solution_names.emplace_back("high_slack");
      data_component_interpretation.push_back(
        DataComponentInterpretation::component_is_scalar);
      solution_names.emplace_back("high_slack_multiplier");
      data_component_interpretation.push_back(
        DataComponentInterpretation::component_is_scalar);
      DataOut < dim > data_out;
      data_out.attach_dof_handler (dof_handler);
      data_out.add_data_vector (nonlinear_solution, solution_names,
          DataOut<dim>::type_dof_data, data_component_interpretation);
      data_out.build_patches ();
      std::ofstream output ("solution" + std::to_string (j) + ".vtk");
      data_out.write_vtk (output);
    }

  template <int dim>
    void
    SANDTopOpt<dim>::run ()
    {
      double barrier_size = .25;

      create_triangulation ();
      setup_block_system ();
      set_bcids ();

       for (unsigned int loop = 0; loop < 2; loop++)
        {
          assemble_block_system (barrier_size);
          solve ();
          update_step ();
          output (loop);
          barrier_size = barrier_size * .9;
          std::cout << "current_values" << std::endl;
          nonlinear_solution.block(0).print(std::cout);


        }
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
      << "----------------------------------------------------" << std::endl;
      std::cerr << "Exception on processing: " << std::endl << exc.what ()
      << std::endl << "Aborting!" << std::endl
      << "----------------------------------------------------" << std::endl;

      return 1;
    }
  catch (...)
    {
      std::cerr << std::endl << std::endl
      << "----------------------------------------------------" << std::endl;
      std::cerr << "Unknown exception!" << std::endl << "Aborting!" << std::endl
      << "----------------------------------------------------" << std::endl;
      return 1;
    }

  return 0;
}
