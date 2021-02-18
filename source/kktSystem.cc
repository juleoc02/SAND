//
// Created by justin on 2/17/21.
//


#include "../include/kktSystem.h"
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/function.h>
#include <deal.II/base/tensor.h>
#include <deal.II/base/timer.h>

#include <deal.II/lac/block_vector.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/block_sparse_matrix.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/precondition.h>
#include <deal.II/lac/linear_operator.h>
#include <deal.II/lac/packaged_operation.h>
#include <deal.II/lac/sparse_direct.h>

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
#include <algorithm>

///This problem initializes with a FESystem composed of 2×dim FE_Q(1) elements, and 7 FE_DGQ(0)  elements.
/// The  piecewise  constant  functions  are  for  density-related  variables,and displacement-related variables are assigned to the FE_Q(1) elements.

template<int dim>
KktSystem<dim>::KktSystem()
        :
        dof_handler(triangulation),
        /*fe should have 1 FE_DGQ<dim>(0) element for density, dim FE_Q finite elements for displacement,
         * another dim FE_Q elements for the lagrange multiplier on the FE constraint, and 2 more FE_DGQ<dim>(0)
         * elements for the upper and lower bound constraints */
        fe(FE_DGQ<dim>(0) ^ 1,
           (FESystem<dim>(FE_Q<dim>(1) ^ dim)) ^ 1,
           FE_DGQ<dim>(0) ^ 1,
           (FESystem<dim>(FE_Q<dim>(1) ^ dim)) ^ 1,
           FE_DGQ<dim>(0) ^ 5),
        density_ratio (.5),
        density_penalty_exponent (3),
        filter_r (.25)
{
}


///A  function  used  once  at  the  beginning  of  the  program,  this  creates  a  matrix  H  so  that H* unfiltered density = filtered density

    template<int dim>
    void
    KktSystem<dim>::setup_filter_matrix() {

        filter_dsp.reinit(dof_handler.get_triangulation().n_active_cells(),
                          dof_handler.get_triangulation().n_active_cells());
        std::set<unsigned int> neighbor_ids;
        std::set<typename Triangulation<dim>::cell_iterator> cells_to_check;
        std::set<typename Triangulation<dim>::cell_iterator> cells_to_check_temp;
        double distance;

        /*finds neighbors-of-neighbors until it is out to specified radius*/
        for (const auto &cell : dof_handler.active_cell_iterators()) {
            const unsigned int i = cell->active_cell_index();
            neighbor_ids = {i};
            cells_to_check = {cell};

            unsigned int n_neighbors = 1;
            while (true) {
                cells_to_check_temp.clear();
                for (auto check_cell : cells_to_check) {
                    for (unsigned int n = 0;
                         n < GeometryInfo<dim>::faces_per_cell; ++n) {
                        if (!(check_cell->face(n)->at_boundary())) {
                            distance = cell->center().distance(
                                    check_cell->neighbor(n)->center());
                            if ((distance < filter_r) &&
                                !(neighbor_ids.count(check_cell->neighbor(n)->active_cell_index()))) {
                                cells_to_check_temp.insert(check_cell->neighbor(n));
                                neighbor_ids.insert(check_cell->neighbor(n)->active_cell_index());
                            }
                        }
                    }
                }

                if (neighbor_ids.size() == n_neighbors) {
                    break;
                } else {
                    cells_to_check = cells_to_check_temp;
                    n_neighbors = neighbor_ids.size();
                }
            }
/*add all of these to the sparsity pattern*/
            for (auto j : neighbor_ids) {
                filter_dsp.add(i, j);
            }
        }

        filter_sparsity_pattern.copy_from(filter_dsp);
        filter_matrix.reinit(filter_sparsity_pattern);

/*find these cells again to add values to matrix*/
        for (const auto &cell : dof_handler.active_cell_iterators()) {
            const unsigned int i = cell->active_cell_index();
            neighbor_ids = {i};
            cells_to_check = {cell};
            cells_to_check_temp = {};

            unsigned int n_neighbors = 1;
            filter_matrix.add(i, i, filter_r);
            while (true) {
                cells_to_check_temp.clear();
                for (auto check_cell : cells_to_check) {
                    for (unsigned int n = 0; n < GeometryInfo<dim>::faces_per_cell; ++n) {
                        if (!(check_cell->face(n)->at_boundary())) {
                            distance = cell->center().distance(
                                    check_cell->neighbor(n)->center());
                            if ((distance < filter_r) && !(neighbor_ids.count(
                                    check_cell->neighbor(n)->active_cell_index()))) {
                                cells_to_check_temp.insert(
                                        check_cell->neighbor(n));
                                neighbor_ids.insert(
                                        check_cell->neighbor(n)->active_cell_index());
/*value should be max radius - distance between cells*/
                                filter_matrix.add(i, check_cell->neighbor(n)->active_cell_index(),
                                                  filter_r - distance);
                            }
                        }
                    }
                }

                if (neighbor_ids.size() == n_neighbors) {
                    break;
                } else {
                    cells_to_check = cells_to_check_temp;
                    n_neighbors = neighbor_ids.size();
                }
            }
        }

        for (const auto &cell : dof_handler.active_cell_iterators()) {
            const unsigned int i = cell->active_cell_index();
            double denominator = 0;
            typename SparseMatrix<double>::iterator iter = filter_matrix.begin(
                    i);
            for (; iter != filter_matrix.end(i); iter++) {
                denominator = denominator + iter->value();
            }
            iter = filter_matrix.begin(i);
            for (; iter != filter_matrix.end(i); iter++) {
                iter->value() = iter->value() / denominator;
            }
        }
    }

    ///This triangulation matches the problem description in the introduction -
    /// a 6-by-1 rectangle where a force will be applied in the top center.

    template<int dim>
    void
    KktSystem<dim>::create_triangulation() {
        /*Make a square*/
        Triangulation<dim> triangulation_temp;
        Point<dim> point_1, point_2;
        point_1(0) = 0;
        point_1(1) = 0;
        point_2(0) = 1;
        point_2(1) = 1;
        GridGenerator::hyper_rectangle(triangulation, point_1, point_2);

        /*make 5 more squares*/
        for (unsigned int n = 1; n < 6; n++) {
            triangulation_temp.clear();
            point_1(0) = n;
            point_2(0) = n + 1;
            GridGenerator::hyper_rectangle(triangulation_temp, point_1, point_2);
            /*glue squares together*/
            GridGenerator::merge_triangulations(triangulation_temp,
                                                triangulation, triangulation);
        }
        triangulation.refine_global(4);

        /*Set BCIDs   */
        for (const auto &cell : triangulation.active_cell_iterators()) {
            for (unsigned int face_number = 0;
                 face_number < GeometryInfo<dim>::faces_per_cell;
                 ++face_number) {
                if (cell->face(face_number)->at_boundary()) {
                    const auto center = cell->face(face_number)->center();
                    if (std::fabs(center(1) - 0) < 1e-12) {
                        /*Boundary ID of 2 is the 0 neumann, so no external force*/
                        cell->face(face_number)->set_boundary_id(2);
                    }
                    if (std::fabs(center(1) - 1) < 1e-12) {
                        /*Find top middle*/
                        if ((std::fabs(center(0) - 3) < .1)) {
                            /*downward force is boundary id of 1*/
                            cell->face(face_number)->set_boundary_id(1);
                        } else {
                            cell->face(face_number)->set_boundary_id(2);
                        }
                    }
                    if (std::fabs(center(0) - 0) < 1e-12) {
                        cell->face(face_number)->set_boundary_id(2);
                    }
                    if (std::fabs(center(0) - 6) < 1e-12) {
                        cell->face(face_number)->set_boundary_id(2);
                    }
                }
            }
        }

        dof_handler.distribute_dofs(fe);

        DoFRenumbering::component_wise(dof_handler);

    }

///The  bottom  corners  are  kept  in  place  in  the  y  direction  -  the  bottom  left  also  in  the  x direction.
/// Because deal.ii is formulated to enforce boundary conditions along regions of the boundary,
/// we do this to ensure these BCs are only enforced at points.
    template<int dim>
    void
    KktSystem<dim>::setup_boundary_values() {
        for (const auto &cell : dof_handler.active_cell_iterators()) {
            for (unsigned int face_number = 0;
                 face_number < GeometryInfo<dim>::faces_per_cell;
                 ++face_number) {
                if (cell->face(face_number)->at_boundary()) {
                    const auto center = cell->face(face_number)->center();
                    if (std::fabs(center(1) - 0) < 1e-12) {

                        for (unsigned int vertex_number = 0;
                             vertex_number < GeometryInfo<dim>::vertices_per_cell;
                             ++vertex_number) {
                            const auto vert = cell->vertex(vertex_number);
                            /*Find bottom left corner*/
                            if (std::fabs(vert(0) - 0) < 1e-12 && std::fabs(
                                    vert(1) - 0) < 1e-12) {

                                const unsigned int x_displacement =
                                        cell->vertex_dof_index(vertex_number, 0);
                                const unsigned int y_displacement =
                                        cell->vertex_dof_index(vertex_number, 1);
                                const unsigned int x_displacement_multiplier =
                                        cell->vertex_dof_index(vertex_number, 2);
                                const unsigned int y_displacement_multiplier =
                                        cell->vertex_dof_index(vertex_number, 3);
                                /*set bottom left BC*/
                                boundary_values[x_displacement] = 0;
                                boundary_values[y_displacement] = 0;
                                boundary_values[x_displacement_multiplier] = 0;
                                boundary_values[y_displacement_multiplier] = 0;
                            }
                            /*Find bottom right corner*/
                            if (std::fabs(vert(0) - 6) < 1e-12 && std::fabs(
                                    vert(
                                            1)
                                    - 0)
                                                                  < 1e-12) {
                                const unsigned int y_displacement =
                                        cell->vertex_dof_index(vertex_number, 1);
                                const unsigned int y_displacement_multiplier =
                                        cell->vertex_dof_index(vertex_number, 3);
                                // const unsigned int x_displacement =
                                //         cell->vertex_dof_index(vertex_number, 0);
                                // const unsigned int x_displacement_multiplier =
                                //         cell->vertex_dof_index(vertex_number, 2);
                                /*set bottom left BC*/
                                boundary_values[y_displacement] = 0;
                                boundary_values[y_displacement_multiplier] = 0;
                            }
                        }
                    }
                }
            }
        }
    }


    ///This makes a giant 9-by-9 block matrix, and also sets up the necessary block vectors.  The
    /// sparsity pattern for this matrix includes the sparsity pattern for the filter matrix. It also initializes
    /// any block vectors we will use.
    template<int dim>
    void
    KktSystem<dim>::setup_block_system() {
        const FEValuesExtractors::Scalar densities(0);

        //MAKE n_u and n_P*****************************************************************

        /*Setup 9 by 9 block matrix*/

        std::vector<unsigned int> block_component(9, 2);
        block_component[0] = 0;
        block_component[1] = 1;
        const std::vector<types::global_dof_index> dofs_per_block =
                DoFTools::count_dofs_per_fe_block(dof_handler, block_component);

        const unsigned int n_p = dofs_per_block[0];
        const unsigned int n_u = dofs_per_block[1];
        std::cout << "n_p:  " << n_p << "   n_u   " << n_u << std::endl;
        const std::vector<unsigned int> block_sizes = {n_p, n_u, n_p, n_u, n_p, n_p, n_p, n_p, n_p};

        BlockDynamicSparsityPattern dsp(9, 9);

        for (unsigned int k = 0; k < 9; k++) {
            for (unsigned int j = 0; j < 9; j++) {
                dsp.block(j, k).reinit(block_sizes[j], block_sizes[k]);
            }
        }

        dsp.collect_sizes();

        Table<2, DoFTools::Coupling> coupling(2 * dim + 7, 2 * dim + 7);

        coupling[0][0] = DoFTools::always;

        for (unsigned int i = 0; i < dim; i++) {
            coupling[0][1 + i] = DoFTools::always;
            coupling[1 + i][0] = DoFTools::always;
        }

        coupling[0][1 + dim] = DoFTools::none;
        coupling[1 + dim][0] = DoFTools::none;

        for (unsigned int i = 0; i < dim; i++) {
            coupling[0][2 + dim + i] = DoFTools::always;
            coupling[2 + dim + i][0] = DoFTools::always;
        }

        coupling[0][2 + 2 * dim] = DoFTools::always;
        coupling[2 + 2 * dim][0] = DoFTools::always;


        coupling[0][2 + 2 * dim + 1] = DoFTools::none;
        coupling[0][2 + 2 * dim + 2] = DoFTools::none;
        coupling[0][2 + 2 * dim + 3] = DoFTools::none;
        coupling[0][2 + 2 * dim + 4] = DoFTools::none;
        coupling[2 + 2 * dim + 1][0] = DoFTools::none;
        coupling[2 + 2 * dim + 2][0] = DoFTools::none;
        coupling[2 + 2 * dim + 3][0] = DoFTools::none;
        coupling[2 + 2 * dim + 4][0] = DoFTools::none;




//Coupling for displacement

        for (unsigned int i = 0; i < dim; i++) {
            for (unsigned int k = 0; k < dim; k++) {
                coupling[1 + i][1 + k] = DoFTools::none;
            }
            coupling[1 + i][1 + dim ] = DoFTools::none;
            coupling[1 + dim ][1 + i] = DoFTools::none;

            for (unsigned int k = 0; k < dim; k++) {
                coupling[1 + i][2 + dim + k] = DoFTools::always;
                coupling[2 + dim + k][1 + i] = DoFTools::always;
            }
            for (unsigned int k = 0; k < 5; k++) {
                coupling[1 + i][2 + 2 * dim + k] = DoFTools::none;
                coupling[2 + 2 * dim + k][1 + i] = DoFTools::none;
            }
        }

// coupling for unfiltered density
        coupling[1+dim][1+dim]= DoFTools::none;
        for (unsigned int i = 0; i < dim; i++) {
            coupling[1 + dim][2 + dim + i] = DoFTools::none;
            coupling[2 + dim + i][1 + dim] = DoFTools::none;
        }

        coupling[1 + dim][3 + 2 * dim] = DoFTools::none;
        coupling[3 + 2 * dim][1 + dim] = DoFTools::none;
        coupling[1 + dim][4 + 2 * dim] = DoFTools::none;
        coupling[4 + 2 * dim][1 + dim] = DoFTools::none;
        coupling[1 + dim][5 + 2 * dim] = DoFTools::always;
        coupling[5 + 2 * dim][1 + dim] = DoFTools::always;
        coupling[1 + dim][6 + 2 * dim] = DoFTools::always;
        coupling[6 + 2 * dim][1 + dim] = DoFTools::always;

//Coupling for equality multipliers
        for (unsigned int i = 0; i < dim + 1; i++) {
            for (unsigned int k = 0; k < dim + 5; k++) {
                coupling[2 + dim + i][2 + dim + k] = DoFTools::none;
                coupling[2 + dim + k][2 + dim + i] = DoFTools::none;
            }
        }

//        Coupling for lower slack
        coupling[3 + 2 * dim][3 + 2 * dim] = DoFTools::always;
        coupling[3 + 2 * dim][4 + 2 * dim] = DoFTools::none;
        coupling[4 + 2 * dim][3 + 2 * dim] = DoFTools::none;
        coupling[3 + 2 * dim][5 + 2 * dim] = DoFTools::always;
        coupling[5 + 2 * dim][3 + 2 * dim] = DoFTools::always;
        coupling[3 + 2 * dim][6 + 2 * dim] = DoFTools::none;
        coupling[6 + 2 * dim][3 + 2 * dim] = DoFTools::none;

//
        coupling[4 + 2 * dim][4 + 2 * dim] = DoFTools::always;
        coupling[4 + 2 * dim][5 + 2 * dim] = DoFTools::none;
        coupling[5 + 2 * dim][4 + 2 * dim] = DoFTools::none;
        coupling[4 + 2 * dim][6 + 2 * dim] = DoFTools::always;
        coupling[6 + 2 * dim][4 + 2 * dim] = DoFTools::always;

//
        coupling[5 + 2 * dim][5 + 2 * dim] = DoFTools::none;
        coupling[5 + 2 * dim][6 + 2 * dim] = DoFTools::none;
        coupling[6 + 2 * dim][5 + 2 * dim] = DoFTools::none;
        coupling[6 + 2 * dim][6 + 2 * dim] = DoFTools::none;

        constraints.clear();

        const ComponentMask density_mask = fe.component_mask(densities);

        const IndexSet density_dofs = DoFTools::extract_dofs(dof_handler,
                                                       density_mask);


        const unsigned int last_density_dof = density_dofs.nth_index_in_set(density_dofs.n_elements() - 1);
        constraints.add_line(last_density_dof);
        for (unsigned int i = 1;
             i < density_dofs.n_elements(); ++i) {
            constraints.add_entry(last_density_dof,
                                  density_dofs.nth_index_in_set(i - 1), -1);
        }


      constraints.set_inhomogeneity (last_density_dof, 0);

        constraints.close();

        DoFTools::make_sparsity_pattern(dof_handler, coupling,dsp, constraints);

        std::set<unsigned int> neighbor_ids;
        std::set<typename Triangulation<dim>::cell_iterator> cells_to_check;
        std::set<typename Triangulation<dim>::cell_iterator> cells_to_check_temp;
        unsigned int n_neighbors, i;
        double distance;
        for (const auto &cell : dof_handler.active_cell_iterators())
        {
            i = cell->active_cell_index();
            neighbor_ids.clear();
            neighbor_ids.insert(i);
            cells_to_check.clear();
            cells_to_check.insert(cell);
            n_neighbors = 1;
            while (true) {
                cells_to_check_temp.clear();
                for (auto check_cell : cells_to_check) {
                    for (unsigned int n = 0;
                         n < GeometryInfo<dim>::faces_per_cell; ++n) {
                        if (!(check_cell->face(n)->at_boundary())) {
                            distance = cell->center().distance(
                                    check_cell->neighbor(n)->center());
                            if ((distance < filter_r) &&
                                !(neighbor_ids.count(check_cell->neighbor(n)->active_cell_index()))) {
                                cells_to_check_temp.insert(check_cell->neighbor(n));
                                neighbor_ids.insert(check_cell->neighbor(n)->active_cell_index());
                            }
                        }
                    }
                }

                if (neighbor_ids.size() == n_neighbors) {
                    break;
                } else {
                    cells_to_check = cells_to_check_temp;
                    n_neighbors = neighbor_ids.size();
                }
            }
/*add all of these to the sparsity pattern*/
            for (auto j : neighbor_ids) {
                dsp.block(2, 4).add(i, j);
                dsp.block(4, 2).add(i, j);
            }
        }

//        constraints.condense(dsp);
        sparsity_pattern.copy_from(dsp);

//        This also breaks everything
//        sparsity_pattern.block(4,2).copy_from( filter_sparsity_pattern);
//        sparsity_pattern.block(2,4).copy_from( filter_sparsity_pattern);

        std::ofstream out("sparsity.plt");
        sparsity_pattern.print_gnuplot(out);

        system_matrix.reinit(sparsity_pattern);


        linear_solution.reinit(9);
        system_rhs.reinit(9);

        for (unsigned int j = 0; j < 9; j++) {
            linear_solution.block(j).reinit(block_sizes[j]);
            system_rhs.block(j).reinit(block_sizes[j]);
        }

        linear_solution.collect_sizes();
        system_rhs.collect_sizes();
    }

    ///This  is  where  the  magic  happens.   The  equations  describing  the newtons method for finding 0s in the KKT conditions are implemented here.


    template<int dim>
    void
    KktSystem<dim>::assemble_block_system(const BlockVector<double> &state,const double barrier_size) {
        const FEValuesExtractors::Scalar densities(0);
        const FEValuesExtractors::Vector displacements(1);
        const FEValuesExtractors::Scalar unfiltered_densities(1 + dim);
        const FEValuesExtractors::Vector displacement_multipliers(2 + dim);
        const FEValuesExtractors::Scalar unfiltered_density_multipliers(2 + 2 * dim);
        const FEValuesExtractors::Scalar density_lower_slacks(3 + 2 * dim);
        const FEValuesExtractors::Scalar density_lower_slack_multipliers(
                4 + 2 * dim);
        const FEValuesExtractors::Scalar density_upper_slacks(5 + 2 * dim);
        const FEValuesExtractors::Scalar density_upper_slack_multipliers(
                6 + 2 * dim);

        /*Remove any values from old iterations*/
        system_matrix.reinit(sparsity_pattern);
        linear_solution = 0;
        system_rhs = 0;

        QGauss<dim> quadrature_formula(fe.degree + 1);
        QGauss<dim - 1> face_quadrature_formula(fe.degree + 1);
        FEValues<dim> fe_values(fe, quadrature_formula,
                                update_values | update_gradients | update_quadrature_points
                                | update_JxW_values);
        FEFaceValues<dim> fe_face_values(fe, face_quadrature_formula,
                                         update_values | update_quadrature_points | update_normal_vectors
                                         | update_JxW_values);

        const unsigned int dofs_per_cell = fe.dofs_per_cell;
        const unsigned int n_q_points = quadrature_formula.size();
        const unsigned int n_face_q_points = face_quadrature_formula.size();

        FullMatrix<double> cell_matrix(dofs_per_cell, dofs_per_cell);
        Vector<double> cell_rhs(dofs_per_cell);
        FullMatrix<double> full_density_cell_matrix(dofs_per_cell,
                                                    dofs_per_cell);
        FullMatrix<double> full_density_cell_matrix_for_Au(dofs_per_cell,
                                                           dofs_per_cell);

        std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);

        std::vector<double> lambda_values(n_q_points);
        std::vector<double> mu_values(n_q_points);

        const Functions::ConstantFunction<dim> lambda(1.), mu(1.);
        std::vector<Tensor<1, dim>> rhs_values(n_q_points);

        BlockVector<double> filtered_unfiltered_density_solution = state;
        BlockVector<double> filter_adjoint_unfiltered_density_multiplier_solution = state;
        filtered_unfiltered_density_solution.block(2) = 0;
        filter_adjoint_unfiltered_density_multiplier_solution.block(4) = 0;

        filter_matrix.vmult(filtered_unfiltered_density_solution.block(2), state.block(2));
        filter_matrix.Tvmult(filter_adjoint_unfiltered_density_multiplier_solution.block(4),
                             state.block(4));


        std::vector<double> old_density_values(n_q_points);
        std::vector<Tensor<1, dim>> old_displacement_values(n_q_points);
        std::vector<double> old_displacement_divs(n_q_points);
        std::vector<SymmetricTensor<2, dim>> old_displacement_symmgrads(
          n_q_points);
        std::vector<Tensor<1, dim>> old_displacement_multiplier_values(
          n_q_points);
        std::vector<double> old_displacement_multiplier_divs(n_q_points);
        std::vector<SymmetricTensor<2, dim>> old_displacement_multiplier_symmgrads(
          n_q_points);
        std::vector<double> old_lower_slack_multiplier_values(n_q_points);
        std::vector<double> old_upper_slack_multiplier_values(n_q_points);
        std::vector<double> old_lower_slack_values(n_q_points);
        std::vector<double> old_upper_slack_values(n_q_points);
        std::vector<double> old_unfiltered_density_values(n_q_points);
        std::vector<double> old_unfiltered_density_multiplier_values(n_q_points);
        std::vector<double> filtered_unfiltered_density_values(n_q_points);
        std::vector<double> filter_adjoint_unfiltered_density_multiplier_values(n_q_points);


        for (const auto &cell : dof_handler.active_cell_iterators()) {
            cell_matrix = 0;
            full_density_cell_matrix = 0;
            full_density_cell_matrix_for_Au = 0;
            cell_rhs = 0;

            cell->get_dof_indices(local_dof_indices);

            fe_values.reinit(cell);

            lambda.value_list(fe_values.get_quadrature_points(), lambda_values);
            mu.value_list(fe_values.get_quadrature_points(), mu_values);

            fe_values[densities].get_function_values(state,
                                                     old_density_values);
            fe_values[displacements].get_function_values(state,
                                                         old_displacement_values);
            fe_values[displacements].get_function_divergences(state,
                                                              old_displacement_divs);
            fe_values[displacements].get_function_symmetric_gradients(
                    state, old_displacement_symmgrads);
            fe_values[displacement_multipliers].get_function_values(
                    state, old_displacement_multiplier_values);
            fe_values[displacement_multipliers].get_function_divergences(
                    state, old_displacement_multiplier_divs);
            fe_values[displacement_multipliers].get_function_symmetric_gradients(
                    state, old_displacement_multiplier_symmgrads);
            fe_values[density_lower_slacks].get_function_values(
                    state, old_lower_slack_values);
            fe_values[density_lower_slack_multipliers].get_function_values(
                    state, old_lower_slack_multiplier_values);
            fe_values[density_upper_slacks].get_function_values(
                    state, old_upper_slack_values);
            fe_values[density_upper_slack_multipliers].get_function_values(
                    state, old_upper_slack_multiplier_values);
            fe_values[unfiltered_densities].get_function_values(
                    state, old_unfiltered_density_values);
            fe_values[unfiltered_density_multipliers].get_function_values(
                    state, old_unfiltered_density_multiplier_values);
            fe_values[unfiltered_densities].get_function_values(
                    filtered_unfiltered_density_solution, filtered_unfiltered_density_values);
            fe_values[unfiltered_density_multipliers].get_function_values(
                    filter_adjoint_unfiltered_density_multiplier_solution,
                    filter_adjoint_unfiltered_density_multiplier_values);

            Tensor<1, dim> traction;
            traction[1] = -1;

            for (unsigned int q_point = 0; q_point < n_q_points; ++q_point) {

                for (unsigned int i = 0; i < dofs_per_cell; ++i) {
                    const SymmetricTensor<2, dim> displacement_phi_i_symmgrad =
                            fe_values[displacements].symmetric_gradient(i, q_point);
                    const double displacement_phi_i_div =
                            fe_values[displacements].divergence(i, q_point);

                    const SymmetricTensor<2, dim> displacement_multiplier_phi_i_symmgrad =
                            fe_values[displacement_multipliers].symmetric_gradient(i,
                                                                                   q_point);
                    const double displacement_multiplier_phi_i_div =
                            fe_values[displacement_multipliers].divergence(i,
                                                                           q_point);


                    const double density_phi_i = fe_values[densities].value(i,
                                                                            q_point);
                    const double unfiltered_density_phi_i = fe_values[unfiltered_densities].value(i,
                                                                                                  q_point);
                    const double unfiltered_density_multiplier_phi_i = fe_values[unfiltered_density_multipliers].value(
                            i, q_point);

                    const double lower_slack_multiplier_phi_i =
                            fe_values[density_lower_slack_multipliers].value(i,
                                                                             q_point);

                    const double lower_slack_phi_i =
                            fe_values[density_lower_slacks].value(i, q_point);

                    const double upper_slack_phi_i =
                            fe_values[density_upper_slacks].value(i, q_point);

                    const double upper_slack_multiplier_phi_i =
                            fe_values[density_upper_slack_multipliers].value(i,
                                                                             q_point);


                    for (unsigned int j = 0; j < dofs_per_cell; ++j) {
                        const SymmetricTensor<2, dim> displacement_phi_j_symmgrad =
                                fe_values[displacements].symmetric_gradient(j,
                                                                            q_point);
                        const double displacement_phi_j_div =
                                fe_values[displacements].divergence(j, q_point);

                        const SymmetricTensor<2, dim> displacement_multiplier_phi_j_symmgrad =
                                fe_values[displacement_multipliers].symmetric_gradient(
                                        j, q_point);
                        const double displacement_multiplier_phi_j_div =
                                fe_values[displacement_multipliers].divergence(j,
                                                                               q_point);

                        const double density_phi_j = fe_values[densities].value(
                                j, q_point);

                        const double unfiltered_density_phi_j = fe_values[unfiltered_densities].value(j,
                                                                                                      q_point);
                        const double unfiltered_density_multiplier_phi_j = fe_values[unfiltered_density_multipliers].value(
                                j, q_point);


                        const double lower_slack_phi_j =
                                fe_values[density_lower_slacks].value(j, q_point);

                        const double upper_slack_phi_j =
                                fe_values[density_upper_slacks].value(j, q_point);

                        const double lower_slack_multiplier_phi_j =
                                fe_values[density_lower_slack_multipliers].value(j,
                                                                                 q_point);

                        const double upper_slack_multiplier_phi_j =
                                fe_values[density_upper_slack_multipliers].value(j,
                                                                                 q_point);

                        //Equation 0
                        cell_matrix(i, j) +=
                                fe_values.JxW(q_point) *
                                (

                                        -density_phi_i * unfiltered_density_multiplier_phi_j

                                        + density_penalty_exponent * (density_penalty_exponent
                                                                      - 1)
                                          * std::pow(
                                                old_density_values[q_point],
                                                density_penalty_exponent - 2)
                                          * density_phi_i
                                          * density_phi_j
                                          * (old_displacement_multiplier_divs[q_point] * old_displacement_divs[q_point]
                                             * lambda_values[q_point]
                                             + 2 * mu_values[q_point]
                                               * (old_displacement_symmgrads[q_point] *
                                                  old_displacement_multiplier_symmgrads[q_point]))

                                        + density_penalty_exponent * std::pow(
                                                old_density_values[q_point],
                                                density_penalty_exponent - 1)
                                          * density_phi_i
                                          * (displacement_multiplier_phi_j_div * old_displacement_divs[q_point]
                                             * lambda_values[q_point]
                                             + 2 * mu_values[q_point]
                                               *
                                               (old_displacement_symmgrads[q_point] *
                                                displacement_multiplier_phi_j_symmgrad))

                                        + density_penalty_exponent * std::pow(
                                                old_density_values[q_point],
                                                density_penalty_exponent - 1)
                                          * density_phi_i
                                          * (displacement_phi_j_div * old_displacement_multiplier_divs[q_point]
                                             * lambda_values[q_point]
                                             + 2 * mu_values[q_point]
                                               * (old_displacement_multiplier_symmgrads[q_point] *
                                                  displacement_phi_j_symmgrad)));

                        //Equation 1

                        cell_matrix(i, j) +=
                                fe_values.JxW(q_point) * (
                                        density_penalty_exponent * std::pow(
                                                old_density_values[q_point],
                                                density_penalty_exponent - 1)
                                        * density_phi_j
                                        * (old_displacement_multiplier_divs[q_point] * displacement_phi_i_div
                                           * lambda_values[q_point]
                                           + 2 * mu_values[q_point]
                                             * (old_displacement_multiplier_symmgrads[q_point] *
                                                displacement_phi_i_symmgrad))

                                        + std::pow(old_density_values[q_point],
                                                   density_penalty_exponent)
                                          * (displacement_multiplier_phi_j_div * displacement_phi_i_div
                                             * lambda_values[q_point]
                                             + 2 * mu_values[q_point]
                                               * (displacement_multiplier_phi_j_symmgrad * displacement_phi_i_symmgrad))

                                );

                        //Equation 2 has to do with the filter, which is calculated elsewhere.
                        cell_matrix(i, j) +=
                                fe_values.JxW(q_point) * (
                                        -1 * unfiltered_density_phi_i * lower_slack_multiplier_phi_j
                                        + unfiltered_density_phi_i * upper_slack_multiplier_phi_j);


                        //Equation 3 - Primal Feasibility

                        cell_matrix(i, j) +=
                                fe_values.JxW(q_point) * (

                                        density_penalty_exponent * std::pow(
                                                old_density_values[q_point],
                                                density_penalty_exponent - 1)
                                        * density_phi_j
                                        * (old_displacement_divs[q_point] * displacement_multiplier_phi_i_div
                                           * lambda_values[q_point]
                                           + 2 * mu_values[q_point]
                                             * (old_displacement_symmgrads[q_point] *
                                                displacement_multiplier_phi_i_symmgrad))

                                        + std::pow(old_density_values[q_point],
                                                   density_penalty_exponent)
                                          * (displacement_phi_j_div * displacement_multiplier_phi_i_div
                                             * lambda_values[q_point]
                                             + 2 * mu_values[q_point]
                                               *
                                               (displacement_phi_j_symmgrad * displacement_multiplier_phi_i_symmgrad)));

                        //Equation 4 - more primal feasibility
                        cell_matrix(i, j) +=
                                -1 * fe_values.JxW(q_point) * lower_slack_multiplier_phi_i *
                                (unfiltered_density_phi_j - lower_slack_phi_j);

                        //Equation 5 - more primal feasibility
                        cell_matrix(i, j) +=
                                -1 * fe_values.JxW(q_point) * upper_slack_multiplier_phi_i * (
                                        -1 * unfiltered_density_phi_j - upper_slack_phi_j);

                        //Equation 6 - more primal feasibility - part with filter added later
                        cell_matrix(i, j) +=
                                -1 * fe_values.JxW(q_point) * unfiltered_density_multiplier_phi_i * (
                                        density_phi_j);

                        //Equation 7 - complementary slackness
                        cell_matrix(i, j) += fe_values.JxW(q_point)
                                             * (lower_slack_phi_i * lower_slack_multiplier_phi_j

                                                + lower_slack_phi_i * lower_slack_phi_j *
                                                  old_lower_slack_multiplier_values[q_point] /
                                                  old_lower_slack_values[q_point]);

                        //Equation 8 - complementary slackness
                        cell_matrix(i, j) += fe_values.JxW(q_point)
                                             * (upper_slack_phi_i * upper_slack_multiplier_phi_j


                                                + upper_slack_phi_i * upper_slack_phi_j
                                                  * old_upper_slack_multiplier_values[q_point] /
                                                  old_upper_slack_values[q_point]);

                    }

                    //rhs eqn 0
                    cell_rhs(i) +=
                            -1 * fe_values.JxW(q_point) * (
                                    density_penalty_exponent *
                                    std::pow(old_density_values[q_point], density_penalty_exponent - 1) * density_phi_i
                                    * (old_displacement_multiplier_divs[q_point] * old_displacement_divs[q_point]
                                       * lambda_values[q_point]
                                       + 2 * mu_values[q_point] * (old_displacement_symmgrads[q_point]
                                                                   * old_displacement_multiplier_symmgrads[q_point]))
                                    - density_phi_i * old_unfiltered_density_multiplier_values[q_point]);

                    //rhs eqn 1 - boundary terms counted later
                    cell_rhs(i) +=
                            -1 * fe_values.JxW(q_point) * (
                                    std::pow(old_density_values[q_point], density_penalty_exponent)
                                    * (old_displacement_multiplier_divs[q_point] * displacement_phi_i_div
                                       * lambda_values[q_point]
                                       + 2 * mu_values[q_point] * (old_displacement_multiplier_symmgrads[q_point]
                                                                   * displacement_phi_i_symmgrad))
                            );

                    //rhs eqn 2
                    cell_rhs(i) +=
                            -1 * fe_values.JxW(q_point) * (
                                    unfiltered_density_phi_i *
                                    filter_adjoint_unfiltered_density_multiplier_values[q_point]
                                    + unfiltered_density_phi_i * old_upper_slack_multiplier_values[q_point]
                                    + -1 * unfiltered_density_phi_i * old_lower_slack_multiplier_values[q_point]
                            );




                    //rhs eqn 3 - boundary terms counted later
                    cell_rhs(i) +=
                            -1 * fe_values.JxW(q_point) * (
                                    std::pow(old_density_values[q_point], density_penalty_exponent)
                                    * (old_displacement_divs[q_point] * displacement_multiplier_phi_i_div
                                       * lambda_values[q_point]
                                       + 2 * mu_values[q_point] * (displacement_multiplier_phi_i_symmgrad
                                                                   * old_displacement_symmgrads[q_point]))
                            );

                    //rhs eqn 4
                    cell_rhs(i) +=
                            fe_values.JxW(q_point) *
                            (lower_slack_multiplier_phi_i
                             * (old_unfiltered_density_values[q_point] - old_lower_slack_values[q_point])
                            );

                    //rhs eqn 5
                    cell_rhs(i) +=
                            fe_values.JxW(q_point) * (
                                    upper_slack_multiplier_phi_i
                                    * (1 - old_unfiltered_density_values[q_point]
                                       - old_upper_slack_values[q_point]));

                    //rhs eqn 6
                    cell_rhs(i) +=
                            fe_values.JxW(q_point) * (
                                    unfiltered_density_multiplier_phi_i
                                    * (old_density_values[q_point] - filtered_unfiltered_density_values[q_point])
                            );

                    //rhs eqn 7
                    cell_rhs(i) +=
                            -1 * fe_values.JxW(q_point) * (lower_slack_phi_i
                                                           * (old_lower_slack_multiplier_values[q_point] -
                                                              barrier_size / old_lower_slack_values[q_point])
                            );

                    //rhs eqn 8
                    cell_rhs(i) +=
                            -1 * fe_values.JxW(q_point) * (upper_slack_phi_i
                                                           * (old_upper_slack_multiplier_values[q_point] -
                                                              barrier_size / old_upper_slack_values[q_point]));

                }

            }
            for (unsigned int face_number = 0;
                 face_number < GeometryInfo<dim>::faces_per_cell;
                 ++face_number) {
                if (cell->face(face_number)->at_boundary() && cell->face(
                        face_number)->boundary_id()
                                                              == 1) {
                    fe_face_values.reinit(cell, face_number);

                    for (unsigned int face_q_point = 0;
                         face_q_point < n_face_q_points; ++face_q_point) {
                        for (unsigned int i = 0; i < dofs_per_cell; ++i) {
                            cell_rhs(i) += -1
                                           * traction
                                           * fe_face_values[displacements].value(i,
                                                                                 face_q_point)
                                           * fe_face_values.JxW(face_q_point);

                            cell_rhs(i) += traction
                                           * fe_face_values[displacement_multipliers].value(
                                    i, face_q_point)
                                           * fe_face_values.JxW(face_q_point);
                        }
                    }
                }
            }


            MatrixTools::local_apply_boundary_values(boundary_values, local_dof_indices,
                                                     cell_matrix, cell_rhs, true);

            constraints.distribute_local_to_global(
                    cell_matrix, cell_rhs, local_dof_indices, system_matrix, system_rhs);


        }


        for (const auto &cell : dof_handler.active_cell_iterators()) {
            unsigned int i = cell->active_cell_index();
            typename SparseMatrix<double>::iterator iter = filter_matrix.begin(
                    i);
            for (; iter != filter_matrix.end(i); iter++) {
                unsigned int j = iter->column();
                double value = iter->value() * cell->measure();

                system_matrix.block(4, 2).add(i, j, value);
                system_matrix.block(2, 4).add(j, i, value);
            }
        }
    }

    template<int dim>
    double
    KktSystem<dim>::calculate_objective_value(const BlockVector<double> &state, const double barrier_size) const
    {
            BlockVector<double> test_rhs = calculate_test_rhs(state, barrier_size);
            double objective_value;
            objective_value = test_rhs.block(0)*state.block(0);
            return objective_value;
    }

    template<int dim>
    double
    KktSystem<dim>::calculate_barrier_distance(const BlockVector<double> &state) const
    {
        double barrier_distance_log_sum = 0;
        unsigned int vect_size = state.block(2).size();
        for (unsigned int k=0; k < vect_size; k++)
        {
            barrier_distance_log_sum += std::log(state.block(2)[k]);
        }
        for (unsigned int k=0; k < vect_size; k++)
        {
            barrier_distance_log_sum += std::log(1 - state.block(2)[k]);
        }
        return barrier_distance_log_sum;
    }

    template<int dim>
    double
    KktSystem<dim>::calculate_feasibility(const BlockVector<double> &state, const double barrier_size) const
    {
        BlockVector<double> test_rhs = calculate_test_rhs(state, barrier_size);
        double feasibility = 0;
        feasibility += test_rhs.block(3).l2_norm() + test_rhs.block(4).l2_norm() + test_rhs.block(5).l2_norm()
                + test_rhs.block(6).l2_norm() + test_rhs.block(7).l2_norm() + test_rhs.block(8).l2_norm();
        return feasibility;
    }

template<int dim>
double
KktSystem<dim>::calculate_rhs_norm(const BlockVector<double> &state, const double barrier_size) const
{
    BlockVector<double> test_rhs = calculate_test_rhs(state, barrier_size);

    return test_rhs.l2_norm();
}

     template<int dim>
    BlockVector<double>
    KktSystem<dim>::calculate_test_rhs(const BlockVector<double> &test_state, const double barrier_size) const
    {
        const FEValuesExtractors::Scalar densities(0);
        const FEValuesExtractors::Vector displacements(1);
        const FEValuesExtractors::Scalar unfiltered_densities(1 + dim);
        const FEValuesExtractors::Vector displacement_multipliers(2 + dim);
        const FEValuesExtractors::Scalar unfiltered_density_multipliers(2 + 2 * dim);
        const FEValuesExtractors::Scalar density_lower_slacks(3 + 2 * dim);
        const FEValuesExtractors::Scalar density_lower_slack_multipliers(
                4 + 2 * dim);
        const FEValuesExtractors::Scalar density_upper_slacks(5 + 2 * dim);
        const FEValuesExtractors::Scalar density_upper_slack_multipliers(
                6 + 2 * dim);

        /*Remove any values from old iterations*/

        BlockVector<double> test_rhs;
        test_rhs = system_rhs;
        test_rhs = 0;

        const QGauss<dim> quadrature_formula(fe.degree + 1);
        const QGauss<dim - 1> face_quadrature_formula(fe.degree + 1);
        FEValues<dim> fe_values(fe, quadrature_formula,
                                update_values | update_gradients | update_quadrature_points
                                | update_JxW_values);
        FEFaceValues<dim> fe_face_values(fe, face_quadrature_formula,
                                         update_values | update_quadrature_points | update_normal_vectors
                                         | update_JxW_values);

        const unsigned int dofs_per_cell = fe.dofs_per_cell;
        const unsigned int n_q_points = quadrature_formula.size();
        const unsigned int n_face_q_points = face_quadrature_formula.size();

        Vector<double> cell_rhs(dofs_per_cell);
        FullMatrix<double> dummy_cell_matrix(dofs_per_cell,dofs_per_cell);

        std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);

        std::vector<double> lambda_values(n_q_points);
        std::vector<double> mu_values(n_q_points);

        const Functions::ConstantFunction<dim> lambda(1.), mu(1.);
        std::vector<Tensor<1, dim>> rhs_values(n_q_points);

        BlockVector<double> filtered_unfiltered_density_solution = test_state;
        BlockVector<double> filter_adjoint_unfiltered_density_multiplier_solution = test_state;
        filtered_unfiltered_density_solution.block(2) = 0;
        filter_adjoint_unfiltered_density_multiplier_solution.block(4) = 0;

        filter_matrix.vmult(filtered_unfiltered_density_solution.block(2), test_state.block(2));
        filter_matrix.Tvmult(filter_adjoint_unfiltered_density_multiplier_solution.block(4),
                             test_state.block(4));


        std::vector<double> old_density_values(n_q_points);
        std::vector<Tensor<1, dim>> old_displacement_values(n_q_points);
        std::vector<double> old_displacement_divs(n_q_points);
        std::vector<SymmetricTensor<2, dim>> old_displacement_symmgrads(
          n_q_points);
        std::vector<Tensor<1, dim>> old_displacement_multiplier_values(
          n_q_points);
        std::vector<double> old_displacement_multiplier_divs(n_q_points);
        std::vector<SymmetricTensor<2, dim>> old_displacement_multiplier_symmgrads(
          n_q_points);
        std::vector<double> old_lower_slack_multiplier_values(n_q_points);
        std::vector<double> old_upper_slack_multiplier_values(n_q_points);
        std::vector<double> old_lower_slack_values(n_q_points);
        std::vector<double> old_upper_slack_values(n_q_points);
        std::vector<double> old_unfiltered_density_values(n_q_points);
        std::vector<double> old_unfiltered_density_multiplier_values(n_q_points);
        std::vector<double> filtered_unfiltered_density_values(n_q_points);
        std::vector<double> filter_adjoint_unfiltered_density_multiplier_values(n_q_points);

        for (const auto &cell : dof_handler.active_cell_iterators()) {
            cell_rhs = 0;

            cell->get_dof_indices(local_dof_indices);

            fe_values.reinit(cell);

            lambda.value_list(fe_values.get_quadrature_points(), lambda_values);
            mu.value_list(fe_values.get_quadrature_points(), mu_values);

            fe_values[densities].get_function_values(test_state,
                                                     old_density_values);
            fe_values[displacements].get_function_values(test_state,
                                                         old_displacement_values);
            fe_values[displacements].get_function_divergences(test_state,
                                                              old_displacement_divs);
            fe_values[displacements].get_function_symmetric_gradients(
                    test_state, old_displacement_symmgrads);
            fe_values[displacement_multipliers].get_function_values(
                    test_state, old_displacement_multiplier_values);
            fe_values[displacement_multipliers].get_function_divergences(
                    test_state, old_displacement_multiplier_divs);
            fe_values[displacement_multipliers].get_function_symmetric_gradients(
                    test_state, old_displacement_multiplier_symmgrads);
            fe_values[density_lower_slacks].get_function_values(
                    test_state, old_lower_slack_values);
            fe_values[density_lower_slack_multipliers].get_function_values(
                    test_state, old_lower_slack_multiplier_values);
            fe_values[density_upper_slacks].get_function_values(
                    test_state, old_upper_slack_values);
            fe_values[density_upper_slack_multipliers].get_function_values(
                    test_state, old_upper_slack_multiplier_values);
            fe_values[unfiltered_densities].get_function_values(
                    test_state, old_unfiltered_density_values);
            fe_values[unfiltered_density_multipliers].get_function_values(
                    test_state, old_unfiltered_density_multiplier_values);
            fe_values[unfiltered_densities].get_function_values(
                    filtered_unfiltered_density_solution, filtered_unfiltered_density_values);
            fe_values[unfiltered_density_multipliers].get_function_values(
                    filter_adjoint_unfiltered_density_multiplier_solution,
                    filter_adjoint_unfiltered_density_multiplier_values);

            Tensor<1, dim> traction;
            traction[1] = -1;

            for (unsigned int q_point = 0; q_point < n_q_points; ++q_point) {

                for (unsigned int i = 0; i < dofs_per_cell; ++i) {
                    const SymmetricTensor<2, dim> displacement_phi_i_symmgrad =
                            fe_values[displacements].symmetric_gradient(i, q_point);
                    const double displacement_phi_i_div =
                            fe_values[displacements].divergence(i, q_point);

                    const SymmetricTensor<2, dim> displacement_multiplier_phi_i_symmgrad =
                            fe_values[displacement_multipliers].symmetric_gradient(i,
                                                                                   q_point);
                    const double displacement_multiplier_phi_i_div =
                            fe_values[displacement_multipliers].divergence(i,
                                                                           q_point);


                    const double density_phi_i = fe_values[densities].value(i,
                                                                            q_point);
                    const double unfiltered_density_phi_i = fe_values[unfiltered_densities].value(i,
                                                                                                  q_point);
                    const double unfiltered_density_multiplier_phi_i = fe_values[unfiltered_density_multipliers].value(
                            i, q_point);

                    const double lower_slack_multiplier_phi_i =
                            fe_values[density_lower_slack_multipliers].value(i,
                                                                             q_point);

                    const double lower_slack_phi_i =
                            fe_values[density_lower_slacks].value(i, q_point);

                    const double upper_slack_phi_i =
                            fe_values[density_upper_slacks].value(i, q_point);

                    const double upper_slack_multiplier_phi_i =
                            fe_values[density_upper_slack_multipliers].value(i,
                                                                             q_point);

                    //rhs eqn 0
                    cell_rhs(i) +=
                            -1 * fe_values.JxW(q_point) * (
                                    density_penalty_exponent *
                                    std::pow(old_density_values[q_point], density_penalty_exponent - 1) * density_phi_i
                                    * (old_displacement_multiplier_divs[q_point] * old_displacement_divs[q_point]
                                       * lambda_values[q_point]
                                       + 2 * mu_values[q_point] * (old_displacement_symmgrads[q_point]
                                                                   * old_displacement_multiplier_symmgrads[q_point]))
                                    - density_phi_i * old_unfiltered_density_multiplier_values[q_point]);

                    //rhs eqn 1 - boundary terms counted later
                    cell_rhs(i) +=
                            -1 * fe_values.JxW(q_point) * (
                                    std::pow(old_density_values[q_point], density_penalty_exponent)
                                    * (old_displacement_multiplier_divs[q_point] * displacement_phi_i_div
                                       * lambda_values[q_point]
                                       + 2 * mu_values[q_point] * (old_displacement_multiplier_symmgrads[q_point]
                                                                   * displacement_phi_i_symmgrad))
                            );

                    //rhs eqn 2
                    cell_rhs(i) +=
                            -1 * fe_values.JxW(q_point) * (
                                    unfiltered_density_phi_i *
                                    filter_adjoint_unfiltered_density_multiplier_values[q_point]
                                    + unfiltered_density_phi_i * old_upper_slack_multiplier_values[q_point]
                                    + -1 * unfiltered_density_phi_i * old_lower_slack_multiplier_values[q_point]
                            );




                    //rhs eqn 3 - boundary terms counted later
                    cell_rhs(i) +=
                            -1 * fe_values.JxW(q_point) * (
                                    std::pow(old_density_values[q_point], density_penalty_exponent)
                                    * (old_displacement_divs[q_point] * displacement_multiplier_phi_i_div
                                       * lambda_values[q_point]
                                       + 2 * mu_values[q_point] * (displacement_multiplier_phi_i_symmgrad
                                                                   * old_displacement_symmgrads[q_point]))
                            );

                    //rhs eqn 4
                    cell_rhs(i) +=
                            fe_values.JxW(q_point) *
                            (lower_slack_multiplier_phi_i
                             * (old_unfiltered_density_values[q_point] - old_lower_slack_values[q_point])
                            );

                    //rhs eqn 5
                    cell_rhs(i) +=
                            fe_values.JxW(q_point) * (
                                    upper_slack_multiplier_phi_i * (1 - old_unfiltered_density_values[q_point]
                                       - old_upper_slack_values[q_point]));

                    //rhs eqn 6
                    cell_rhs(i) +=
                            fe_values.JxW(q_point) * (
                                    unfiltered_density_multiplier_phi_i
                                    * (old_density_values[q_point] - filtered_unfiltered_density_values[q_point])
                            );

                    //rhs eqn 7
                    cell_rhs(i) +=
                            -1 * fe_values.JxW(q_point) * (lower_slack_phi_i
                                                           * (old_lower_slack_multiplier_values[q_point] -
                                                              barrier_size / old_lower_slack_values[q_point])
                            );

                    //rhs eqn 8
                    cell_rhs(i) +=
                            -1 * fe_values.JxW(q_point) * (upper_slack_phi_i
                                                           * (old_upper_slack_multiplier_values[q_point] -
                                                              barrier_size / old_upper_slack_values[q_point]));

                }


            }

            for (unsigned int face_number = 0;
                 face_number < GeometryInfo<dim>::faces_per_cell;
                 ++face_number) {
                if (cell->face(face_number)->at_boundary() && cell->face(
                        face_number)->boundary_id()
                                                              == 1) {
                    fe_face_values.reinit(cell, face_number);

                    for (unsigned int face_q_point = 0;
                         face_q_point < n_face_q_points; ++face_q_point) {
                        for (unsigned int i = 0; i < dofs_per_cell; ++i) {
                            cell_rhs(i) += -1
                                           * traction
                                           * fe_face_values[displacements].value(i,
                                                                                 face_q_point)
                                           * fe_face_values.JxW(face_q_point);

                            cell_rhs(i) += traction
                                           * fe_face_values[displacement_multipliers].value(
                                    i, face_q_point)
                                           * fe_face_values.JxW(face_q_point);
                        }
                    }
                }
            }

            MatrixTools::local_apply_boundary_values(boundary_values, local_dof_indices,
                                                     dummy_cell_matrix, cell_rhs, true);

            constraints.distribute_local_to_global(
                    cell_rhs, local_dof_indices, test_rhs);


        }
        return test_rhs;

    }


    ///A direct solver, for now. The complexity of the system means that an iterative solver algorithm will take some more work in the future.
    template <int dim>
    BlockVector<double>
    KktSystem<dim>::solve() {
        //This broke everything. Unsure why.
//      constraints.condense(system_matrix);
//      constraints.condense(system_rhs);

        SparseDirectUMFPACK A_direct;
        A_direct.initialize(system_matrix);
        A_direct.vmult(linear_solution, system_rhs);
        constraints.distribute(linear_solution);
        return linear_solution;
    }

///A direct solver, for now. The complexity of the system means that an iterative solver algorithm will take some more work in the future.
template<int dim>
BlockVector<double>
KktSystem<dim>::get_initial_state() {

    std::vector<unsigned int> block_component(9, 2);
    block_component[0] = 0;
    block_component[1] = 1;
    const std::vector<types::global_dof_index> dofs_per_block =
            DoFTools::count_dofs_per_fe_block(dof_handler, block_component);
    const unsigned int n_p = dofs_per_block[0];
    const unsigned int n_u = dofs_per_block[1];
    std::cout << n_p << "  " << n_u << std::endl;
    const std::vector<unsigned int> block_sizes = {n_p, n_u, n_p, n_u, n_p, n_p, n_p, n_p, n_p};

    BlockVector<double> state(9);

    for (unsigned int j = 0; j < 9; j++) {
            state.block(j).reinit(block_sizes[j]);
        }

    state.collect_sizes();

    for (unsigned int k = 0; k < n_u; k++) {
        state.block(1)[k] = 0;
    }
    for (unsigned int k = 0; k < n_p; k++) {
        state.block(0)[k] = density_ratio;
        state.block(2)[k] = density_ratio;
        state.block(4)[k] = density_ratio;
        state.block(5)[k] = density_ratio;
        state.block(6)[k] = 50;
        state.block(7)[k] = 1 - density_ratio;
        state.block(8)[k] = 50;
    }

    return state;

}

template<int dim>
void
KktSystem<dim>::output(const BlockVector<double> &state, const unsigned int j) const {
    std::vector<std::string> solution_names(1, "density");
    std::vector<DataComponentInterpretation::DataComponentInterpretation> data_component_interpretation(
            1, DataComponentInterpretation::component_is_scalar);
    for (unsigned int i = 0; i < dim; i++) {
        solution_names.emplace_back("displacement");
        data_component_interpretation.push_back(
                DataComponentInterpretation::component_is_part_of_vector);
    }
    solution_names.emplace_back("unfiltered_density");
    data_component_interpretation.push_back(
            DataComponentInterpretation::component_is_scalar);
    for (unsigned int i = 0; i < dim; i++) {
        solution_names.emplace_back("displacement_multiplier");
        data_component_interpretation.push_back(
                DataComponentInterpretation::component_is_part_of_vector);
    }
    solution_names.emplace_back("unfiltered_density_multiplier");
    data_component_interpretation.push_back(
            DataComponentInterpretation::component_is_scalar);
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
    DataOut<dim> data_out;
    data_out.attach_dof_handler(dof_handler);
    data_out.add_data_vector(state, solution_names,
                             DataOut<dim>::type_dof_data, data_component_interpretation);
//      data_out.add_data_vector (linear_solution, solution_names,
//          DataOut<dim>::type_dof_data, data_component_interpretation);
    data_out.build_patches();
    std::ofstream output("solution" + std::to_string(j) + ".vtk");
    data_out.write_vtk(output);
}

template class KktSystem<2>;
template class KktSystem<3>;