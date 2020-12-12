#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/function.h>
#include <deal.II/base/tensor.h>

#include <deal.II/lac/block_vector.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/block_sparse_matrix.h>
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

namespace SAND {
    using namespace dealii;

    template<int dim>
    class SANDTopOpt {
    public:
        SANDTopOpt();

        void
        run();

    private:
        //Sets up sparsity patterns with constraints and everything
        void
        setup_block_system();

        //Fills in sparse matrix and rhs
        void
        assemble_block_system(double barrier_size);

        //Creates a mesh for a 6-by-1 rectangle and identifies those points that have 0 Dirichlet boundary. As these are nodes, this cannot be done through BCIDs.
        void
        create_triangulation();

        //Really does the Neumann BCs as these take up edges
        void
        set_bcids();

        //Right now I solve with a direct solver - want to look into better versions in the future
        void
        solve();

        //Finds biggest steps for both primal and dual parts of the problem to maintain feasibility
        std::vector<double>
        calculate_maximum_step_size();

        //Checks the l2 norm of the rhs - keeps step if it went down, backsteps if not.
        void
        update_step(std::vector<double> maximum_step_size, double barrier_size, double penalty_parameter);

        //outputs all data in vtk file format
        void
        output(int j);

        //Makes the filter matrix H so that H*unfiltered_density = filtered_density
        void
        setup_filter_matrix();

        //Makes an "exact merit function"
        double
        calculate_solution_merit(BlockVector<double> test_solution, double barrier_size, double penalty_parameter);

        //Calculates the next iteration rhs prematurely - we want this to be 0, so it works out better than the exact one for now.
        double
        calculate_solution_rhs_merit(BlockVector<double> test_solution, double barrier_size, double penalty_parameter);

        //This is used in the  "calculate_solution_merit" function to get one part of it.
        double
        get_compliance_plus_elasticity_error(BlockVector<double> test_solution, double penalty_parameter);

        BlockSparsityPattern sparsity_pattern;
        BlockSparseMatrix<double> system_matrix;
        SparsityPattern filter_sparsity_pattern;
        SparseMatrix<double> filter_matrix;
        BlockVector<double> linear_solution;
        BlockVector<double> system_rhs;
        BlockVector<double> nonlinear_solution;
        Triangulation<dim> triangulation;
        DoFHandler<dim> dof_handler;
        AffineConstraints<double> constraints;
        FESystem<dim> fe;
        double density_ratio;
        double density_penalty_exponent;
        double filter_r;
        std::map<types::global_dof_index, double> boundary_values;
    };

    ///Constructor
    template<int dim>
    SANDTopOpt<dim>::SANDTopOpt()
            :
            dof_handler(triangulation),
            /*fe should have 1 FE_DGQ<dim>(0) element for density, dim FE_Q finite elements for displacement,
             * another FE_DGQ<dim>(0) element for unfiltered density, another dim FE_Q elements for the lagrange multiplier on the displacement constraint,
             * and 5 more FE_DGQ<dim>(0) elements for the upper and lower bound slacks, multipliers, and the  filter multiplier*/
            fe(FE_DGQ<dim>(0) ^ 1,
               (FESystem<dim>(FE_Q<dim>(1) ^ dim)) ^ 1,
               FE_DGQ<dim>(0) ^ 1,
               (FESystem<dim>(FE_Q<dim>(1) ^ dim)) ^ 1,
               FE_DGQ<dim>(0) ^ 5) {

    }

    /** This sets up and assembles the filter matrix H, which is made so that H* unfiltered_density_vector = filtered_density_vector*/
    template<int dim>
    void
    SANDTopOpt<dim>::setup_filter_matrix() {


        DynamicSparsityPattern filter_dsp;
        filter_dsp.reinit(dof_handler.get_triangulation().n_active_cells(),
                          dof_handler.get_triangulation().n_active_cells());
        std::set<unsigned int> neighbor_ids;
        std::set<typename Triangulation<dim>::cell_iterator> cells_to_check;
        std::set<typename Triangulation<dim>::cell_iterator> cells_to_check_temp;
        unsigned int n_neighbors, i;
        double distance;

        //iterates over all cells
        for (const auto &cell : dof_handler.active_cell_iterators()) {
            i = cell->active_cell_index();
            neighbor_ids.clear();
            neighbor_ids.insert(i);
            cells_to_check.clear();
            //cells_to_check is a set of cell IDs that are on the edge of the region we have searched
            cells_to_check.insert(cell);
            n_neighbors = 1;
            while (true) {
                //cells_to_check_temp is a set that holds the cell IDs for the next iteration of this search
                cells_to_check_temp.clear();
                //iterates over all cells on edge of region
                for (auto check_cell : cells_to_check) {
                    //Iterates over all the boundaries
                    for (unsigned int n = 0;
                         n < GeometryInfo<dim>::faces_per_cell; ++n) {
                        //This makes sure we don't look for cells that aren't there - stays away from the boundary of the domain.
                        if (!(check_cell->face(n)->at_boundary())) {
                            //finds distance between this new cell and the edge
                            distance = cell->center().distance(
                                    check_cell->neighbor(n)->center());
                            if ((distance < filter_r) &&
                                !(neighbor_ids.count(check_cell->neighbor(n)->active_cell_index()))) {
                                /* if the new cell is not already in the list and it is within the radius,
                                * we add it to the list to check the next iteration (as it is on the edge now))
                                * and the list of neighbors */
                                cells_to_check_temp.insert(check_cell->neighbor(n));
                                neighbor_ids.insert(check_cell->neighbor(n)->active_cell_index());
                            }
                        }
                    }
                }
                /*Once we stop adding neighbors to the list of ids, the size of the neighbor list doesn't change,
                 and so all the neighbors in the radius have been found.*/
                if (neighbor_ids.size() == n_neighbors) {
                    break;
                } else {
                    cells_to_check = cells_to_check_temp;
                    n_neighbors = neighbor_ids.size();
                }
            }
//add all of these to the sparsity pattern
            for (auto j : neighbor_ids) {
                filter_dsp.add(i, j);
            }
        }

        filter_sparsity_pattern.copy_from(filter_dsp);
        filter_matrix.reinit(filter_sparsity_pattern);

//find these cells again to add values to matrix - algorithm works the same way as before.
        for (const auto &cell : dof_handler.active_cell_iterators()) {
            i = cell->active_cell_index();
            neighbor_ids.clear();
            neighbor_ids.insert(i);
            cells_to_check.clear();
            cells_to_check.insert(cell);
            cells_to_check_temp.clear();
            n_neighbors = 1;
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
                                //value should be max radius - distance between cells
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
        /*After we have added all of these values to the matrix,
        * we normalize it by dividing each row by the sum of the values in the row.*/
        for (const auto &cell : dof_handler.active_cell_iterators()) {
            i = cell->active_cell_index();
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
        std::cout << "filled in filter matrix" << std::endl;
    }


    /** This creates the triangulation we will be working on , which is a 6-by-1 rectangle. It refines it, and
     * identifies the external forces as Neumann Boundary Condition*/
    template<int dim>
    void
    SANDTopOpt<dim>::create_triangulation() {
        /*Make a square*/
        Triangulation<dim> triangulation_temp;
        Point<dim> point_1, point_2;
        point_1(0) = 0;
        point_1(1) = 0;
        point_2(0) = 1;
        point_2(1) = 1;
        GridGenerator::hyper_rectangle(triangulation, point_1, point_2);

        /*make 5 more squares*/
        for (int n = 1; n < 6; n++) {
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
                    if (std::fabs(center(1) - 1) < 1e-12)
                    {
                        if (std::fabs(center(0) - 3) < .25)
                        {
                            cell->face(face_number)->set_boundary_id(1);
                        }
                        else
                        {
                            cell->face(face_number)->set_boundary_id(2);
                        }

                        cell->face(face_number)->set_boundary_id(2);
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


    /** Because the Dirichlet BCs occur at points, and not along edges, a little more work has to be done to make the
     * Dirichlet BCs work out. That is done here. */
    template<int dim>
    void
    SANDTopOpt<dim>::set_bcids() {
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
                                    vert(1)- 0) < 1e-12) {
                                const unsigned int y_displacement =
                                        cell->vertex_dof_index(vertex_number, 1);
                                const unsigned int y_displacement_multiplier =
                                        cell->vertex_dof_index(vertex_number, 3);
                                const unsigned int x_displacement =
                                        cell->vertex_dof_index(vertex_number, 0);
                                const unsigned int x_displacement_multiplier =
                                        cell->vertex_dof_index(vertex_number, 2);
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

    /**This function creates the sparsity pattern for the large block matrix, and initializes the block vectors.
     * This also takes care of creating the constraint on the total volume, as that effects the sparsity pattern. */
    template<int dim>
    void
    SANDTopOpt<dim>::setup_block_system() {
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
        std::vector<unsigned int> block_sizes = {n_p, n_u, n_p, n_u, n_p, n_p, n_p, n_p, n_p};

        BlockDynamicSparsityPattern dsp(9, 9);

        //Assign each block the appropriate size
        for (int k = 0; k < 9; k++) {
            for (int j = 0; j < 9; j++) {
                dsp.block(j, k).reinit(block_sizes[j], block_sizes[k]);
            }
        }

        dsp.collect_sizes();

        Table<2, DoFTools::Coupling> coupling(2 * dim + 7, 2 * dim + 7);

        coupling[0][0] = DoFTools::always;

        for (int i = 0; i < dim; i++) {
            coupling[0][1 + i] = DoFTools::always;
            coupling[1 + i][0] = DoFTools::always;
        }

        coupling[0][1 + dim] = DoFTools::always;
        coupling[1 + dim][0] = DoFTools::always;

        for (int i = 0; i < dim; i++) {
            coupling[0][2 + dim + i] = DoFTools::always;
            coupling[2 + dim + i][0] = DoFTools::always;
        }

        coupling[0][2 + 2 * dim] = DoFTools::always;
        coupling[2 + 2 * dim][0] = DoFTools::always;

        for (int i = 0; i < dim; i++) {


            for (int k = 0; k < dim; k++) {
                coupling[1 + i][2 + dim + k] = DoFTools::always;
                coupling[2 + dim + k][1 + i] = DoFTools::always;
            }
        }


        constraints.clear();

        ComponentMask density_mask = fe.component_mask(densities);

        IndexSet density_dofs = DoFTools::extract_dofs(dof_handler,
                                                       density_mask);


        const unsigned int last_density_dof = density_dofs.nth_index_in_set(density_dofs.n_elements() - 1);
        constraints.add_line(last_density_dof);
        for (unsigned int i = 1;
             i < density_dofs.n_elements(); ++i) {
            constraints.add_entry(last_density_dof,
                                  density_dofs.nth_index_in_set(i - 1), -1);
        }

        constraints.close();

//      DoFTools::make_sparsity_pattern (dof_handler, dsp, constraints,
//          false);
//changed it to below - works now? Even when I just put the coupling in something breaks - want to fix this when I'm trying to speed stuff up.
        DoFTools::make_sparsity_pattern(dof_handler, coupling, dsp, constraints);

        std::set<unsigned int> neighbor_ids;
        std::set<typename Triangulation<dim>::cell_iterator> cells_to_check;
        std::set<typename Triangulation<dim>::cell_iterator> cells_to_check_temp;
        unsigned int n_neighbors, i;
        double distance;

        //I also add the filter sparsity pattern to where the filter belongs in the block matrix.
        for (const auto &cell : dof_handler.active_cell_iterators()) {
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

        constraints.condense(dsp);
        sparsity_pattern.copy_from(dsp);

        std::ofstream out("sparsity.plt");
        sparsity_pattern.print_gnuplot(out);

        system_matrix.reinit(sparsity_pattern);


        //The linear solution is the solution at each iteration. The nonlinear solution is my current vector values.
        //These and the system_rhs all have the same block structure so I handle them all at the same time.
        linear_solution.reinit(9);
        nonlinear_solution.reinit(9);
        system_rhs.reinit(9);

        for (int j = 0; j < 9; j++) {
            linear_solution.block(j).reinit(block_sizes[j]);
            nonlinear_solution.block(j).reinit(block_sizes[j]);
            system_rhs.block(j).reinit(block_sizes[j]);
        }

        linear_solution.collect_sizes();
        nonlinear_solution.collect_sizes();
        system_rhs.collect_sizes();

        density_ratio = .5;

        for (unsigned int k = 0; k < n_u; k++) {
            //I start the displacements as 0.
            nonlinear_solution.block(1)[k] = 0;
            nonlinear_solution.block(3)[k] = 0;
        }
        for (unsigned int k = 0; k < n_p; k++) {
            //I start all the unfiltered and filtered density at their average value, and give the multipliers the same value, because what else am I gonna make them?
            nonlinear_solution.block(0)[i] = density_ratio;
            nonlinear_solution.block(2)[i] = density_ratio;
            nonlinear_solution.block(4)[i] = density_ratio;
            nonlinear_solution.block(5)[i] = density_ratio;
            //Slack multipliers chosen so that they multiply with slack values to get barrier size.
            nonlinear_solution.block(6)[i] = 50;
            //Upper slack chosen to be distance from upper bound, which is 1.
            nonlinear_solution.block(7)[i] = 1 - density_ratio;
            nonlinear_solution.block(8)[i] = 50;
        }

    }


    /** Assembles a linear system to solve a newton iteration of the current subproblem. The subproblem is determined
     * by the KKT conditions for optimality, and consists of 8 equations*/
    template<int dim>
    void
    SANDTopOpt<dim>::assemble_block_system(double barrier_size) {
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

        Functions::ConstantFunction<dim> lambda(1.), mu(1.);
        std::vector<Tensor<1, dim>> rhs_values(n_q_points);

        BlockVector<double> filtered_unfiltered_density_solution = nonlinear_solution;
        BlockVector<double> filter_adjoint_unfiltered_density_multiplier_solution = nonlinear_solution;
        filtered_unfiltered_density_solution.block(2) = 0;
        filter_adjoint_unfiltered_density_multiplier_solution.block(4) = 0;

        filter_matrix.vmult(filtered_unfiltered_density_solution.block(2), nonlinear_solution.block(2));
        filter_matrix.Tvmult(filter_adjoint_unfiltered_density_multiplier_solution.block(4),
                             nonlinear_solution.block(4));


        for (const auto &cell : dof_handler.active_cell_iterators()) {
            cell_matrix = 0;
            full_density_cell_matrix = 0;
            full_density_cell_matrix_for_Au = 0;
            cell_rhs = 0;

            cell->get_dof_indices(local_dof_indices);

            fe_values.reinit(cell);

            lambda.value_list(fe_values.get_quadrature_points(), lambda_values);
            mu.value_list(fe_values.get_quadrature_points(), mu_values);

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


            fe_values[densities].get_function_values(nonlinear_solution,
                                                     old_density_values);
            fe_values[displacements].get_function_values(nonlinear_solution,
                                                         old_displacement_values);
            fe_values[displacements].get_function_divergences(nonlinear_solution,
                                                              old_displacement_divs);
            fe_values[displacements].get_function_symmetric_gradients(
                    nonlinear_solution, old_displacement_symmgrads);
            fe_values[displacement_multipliers].get_function_values(
                    nonlinear_solution, old_displacement_multiplier_values);
            fe_values[displacement_multipliers].get_function_divergences(
                    nonlinear_solution, old_displacement_multiplier_divs);
            fe_values[displacement_multipliers].get_function_symmetric_gradients(
                    nonlinear_solution, old_displacement_multiplier_symmgrads);
            fe_values[density_lower_slacks].get_function_values(
                    nonlinear_solution, old_lower_slack_values);
            fe_values[density_lower_slack_multipliers].get_function_values(
                    nonlinear_solution, old_lower_slack_multiplier_values);
            fe_values[density_upper_slacks].get_function_values(
                    nonlinear_solution, old_upper_slack_values);
            fe_values[density_upper_slack_multipliers].get_function_values(
                    nonlinear_solution, old_upper_slack_multiplier_values);
            fe_values[unfiltered_densities].get_function_values(
                    nonlinear_solution, old_unfiltered_density_values);
            fe_values[unfiltered_density_multipliers].get_function_values(
                    nonlinear_solution, old_unfiltered_density_multiplier_values);
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

                    //rhs of eqn 1 - boundary terms counted later
                    cell_rhs(i) +=
                            -1 * fe_values.JxW(q_point) * (
                                    std::pow(old_density_values[q_point], density_penalty_exponent)
                                    * (old_displacement_multiplier_divs[q_point] * displacement_phi_i_div
                                       * lambda_values[q_point]
                                       + 2 * mu_values[q_point] * (old_displacement_multiplier_symmgrads[q_point]
                                                                   * displacement_phi_i_symmgrad))
                            );

                    //rhs of eqn 2
                    cell_rhs(i) +=
                            -1 * fe_values.JxW(q_point) * (
                                    unfiltered_density_phi_i *
                                    filter_adjoint_unfiltered_density_multiplier_values[q_point]
                                    + unfiltered_density_phi_i * old_upper_slack_multiplier_values[q_point]
                                    + -1 * unfiltered_density_phi_i * old_lower_slack_multiplier_values[q_point]
                            );




                    //rhs of eqn 3 - boundary terms counted later
                    cell_rhs(i) +=
                            -1 * fe_values.JxW(q_point) * (
                                    std::pow(old_density_values[q_point], density_penalty_exponent)
                                    * (old_displacement_divs[q_point] * displacement_multiplier_phi_i_div
                                       * lambda_values[q_point]
                                       + 2 * mu_values[q_point] * (displacement_multiplier_phi_i_symmgrad
                                                                   * old_displacement_symmgrads[q_point]))
                            );

                    //rhs of eqn 4
                    cell_rhs(i) +=
                            fe_values.JxW(q_point) *
                            (lower_slack_multiplier_phi_i
                             * (old_unfiltered_density_values[q_point] - old_lower_slack_values[q_point])
                            );

                    //rhs of eqn 5
                    cell_rhs(i) +=
                            fe_values.JxW(q_point) * (
                                    upper_slack_multiplier_phi_i
                                    * (1 - old_unfiltered_density_values[q_point]
                                       - old_upper_slack_values[q_point]));

                    //rhs of eqn 6
                    cell_rhs(i) +=
                            fe_values.JxW(q_point) * (
                                    unfiltered_density_multiplier_phi_i
                                    * (old_density_values[q_point] - filtered_unfiltered_density_values[q_point])
                            );

                    //rhs of eqn 7
                    cell_rhs(i) +=
                            -1 * fe_values.JxW(q_point) * (lower_slack_phi_i
                                                           * (old_lower_slack_multiplier_values[q_point] -
                                                              barrier_size / old_lower_slack_values[q_point])
                            );

                    //rhs of eqn 8
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


    /**Right now, this uses a direct solver to solve each subproblem. As I begin working on making this faster,
     * will have to include a preconditioning step as well as  using an iterative solver.*/
    template<int dim>
    void
    SANDTopOpt<dim>::solve() {
        //This broke everything. Unsure why.
//      constraints.condense(system_matrix);
//      constraints.condense(system_rhs);

        SparseDirectUMFPACK A_direct;
        A_direct.initialize(system_matrix);
        A_direct.vmult(linear_solution, system_rhs);

        constraints.distribute(linear_solution);

    }

    /**This figures out the maximum feasible step size (up to 1) for both the primal and dual parts of the problem using a binary search method.
     * It only needs to check that the slack variables and the lagrange multipliers corresponding to those variables stay positive*/
    template<int dim>
    std::vector<double>
    SANDTopOpt<dim>::calculate_maximum_step_size() {
        double fraction_to_boundary = .995;

        double step_size_s_low = 0;
        double step_size_z_low = 0;
        double step_size_s_high = 1;
        double step_size_z_high = 1;
        double step_size_s, step_size_z;
        bool accept_s, accept_z;

        for (unsigned int k = 0; k < 50; k++) {
            step_size_s = (step_size_s_low + step_size_s_high) / 2;
            step_size_z = (step_size_z_low + step_size_z_high) / 2;

            BlockVector<double> nonlinear_solution_test_s =
                    (fraction_to_boundary * nonlinear_solution) + (step_size_s
                                                                   * linear_solution);

            BlockVector<double> nonlinear_solution_test_z =
                    (fraction_to_boundary * nonlinear_solution) + (step_size_z
                                                                   * linear_solution);

            accept_s = (nonlinear_solution_test_s.block(5).is_non_negative())
                       && (nonlinear_solution_test_s.block(7).is_non_negative());
            accept_z = (nonlinear_solution_test_z.block(6).is_non_negative())
                       && (nonlinear_solution_test_z.block(8).is_non_negative());

            if (accept_s) {
                step_size_s_low = step_size_s;
            } else {
                step_size_s_high = step_size_s;
            }
            if (accept_z) {
                step_size_z_low = step_size_z;
            } else {
                step_size_z_high = step_size_z;
            }
        }
        std::cout << step_size_s_low << "    " << step_size_z_low << std::endl;
        return {step_size_s_low,step_size_z_low};
            }

    /**This checks to see if the maximal feasible step size in fact decreases the value of our merit function.
     * If so, it updates the solution values accordingly. If not, it backsteps until it does.
     * Right now, the l_2 norm of the RHS is used as a (very bad) merit function */
    template<int dim>
    void
    SANDTopOpt<dim>::update_step(std::vector<double> maximum_step_size, double barrier_size, double penalty_parameter)
    {

        BlockVector<double> test_solution;
        test_solution.reinit(nonlinear_solution.n_blocks());
        for (unsigned int k = 0; k < nonlinear_solution.n_blocks(); k++)
        {
            test_solution.block(k).reinit(nonlinear_solution.block(k).size());
        }
        double current_merit = calculate_solution_rhs_merit(nonlinear_solution,barrier_size,penalty_parameter);
        std::cout << "current_merit:   " << current_merit << std::endl;
        double test_merit;

        bool found_step_size = false;


        for(int k = 0; k<7 && !found_step_size; k++)
        {
            test_solution.block(0) = nonlinear_solution.block(0)
                                          + maximum_step_size[0] * linear_solution.block(0);
            test_solution.block(1) = nonlinear_solution.block(1)
                                          + maximum_step_size[0] * linear_solution.block(1);
            test_solution.block(2) = nonlinear_solution.block(2)
                                          + maximum_step_size[0] * linear_solution.block(2);
            test_solution.block(3) = nonlinear_solution.block(3)
                                          + maximum_step_size[1] * linear_solution.block(3);
            test_solution.block(4) = nonlinear_solution.block(4)
                                          + maximum_step_size[1] * linear_solution.block(4);
            test_solution.block(5) = nonlinear_solution.block(5)
                                          + maximum_step_size[0] * linear_solution.block(5);
            test_solution.block(6) = nonlinear_solution.block(6)
                                          + maximum_step_size[1] * linear_solution.block(6);
            test_solution.block(7) = nonlinear_solution.block(7)
                                          + maximum_step_size[0] * linear_solution.block(7);
            test_solution.block(8) = nonlinear_solution.block(8)
                                          + maximum_step_size[1] * linear_solution.block(8);
            test_merit = calculate_solution_rhs_merit(test_solution,barrier_size,penalty_parameter);
            std::cout << "test merit:   " << test_merit << std::endl;

            //Typically if (test_merit < current_merit)
            if(0 < current_merit)
            {
                found_step_size = true;
            }
            else
            {
                maximum_step_size[0] = maximum_step_size[0]/2;
                maximum_step_size[1] = maximum_step_size[1]/2;
            }


        }

        std::cout << maximum_step_size[0] <<"   "<< maximum_step_size[1] << std::endl;

        nonlinear_solution.block(0) = nonlinear_solution.block(0)
                                 + maximum_step_size[0] * linear_solution.block(0);
        nonlinear_solution.block(1) = nonlinear_solution.block(1)
                                 + maximum_step_size[0] * linear_solution.block(1);
        nonlinear_solution.block(2) = nonlinear_solution.block(2)
                                 + maximum_step_size[0] * linear_solution.block(2);
        nonlinear_solution.block(3) = nonlinear_solution.block(3)
                                 + maximum_step_size[1] * linear_solution.block(3);
        nonlinear_solution.block(4) = nonlinear_solution.block(4)
                                 + maximum_step_size[1] * linear_solution.block(4);
        nonlinear_solution.block(5) = nonlinear_solution.block(5)
                                 + maximum_step_size[0] * linear_solution.block(5);
        nonlinear_solution.block(6) = nonlinear_solution.block(6)
                                 + maximum_step_size[1] * linear_solution.block(6);
        nonlinear_solution.block(7) = nonlinear_solution.block(7)
                                 + maximum_step_size[0] * linear_solution.block(7);
        nonlinear_solution.block(8) = nonlinear_solution.block(8)
                                 + maximum_step_size[1] * linear_solution.block(8);

    }

    /**This is an "exact merit function" - except it needs a lot of scaling to actually work. Rather than always getting to 0,
     * this merit function would actually be directly tied to the cost function, which is nice. */
    template<int dim>
    double
    SANDTopOpt<dim>::calculate_solution_merit(BlockVector<double> test_solution, double barrier_size,
                                              double penalty_parameter) {
        double solution_merit = 0;

        //term for elasticity constraint - I need something like A(rho)u - f
        solution_merit = solution_merit + get_compliance_plus_elasticity_error(test_solution, penalty_parameter);
        std::cout << "compliance and elasticity error   " << solution_merit << std::endl;

        //term for filter constraint
        Vector<double> filtered_unfiltered_density;
        filtered_unfiltered_density.reinit(filter_matrix.m());
        filter_matrix.vmult(filtered_unfiltered_density, test_solution.block(2));
        filtered_unfiltered_density = (filtered_unfiltered_density - test_solution.block(0));

        solution_merit = solution_merit + penalty_parameter * filtered_unfiltered_density.l2_norm();

        //term for inequality constraints on slacks
        for (unsigned int k = 0; k < test_solution.block(5).size(); k++) {
            solution_merit = solution_merit - barrier_size * std::log(test_solution.block(5)[k]);
        }
        for (unsigned int k = 0; k < test_solution.block(7).size(); k++) {
            solution_merit = solution_merit - barrier_size * std::log(test_solution.block(7)[k]);
        }


        //term for slack variable equality constraint
        Vector<double> slack_error;
        slack_error = test_solution.block(2) - test_solution.block(5);
        solution_merit = solution_merit + penalty_parameter * slack_error.l2_norm();
        //above is lower slack, below is upper
        double sum = 0;
        for (unsigned int k = 0; k < test_solution.block(7).size(); k++) {
            sum = sum + std::pow(1 - test_solution.block(2)[k] - test_solution.block(7)[k], 2);
        }
        solution_merit = solution_merit + penalty_parameter * std::pow(sum, .5);


        return solution_merit;
    }

    /**My current stand-in for the exact merit function, this calculates a rhs vector with the step that would have been taken,
     * and finds the l2 norm. Because the goal is to get this to 0, I can use the l2 norm as a fake merit function. */
    template<int dim>
    double
    SANDTopOpt<dim>::calculate_solution_rhs_merit(BlockVector<double> test_solution, double barrier_size,
                                                  double penalty_parameter) {
        double solution_merit = 0;
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
        Vector<double> test_rhs;
        test_rhs=system_rhs;
        test_rhs = 0;

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

        Functions::ConstantFunction<dim> lambda(1.), mu(1.);
        std::vector<Tensor<1, dim>> rhs_values(n_q_points);

        BlockVector<double> filtered_unfiltered_density_solution = test_solution;
        BlockVector<double> filter_adjoint_unfiltered_density_multiplier_solution = test_solution;
        filtered_unfiltered_density_solution.block(2) = 0;
        filter_adjoint_unfiltered_density_multiplier_solution.block(4) = 0;

        filter_matrix.vmult(filtered_unfiltered_density_solution.block(2), test_solution.block(2));
        filter_matrix.Tvmult(filter_adjoint_unfiltered_density_multiplier_solution.block(4),
                             test_solution.block(4));


        for (const auto &cell : dof_handler.active_cell_iterators()) {
            cell_matrix = 0;
            full_density_cell_matrix = 0;
            full_density_cell_matrix_for_Au = 0;
            cell_rhs = 0;

            cell->get_dof_indices(local_dof_indices);

            fe_values.reinit(cell);

            lambda.value_list(fe_values.get_quadrature_points(), lambda_values);
            mu.value_list(fe_values.get_quadrature_points(), mu_values);

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


            fe_values[densities].get_function_values(test_solution,
                                                     old_density_values);
            fe_values[displacements].get_function_values(test_solution,
                                                         old_displacement_values);
            fe_values[displacements].get_function_divergences(test_solution,
                                                              old_displacement_divs);
            fe_values[displacements].get_function_symmetric_gradients(
                    test_solution, old_displacement_symmgrads);
            fe_values[displacement_multipliers].get_function_values(
                    test_solution, old_displacement_multiplier_values);
            fe_values[displacement_multipliers].get_function_divergences(
                    test_solution, old_displacement_multiplier_divs);
            fe_values[displacement_multipliers].get_function_symmetric_gradients(
                    test_solution, old_displacement_multiplier_symmgrads);
            fe_values[density_lower_slacks].get_function_values(
                    test_solution, old_lower_slack_values);
            fe_values[density_lower_slack_multipliers].get_function_values(
                    test_solution, old_lower_slack_multiplier_values);
            fe_values[density_upper_slacks].get_function_values(
                    test_solution, old_upper_slack_values);
            fe_values[density_upper_slack_multipliers].get_function_values(
                    test_solution, old_upper_slack_multiplier_values);
            fe_values[unfiltered_densities].get_function_values(
                    test_solution, old_unfiltered_density_values);
            fe_values[unfiltered_density_multipliers].get_function_values(
                    test_solution, old_unfiltered_density_multiplier_values);
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
                    cell_matrix, cell_rhs, local_dof_indices, system_matrix, test_rhs);


        }
        return test_rhs.l2_norm();

    }


    /**Only here because it is called by the exact merit function. Also, there is a much better way to do this,
     * where I don't try to build a whole matrix. Oops. */
    template<int dim>
    double
    SANDTopOpt<dim>::get_compliance_plus_elasticity_error(BlockVector<double> test_solution, double penalty_parameter) {
        BlockSparseMatrix<double> elasticity_matrix;
        elasticity_matrix.reinit(sparsity_pattern);
        BlockVector<double> elasticity_rhs;
        elasticity_rhs=system_rhs;
        elasticity_rhs = 0;

        const FEValuesExtractors::Scalar densities(0);
        const FEValuesExtractors::Vector displacements(1);


        /*Remove any values from old iterations*/
        system_matrix.reinit(sparsity_pattern);

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

        std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);

        std::vector<double> lambda_values(n_q_points);
        std::vector<double> mu_values(n_q_points);

        Functions::ConstantFunction<dim> lambda(1.), mu(1.);
        std::vector<Tensor<1, dim>> rhs_values(n_q_points);

        BlockVector<double> filtered_unfiltered_density_solution = test_solution;
        BlockVector<double> filter_adjoint_unfiltered_density_multiplier_solution = test_solution;
        filtered_unfiltered_density_solution.block(2) = 0;
        filter_adjoint_unfiltered_density_multiplier_solution.block(4) = 0;

        filter_matrix.vmult(filtered_unfiltered_density_solution.block(2), test_solution.block(2));
        filter_matrix.Tvmult(filter_adjoint_unfiltered_density_multiplier_solution.block(4),
                             nonlinear_solution.block(4));


        for (const auto &cell : dof_handler.active_cell_iterators()) {
            cell_matrix = 0;
            cell_rhs = 0;

            cell->get_dof_indices(local_dof_indices);

            fe_values.reinit(cell);

            lambda.value_list(fe_values.get_quadrature_points(), lambda_values);
            mu.value_list(fe_values.get_quadrature_points(), mu_values);

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


            fe_values[densities].get_function_values(test_solution,
                                                     old_density_values);
            fe_values[displacements].get_function_values(test_solution,
                                                         old_displacement_values);
            fe_values[displacements].get_function_divergences(test_solution,
                                                              old_displacement_divs);
            fe_values[displacements].get_function_symmetric_gradients(
                    test_solution, old_displacement_symmgrads);


            Tensor<1, dim> traction;
            traction[1] = -1;

            for (unsigned int q_point = 0; q_point < n_q_points; ++q_point) {

                for (unsigned int i = 0; i < dofs_per_cell; ++i) {
                    const SymmetricTensor<2, dim> displacement_phi_i_symmgrad =
                            fe_values[displacements].symmetric_gradient(i, q_point);
                    const double displacement_phi_i_div =
                            fe_values[displacements].divergence(i, q_point);


                    for (unsigned int j = 0; j < dofs_per_cell; ++j) {
                        const SymmetricTensor<2, dim> displacement_phi_j_symmgrad =
                                fe_values[displacements].symmetric_gradient(j,
                                                                            q_point);
                        const double displacement_phi_j_div =
                                fe_values[displacements].divergence(j, q_point);

                        const double density_phi_j = fe_values[densities].value(
                                j, q_point);




                        //Storing this in a new big matrix because I can...

                        cell_matrix(i, j) +=
                                fe_values.JxW(q_point) * (
                                        std::pow(
                                                old_density_values[q_point],
                                                density_penalty_exponent)
                                        * (displacement_phi_i_div * displacement_phi_j_div
                                           * lambda_values[q_point]
                                           + 2 * mu_values[q_point]
                                             * (displacement_phi_i_symmgrad*
                                                displacement_phi_j_symmgrad))
                                                );


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
                                cell_rhs(i) += traction
                                               * fe_face_values[displacements].value(i,
                                                                                     face_q_point)
                                               * fe_face_values.JxW(face_q_point);
                            }
                        }
                    }
                }


                MatrixTools::local_apply_boundary_values(boundary_values, local_dof_indices,
                                                         cell_matrix, cell_rhs, true);

                constraints.distribute_local_to_global(
                        cell_matrix, cell_rhs, local_dof_indices, elasticity_matrix, elasticity_rhs);


            }


        }
        Vector<double> lhs_vector;
        lhs_vector.reinit(elasticity_matrix.block(1,1).m());
        elasticity_matrix.block(1,1).vmult(lhs_vector,test_solution.block(1));
        lhs_vector = lhs_vector - elasticity_rhs.block(1);
        return penalty_parameter*lhs_vector.l2_norm() + test_solution.block(1) * elasticity_rhs.block(1);
    }


    /**This outputs all of the variables into a vtk file format. */
    template<int dim>
    void
    SANDTopOpt<dim>::output(int j) {
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
        data_out.add_data_vector(nonlinear_solution, solution_names,
                                 DataOut<dim>::type_dof_data, data_component_interpretation);
//      data_out.add_data_vector (linear_solution, solution_names,
//          DataOut<dim>::type_dof_data, data_component_interpretation);
        data_out.build_patches();
        std::ofstream output("solution" + std::to_string(j) + ".vtk");
        data_out.write_vtk(output);
    }

    /** The main function of this class. Runs a nonlinear optimization search using interior point methods, reducing barrier size when appropriate*/
    template<int dim>
    void
    SANDTopOpt<dim>::run() {
        double barrier_size = 25;
        density_penalty_exponent = 3;
        filter_r = .25;
        create_triangulation();
        setup_block_system();
        set_bcids();
        setup_filter_matrix();
        std::vector<double> max_step_size;
        for (unsigned int loop = 0; loop < 100; loop++)
        {
            assemble_block_system(barrier_size);
            solve();
            max_step_size =calculate_maximum_step_size();
            update_step(max_step_size,barrier_size, 1);
            if (loop % 1 == 0)
            {
                output(loop / 1);
                std::cout << "finished loop number " <<  loop << std::endl;
            }


            if (system_rhs.l2_norm() < 1e-10) {
                barrier_size = barrier_size * .5;
                std::cout << "barrier size is   " << barrier_size << std::endl;
            }
        }
    }

} // namespace SAND

int
main() {
    try {
        SAND::SANDTopOpt<2> elastic_problem_2d;
        elastic_problem_2d.run();
    }
    catch (std::exception &exc) {
        std::cerr << std::endl << std::endl
                  << "----------------------------------------------------" << std::endl;
        std::cerr << "Exception on processing: " << std::endl << exc.what()
                  << std::endl << "Aborting!" << std::endl
                  << "----------------------------------------------------" << std::endl;

        return 1;
    }
    catch (...) {
        std::cerr << std::endl << std::endl
                  << "----------------------------------------------------" << std::endl;
        std::cerr << "Unknown exception!" << std::endl << "Aborting!" << std::endl
                  << "----------------------------------------------------" << std::endl;
        return 1;
    }

    return 0;
}
