//
// Created by justin on 2/17/21.
//
#include <deal.II/lac/linear_operator.h>
#include <deal.II/lac/sparse_direct.h>
#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/base/timer.h>
#include "../include/schur_preconditioner.h"
#include "../include/input_information.h"
#include <fstream>

namespace SAND {

    using namespace dealii;

    template<int dim>
    TopOptSchurPreconditioner<dim>::TopOptSchurPreconditioner()
            :
            n_rows(0),
            n_columns(0),
            n_block_rows(0),
            n_block_columns(0),
            diag_solver_control(10000, 1e-6),
            diag_cg(diag_solver_control),
            other_solver_control(10000, 1e-6),
            other_bicgstab(other_solver_control),
            timer(std::cout, TimerOutput::summary,
                  TimerOutput::wall_times) {
    }

    template<int dim>
    void TopOptSchurPreconditioner<dim>::initialize(BlockSparseMatrix<double> &matrix,
                                                    std::map<types::global_dof_index, double> &boundary_values) {
        TimerOutput::Scope t(timer, "initialize");
        for (auto&[dof_index, boundary_value] : boundary_values) {
            const types::global_dof_index disp_start_index = matrix.get_row_indices().block_start(
                    SolutionBlocks::displacement);
            const types::global_dof_index disp_mult_start_index = matrix.get_row_indices().block_start(
                    SolutionBlocks::displacement_multiplier);
            const types::global_dof_index n_u = matrix.block(SolutionBlocks::displacement,
                                                             SolutionBlocks::displacement).m();
            if ((dof_index >= disp_start_index) && (dof_index < disp_start_index + n_u)) {
                double diag_val = matrix.block(SolutionBlocks::displacement, SolutionBlocks::displacement).el(
                        dof_index - disp_start_index, dof_index - disp_start_index);
                matrix.block(SolutionBlocks::displacement, SolutionBlocks::displacement_multiplier).set(
                        dof_index - disp_start_index, dof_index - disp_start_index, diag_val);
            } else if ((dof_index >= disp_mult_start_index) && (dof_index < disp_mult_start_index + n_u)) {
                double diag_val = matrix.block(SolutionBlocks::displacement_multiplier,
                                               SolutionBlocks::displacement_multiplier).el(
                        dof_index - disp_mult_start_index, dof_index - disp_mult_start_index);
                matrix.block(SolutionBlocks::displacement_multiplier, SolutionBlocks::displacement).set(
                        dof_index - disp_mult_start_index, dof_index - disp_mult_start_index, diag_val);
            }
        }

        //set diagonal to 0?
        for (auto&[dof_index, boundary_value] : boundary_values) {
            const types::global_dof_index disp_start_index = matrix.get_row_indices().block_start(
                    SolutionBlocks::displacement);
            const types::global_dof_index disp_mult_start_index = matrix.get_row_indices().block_start(
                    SolutionBlocks::displacement_multiplier);
            const types::global_dof_index n_u = matrix.block(SolutionBlocks::displacement,
                                                             SolutionBlocks::displacement).m();
            if ((dof_index >= disp_start_index) && (dof_index < disp_start_index + n_u)) {
                matrix.block(SolutionBlocks::displacement, SolutionBlocks::displacement).set(
                        dof_index - disp_start_index, dof_index - disp_start_index, 0);
            } else if ((dof_index >= disp_mult_start_index) && (dof_index < disp_mult_start_index + n_u)) {
                matrix.block(SolutionBlocks::displacement_multiplier, SolutionBlocks::displacement_multiplier).set(
                        dof_index - disp_mult_start_index, dof_index - disp_mult_start_index, 0);
            }
        }

        op_elastic = linear_operator(
                matrix.block(SolutionBlocks::displacement, SolutionBlocks::displacement_multiplier));

        op_filter = linear_operator(
                matrix.block(SolutionBlocks::unfiltered_density_multiplier, SolutionBlocks::unfiltered_density));

        op_diag_1 = linear_operator(
                matrix.block(SolutionBlocks::density_lower_slack, SolutionBlocks::density_lower_slack));

        op_diag_2 = linear_operator(
                matrix.block(SolutionBlocks::density_upper_slack, SolutionBlocks::density_upper_slack));

        op_displacement_density = linear_operator(matrix.block(SolutionBlocks::displacement, SolutionBlocks::density));

        op_displacement_multiplier_density = linear_operator(
                matrix.block(SolutionBlocks::displacement_multiplier, SolutionBlocks::density));

        op_density_density = linear_operator(matrix.block(SolutionBlocks::density, SolutionBlocks::density));

//        diag_sum_direct.initialize(matrix.block(SolutionBlocks::density_lower_slack,SolutionBlocks::density_lower_slack)
//                                    + matrix.block(SolutionBlocks::density_lower_slack,SolutionBlocks::density_upper_slack));

        op_diag_sum_inverse = inverse_operator(op_diag_1 + op_diag_2, diag_cg, PreconditionIdentity());

        elastic_direct.initialize(matrix.block(SolutionBlocks::displacement, SolutionBlocks::displacement_multiplier));
        op_elastic_inverse = linear_operator(elastic_direct);

        op_scaled_identity = linear_operator(
                matrix.block(SolutionBlocks::density_upper_slack, SolutionBlocks::density_upper_slack_multiplier));

        scaled_direct.initialize(
                matrix.block(SolutionBlocks::density_upper_slack, SolutionBlocks::density_upper_slack_multiplier));
        op_scaled_inverse = linear_operator(scaled_direct);

        op_fddf_chunk = -1 * op_filter * op_diag_sum_inverse * transpose_operator(op_filter);
        op_simplified_fddf_chunk = -1 * op_diag_sum_inverse;
        op_bcaeeac_chunk = (op_density_density
                            -
                            transpose_operator(op_displacement_density) * op_elastic_inverse *
                            op_displacement_multiplier_density
                            -
                            transpose_operator(op_displacement_multiplier_density) *
                            op_elastic_inverse * op_displacement_density);
        op_simplified_bcaeeac_chunk = linear_operator(
                weighted_mass_matrix.block(SolutionBlocks::unfiltered_density, SolutionBlocks::unfiltered_density));
        op_lumped_bcaeeac_chunk = linear_operator(
                weighted_mass_matrix.block(SolutionBlocks::unfiltered_density_multiplier,
                                           SolutionBlocks::unfiltered_density_multiplier));

        op_top_big_inverse = inverse_operator(op_bcaeeac_chunk * op_scaled_inverse * op_fddf_chunk
                                              - op_scaled_identity,
                                              other_bicgstab,
                                              PreconditionIdentity());

        op_simplified_top_big_inverse = inverse_operator(
                op_lumped_bcaeeac_chunk * op_scaled_inverse * op_simplified_fddf_chunk
                - op_scaled_identity,
                other_bicgstab,
                PreconditionIdentity());

        op_bot_big_inverse = inverse_operator(op_fddf_chunk * op_scaled_inverse * op_bcaeeac_chunk
                                              - op_scaled_identity,
                                              other_bicgstab,
                                              PreconditionIdentity());

        op_simplified_bot_big_inverse = inverse_operator(
                op_simplified_fddf_chunk * op_scaled_inverse * op_lumped_bcaeeac_chunk
                - op_scaled_identity,
                other_bicgstab,
                PreconditionIdentity());

//        weighted_mass_matrix.reinit(mass_sparsity);
    }

    template<int dim>
    void TopOptSchurPreconditioner<dim>::vmult(BlockVector<double> &dst, const BlockVector<double> &src) const {
        BlockVector<double> temp_src;
        {
            TimerOutput::Scope t(timer, "part 1");
            vmult_step_1(dst, src);
            temp_src = dst;
        }

        {
            TimerOutput::Scope t(timer, "part 2");
            vmult_step_2(dst, temp_src);
            temp_src = dst;
        }

        {
            TimerOutput::Scope t(timer, "part 3");
            vmult_step_3(dst, temp_src);
            temp_src = dst;
        }

        vmult_step_4(dst, temp_src);

        timer.print_summary();
    }

    template<int dim>
    void TopOptSchurPreconditioner<dim>::Tvmult(BlockVector<double> &dst, const BlockVector<double> &src) const {
        dst = src;
    }

    template<int dim>
    void TopOptSchurPreconditioner<dim>::vmult_add(BlockVector<double> &dst, const BlockVector<double> &src) const {
        BlockVector<double> dst_temp = dst;
        vmult(dst_temp, src);
        dst += dst_temp;
    }

    template<int dim>
    void TopOptSchurPreconditioner<dim>::Tvmult_add(BlockVector<double> &dst, const BlockVector<double> &src) const {
        dst = dst + src;
    }

    template<int dim>
    void TopOptSchurPreconditioner<dim>::vmult_step_1(BlockVector<double> &dst, const BlockVector<double> &src) const {
        dst = src;
        dst.block(SolutionBlocks::unfiltered_density) +=
                -1 * op_diag_1 * op_scaled_inverse * src.block(SolutionBlocks::density_lower_slack_multiplier)
                +
                op_diag_2 * op_scaled_inverse * src.block(SolutionBlocks::density_upper_slack_multiplier)
                +
                src.block(SolutionBlocks::density_lower_slack)
                -
                src.block(SolutionBlocks::density_upper_slack);
    }

    template<int dim>
    void TopOptSchurPreconditioner<dim>::vmult_step_2(BlockVector<double> &dst, const BlockVector<double> &src) const {
        dst = src;
        dst.block(SolutionBlocks::unfiltered_density_multiplier) +=
                -1 * op_filter * op_diag_sum_inverse * src.block(SolutionBlocks::unfiltered_density);

    }

    template<int dim>
    void TopOptSchurPreconditioner<dim>::vmult_step_3(BlockVector<double> &dst, const BlockVector<double> &src) const {
        dst = src;
        dst.block(SolutionBlocks::density) +=
                -1 * transpose_operator(op_displacement_density) * op_elastic_inverse *
                src.block(SolutionBlocks::displacement_multiplier)
                +
                -1 * transpose_operator(op_displacement_multiplier_density) * op_elastic_inverse *
                src.block(SolutionBlocks::displacement);
    }

    template<int dim>
    void TopOptSchurPreconditioner<dim>::vmult_step_4(BlockVector<double> &dst, const BlockVector<double> &src) const {
        {
            //First Block Inverse
            TimerOutput::Scope t(timer, "inverse 1");
            dst.block(SolutionBlocks::density_lower_slack_multiplier) =
                    -1 * op_diag_1 * op_scaled_inverse * op_scaled_inverse *
                    src.block(SolutionBlocks::density_lower_slack_multiplier)
                    +
                    op_scaled_inverse * src.block(SolutionBlocks::density_lower_slack);

            dst.block(SolutionBlocks::density_upper_slack_multiplier) =
                    -1 * op_diag_2 * op_scaled_inverse * op_scaled_inverse *
                    src.block(SolutionBlocks::density_upper_slack_multiplier)
                    +
                    op_scaled_inverse * src.block(SolutionBlocks::density_upper_slack);

            dst.block(SolutionBlocks::density_lower_slack) =
                    op_scaled_inverse * src.block(SolutionBlocks::density_lower_slack_multiplier);

            dst.block(SolutionBlocks::density_upper_slack) =
                    op_scaled_inverse * src.block(SolutionBlocks::density_upper_slack_multiplier);
        }

        {
            //Second Block Inverse
            TimerOutput::Scope t(timer, "inverse 2");
            dst.block(SolutionBlocks::unfiltered_density) =
                    op_diag_sum_inverse * src.block(SolutionBlocks::unfiltered_density);
        }

        {
            //Third Block Inverse
            TimerOutput::Scope t(timer, "inverse 3");
            dst.block(SolutionBlocks::displacement) =
                    op_elastic_inverse * src.block(SolutionBlocks::displacement_multiplier);
            dst.block(SolutionBlocks::displacement_multiplier) =
                    op_elastic_inverse * src.block(SolutionBlocks::displacement);
        }

        {
            //Fourth (ugly) Block Inverse
            TimerOutput::Scope t(timer, "inverse 4");
            dst.block(SolutionBlocks::unfiltered_density_multiplier) =
                    (op_bcaeeac_chunk * op_scaled_inverse * src.block(SolutionBlocks::unfiltered_density_multiplier));
            dst.block(SolutionBlocks::unfiltered_density_multiplier) =
                    (op_simplified_top_big_inverse * dst.block(SolutionBlocks::unfiltered_density_multiplier));

            dst.block(SolutionBlocks::unfiltered_density_multiplier) +=
                    (op_simplified_top_big_inverse * src.block(SolutionBlocks::density));

            dst.block(SolutionBlocks::density) =
                    (op_fddf_chunk * op_scaled_inverse * src.block(SolutionBlocks::density));

            dst.block(SolutionBlocks::density) = op_simplified_bot_big_inverse * dst.block(SolutionBlocks::density);

            dst.block(SolutionBlocks::density) +=
                    (op_simplified_bot_big_inverse * src.block(SolutionBlocks::unfiltered_density_multiplier));
        }
    }

    template<int dim>
    void TopOptSchurPreconditioner<dim>::get_sparsity_pattern(BlockDynamicSparsityPattern &bdsp) {
        mass_sparsity.copy_from(bdsp);
    }

    template<int dim>
    void TopOptSchurPreconditioner<dim>::assemble_mass_matrix(const BlockVector<double> &state, const FESystem<dim> &fe,
                                                              const DoFHandler<dim> &dof_handler,
                                                              const AffineConstraints<double> &constraints) {
        timer.reset();
        const unsigned int density_penalty_exponent = Input::density_penalty_exponent;
        const FEValuesExtractors::Scalar densities(SolutionComponents::density<dim>);
        const FEValuesExtractors::Vector displacements(SolutionComponents::displacement<dim>);
        const FEValuesExtractors::Scalar unfiltered_densities(SolutionComponents::unfiltered_density<dim>);
        const FEValuesExtractors::Vector displacement_multipliers(SolutionComponents::displacement_multiplier<dim>);
        const FEValuesExtractors::Scalar unfiltered_density_multipliers(
                SolutionComponents::unfiltered_density_multiplier<dim>);


        /*Remove any values from old iterations*/
        weighted_mass_matrix.reinit(mass_sparsity);

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
            fe_values[unfiltered_densities].get_function_values(
                    state, old_unfiltered_density_values);
            fe_values[unfiltered_density_multipliers].get_function_values(
                    state, old_unfiltered_density_multiplier_values);
            Tensor<1, dim> traction;
            traction[1] = -1;

            for (unsigned int q_point = 0; q_point < n_q_points; ++q_point) {

                for (unsigned int i = 0; i < dofs_per_cell; ++i) {
                    const double density_phi_i = fe_values[densities].value(i,
                                                                            q_point);

                    const double unfiltered_density_multiplier_phi_i = fe_values[unfiltered_density_multipliers].value(
                            i, q_point);
                    const double unfiltered_density_phi_i = fe_values[unfiltered_densities].value(
                            i, q_point);



                    for (unsigned int j = 0; j < dofs_per_cell; ++j) {

                        const double density_phi_j = fe_values[densities].value(
                                j, q_point);
                        const double unfiltered_density_multiplier_phi_j = fe_values[unfiltered_density_multipliers].value(
                                j, q_point);
                        const double unfiltered_density_phi_j = fe_values[unfiltered_densities].value(
                                j, q_point);

                        //Equation 0
                        cell_matrix(i, j) +=
                                fe_values.JxW(q_point) *
                                (
                                        -1 * density_penalty_exponent * (density_penalty_exponent + 1)
                                        *
                                        std::pow(old_density_values[q_point],
                                                 density_penalty_exponent - 2)
                                        *
                                        (old_displacement_divs[q_point] * old_displacement_multiplier_divs[q_point]
                                         * lambda_values[q_point]
                                         +
                                         2 * mu_values[q_point] * (old_displacement_symmgrads[q_point] *
                                                                   old_displacement_multiplier_symmgrads[q_point]))
                                        * unfiltered_density_phi_i * unfiltered_density_phi_j
                                );
                        //Equation 1

                        cell_matrix(i, i) +=
                                fe_values.JxW(q_point) *
                                (
                                        -1 * density_penalty_exponent * (density_penalty_exponent + 1)
                                        *
                                        std::pow(old_density_values[q_point],
                                                 density_penalty_exponent - 2)
                                        *
                                        (old_displacement_divs[q_point] * old_displacement_multiplier_divs[q_point]
                                         * lambda_values[q_point]
                                         +
                                         2 * mu_values[q_point] * (old_displacement_symmgrads[q_point] *
                                                                   old_displacement_multiplier_symmgrads[q_point]))
                                        * unfiltered_density_multiplier_phi_i * unfiltered_density_multiplier_phi_j
                                );
                    }

                }

            }

            constraints.distribute_local_to_global(
                    cell_matrix, local_dof_indices, weighted_mass_matrix);
        }

    }


    template<int dim>
    void TopOptSchurPreconditioner<dim>::print_stuff(BlockSparseMatrix<double> &matrix) {

        const unsigned int vec_size = matrix.block(SolutionBlocks::density, SolutionBlocks::density).n();
        FullMatrix<double> orig_mat(vec_size, vec_size);
        FullMatrix<double> est_mat(vec_size, vec_size);
        FullMatrix<double> est_mass_mat(vec_size, vec_size);
        FullMatrix<double> est_lumped_mat(vec_size, vec_size);
        FullMatrix<double> just_mass_mat(vec_size, vec_size);

        for (unsigned int j = 0; j < vec_size; j++) {
            Vector<double> unit_vector;
            unit_vector.reinit(vec_size);
            unit_vector=0;
            unit_vector[j] = 1;
            Vector<double> transformed_unit_vector_orig;
            Vector<double> transformed_unit_vector_est;
            Vector<double> transformed_unit_vector_mass;
            Vector<double> transformed_unit_vector_lumped;
            Vector<double> transformed_unit_vector_just_mass(vec_size);
            transformed_unit_vector_orig = op_density_density* unit_vector
                    - transpose_operator(op_displacement_density) * op_elastic_inverse *
                                      op_displacement_multiplier_density * unit_vector
                                      -
                                      transpose_operator(op_displacement_multiplier_density) * op_elastic_inverse *
                                      op_displacement_density * unit_vector;
            transformed_unit_vector_est = op_bcaeeac_chunk * unit_vector;
            transformed_unit_vector_mass = op_simplified_bcaeeac_chunk * unit_vector;
            transformed_unit_vector_lumped = op_lumped_bcaeeac_chunk * unit_vector;
            weighted_mass_matrix.block(SolutionBlocks::unfiltered_density, SolutionBlocks::unfiltered_density).template vmult(transformed_unit_vector_just_mass, unit_vector);

            for (unsigned int i = 0; i < vec_size; i++) {
                orig_mat(i,j) = transformed_unit_vector_orig[i];
                est_mat(i,j) = transformed_unit_vector_est[i];
                est_mass_mat(i,j) = transformed_unit_vector_mass[i];
                est_lumped_mat(i,j) = transformed_unit_vector_lumped[i];
                just_mass_mat(i,j) = transformed_unit_vector_just_mass[i];
            }
        }

        std::ofstream OGMat("original_matrix.csv");
        std::ofstream EstMat("estimated.csv");
        std::ofstream MassMat("mass_estimated.csv");
        std::ofstream LumpMat("lumped_mass_estimated.csv");
        std::ofstream JustMassMat("just_mass.csv");
        for (unsigned int i = 0; i < vec_size; i++)
        {
            OGMat << orig_mat(i, 0);
            EstMat << est_mat(i, 0);
            MassMat << est_mass_mat(i, 0);
            LumpMat << est_lumped_mat(i, 0);
            JustMassMat << just_mass_mat(i,0);
            for (unsigned int j = 1; j < vec_size; j++)
            {
                OGMat << "," << orig_mat(i, j);
                EstMat << ","  << est_mat(i, j);
                MassMat << "," << est_mass_mat(i, j);
                LumpMat << "," << est_lumped_mat(i, j);
                JustMassMat << "," << just_mass_mat(i,j);
            }
            OGMat << "\n";
            EstMat << "\n";
            MassMat << "\n";
            LumpMat << "\n";
            JustMassMat << "\n";
        }
    OGMat.close();
    EstMat.close();
    MassMat.close();
    LumpMat.close();
    JustMassMat.close();
    }
}
template class SAND::TopOptSchurPreconditioner<2>;
template class SAND::TopOptSchurPreconditioner<3>;