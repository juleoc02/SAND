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
    TopOptSchurPreconditioner<dim>::TopOptSchurPreconditioner(BlockSparseMatrix<double> &matrix_in)
            :
            system_matrix(matrix_in),
            n_rows(0),
            n_columns(0),
            n_block_rows(0),
            n_block_columns(0),
            other_solver_control(100000, 1e-10),
            other_bicgstab(other_solver_control),
            other_gmres(other_solver_control),
            other_cg(other_solver_control),
            a_mat(matrix_in.block(SolutionBlocks::displacement, SolutionBlocks::displacement_multiplier)),
            b_mat(matrix_in.block(SolutionBlocks::density, SolutionBlocks::density)),
            c_mat(matrix_in.block(SolutionBlocks::displacement,SolutionBlocks::density)),
            e_mat(matrix_in.block(SolutionBlocks::displacement_multiplier,SolutionBlocks::density)),
            f_mat(matrix_in.block(SolutionBlocks::unfiltered_density_multiplier,SolutionBlocks::unfiltered_density)),
            d_m_mat(matrix_in.block(SolutionBlocks::density_upper_slack_multiplier, SolutionBlocks::density_upper_slack)),
            d_1_mat(matrix_in.block(SolutionBlocks::density_lower_slack, SolutionBlocks::density_lower_slack)),
            d_2_mat(matrix_in.block(SolutionBlocks::density_upper_slack, SolutionBlocks::density_upper_slack)),
            m_vect(matrix_in.block(SolutionBlocks::density, SolutionBlocks::total_volume_multiplier)),
            timer(std::cout, TimerOutput::summary,
                  TimerOutput::wall_times)
    {

    }

    template<int dim>
    void TopOptSchurPreconditioner<dim>::initialize(BlockSparseMatrix<double> &matrix, const std::map<types::global_dof_index, double> &boundary_values,const DoFHandler<dim> &dof_handler, const double barrier_size, const BlockVector<double> &state)
    {
        TimerOutput::Scope t(timer, "initialize");
        for (auto&[dof_index, boundary_value] : boundary_values) {
            const types::global_dof_index disp_start_index = system_matrix.get_row_indices().block_start(
                    SolutionBlocks::displacement);
            const types::global_dof_index disp_mult_start_index = system_matrix.get_row_indices().block_start(
                    SolutionBlocks::displacement_multiplier);
            const types::global_dof_index n_u = system_matrix.block(SolutionBlocks::displacement,
                                                             SolutionBlocks::displacement).m();
            if ((dof_index >= disp_start_index) && (dof_index < disp_start_index + n_u)) {
                double diag_val = system_matrix.block(SolutionBlocks::displacement, SolutionBlocks::displacement).el(
                        dof_index - disp_start_index, dof_index - disp_start_index);
                system_matrix.block(SolutionBlocks::displacement, SolutionBlocks::displacement_multiplier).set(
                        dof_index - disp_start_index, dof_index - disp_start_index, diag_val);
            } else if ((dof_index >= disp_mult_start_index) && (dof_index < disp_mult_start_index + n_u)) {
                double diag_val = system_matrix.block(SolutionBlocks::displacement_multiplier,
                                               SolutionBlocks::displacement_multiplier).el(
                        dof_index - disp_mult_start_index, dof_index - disp_mult_start_index);
                system_matrix.block(SolutionBlocks::displacement_multiplier, SolutionBlocks::displacement).set(
                        dof_index - disp_mult_start_index, dof_index - disp_mult_start_index, diag_val);
            }
        }


        //set diagonal to 0?
        for (auto&[dof_index, boundary_value] : boundary_values) {
            const types::global_dof_index disp_start_index = system_matrix.get_row_indices().block_start(
                    SolutionBlocks::displacement);
            const types::global_dof_index disp_mult_start_index = system_matrix.get_row_indices().block_start(
                    SolutionBlocks::displacement_multiplier);
            const types::global_dof_index n_u = system_matrix.block(SolutionBlocks::displacement,
                                                             SolutionBlocks::displacement).m();
            if ((dof_index >= disp_start_index) && (dof_index < disp_start_index + n_u)) {
                system_matrix.block(SolutionBlocks::displacement, SolutionBlocks::displacement).set(
                        dof_index - disp_start_index, dof_index - disp_start_index, 0);
            } else if ((dof_index >= disp_mult_start_index) && (dof_index < disp_mult_start_index + n_u)) {
                system_matrix.block(SolutionBlocks::displacement_multiplier, SolutionBlocks::displacement_multiplier).set(
                        dof_index - disp_mult_start_index, dof_index - disp_mult_start_index, 0);
            }
        }

        a_inv_direct.initialize(a_mat);

        d_3_mat.reinit(matrix.block(SolutionBlocks::density,SolutionBlocks::density).get_sparsity_pattern());
        d_4_mat.reinit(matrix.block(SolutionBlocks::density,SolutionBlocks::density).get_sparsity_pattern());
        d_5_mat.reinit(matrix.block(SolutionBlocks::density,SolutionBlocks::density).get_sparsity_pattern());
        d_6_mat.reinit(matrix.block(SolutionBlocks::density,SolutionBlocks::density).get_sparsity_pattern());
        d_7_mat.reinit(matrix.block(SolutionBlocks::density,SolutionBlocks::density).get_sparsity_pattern());
        d_8_mat.reinit(matrix.block(SolutionBlocks::density,SolutionBlocks::density).get_sparsity_pattern());
        d_m_inv_mat.reinit(matrix.block(SolutionBlocks::density,SolutionBlocks::density).get_sparsity_pattern());

        for (const auto cell : dof_handler.active_cell_iterators())
        {
            const double i = cell->active_cell_index();
            const double m = cell->measure();
            double d_3_value = -1 * state.block(SolutionBlocks::density_lower_slack_multiplier)[i] / (m*state.block(SolutionBlocks::density_lower_slack)[i]);
            double d_4_value = -1 * state.block(SolutionBlocks::density_upper_slack_multiplier)[i] / (m*state.block(SolutionBlocks::density_upper_slack)[i]);
            double d_5_value = state.block(SolutionBlocks::density_lower_slack_multiplier)[i] / (state.block(SolutionBlocks::density_lower_slack)[i]);
            double d_6_value = state.block(SolutionBlocks::density_upper_slack_multiplier)[i] / (state.block(SolutionBlocks::density_upper_slack)[i]);
            double d_7_value = (m * (state.block(SolutionBlocks::density_lower_slack_multiplier)[i] * state.block(SolutionBlocks::density_upper_slack)[i] + state.block(SolutionBlocks::density_upper_slack_multiplier)[i] * state.block(SolutionBlocks::density_lower_slack)[i]))
                                /(state.block(SolutionBlocks::density_lower_slack)[i]*state.block(SolutionBlocks::density_upper_slack)[i]);
            double d_8_value =(state.block(SolutionBlocks::density_lower_slack)[i]*state.block(SolutionBlocks::density_upper_slack)[i])
                                /(m * (state.block(SolutionBlocks::density_lower_slack_multiplier)[i] * state.block(SolutionBlocks::density_upper_slack)[i] + state.block(SolutionBlocks::density_upper_slack_multiplier)[i] * state.block(SolutionBlocks::density_lower_slack)[i]));
            d_3_mat.set(i,i,d_3_value);
            d_4_mat.set(i,i,d_4_value);
            d_5_mat.set(i,i,d_5_value);
            d_6_mat.set(i,i,d_6_value);
            d_7_mat.set(i,i,d_7_value);
            d_8_mat.set(i,i,d_8_value);
            d_m_inv_mat.set(i,i,1/m);

        }

        pre_j.reinit(matrix.block(SolutionBlocks::density,SolutionBlocks::density).n());
        pre_k.reinit(matrix.block(SolutionBlocks::density,SolutionBlocks::density).n());
        g_d_m_inv_density.reinit(matrix.block(SolutionBlocks::density,SolutionBlocks::density).n());
        k_g_d_m_inv_density.reinit(matrix.block(SolutionBlocks::density,SolutionBlocks::density).n());

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
        {
            TimerOutput::Scope t(timer, "part 4");
            vmult_step_4(dst, temp_src);
            temp_src = dst;
        }
        vmult_step_5(dst, temp_src);

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
        dst.block(SolutionBlocks::unfiltered_density) += -1 * linear_operator(d_5_mat)*src.block(SolutionBlocks::density_lower_slack_multiplier) +
                linear_operator(d_6_mat) * src.block(SolutionBlocks::density_upper_slack_multiplier) + src.block(SolutionBlocks::density_lower_slack)
                - src.block(SolutionBlocks::density_upper_slack);
    }

    template<int dim>
    void TopOptSchurPreconditioner<dim>::vmult_step_2(BlockVector<double> &dst, const BlockVector<double> &src) const {
        dst = src;
        dst.block(SolutionBlocks::unfiltered_density_multiplier) += -1 * linear_operator(f_mat) * linear_operator(d_8_mat) * src.block(SolutionBlocks::unfiltered_density);
    }

    template<int dim>
    void TopOptSchurPreconditioner<dim>::vmult_step_3(BlockVector<double> &dst, const BlockVector<double> &src) const {
        dst = src;
        dst.block(SolutionBlocks::density)+= -1 * transpose_operator(linear_operator(e_mat)) * linear_operator(a_inv_direct) * src.block(SolutionBlocks::displacement)
                                                - transpose_operator(linear_operator(c_mat)) * linear_operator(a_inv_direct) * src.block(SolutionBlocks::displacement_multiplier);
    }

    template<int dim>
    void TopOptSchurPreconditioner<dim>::vmult_step_4(BlockVector<double> &dst, const BlockVector<double> &src) const {
        dst = src;

        auto op_h = linear_operator(b_mat)
                    - transpose_operator(linear_operator(c_mat)) * linear_operator(a_inv_direct) * linear_operator(e_mat)
                    - transpose_operator(linear_operator(e_mat)) * linear_operator(a_inv_direct) * linear_operator(c_mat);

//       auto op_h = linear_operator(approx_h_mat.block(SolutionBlocks::unfiltered_density,SolutionBlocks::unfiltered_density));

        auto op_g = linear_operator(f_mat) * linear_operator(d_8_mat) *
                      transpose_operator(linear_operator(f_mat));

        auto op_k_inv = -1 * op_g * linear_operator(d_m_inv_mat) * op_h -
                        linear_operator(d_m_mat);

        auto op_k = inverse_operator(op_k_inv, other_gmres, PreconditionIdentity());



        g_d_m_inv_density = op_g * linear_operator(d_m_inv_mat) * src.block(SolutionBlocks::density);
        k_g_d_m_inv_density = op_k * g_d_m_inv_density;

        dst.block(SolutionBlocks::total_volume_multiplier) += transpose_operator(linear_operator(m_vect))*k_g_d_m_inv_density;

        Vector<double> k_density_mult;
        k_density_mult.reinit(src.block(SolutionBlocks::density).size());
        k_density_mult = op_k * src.block(SolutionBlocks::unfiltered_density_multiplier);
        dst.block(SolutionBlocks::total_volume_multiplier) -= transpose_operator(linear_operator(m_vect))*k_density_mult;

    }




    template<int dim>
    void TopOptSchurPreconditioner<dim>::vmult_step_5(BlockVector<double> &dst, const BlockVector<double> &src) const {
        {
            //First Block Inverse
            TimerOutput::Scope t(timer, "inverse 1");
            dst.block(SolutionBlocks::density_lower_slack_multiplier) = linear_operator(d_3_mat) * src.block(SolutionBlocks::density_lower_slack_multiplier) +
                    linear_operator(d_m_inv_mat) * src.block(SolutionBlocks::density_lower_slack);
            dst.block(SolutionBlocks::density_upper_slack_multiplier) = linear_operator(d_4_mat) * src.block(SolutionBlocks::density_upper_slack_multiplier) +
                    linear_operator(d_m_inv_mat) * src.block(SolutionBlocks::density_upper_slack);
            dst.block(SolutionBlocks::density_lower_slack) = linear_operator(d_m_inv_mat) * src.block(SolutionBlocks::density_lower_slack_multiplier);
            dst.block(SolutionBlocks::density_upper_slack) = linear_operator(d_m_inv_mat) * src.block(SolutionBlocks::density_upper_slack_multiplier);

        }
        std::cout << "inverse 1" << std::endl;

        {
            //Second Block Inverse
            TimerOutput::Scope t(timer, "inverse 2");
            dst.block(SolutionBlocks::unfiltered_density) =
                    linear_operator(d_8_mat) * src.block(SolutionBlocks::unfiltered_density);
        }

        std::cout << "inverse 2" << std::endl;
        {
            //Third Block Inverse
            TimerOutput::Scope t(timer, "inverse 3");

            dst.block(SolutionBlocks::displacement) = linear_operator(a_inv_direct) * src.block(SolutionBlocks::displacement_multiplier);
            dst.block(SolutionBlocks::displacement_multiplier) = linear_operator(a_inv_direct) * src.block(SolutionBlocks::displacement);
        }
        std::cout << "inverse 3" << std::endl;
        {
            //Fourth (ugly) Block Inverse
            TimerOutput::Scope t(timer, "inverse 4");

            auto op_h = linear_operator(b_mat)
                    - transpose_operator(linear_operator(c_mat)) * linear_operator(a_inv_direct) * linear_operator(e_mat)
                    - transpose_operator(linear_operator(e_mat)) * linear_operator(a_inv_direct) * linear_operator(c_mat);

//            auto op_h = linear_operator(approx_h_mat.block(SolutionBlocks::unfiltered_density,SolutionBlocks::unfiltered_density));

            auto op_g = linear_operator(f_mat) * linear_operator(d_8_mat) *
                      transpose_operator(linear_operator(f_mat));

            auto op_k_inv = -1 * op_g * linear_operator(d_m_inv_mat) * op_h -
                        linear_operator(d_m_mat);

            auto op_k = inverse_operator(op_k_inv, other_gmres, PreconditionIdentity());

            auto op_j_inv = -1 * op_h * linear_operator(d_m_inv_mat) * op_g -
                        linear_operator(d_m_mat);

            auto op_j = inverse_operator(op_j_inv, other_gmres, PreconditionIdentity());
            {
                TimerOutput::Scope t(timer, "inverse 4.0");

                pre_j = src.block(SolutionBlocks::density) + op_h * linear_operator(d_m_inv_mat) * src.block(SolutionBlocks::unfiltered_density_multiplier);

                pre_k = -1*op_g * linear_operator(d_m_inv_mat) * src.block(SolutionBlocks::density) + src.block(SolutionBlocks::unfiltered_density_multiplier);
            }
            {
                TimerOutput::Scope t(timer, "inverse 4.1");
                dst.block(SolutionBlocks::unfiltered_density_multiplier) = op_j * pre_j;
                dst.block(SolutionBlocks::density) = op_k * pre_k;
            }

        }
        std::cout << "inverse 4" << std::endl;
        {
            dst.block(SolutionBlocks::total_volume_multiplier) = src.block(SolutionBlocks::total_volume_multiplier);
        }
        std::cout << "inverse 5" << std::endl;
    }

    template<int dim>
    void TopOptSchurPreconditioner<dim>::get_sparsity_pattern(BlockDynamicSparsityPattern &bdsp) {
        mass_sparsity.copy_from(bdsp);
    }

    template<int dim>
    void TopOptSchurPreconditioner<dim>::assemble_mass_matrix(const BlockVector<double> &state,
                                                              const hp::FECollection<dim> &fe_collection,
                                                              const DoFHandler<dim> &dof_handler,
                                                              const AffineConstraints<double> &constraints,
                                                              const BlockSparsityPattern &bsp) {
        timer.reset();

        approx_h_mat.reinit(bsp);

        std::cout << approx_h_mat.n() << std::endl;

        /*Remove any values from old iterations*/
        QGauss<dim> nine_quadrature(fe_collection[0].degree + 1);
        QGauss<dim> ten_quadrature(fe_collection[1].degree + 1);

        hp::QCollection<dim> q_collection;
        q_collection.push_back(nine_quadrature);
        q_collection.push_back(ten_quadrature);

        hp::FEValues<dim> hp_fe_values(fe_collection,
                                       q_collection,
                                       update_values | update_quadrature_points |
                                       update_JxW_values | update_gradients);
        FullMatrix<double> cell_matrix;
        Vector<double>     cell_rhs;
        std::vector<types::global_dof_index> local_dof_indices;

        const FEValuesExtractors::Scalar densities(SolutionComponents::density<dim>);
        const FEValuesExtractors::Vector displacements(SolutionComponents::displacement<dim>);
        const FEValuesExtractors::Scalar unfiltered_densities(SolutionComponents::unfiltered_density<dim>);
        const FEValuesExtractors::Vector displacement_multipliers(SolutionComponents::displacement_multiplier<dim>);
        const FEValuesExtractors::Scalar unfiltered_density_multipliers(
                SolutionComponents::unfiltered_density_multiplier<dim>);
        const FEValuesExtractors::Scalar density_lower_slacks(SolutionComponents::density_lower_slack<dim>);
        const FEValuesExtractors::Scalar density_lower_slack_multipliers(
                SolutionComponents::density_lower_slack_multiplier<dim>);
        const FEValuesExtractors::Scalar density_upper_slacks(SolutionComponents::density_upper_slack<dim>);
        const FEValuesExtractors::Scalar density_upper_slack_multipliers(
                SolutionComponents::density_upper_slack_multiplier<dim>);
        const FEValuesExtractors::Scalar total_volume_multiplier(
                SolutionComponents::total_volume_multiplier<dim>);

        const Functions::ConstantFunction<dim> lambda(1.), mu(1.);

        BlockVector<double> filtered_unfiltered_density_solution = state;
        BlockVector<double> filter_adjoint_unfiltered_density_multiplier_solution = state;
        filtered_unfiltered_density_solution.block(SolutionBlocks::unfiltered_density) = 0;
        filter_adjoint_unfiltered_density_multiplier_solution.block(SolutionBlocks::unfiltered_density_multiplier) = 0;


        for (const auto &cell : dof_handler.active_cell_iterators()) {
            hp_fe_values.reinit(cell);
            const FEValues<dim> &fe_values = hp_fe_values.get_present_fe_values();
            cell_matrix.reinit(cell->get_fe().n_dofs_per_cell(),
                               cell->get_fe().n_dofs_per_cell());
            cell_rhs.reinit(cell->get_fe().n_dofs_per_cell());

            const unsigned int n_q_points = fe_values.n_quadrature_points;

            std::vector<double> old_density_values(n_q_points);
            std::vector<double> old_displacement_divs(n_q_points);
            std::vector<SymmetricTensor<2, dim>> old_displacement_symmgrads(
                    n_q_points);
            std::vector<double> old_displacement_multiplier_divs(n_q_points);
            std::vector<SymmetricTensor<2, dim>> old_displacement_multiplier_symmgrads(
                    n_q_points);
            std::vector<double> lambda_values(n_q_points);
            std::vector<double> mu_values(n_q_points);

            const unsigned int dofs_per_cell = cell->get_fe().n_dofs_per_cell();

            cell_matrix = 0;
            cell_rhs = 0;
            local_dof_indices.resize(cell->get_fe().n_dofs_per_cell());
            cell->get_dof_indices(local_dof_indices);

            lambda.value_list(fe_values.get_quadrature_points(), lambda_values);
            mu.value_list(fe_values.get_quadrature_points(), mu_values);

            fe_values[densities].get_function_values(state,
                                                     old_density_values);
            fe_values[displacements].get_function_divergences(state,
                                                              old_displacement_divs);
            fe_values[displacements].get_function_symmetric_gradients(
                    state, old_displacement_symmgrads);
            fe_values[displacement_multipliers].get_function_divergences(
                    state, old_displacement_multiplier_divs);
            fe_values[displacement_multipliers].get_function_symmetric_gradients(
                    state, old_displacement_multiplier_symmgrads);

            Tensor<1, dim> traction;
            traction[1] = -1;

            for (unsigned int q_point = 0; q_point < n_q_points; ++q_point) {

                for (unsigned int i = 0; i < dofs_per_cell; ++i) {

                    const double unfiltered_density_phi_i = fe_values[unfiltered_densities].value(i,
                                                                                                  q_point);

                    for (unsigned int j = 0; j < dofs_per_cell; ++j) {

                        const double unfiltered_density_phi_j = fe_values[unfiltered_densities].value(j,
                                                                                                      q_point);


                        //Equation 0
                        double value =   unfiltered_density_phi_i
                                       * unfiltered_density_phi_j
                                       * (-1 * Input::density_penalty_exponent * Input::density_penalty_exponent - Input::density_penalty_exponent)
                                       * std::pow(old_density_values[q_point],Input::density_penalty_exponent - 2)
                                       *
                                       (old_displacement_divs[q_point] * old_displacement_multiplier_divs[q_point]
                                        * lambda_values[q_point]
                                        +
                                        2 * mu_values[q_point] * (old_displacement_symmgrads[q_point] *
                                                                  old_displacement_multiplier_symmgrads[q_point]));

                        if (value != 0)
                        {
                            cell_matrix(i, j) +=
                                    fe_values.JxW(q_point) * value ;
                        }
                    }

                }

            }

            constraints.distribute_local_to_global(cell_matrix, local_dof_indices, approx_h_mat);
        }

    }


    template<int dim>
    void TopOptSchurPreconditioner<dim>::print_stuff(const BlockSparseMatrix<double> &matrix) {

        const unsigned int vec_size = matrix.block(SolutionBlocks::density, SolutionBlocks::density).n();
        FullMatrix<double> orig_mat(vec_size, vec_size);
        FullMatrix<double> est_mass_mat(vec_size, vec_size);

        for (unsigned int j = 0; j < vec_size; j++) {
            Vector<double> unit_vector;
            unit_vector.reinit(vec_size);
            unit_vector=0;
            unit_vector[j] = 1;
            Vector<double> transformed_unit_vector_orig;
            Vector<double> transformed_unit_vector_mass;
            transformed_unit_vector_orig = linear_operator(b_mat)* unit_vector
                    - transpose_operator(linear_operator(c_mat)) * linear_operator(a_inv_direct)
                    * linear_operator(e_mat) * unit_vector
                    -
                    transpose_operator(linear_operator(e_mat)) * linear_operator(a_inv_direct)  *
                    linear_operator(c_mat) * unit_vector;
            transformed_unit_vector_mass = linear_operator(approx_h_mat.block(SolutionBlocks::unfiltered_density, SolutionBlocks::unfiltered_density)) * unit_vector;

            for (unsigned int i = 0; i < vec_size; i++) {
                orig_mat(i,j) = transformed_unit_vector_orig[i];
                est_mass_mat(i,j) = transformed_unit_vector_mass[i];
            }
        }

        std::ofstream OGMat("original_matrix.csv");
        std::ofstream MassMat("mass_estimated.csv");

        for (unsigned int i = 0; i < vec_size; i++)
        {
            OGMat << orig_mat(i, 0);
            MassMat << est_mass_mat(i, 0);
            for (unsigned int j = 1; j < vec_size; j++)
            {
                OGMat << "," << orig_mat(i, j);
                MassMat << "," << est_mass_mat(i, j);
            }
            OGMat << "\n";
            MassMat << "\n";
        }
    OGMat.close();
    MassMat.close();
    }
}
template class SAND::TopOptSchurPreconditioner<2>;
template class SAND::TopOptSchurPreconditioner<3>;