//
// Created by justin on 2/17/21.
//
#include <deal.II/lac/linear_operator.h>
#include <deal.II/lac/sparse_direct.h>
#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/base/timer.h>
#include "../include/schur_preconditioner.h"
#include "../include/input_information.h"
#include "../include/sand_tools.h"
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
        {
            TimerOutput::Scope t(timer, "diag stuff");
            for (auto&[dof_index, boundary_value]: boundary_values) {
                const types::global_dof_index disp_start_index = system_matrix.get_row_indices().block_start(
                        SolutionBlocks::displacement);
                const types::global_dof_index disp_mult_start_index = system_matrix.get_row_indices().block_start(
                        SolutionBlocks::displacement_multiplier);
                const types::global_dof_index n_u = system_matrix.block(SolutionBlocks::displacement,
                                                                        SolutionBlocks::displacement).m();
                if ((dof_index >= disp_start_index) && (dof_index < disp_start_index + n_u)) {
                    double diag_val = system_matrix.block(SolutionBlocks::displacement,
                                                          SolutionBlocks::displacement).el(
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
            for (auto&[dof_index, boundary_value]: boundary_values) {
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
                    system_matrix.block(SolutionBlocks::displacement_multiplier,
                                        SolutionBlocks::displacement_multiplier).set(
                            dof_index - disp_mult_start_index, dof_index - disp_mult_start_index, 0);
                }
            }
        }
        if (Input::solver_choice==SolverOptions::inexact_K_with_inexact_A_gmres)
        {

        }
        else
        {
            TimerOutput::Scope t(timer, "build A inv");
            a_inv_direct.initialize(a_mat);
        }
        {
            TimerOutput::Scope t(timer, "reinit diag matrices");
            d_3_mat.reinit(matrix.block(SolutionBlocks::density, SolutionBlocks::density).get_sparsity_pattern());
            d_4_mat.reinit(matrix.block(SolutionBlocks::density, SolutionBlocks::density).get_sparsity_pattern());
            d_5_mat.reinit(matrix.block(SolutionBlocks::density, SolutionBlocks::density).get_sparsity_pattern());
            d_6_mat.reinit(matrix.block(SolutionBlocks::density, SolutionBlocks::density).get_sparsity_pattern());
            d_7_mat.reinit(matrix.block(SolutionBlocks::density, SolutionBlocks::density).get_sparsity_pattern());
            d_8_mat.reinit(matrix.block(SolutionBlocks::density, SolutionBlocks::density).get_sparsity_pattern());
            d_m_inv_mat.reinit(matrix.block(SolutionBlocks::density, SolutionBlocks::density).get_sparsity_pattern());
        }
        {
            TimerOutput::Scope t(timer, "build diag matrices");
            for (const auto cell: dof_handler.active_cell_iterators())
            {
                const double i = cell->active_cell_index();
                const double m = cell->measure();
                double d_3_value = -1 * state.block(SolutionBlocks::density_lower_slack_multiplier)[i] /
                                   (m * state.block(SolutionBlocks::density_lower_slack)[i]);
                double d_4_value = -1 * state.block(SolutionBlocks::density_upper_slack_multiplier)[i] /
                                   (m * state.block(SolutionBlocks::density_upper_slack)[i]);
                double d_5_value = state.block(SolutionBlocks::density_lower_slack_multiplier)[i] /
                                   (state.block(SolutionBlocks::density_lower_slack)[i]);
                double d_6_value = state.block(SolutionBlocks::density_upper_slack_multiplier)[i] /
                                   (state.block(SolutionBlocks::density_upper_slack)[i]);
                double d_7_value = (m * (state.block(SolutionBlocks::density_lower_slack_multiplier)[i] *
                                         state.block(SolutionBlocks::density_upper_slack)[i] +
                                         state.block(SolutionBlocks::density_upper_slack_multiplier)[i] *
                                         state.block(SolutionBlocks::density_lower_slack)[i]))
                                   / (state.block(SolutionBlocks::density_lower_slack)[i] *
                                      state.block(SolutionBlocks::density_upper_slack)[i]);
                double d_8_value = (state.block(SolutionBlocks::density_lower_slack)[i] *
                                    state.block(SolutionBlocks::density_upper_slack)[i])
                                   / (m * (state.block(SolutionBlocks::density_lower_slack_multiplier)[i] *
                                           state.block(SolutionBlocks::density_upper_slack)[i] +
                                           state.block(SolutionBlocks::density_upper_slack_multiplier)[i] *
                                           state.block(SolutionBlocks::density_lower_slack)[i]));
                d_3_mat.set(i, i, d_3_value);
                d_4_mat.set(i, i, d_4_value);
                d_5_mat.set(i, i, d_5_value);
                d_6_mat.set(i, i, d_6_value);
                d_7_mat.set(i, i, d_7_value);
                d_8_mat.set(i, i, d_8_value);
                d_m_inv_mat.set(i, i, 1 / m);

            }
        }

        pre_j.reinit(matrix.block(SolutionBlocks::density,SolutionBlocks::density).n());
        pre_k.reinit(matrix.block(SolutionBlocks::density,SolutionBlocks::density).n());
        g_d_m_inv_density.reinit(matrix.block(SolutionBlocks::density,SolutionBlocks::density).n());
        k_g_d_m_inv_density.reinit(matrix.block(SolutionBlocks::density,SolutionBlocks::density).n());

        auto op_g = linear_operator(f_mat) * linear_operator(d_8_mat) *
                    transpose_operator(linear_operator(f_mat));

        auto op_h = op_g;
        PreconditionSSOR<SparseMatrix<double>> preconditioner;
        preconditioner.initialize(a_mat, 1.2);
        SolverControl            solver_control(1000, 1e-12);
        SolverCG<Vector<double>> a_solver_cg(solver_control);
        auto a_inv_op = inverse_operator(linear_operator(a_mat),a_solver_cg,preconditioner);

        if (Input::solver_choice==SolverOptions::inexact_K_with_inexact_A_gmres)
        {
            op_h = linear_operator(b_mat)
                   - transpose_operator(linear_operator(c_mat)) * a_inv_op * linear_operator(e_mat)
                   - transpose_operator(linear_operator(e_mat)) * a_inv_op * linear_operator(c_mat);
        }
        else
        {
            op_h = linear_operator(b_mat)
                   - transpose_operator(linear_operator(c_mat)) * linear_operator(a_inv_direct) * linear_operator(e_mat)
                   - transpose_operator(linear_operator(e_mat)) * linear_operator(a_inv_direct) * linear_operator(c_mat);
        }

        if(Input::solver_choice == SolverOptions::inexact_K_with_inexact_A_gmres || Input::solver_choice == SolverOptions::inexact_K_with_exact_A_gmres)
        {

        }
        else
        {
            {
                TimerOutput::Scope t(timer, "build g_mat");
                g_mat.reinit(b_mat.n(), b_mat.n());
                build_matrix_element_by_element(op_g, g_mat);
            }


            {
                TimerOutput::Scope t(timer, "build h_mat");
                h_mat.reinit(b_mat.n(), b_mat.n());
                build_matrix_element_by_element(op_h, h_mat);
            }

            {
                TimerOutput::Scope t(timer, "build k_inv_mat");
                auto op_k_inv = -1 * op_g * linear_operator(d_m_inv_mat) * op_h -
                                linear_operator(d_m_mat);
                k_inv_mat.reinit(b_mat.n(), b_mat.n());
                build_matrix_element_by_element(op_k_inv, k_inv_mat);
            }
        }



        if (Input::solver_choice == SolverOptions::exact_preconditioner_with_gmres)
        {
            TimerOutput::Scope t(timer, "invert k_mat");
            k_mat.copy_from(k_inv_mat);
            k_mat.invert();
        }

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
        dst = src;
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
        if (Input::solver_choice == SolverOptions::inexact_K_with_inexact_A_gmres)
        {
            PreconditionSSOR<SparseMatrix<double>> preconditioner;
            preconditioner.initialize(a_mat, 1.2);
            SolverControl            solver_control(1000, 1e-12);
            SolverCG<Vector<double>> a_solver_cg(solver_control);
            auto a_inv_op = inverse_operator(linear_operator(a_mat),a_solver_cg,preconditioner);
            dst.block(SolutionBlocks::density)+= -1 * transpose_operator(linear_operator(e_mat)) * a_inv_op * src.block(SolutionBlocks::displacement)
                                                 - transpose_operator(linear_operator(c_mat)) * a_inv_op * src.block(SolutionBlocks::displacement_multiplier);
        }
        else
        {
            dst.block(SolutionBlocks::density)+= -1 * transpose_operator(linear_operator(e_mat)) * linear_operator(a_inv_direct) * src.block(SolutionBlocks::displacement)
                                                 - transpose_operator(linear_operator(c_mat)) * linear_operator(a_inv_direct) * src.block(SolutionBlocks::displacement_multiplier);
        }

    }

    template<int dim>
    void TopOptSchurPreconditioner<dim>::vmult_step_4(BlockVector<double> &dst, const BlockVector<double> &src) const {
        dst = src;
        Vector<double> k_density_mult;
        k_density_mult.reinit(src.block(SolutionBlocks::density).size());



        if (Input::solver_choice == SolverOptions::exact_preconditioner_with_gmres)
        {
            g_d_m_inv_density = linear_operator(g_mat) * linear_operator(d_m_inv_mat) * src.block(SolutionBlocks::density);
            k_g_d_m_inv_density = linear_operator(k_mat) * g_d_m_inv_density;
            k_density_mult = linear_operator(k_mat) * src.block(SolutionBlocks::unfiltered_density_multiplier);
        }

        else if (Input::solver_choice == SolverOptions::inexact_K_with_exact_A_gmres)
        {
            auto op_g = linear_operator(f_mat) * linear_operator(d_8_mat) *
                    transpose_operator(linear_operator(f_mat));

            auto op_h = linear_operator(b_mat)
                   - transpose_operator(linear_operator(c_mat)) * linear_operator(a_inv_direct) * linear_operator(e_mat)
                   - transpose_operator(linear_operator(e_mat)) * linear_operator(a_inv_direct) * linear_operator(c_mat);

            auto op_k_inv = -1 * op_g * linear_operator(d_m_inv_mat) * op_h - linear_operator(d_m_mat);

            g_d_m_inv_density = op_g * linear_operator(d_m_inv_mat) * src.block(SolutionBlocks::density);

            SolverControl step_4_gmres_control_1 (10000, g_d_m_inv_density.l2_norm()*1e-6);
            SolverGMRES<Vector<double>> step_4_gmres_1 (step_4_gmres_control_1);
            try {
                k_g_d_m_inv_density = inverse_operator(op_k_inv, step_4_gmres_1, PreconditionIdentity()) *
                                      g_d_m_inv_density;
            } catch (std::exception &exc)
            {
                std::cerr << "Failure of linear solver step_4_gmres_1" << std::endl;
                std::cout << "first residual: " << step_4_gmres_control_1.initial_value() << std::endl;
                std::cout << "last residual: " << step_4_gmres_control_1.last_value() << std::endl;
                throw;
            }

            SolverControl step_4_gmres_control_2 (10000, src.block(SolutionBlocks::unfiltered_density_multiplier).l2_norm()*1e-6);
            SolverGMRES<Vector<double>> step_4_gmres_2 (step_4_gmres_control_2);
            try {
                k_density_mult = inverse_operator(op_k_inv,step_4_gmres_2, PreconditionIdentity()) *
                                 src.block(SolutionBlocks::unfiltered_density_multiplier);
            } catch (std::exception &exc)
            {
                std::cerr << "Failure of linear solver step_4_gmres_2" << std::endl;
                std::cout << "first residual: " << step_4_gmres_control_2.initial_value() << std::endl;
                std::cout << "last residual: " << step_4_gmres_control_2.last_value() << std::endl;
                throw;
            }
        }
        else if (Input::solver_choice == SolverOptions::inexact_K_with_inexact_A_gmres)
        {
            PreconditionSSOR<SparseMatrix<double>> preconditioner;
            preconditioner.initialize(a_mat, 1.2);
            SolverControl            solver_control(1000, 1e-12);
            SolverCG<Vector<double>> a_solver_cg(solver_control);
            auto a_inv_op = inverse_operator(linear_operator(a_mat),a_solver_cg,preconditioner);

            auto op_g = linear_operator(f_mat) * linear_operator(d_8_mat) *
                        transpose_operator(linear_operator(f_mat));

            auto op_h = linear_operator(b_mat)
                        - transpose_operator(linear_operator(c_mat)) * a_inv_op * linear_operator(e_mat)
                        - transpose_operator(linear_operator(e_mat)) * a_inv_op * linear_operator(c_mat);

            auto op_k_inv = -1 * op_g * linear_operator(d_m_inv_mat) * op_h - linear_operator(d_m_mat);

            g_d_m_inv_density = op_g * linear_operator(d_m_inv_mat) * src.block(SolutionBlocks::density);

            SolverControl step_4_gmres_control_1 (10000, g_d_m_inv_density.l2_norm()*1e-6);
            SolverGMRES<Vector<double>> step_4_gmres_1 (step_4_gmres_control_1);
            try {
                k_g_d_m_inv_density = inverse_operator(op_k_inv, step_4_gmres_1, PreconditionIdentity()) *
                                      g_d_m_inv_density;
            } catch (std::exception &exc)
            {
                std::cerr << "Failure of linear solver step_4_gmres_1" << std::endl;
                std::cout << "first residual: " << step_4_gmres_control_1.initial_value() << std::endl;
                std::cout << "last residual: " << step_4_gmres_control_1.last_value() << std::endl;
                throw;
            }

            SolverControl step_4_gmres_control_2 (10000, src.block(SolutionBlocks::unfiltered_density_multiplier).l2_norm()*1e-6);
            SolverGMRES<Vector<double>> step_4_gmres_2 (step_4_gmres_control_2);
            try {
                k_density_mult = inverse_operator(op_k_inv,step_4_gmres_2, PreconditionIdentity()) *
                                 src.block(SolutionBlocks::unfiltered_density_multiplier);
            } catch (std::exception &exc)
            {
                std::cerr << "Failure of linear solver step_4_gmres_2" << std::endl;
                std::cout << "first residual: " << step_4_gmres_control_2.initial_value() << std::endl;
                std::cout << "last residual: " << step_4_gmres_control_2.last_value() << std::endl;
                throw;
            }
        }
        else
        {
            std::cout << "shouldn't get here";
            throw;
        }


        dst.block(SolutionBlocks::total_volume_multiplier) += transpose_operator(linear_operator(m_vect))*k_g_d_m_inv_density;
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


        {
            //Second Block Inverse
            TimerOutput::Scope t(timer, "inverse 2");
            dst.block(SolutionBlocks::unfiltered_density) =
                    linear_operator(d_8_mat) * src.block(SolutionBlocks::unfiltered_density);
        }


        {
            //Third Block Inverse
            TimerOutput::Scope t(timer, "inverse 3");
            if(Input::solver_choice == SolverOptions::inexact_K_with_inexact_A_gmres)
            {
                PreconditionSSOR<SparseMatrix<double>> preconditioner;
                preconditioner.initialize(a_mat, 1.2);
                SolverControl            solver_control(1000, 1e-12);
                SolverCG<Vector<double>> a_solver_cg(solver_control);
                auto a_inv_op = inverse_operator(linear_operator(a_mat),a_solver_cg,preconditioner);
                dst.block(SolutionBlocks::displacement) = a_inv_op * src.block(SolutionBlocks::displacement_multiplier);
                dst.block(SolutionBlocks::displacement_multiplier) = a_inv_op * src.block(SolutionBlocks::displacement);
            } else
            {
                dst.block(SolutionBlocks::displacement) = linear_operator(a_inv_direct) * src.block(SolutionBlocks::displacement_multiplier);
                dst.block(SolutionBlocks::displacement_multiplier) = linear_operator(a_inv_direct) * src.block(SolutionBlocks::displacement);
            }

        }


        {
            //Fourth (ugly) Block Inverse
            TimerOutput::Scope t(timer, "inverse 4");



            if (Input::solver_choice == SolverOptions::exact_preconditioner_with_gmres)
            {
                pre_j = src.block(SolutionBlocks::density) + linear_operator(h_mat) * linear_operator(d_m_inv_mat) * src.block(SolutionBlocks::unfiltered_density_multiplier);
                pre_k = -1* linear_operator(g_mat) * linear_operator(d_m_inv_mat) * src.block(SolutionBlocks::density) + src.block(SolutionBlocks::unfiltered_density_multiplier);
                dst.block(SolutionBlocks::unfiltered_density_multiplier) = transpose_operator(linear_operator(k_mat)) * pre_j;
                dst.block(SolutionBlocks::density) = linear_operator(k_mat) * pre_k;
            }

            else if (Input::solver_choice == SolverOptions::inexact_K_with_exact_A_gmres)
            {
                auto op_g = linear_operator(f_mat) * linear_operator(d_8_mat) *
                            transpose_operator(linear_operator(f_mat));

                auto op_h = linear_operator(b_mat)
                            - transpose_operator(linear_operator(c_mat)) * linear_operator(a_inv_direct) * linear_operator(e_mat)
                            - transpose_operator(linear_operator(e_mat)) * linear_operator(a_inv_direct) * linear_operator(c_mat);

                auto op_k_inv = -1 * op_g * linear_operator(d_m_inv_mat) * op_h - linear_operator(d_m_mat);

                pre_j = src.block(SolutionBlocks::density) + op_h * linear_operator(d_m_inv_mat) * src.block(SolutionBlocks::unfiltered_density_multiplier);
                pre_k = -1* op_g * linear_operator(d_m_inv_mat) * src.block(SolutionBlocks::density) + src.block(SolutionBlocks::unfiltered_density_multiplier);

                SolverControl step_5_gmres_control_1 (10000, pre_j.l2_norm()*1e-6);
                SolverGMRES<Vector<double>> step_5_gmres_1 (step_5_gmres_control_1);
                try {
                    dst.block(SolutionBlocks::unfiltered_density_multiplier) = inverse_operator(transpose_operator(op_k_inv), step_5_gmres_1, PreconditionIdentity()) *
                                          pre_j;
                } catch (std::exception &exc)
                {
                    std::cerr << "Failure of linear solver step_5_gmres_1" << std::endl;
                    std::cout << "first residual: " << step_5_gmres_control_1.initial_value() << std::endl;
                    std::cout << "last residual: " << step_5_gmres_control_1.last_value() << std::endl;
                    throw;
                }

                SolverControl step_5_gmres_control_2 (10000, pre_k.l2_norm()*1e-6);
                SolverGMRES<Vector<double>> step_5_gmres_2 (step_5_gmres_control_2);
                try {
                    dst.block(SolutionBlocks::density) = inverse_operator(op_k_inv, step_5_gmres_2, PreconditionIdentity()) *
                                                                               pre_k;
                } catch (std::exception &exc)
                {
                    std::cerr << "Failure of linear solver step_5_gmres_2" << std::endl;
                    std::cout << "first residual: " << step_5_gmres_control_2.initial_value() << std::endl;
                    std::cout << "last residual: " << step_5_gmres_control_2.last_value() << std::endl;
                    throw;
                }
            }
            else if (Input::solver_choice == SolverOptions::inexact_K_with_inexact_A_gmres)
            {
                PreconditionSSOR<SparseMatrix<double>> preconditioner;
                preconditioner.initialize(a_mat, 1.2);
                SolverControl            solver_control(1000, 1e-12);
                SolverCG<Vector<double>> a_solver_cg(solver_control);
                auto a_inv_op = inverse_operator(linear_operator(a_mat),a_solver_cg,preconditioner);

                auto op_g = linear_operator(f_mat) * linear_operator(d_8_mat) *
                            transpose_operator(linear_operator(f_mat));

                auto op_h = linear_operator(b_mat)
                            - transpose_operator(linear_operator(c_mat)) * a_inv_op * linear_operator(e_mat)
                            - transpose_operator(linear_operator(e_mat)) * a_inv_op * linear_operator(c_mat);

                auto op_k_inv = -1 * op_g * linear_operator(d_m_inv_mat) * op_h - linear_operator(d_m_mat);

                pre_j = src.block(SolutionBlocks::density) + op_h * linear_operator(d_m_inv_mat) * src.block(SolutionBlocks::unfiltered_density_multiplier);
                pre_k = -1* op_g * linear_operator(d_m_inv_mat) * src.block(SolutionBlocks::density) + src.block(SolutionBlocks::unfiltered_density_multiplier);

                SolverControl step_5_gmres_control_1 (10000, pre_j.l2_norm()*1e-6);
                SolverGMRES<Vector<double>> step_5_gmres_1 (step_5_gmres_control_1);
                try {
                    dst.block(SolutionBlocks::unfiltered_density_multiplier) = inverse_operator(transpose_operator(op_k_inv), step_5_gmres_1, PreconditionIdentity()) *
                                                                               pre_j;
                } catch (std::exception &exc)
                {
                    std::cerr << "Failure of linear solver step_5_gmres_1" << std::endl;
                    std::cout << "first residual: " << step_5_gmres_control_1.initial_value() << std::endl;
                    std::cout << "last residual: " << step_5_gmres_control_1.last_value() << std::endl;
                    throw;
                }

                SolverControl step_5_gmres_control_2 (10000, pre_k.l2_norm()*1e-6);
                SolverGMRES<Vector<double>> step_5_gmres_2 (step_5_gmres_control_2);
                try {
                    dst.block(SolutionBlocks::density) = inverse_operator(op_k_inv, step_5_gmres_2, PreconditionIdentity()) *
                                                         pre_k;
                } catch (std::exception &exc)
                {
                    std::cerr << "Failure of linear solver step_5_gmres_2" << std::endl;
                    std::cout << "first residual: " << step_5_gmres_control_2.initial_value() << std::endl;
                    std::cout << "last residual: " << step_5_gmres_control_2.last_value() << std::endl;
                    throw;
                }
            }
            else
            {
                std::cout << "shouldn't get here";
                throw;
            }

        }
        {
            dst.block(SolutionBlocks::total_volume_multiplier) = src.block(SolutionBlocks::total_volume_multiplier);
        }
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
                    const double density_phi_i = fe_values[densities].value(i,q_point);

                    for (unsigned int j = 0; j < dofs_per_cell; ++j) {

                        const double unfiltered_density_phi_j = fe_values[unfiltered_densities].value(j,
                                                                                                      q_point);
                        const double density_phi_j = fe_values[densities].value(j,q_point);



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


                        value +=   density_phi_i
                                * density_phi_j
                                * (-2 * Input::density_penalty_exponent * Input::density_penalty_exponent)
                                * std::pow(old_density_values[q_point],Input::density_penalty_exponent - 2)
                                * old_displacement_symmgrads[q_point].norm() *
                                old_displacement_multiplier_symmgrads[q_point].norm();

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
    void TopOptSchurPreconditioner<dim>::print_stuff(const BlockSparseMatrix<double> &matrix)
    {

//        print_matrix("OAmat.csv",a_mat);
//        print_matrix("OBmat.csv",b_mat);
//        print_matrix("OCmat.csv",c_mat);
//        print_matrix("OEmat.csv",e_mat);
//        print_matrix("OFmat.csv",f_mat);
        FullMatrix<double> g_mat;
        FullMatrix<double> h_mat;
        FullMatrix<double> k_inv_mat;
        g_mat.reinit(b_mat.m(),b_mat.n());
        h_mat.reinit(b_mat.m(),b_mat.n());
        k_inv_mat.reinit(b_mat.m(),b_mat.n());
        auto op_g = linear_operator(f_mat) * linear_operator(d_8_mat) *
                    transpose_operator(linear_operator(f_mat));

        auto op_h = linear_operator(b_mat)
                    - transpose_operator(linear_operator(c_mat)) * linear_operator(a_inv_direct) * linear_operator(e_mat)
                    - transpose_operator(linear_operator(e_mat)) * linear_operator(a_inv_direct) * linear_operator(c_mat);

        auto op_k_inv = -1 * op_g * linear_operator(d_m_inv_mat) * op_h - linear_operator(d_m_mat);
        build_matrix_element_by_element(op_g,g_mat);
        build_matrix_element_by_element(op_h,h_mat);
        build_matrix_element_by_element(op_k_inv,k_inv_mat);
//        print_matrix("OGmat.csv",g_mat);
//        print_matrix("OHmat.csv",h_mat);
//        print_matrix("OKinvmat.csv",k_inv_mat);


    }
}
template class SAND::TopOptSchurPreconditioner<2>;
template class SAND::TopOptSchurPreconditioner<3>;