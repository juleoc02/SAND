//
// Created by justin on 2/17/21.
//
#include "../include/markov_filter.h"
#include <deal.II/lac/linear_operator.h>
#include <deal.II/lac/sparse_direct.h>
#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/solver_bicgstab.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/base/timer.h>
#include "../include/schur_preconditioner.h"
#include "../include/parameters_and_components.h"

namespace SAND {

    using namespace dealii;

    TopOptSchurPreconditioner::TopOptSchurPreconditioner()
            :
            n_rows(0),
            n_columns(0),
            n_block_rows(0),
            n_block_columns(0),
            diag_solver_control(10000, 1e-6),
            diag_cg(diag_solver_control),
            other_solver_control(10000, 1e-6),
            other_bicgstab(other_solver_control),
            timer (std::cout, TimerOutput::summary,
                   TimerOutput::wall_times)
    {
    }

    void TopOptSchurPreconditioner::initialize(BlockSparseMatrix<double> &matrix, std::map<types::global_dof_index, double> boundary_values)
    {
        TimerOutput::Scope t(timer, "initialize");
        for (auto& [dof_index, boundary_value] : boundary_values)
        {
            const types::global_dof_index disp_start_index = matrix.get_row_indices().block_start(SolutionBlocks::displacement);
            const types::global_dof_index disp_mult_start_index = matrix.get_row_indices().block_start(SolutionBlocks::displacement_multiplier);
            const types::global_dof_index n_u= matrix.block(SolutionBlocks::displacement,SolutionBlocks::displacement).m();
            if ((dof_index >= disp_start_index) && (dof_index < disp_start_index+n_u))
            {
                double diag_val = matrix.block(SolutionBlocks::displacement,SolutionBlocks::displacement).el(dof_index - disp_start_index,dof_index - disp_start_index);
                matrix.block(SolutionBlocks::displacement,SolutionBlocks::displacement_multiplier).set(dof_index - disp_start_index,dof_index - disp_start_index, diag_val);
            }
            else if ((dof_index >= disp_mult_start_index) && (dof_index < disp_mult_start_index + n_u))
            {
                double diag_val = matrix.block(SolutionBlocks::displacement_multiplier,SolutionBlocks::displacement_multiplier).el(dof_index-disp_mult_start_index,dof_index-disp_mult_start_index);
                matrix.block(SolutionBlocks::displacement_multiplier,SolutionBlocks::displacement).set(dof_index-disp_mult_start_index,dof_index-disp_mult_start_index, diag_val);
            }
        }

        //set diagonal to 0?
        for (auto& [dof_index, boundary_value] : boundary_values)
        {
            const types::global_dof_index disp_start_index = matrix.get_row_indices().block_start(SolutionBlocks::displacement);
            const types::global_dof_index disp_mult_start_index = matrix.get_row_indices().block_start(SolutionBlocks::displacement_multiplier);
            const types::global_dof_index n_u= matrix.block(SolutionBlocks::displacement,SolutionBlocks::displacement).m();
            if ((dof_index >= disp_start_index) && (dof_index < disp_start_index+n_u))
            {
               matrix.block(SolutionBlocks::displacement,SolutionBlocks::displacement).set(dof_index - disp_start_index,dof_index - disp_start_index, 0);
            }
            else if ((dof_index >= disp_mult_start_index) && (dof_index < disp_mult_start_index + n_u))
            {
                matrix.block(SolutionBlocks::displacement_multiplier,SolutionBlocks::displacement_multiplier).set(dof_index-disp_mult_start_index,dof_index-disp_mult_start_index, 0);
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

        op_diag_sum_inverse = inverse_operator(op_diag_1+op_diag_2, diag_cg, PreconditionIdentity());

        elastic_direct.initialize(matrix.block(SolutionBlocks::displacement,SolutionBlocks::displacement_multiplier));
        op_elastic_inverse = linear_operator(elastic_direct);

        op_scaled_identity = linear_operator(matrix.block(SolutionBlocks::density_upper_slack,SolutionBlocks::density_upper_slack_multiplier));

        scaled_direct.initialize(matrix.block(SolutionBlocks::density_upper_slack,SolutionBlocks::density_upper_slack_multiplier));
        op_scaled_inverse = linear_operator(scaled_direct);

        op_fddf_chunk = -1 * op_filter * op_diag_sum_inverse * transpose_operator(op_filter);
        op_bcaeeac_chunk =  (op_density_density
                                        -
                                        transpose_operator(op_displacement_density) * op_elastic_inverse *
                                        op_displacement_multiplier_density
                                        -
                                        transpose_operator(op_displacement_multiplier_density) *
                                        op_elastic_inverse * op_displacement_density
        );


        op_top_big_inverse = inverse_operator(op_bcaeeac_chunk * op_scaled_inverse * op_fddf_chunk
                                                         - op_scaled_identity,
                                                         other_bicgstab,
                                                         PreconditionIdentity());

        op_bot_big_inverse = inverse_operator(op_fddf_chunk * op_scaled_inverse *  op_bcaeeac_chunk
                                                         - op_scaled_identity,
                                                         other_bicgstab,
                                                         PreconditionIdentity());
    }

    void TopOptSchurPreconditioner::vmult(BlockVector<double> &dst, const BlockVector<double> &src) {
        BlockVector<double> temp_src;
        std::cout << "part 1" << std::endl;
        {
            TimerOutput::Scope t(timer, "part 1");
            vmult_step_1(dst, src);
            temp_src = dst;
        }
        std::cout << "part 2" << std::endl;
        {
            TimerOutput::Scope t(timer, "part 2");
            vmult_step_2(dst, temp_src);
            temp_src = dst;
        }
        std::cout << "part 3" << std::endl;
        {
            TimerOutput::Scope t(timer, "part 3");
            vmult_step_3(dst, temp_src);
            temp_src = dst;
        }
        std::cout << "part 4" << std::endl;

        vmult_step_4(dst, temp_src);

        std::cout << "vmult done" << std::endl;
        timer.print_summary();
    }

    void TopOptSchurPreconditioner::Tvmult(BlockVector<double> &dst, const BlockVector<double> &src) {
        dst = src;
        std::cout << "bad";
    }

    void TopOptSchurPreconditioner::vmult_add(BlockVector<double> &dst, const BlockVector<double> &src) {
        BlockVector<double> dst_temp = dst;
        vmult(dst_temp, src);
        dst += dst_temp;
        std::cout << "bad";
    }

    void TopOptSchurPreconditioner::Tvmult_add(BlockVector<double> &dst, const BlockVector<double> &src) {
        dst = dst + src;
        std::cout << "bad";
    }

    void TopOptSchurPreconditioner::vmult_step_1(BlockVector<double> &dst, const BlockVector<double> &src) {
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

    void TopOptSchurPreconditioner::vmult_step_2(BlockVector<double> &dst, const BlockVector<double> &src) {
        dst = src;
        dst.block(SolutionBlocks::unfiltered_density_multiplier) +=
                -1 * op_filter * op_diag_sum_inverse * src.block(SolutionBlocks::unfiltered_density);

    }

    void TopOptSchurPreconditioner::vmult_step_3(BlockVector<double> &dst, const BlockVector<double> &src) {
        dst = src;
        dst.block(SolutionBlocks::density) +=
                -1 * transpose_operator(op_displacement_density) * op_elastic_inverse *
                src.block(SolutionBlocks::displacement_multiplier)
                +
                -1 * transpose_operator(op_displacement_multiplier_density) * op_elastic_inverse *
                src.block(SolutionBlocks::displacement);
    }

    void TopOptSchurPreconditioner::vmult_step_4(BlockVector<double> &dst, const BlockVector<double> &src) {
        std::cout << "in part 4" << std::endl;
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
        std::cout << "part 4.1 done" << std::endl;
        {
            //Second Block Inverse
            TimerOutput::Scope t(timer, "inverse 2");
            dst.block(SolutionBlocks::unfiltered_density) =
                    op_diag_sum_inverse * src.block(SolutionBlocks::unfiltered_density);
        }
        std::cout << "part 4.2 done" << std::endl;
        {
            //Third Block Inverse
            TimerOutput::Scope t(timer, "inverse 3");
            dst.block(SolutionBlocks::displacement) =
                    op_elastic_inverse * src.block(SolutionBlocks::displacement_multiplier);
            dst.block(SolutionBlocks::displacement_multiplier) =
                    op_elastic_inverse * src.block(SolutionBlocks::displacement);
        }
        std::cout << "part 4.3 done" << std::endl;
        {
            //Fourth (ugly) Block Inverse
            TimerOutput::Scope t(timer, "inverse 4");
            dst.block(SolutionBlocks::unfiltered_density_multiplier) =
                    (op_bcaeeac_chunk * op_scaled_inverse * src.block(SolutionBlocks::unfiltered_density_multiplier));
            std::cout << "part 4.4.1 done" << std::endl;
            dst.block(SolutionBlocks::unfiltered_density_multiplier) =
                    (op_top_big_inverse * dst.block(SolutionBlocks::unfiltered_density_multiplier));

            std::cout << "part 4.4.2 done" << std::endl;
            dst.block(SolutionBlocks::unfiltered_density_multiplier) +=
                    (op_top_big_inverse * src.block(SolutionBlocks::density));

            dst.block(SolutionBlocks::density) =
                    (op_fddf_chunk * op_scaled_inverse * src.block(SolutionBlocks::density));

            dst.block(SolutionBlocks::density) = op_bot_big_inverse * dst.block(SolutionBlocks::density);

            dst.block(SolutionBlocks::density) +=
                    (op_bot_big_inverse * src.block(SolutionBlocks::unfiltered_density_multiplier));
        }
        std::cout << "part 4.4 done" << std::endl;
    }
}