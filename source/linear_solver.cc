//
// Created by justin on 2/17/21.
//
#include "../include/markov.h"
#include <deal.II/lac/linear_operator.h>
#include <deal.II/lac/sparse_direct.h>
#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/solver_bicgstab.h>
#include "../include/linear_solver.h"
#include "../include/parameters_and_components.h"

namespace SAND {

    using namespace dealii;

    TopOptSchurPreconditioner::TopOptSchurPreconditioner()
            :
            n_rows(0),
            n_columns(0),
            n_block_rows(0),
            n_block_columns(0),
            elastic_solver_control(10000, 1e-12),
            elastic_cg(elastic_solver_control),
            diag_solver_control(10000, 1e-12),
            diag_cg(diag_solver_control)

    {
    }

    void TopOptSchurPreconditioner::initialize(BlockSparseMatrix<double> &matrix, std::map<types::global_dof_index, double> boundary_values)
    {

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

//        double diag_val = matrix.block(SolutionBlocks::density,SolutionBlocks::density).el(0,0);
//        matrix.block(SolutionBlocks::density, SolutionBlocks::displacement).set(0,0,diag_val);
//        matrix.block(SolutionBlocks::density, SolutionBlocks::displacement_multiplier).set(0,0,diag_val);
//        matrix.block(SolutionBlocks::displacement, SolutionBlocks::density).set(0,0,diag_val);
//        matrix.block(SolutionBlocks::displacement_multiplier, SolutionBlocks::density).set(0,0,diag_val);

        op_elastic = linear_operator(
                matrix.block(SolutionBlocks::displacement, SolutionBlocks::displacement_multiplier));
        op_elastic_inv = inverse_operator(op_elastic, elastic_cg, PreconditionIdentity());

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

        op_diag_sum_inv = inverse_operator(op_diag_1 + op_diag_2, diag_cg, PreconditionIdentity());

        A_direct.initialize(matrix.block(SolutionBlocks::displacement,SolutionBlocks::displacement_multiplier));

        op_elastic_inv = linear_operator(A_direct);

        op_scaled_identity = linear_operator(matrix.block(SolutionBlocks::density_upper_slack,SolutionBlocks::density_upper_slack_multiplier));

        Scaled_direct.initialize(matrix.block(SolutionBlocks::density_upper_slack,SolutionBlocks::density_upper_slack_multiplier));
        op_scaled_inv = linear_operator(Scaled_direct);
    }

    void TopOptSchurPreconditioner::vmult(BlockVector<double> &dst, const BlockVector<double> &src) const {

        vmult_step_1(dst, src);
        BlockVector<double> temp_src = dst;

        vmult_step_2(dst, temp_src);
        temp_src = dst;

        vmult_step_3(dst, temp_src);
        temp_src = dst;

        vmult_step_4(dst, temp_src);
    }

    void TopOptSchurPreconditioner::Tvmult(BlockVector<double> &dst, const BlockVector<double> &src) const {
        dst = src;
        std::cout << "bad";
    }

    void TopOptSchurPreconditioner::vmult_add(BlockVector<double> &dst, const BlockVector<double> &src) const {
        BlockVector<double> dst_temp = dst;
        vmult(dst_temp, src);
        dst += dst_temp;
        std::cout << "bad";
    }

    void TopOptSchurPreconditioner::Tvmult_add(BlockVector<double> &dst, const BlockVector<double> &src) const {
        dst = dst + src;
        std::cout << "bad";
    }

    void TopOptSchurPreconditioner::vmult_step_1(BlockVector<double> &dst, const BlockVector<double> &src) const {
        dst = src;
        dst.block(SolutionBlocks::unfiltered_density) +=
                -1 * op_diag_1 * op_scaled_inv * src.block(SolutionBlocks::density_lower_slack_multiplier)
                +
                op_diag_2 * op_scaled_inv * src.block(SolutionBlocks::density_upper_slack_multiplier)
                +
                src.block(SolutionBlocks::density_lower_slack)
                -
                src.block(SolutionBlocks::density_upper_slack);
    }

    void TopOptSchurPreconditioner::vmult_step_2(BlockVector<double> &dst, const BlockVector<double> &src) const {
        dst = src;
        dst.block(SolutionBlocks::unfiltered_density_multiplier) +=
                -1 * op_filter * op_diag_sum_inv * src.block(SolutionBlocks::unfiltered_density);

    }

    void TopOptSchurPreconditioner::vmult_step_3(BlockVector<double> &dst, const BlockVector<double> &src) const {
        dst = src;
        dst.block(SolutionBlocks::density) +=
                -1 * transpose_operator(op_displacement_density) * op_elastic_inv *
                src.block(SolutionBlocks::displacement_multiplier)
                +
                -1 * transpose_operator(op_displacement_multiplier_density) * op_elastic_inv *
                src.block(SolutionBlocks::displacement);
    }

    void TopOptSchurPreconditioner::vmult_step_4(BlockVector<double> &dst, const BlockVector<double> &src) const {


        //First Block Inverse
        dst.block(SolutionBlocks::density_lower_slack_multiplier) =
                -1 * op_diag_1 * op_scaled_inv * op_scaled_inv * src.block(SolutionBlocks::density_lower_slack_multiplier)
                +
                op_scaled_inv * src.block(SolutionBlocks::density_lower_slack);

        dst.block(SolutionBlocks::density_upper_slack_multiplier) =
                -1 * op_diag_2 *op_scaled_inv *op_scaled_inv * src.block(SolutionBlocks::density_upper_slack_multiplier)
                +
                op_scaled_inv *src.block(SolutionBlocks::density_upper_slack);

        dst.block(SolutionBlocks::density_lower_slack) =
                op_scaled_inv * src.block(SolutionBlocks::density_lower_slack_multiplier);

        dst.block(SolutionBlocks::density_upper_slack) =
                op_scaled_inv * src.block(SolutionBlocks::density_upper_slack_multiplier);

        //Second Block Inverse
        dst.block(SolutionBlocks::unfiltered_density) =
                op_diag_sum_inv * src.block(SolutionBlocks::unfiltered_density);

        //Third Block Inverse
        dst.block(SolutionBlocks::displacement) =
                op_elastic_inv * src.block(SolutionBlocks::displacement_multiplier);
        dst.block(SolutionBlocks::displacement_multiplier) =
                op_elastic_inv * src.block(SolutionBlocks::displacement);

        //Fourth (ugly) Block Inverse
        SolverControl other_solver_control(10001, 1e-6);
        SolverBicgstab<Vector<double>> other_cg(other_solver_control);

        const auto op_fddf_chunk = -1 * op_filter * op_diag_sum_inv * transpose_operator(op_filter);
        const auto op_bcaeeac_chunk =  (op_density_density
                                        -
                                        transpose_operator(op_displacement_density) * op_elastic_inv *
                                        op_displacement_multiplier_density
                                        -
                                        transpose_operator(op_displacement_multiplier_density) *
                                        op_elastic_inv * op_displacement_density
                                        );


        const auto op_top_big_inverse = inverse_operator(op_bcaeeac_chunk * op_scaled_inv * op_fddf_chunk
                                                        - op_scaled_identity,
                                                          other_cg,
                                                          PreconditionIdentity());

        const auto op_bot_big_inverse = inverse_operator(op_fddf_chunk * op_scaled_inv *  op_bcaeeac_chunk
                                                           - op_scaled_identity,
                                                           other_cg,
                                                           PreconditionIdentity());

        dst.block(SolutionBlocks::unfiltered_density_multiplier) =
                (op_bcaeeac_chunk * op_scaled_inv * src.block(SolutionBlocks::unfiltered_density_multiplier));

        dst.block(SolutionBlocks::unfiltered_density_multiplier) =
                        (op_top_big_inverse * dst.block(SolutionBlocks::unfiltered_density_multiplier));

        dst.block(SolutionBlocks::unfiltered_density_multiplier) +=
                         (op_top_big_inverse * src.block(SolutionBlocks::density));

        dst.block(SolutionBlocks::density) =
                (op_fddf_chunk * op_scaled_inv * src.block(SolutionBlocks::density));

        dst.block(SolutionBlocks::density) = op_bot_big_inverse * dst.block(SolutionBlocks::density);

        dst.block(SolutionBlocks::density) +=
                (op_bot_big_inverse * src.block(SolutionBlocks::unfiltered_density_multiplier));
    }
}