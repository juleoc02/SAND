//
// Created by justin on 2/17/21.
//
#include "../include/markov.h"
#include <deal.II/lac/linear_operator.h>
#include <deal.II/lac/sparse_direct.h>
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
            elastic_solver_control(10000, 1e-3),
            elastic_cg(elastic_solver_control),
            diag_solver_control(10, 1e-1),
            diag_cg(diag_solver_control)
    {
    }

    void TopOptSchurPreconditioner::initialize(const BlockSparseMatrix<double> &matrix)
    {


        op_elastic = linear_operator(matrix.block(SolutionBlocks::displacement,SolutionBlocks::displacement_multiplier));
        op_elastic_inv = inverse_operator(op_elastic,elastic_cg,PreconditionIdentity());

        op_filter = linear_operator(matrix.block(SolutionBlocks::unfiltered_density_multiplier,SolutionBlocks::unfiltered_density));
        op_diag_1 = linear_operator(matrix.block(SolutionBlocks::density_lower_slack,SolutionBlocks::density_lower_slack));
        op_diag_2 = linear_operator(matrix.block(SolutionBlocks::density_upper_slack,SolutionBlocks::density_upper_slack));
        op_displacement_density = linear_operator(matrix.block(SolutionBlocks::displacement,SolutionBlocks::density));

        op_displacement_multiplier_density = linear_operator(matrix.block(SolutionBlocks::displacement_multiplier,SolutionBlocks::density));

        op_diag_sum_inv = inverse_operator(op_diag_1+op_diag_2,diag_cg,PreconditionIdentity());

    }

    void TopOptSchurPreconditioner::vmult(BlockVector<double> &dst, const BlockVector<double> &src) const
    {
        vmult_step_1(dst,src);
        BlockVector<double> temp_src = dst;
        vmult_step_2(dst,temp_src);
        temp_src = dst;
        vmult_step_3(dst,temp_src);
//        temp_src = dst;
//        vmult_step_4(dst,temp_src);
//        dst = src;

    }

    void TopOptSchurPreconditioner::Tvmult(BlockVector<double> &dst, const BlockVector<double> &src) const
    {
        dst = src;
    }

    void TopOptSchurPreconditioner::vmult_add(BlockVector<double> &dst, const BlockVector<double> &src) const
    {
        BlockVector<double> dst_temp = dst;
        vmult(dst_temp,src);
        dst += dst_temp;
    }

    void TopOptSchurPreconditioner::Tvmult_add(BlockVector<double> &dst, const BlockVector<double> &src) const
    {
        dst = dst+src;
    }

    void TopOptSchurPreconditioner::vmult_step_1(BlockVector<double> &dst, const BlockVector<double> &src) const
    {
        dst = src;
        dst.block(SolutionBlocks::unfiltered_density) +=
                -1*op_diag_1 * src.block(SolutionBlocks::density_lower_slack_multiplier)
                +
                op_diag_2 * src.block(SolutionBlocks::density_upper_slack_multiplier)
                +
                src.block(SolutionBlocks::density_lower_slack)
                -
                src.block(SolutionBlocks::density_upper_slack);
    }

    void TopOptSchurPreconditioner::vmult_step_2(BlockVector<double> &dst, const BlockVector<double> &src) const
    {
        dst = src;
        dst.block(SolutionBlocks::unfiltered_density_multiplier) +=
               -1*op_filter * op_diag_sum_inv * src.block(SolutionBlocks::unfiltered_density);

    }

    void TopOptSchurPreconditioner::vmult_step_3(BlockVector<double> &dst, const BlockVector<double> &src) const
    {
        dst = src;
        dst.block(SolutionBlocks::density) +=
                -1 * transpose_operator(op_displacement_multiplier_density) * op_elastic_inv * src.block(SolutionBlocks::displacement_multiplier)
                +
                -1 * transpose_operator(op_displacement_density)*op_elastic_inv * src.block(SolutionBlocks::displacement);
    }

    void TopOptSchurPreconditioner::vmult_step_4(BlockVector<double> &dst, const BlockVector<double> &src) const
    {
        //First Block Inverse
        dst.block(SolutionBlocks::density_lower_slack_multiplier) =
                -1*op_diag_1*src.block(SolutionBlocks::density_lower_slack_multiplier)
                +
                src.block(SolutionBlocks::density_lower_slack);
        dst.block(SolutionBlocks::density_upper_slack_multiplier) =
                -1*op_diag_2*src.block(SolutionBlocks::density_upper_slack_multiplier)
                +
                src.block(SolutionBlocks::density_upper_slack);
        dst.block(SolutionBlocks::density_lower_slack) =
                op_diag_1*src.block(SolutionBlocks::density_lower_slack_multiplier);
        dst.block(SolutionBlocks::density_upper_slack) =
                op_diag_1*src.block(SolutionBlocks::density_upper_slack_multiplier);

        //Second Block Inverse
        dst.block(SolutionBlocks::unfiltered_density) =
                op_diag_sum_inv * src.block(SolutionBlocks::unfiltered_density);

        //Third Block Inverse
        dst.block(SolutionBlocks::displacement) =
                op_elastic_inv*src.block(SolutionBlocks::displacement_multiplier);
        dst.block(SolutionBlocks::displacement_multiplier) =
                op_elastic_inv*src.block(SolutionBlocks::displacement);

        //Fourth (ugly) Block Inverse




    }


}