//
// Created by justin on 2/17/21.
//
#include "../include/markov.h"
#include <deal.II/lac/linear_operator.h>
#include <deal.II/lac/sparse_direct.h>
#include "../include/linear_solver.h"

namespace SAND {

    using namespace dealii;

    TopOptSchurPreconditioner::TopOptSchurPreconditioner()
            :
            n_rows(0),
            n_columns(0),
            n_block_rows(0),
            n_block_columns(0)
    {
    }

    void TopOptSchurPreconditioner::initialize(const BlockSparseMatrix<double> &matrix)
    {

    }

    void TopOptSchurPreconditioner::vmult(BlockVector<double> &dst, const BlockVector<double> &src) const
    {
        dst = src;
    }

    void TopOptSchurPreconditioner::Tvmult(BlockVector<double> &dst, const BlockVector<double> &src) const
    {
        dst = src;
    }

    void TopOptSchurPreconditioner::vmult_add(BlockVector<double> &dst, const BlockVector<double> &src) const
    {
        dst = dst + src;
    }

    void TopOptSchurPreconditioner::Tvmult_add(BlockVector<double> &dst, const BlockVector<double> &src) const
    {
        dst = dst+src;
    }

}