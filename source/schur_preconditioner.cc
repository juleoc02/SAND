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
            n_rows(0),
            n_columns(0),
            n_block_rows(0),
            n_block_columns(0),
            diag_solver_control(10000, 1e-6),
            diag_cg(diag_solver_control),
            other_solver_control(10000, 1e-6),
            other_bicgstab(other_solver_control),
            other_cg(other_solver_control),
            timer(std::cout, TimerOutput::summary,
                  TimerOutput::wall_times),
            system_matrix(matrix_in),
            a_mat(matrix_in.block(SolutionBlocks::displacement, SolutionBlocks::displacement_multiplier)),
            b_mat(matrix_in.block(SolutionBlocks::density, SolutionBlocks::density)),
            c_mat(matrix_in.block(SolutionBlocks::displacement,SolutionBlocks::density)),
            e_mat(matrix_in.block(SolutionBlocks::displacement_multiplier,SolutionBlocks::density)),
            f_mat(matrix_in.block(SolutionBlocks::unfiltered_density_multiplier,SolutionBlocks::unfiltered_density)),
            d_m_mat(matrix_in.block(SolutionBlocks::density_upper_slack_multiplier, SolutionBlocks::density_upper_slack)),
            d_1_mat(matrix_in.block(SolutionBlocks::density_lower_slack, SolutionBlocks::density_lower_slack)),
            d_2_mat(matrix_in.block(SolutionBlocks::density_upper_slack, SolutionBlocks::density_upper_slack)),
            m_mat(matrix_in.block(SolutionBlocks::density, SolutionBlocks::total_volume_multiplier))
    {

    }

    template<int dim>
    void TopOptSchurPreconditioner<dim>::initialize(BlockSparseMatrix<double> &matrix, std::map<types::global_dof_index, double> &boundary_values)
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
        d_m_inv_direct.initialize(d_m_mat);
//        d_1_2_inv_direct.initialize(linear_operator(d_1_mat) + linear_operator(d_2_mat));

    }


    template<int dim>
    void TopOptSchurPreconditioner<dim>::vmult(BlockVector<double> &dst, const BlockVector<double> &src) const {
        BlockVector<double> temp_src;
        std::cout << "starting part 1"<< std::endl;
        {
            TimerOutput::Scope t(timer, "part 1");
            vmult_step_1(dst, src);
            temp_src = dst;
        }
        std::cout << "starting part 2"<< std::endl;
        {
            TimerOutput::Scope t(timer, "part 2");
            vmult_step_2(dst, temp_src);
            temp_src = dst;
        }
        std::cout << "starting part 3"<< std::endl;
        {
            TimerOutput::Scope t(timer, "part 3");
            vmult_step_3(dst, temp_src);
            temp_src = dst;
        }
        std::cout << "starting part 4"<< std::endl;
        {
            TimerOutput::Scope t(timer, "part 4");
            vmult_step_4(dst, temp_src);
            temp_src = dst;
        }
        std::cout << "starting part 5" << std::endl;
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
        dst.block(SolutionBlocks::unfiltered_density) += src.block(SolutionBlocks::density_lower_slack);
        dst.block(SolutionBlocks::unfiltered_density) -= src.block(SolutionBlocks::density_upper_slack);

        BlockVector<double> dst_temp = dst;
        diag_cg.solve(d_m_mat,dst_temp.block(SolutionBlocks::unfiltered_density),src.block(SolutionBlocks::density_lower_slack_multiplier),PreconditionIdentity());
        dst_temp.block(SolutionBlocks::unfiltered_density) = -1 * dst_temp.block(SolutionBlocks::unfiltered_density);
        d_1_mat.vmult_add(dst.block(SolutionBlocks::unfiltered_density),dst_temp.block(SolutionBlocks::unfiltered_density));

        dst_temp=0;
        diag_cg.solve(d_m_mat,dst_temp.block(SolutionBlocks::unfiltered_density),src.block(SolutionBlocks::density_upper_slack_multiplier),PreconditionIdentity());
        d_2_mat.template vmult_add(dst.block(SolutionBlocks::unfiltered_density),dst_temp.block(SolutionBlocks::unfiltered_density));

    }

    template<int dim>
    void TopOptSchurPreconditioner<dim>::vmult_step_2(BlockVector<double> &dst, const BlockVector<double> &src) const {
        dst = src;
        BlockVector<double> dst_temp = dst;
        dst_temp.block(SolutionBlocks::unfiltered_density_multiplier) = inverse_operator(linear_operator(d_1_mat) + linear_operator(d_2_mat),diag_cg, PreconditionIdentity()) * src.block(SolutionBlocks::unfiltered_density);
        dst_temp.block(SolutionBlocks::unfiltered_density_multiplier) = -1 *  dst_temp.block(SolutionBlocks::unfiltered_density_multiplier);
        f_mat.vmult_add(dst.block(SolutionBlocks::unfiltered_density_multiplier), dst_temp.block(SolutionBlocks::unfiltered_density_multiplier));
    }

    template<int dim>
    void TopOptSchurPreconditioner<dim>::vmult_step_3(BlockVector<double> &dst, const BlockVector<double> &src) const {
        dst = src;
        BlockVector<double> dst_temp = dst;
        other_cg.solve(a_mat,dst_temp.block(SolutionBlocks::displacement),src.block(SolutionBlocks::displacement),PreconditionIdentity());
        dst_temp.block(SolutionBlocks::displacement) = -1 * dst_temp.block(SolutionBlocks::displacement);
        e_mat.Tvmult_add(dst.block(SolutionBlocks::density),dst_temp.block(SolutionBlocks::displacement));

        dst_temp = 0;
        other_cg.solve(a_mat,dst_temp.block(SolutionBlocks::displacement_multiplier),src.block(SolutionBlocks::displacement_multiplier),PreconditionIdentity());
        dst_temp.block(SolutionBlocks::displacement_multiplier) = -1 * dst_temp.block(SolutionBlocks::displacement_multiplier);
        c_mat.Tvmult_add(dst.block(SolutionBlocks::density),dst_temp.block(SolutionBlocks::displacement_multiplier));
    }

    template<int dim>
    void TopOptSchurPreconditioner<dim>::vmult_step_4(BlockVector<double> &dst, const BlockVector<double> &src) const {
        dst = src;
        auto op_pre_mass = linear_operator(b_mat)
                - transpose_operator(linear_operator(c_mat)) * linear_operator(a_inv_direct) * linear_operator(e_mat)
                - transpose_operator(linear_operator(e_mat)) * linear_operator(a_inv_direct) * linear_operator(c_mat);

        auto op_big = linear_operator(f_mat) * inverse_operator(linear_operator(d_1_mat) + linear_operator(d_2_mat),other_cg,PreconditionIdentity()) *
                transpose_operator(linear_operator(f_mat)) * linear_operator(d_m_inv_direct) * op_pre_mass -
                linear_operator(d_m_mat);

        auto op_big_inv = inverse_operator(op_big,other_bicgstab,PreconditionIdentity());

        auto op_big_simple = inverse_operator(linear_operator(d_1_mat) + linear_operator(d_2_mat),diag_cg,PreconditionIdentity()) *
                             linear_operator(d_m_inv_direct) *
                             linear_operator(weighted_mass_matrix.block(SolutionBlocks::unfiltered_density,SolutionBlocks::unfiltered_density)) -
                             linear_operator(d_m_mat);

        auto op_big_simple_inv = inverse_operator(op_big_simple,diag_cg,PreconditionIdentity());


        Vector<double> d_m_inv_density;
        Vector<double> f_t_d_m_inv_density;
        Vector<double> d_sum_inv_f_t_d_m_inv_density;
        Vector<double> f_d_sum_inv_f_t_d_m_inv_density;
        Vector<double> before_bot_tvmult;
        Vector<double> before_top_tvmult;
        d_m_inv_density.reinit(src.block(SolutionBlocks::density).size());
        f_t_d_m_inv_density.reinit(src.block(SolutionBlocks::density).size());
        d_sum_inv_f_t_d_m_inv_density.reinit(src.block(SolutionBlocks::density).size());
        f_d_sum_inv_f_t_d_m_inv_density.reinit(src.block(SolutionBlocks::density).size());
        before_bot_tvmult.reinit(src.block(SolutionBlocks::density).size());
        before_top_tvmult.reinit(src.block(SolutionBlocks::density).size());


        before_top_tvmult = op_big_inv * src.block(SolutionBlocks::unfiltered_density_multiplier);

        m_mat.Tvmult_add(dst.block(SolutionBlocks::total_volume_multiplier),before_top_tvmult);


        diag_cg.solve(d_m_mat,d_m_inv_density,src.block(SolutionBlocks::density),PreconditionIdentity());
        f_mat.Tvmult(f_t_d_m_inv_density,d_m_inv_density);
        d_sum_inv_f_t_d_m_inv_density = inverse_operator(linear_operator(d_1_mat) + linear_operator(d_2_mat),diag_cg,PreconditionIdentity()) * f_t_d_m_inv_density;
        std::cout<< "here" <<std::endl;
        f_mat.vmult(f_d_sum_inv_f_t_d_m_inv_density,d_sum_inv_f_t_d_m_inv_density);
        std::cout << "there" << std::endl;
        double scale = f_d_sum_inv_f_t_d_m_inv_density.l2_norm();
        f_d_sum_inv_f_t_d_m_inv_density = f_d_sum_inv_f_t_d_m_inv_density * (1/scale);
        before_bot_tvmult = op_big_inv * f_d_sum_inv_f_t_d_m_inv_density;
        before_bot_tvmult = before_bot_tvmult * scale;
        std::cout << "tthere" << std::endl;
        m_mat.Tvmult_add(dst.block(SolutionBlocks::total_volume_multiplier),before_bot_tvmult);
    }




    template<int dim>
    void TopOptSchurPreconditioner<dim>::vmult_step_5(BlockVector<double> &dst, const BlockVector<double> &src) const {
        {
            //First Block Inverse
            //row 1
            TimerOutput::Scope t(timer, "inverse 1");

            Vector<double> d_m_inv_s_1;
            Vector<double> d_m_inv_z_1;
            Vector<double> d_m_inv_squared_z_1;
            Vector<double> d_1_d_m_inv_squared_z_1;

            d_m_inv_s_1.reinit(src.block(SolutionBlocks::density_lower_slack_multiplier).size());
            d_m_inv_z_1.reinit(src.block(SolutionBlocks::density_lower_slack_multiplier).size());
            d_m_inv_squared_z_1.reinit(src.block(SolutionBlocks::density_lower_slack_multiplier).size());
            d_1_d_m_inv_squared_z_1.reinit(src.block(SolutionBlocks::density_lower_slack_multiplier).size());

            diag_cg.solve(d_m_mat,
                              d_m_inv_z_1, src.block(SolutionBlocks::density_lower_slack_multiplier), PreconditionIdentity());
            diag_cg.solve(d_m_mat,
                              d_m_inv_s_1, src.block(SolutionBlocks::density_lower_slack), PreconditionIdentity());
            diag_cg.solve(d_m_mat,
                              d_m_inv_squared_z_1, d_m_inv_z_1, PreconditionIdentity());

            d_1_mat.vmult(d_1_d_m_inv_squared_z_1, d_m_inv_squared_z_1);
            dst.block(SolutionBlocks::density_lower_slack_multiplier) =
                    d_m_inv_s_1 - d_1_d_m_inv_squared_z_1;

            //row 2
            Vector<double> d_m_inv_s_2;
            Vector<double> d_m_inv_z_2;
            Vector<double> d_m_inv_squared_z_2;
            Vector<double> d_2_d_m_inv_squared_z_2;

            d_m_inv_s_2.reinit(src.block(SolutionBlocks::density_upper_slack_multiplier).size());
            d_m_inv_z_2.reinit(src.block(SolutionBlocks::density_upper_slack_multiplier).size());
            d_m_inv_squared_z_2.reinit(src.block(SolutionBlocks::density_upper_slack_multiplier).size());
            d_2_d_m_inv_squared_z_2.reinit(src.block(SolutionBlocks::density_upper_slack_multiplier).size());

            diag_cg.solve(d_m_mat,
                          d_m_inv_z_2, src.block(SolutionBlocks::density_upper_slack_multiplier), PreconditionIdentity());
            diag_cg.solve(d_m_mat,
                          d_m_inv_s_2, src.block(SolutionBlocks::density_upper_slack), PreconditionIdentity());
            diag_cg.solve(d_m_mat,
                          d_m_inv_squared_z_2, d_m_inv_z_2, PreconditionIdentity());

            d_2_mat.vmult(d_2_d_m_inv_squared_z_2, d_m_inv_squared_z_2);

            dst.block(SolutionBlocks::density_upper_slack_multiplier) =
                    d_m_inv_s_2 - d_2_d_m_inv_squared_z_2;

            //line 3
            diag_cg.solve(d_m_mat,dst.block(SolutionBlocks::density_lower_slack),src.block(SolutionBlocks::density_lower_slack_multiplier),PreconditionIdentity());

            //line 4
            diag_cg.solve(d_m_mat,dst.block(SolutionBlocks::density_upper_slack),src.block(SolutionBlocks::density_upper_slack_multiplier),PreconditionIdentity());

        }
        std::cout << "inverse 1" << std::endl;

        {
            //Second Block Inverse
            TimerOutput::Scope t(timer, "inverse 2");
            auto op_diag_sum_inverse = inverse_operator(linear_operator(d_1_mat) + linear_operator(d_2_mat),diag_cg,PreconditionIdentity());
            dst.block(SolutionBlocks::unfiltered_density) =
                    op_diag_sum_inverse * src.block(SolutionBlocks::unfiltered_density);
        }

        std::cout << "inverse 2" << std::endl;
        {
            //Third Block Inverse
            TimerOutput::Scope t(timer, "inverse 3");

            other_cg.solve(a_mat,dst.block(SolutionBlocks::displacement),src.block(SolutionBlocks::displacement_multiplier),PreconditionIdentity());

            other_cg.solve(a_mat,dst.block(SolutionBlocks::displacement_multiplier),src.block(SolutionBlocks::displacement),PreconditionIdentity());
        }
        std::cout << "inverse 3" << std::endl;
        {
            //Fourth (ugly) Block Inverse
            TimerOutput::Scope t(timer, "inverse 4");

            Vector<double> density_multiplier_pre_vect;
            density_multiplier_pre_vect.reinit(src.block(SolutionBlocks::density));
            auto op_b_chunk = linear_operator(b_mat)
                                - transpose_operator(linear_operator(c_mat)) * linear_operator(a_inv_direct) *
                                                               linear_operator(e_mat)
                                - transpose_operator(linear_operator(e_mat)) * linear_operator(a_inv_direct) *
                                                               linear_operator(c_mat);
            auto op_f_chunk = linear_operator(f_mat) * inverse_operator(linear_operator(d_1_mat) + linear_operator(d_2_mat),diag_cg,PreconditionIdentity()) *
                    transpose_operator(linear_operator(f_mat));

            auto op_big_density_multiplier =
                    op_b_chunk * linear_operator(d_m_inv_direct) * -1 * op_f_chunk -
                    linear_operator(d_m_mat);

            density_multiplier_pre_vect = op_b_chunk * linear_operator(d_m_inv_direct) * src.block(SolutionBlocks::unfiltered_density_multiplier)
                                            + src.block(SolutionBlocks::density);
            dst.block(SolutionBlocks::unfiltered_density_multiplier) = inverse_operator(op_big_density_multiplier,other_bicgstab,PreconditionIdentity()) * density_multiplier_pre_vect;
//            Vector<double> density_multiplier_pre_vect_simple;
//            Vector<double> d_m_inv_density_multiplier;
//            d_m_inv_density_multiplier.reinit(src.block(SolutionBlocks::density));
//            density_multiplier_pre_vect_simple.reinit(src.block(SolutionBlocks::density));
//            diag_cg.solve(d_m_mat,d_m_inv_density_multiplier,src.block(SolutionBlocks::unfiltered_density_multiplier),PreconditionIdentity());
//            auto op_simple_big_density_multiplier =
//                    linear_operator(weighted_mass_matrix.block(SolutionBlocks::unfiltered_density,SolutionBlocks::unfiltered_density))
//                    * inverse_operator(linear_operator(d_m_mat),diag_cg,PreconditionIdentity())
//                    * inverse_operator(-1 * linear_operator(d_1_mat) - linear_operator(d_2_mat),diag_cg,PreconditionIdentity())
//                    -
//                    linear_operator(d_m_mat);
//
//            weighted_mass_matrix.block(SolutionBlocks::unfiltered_density,SolutionBlocks::unfiltered_density).vmult(density_multiplier_pre_vect_simple, d_m_inv_density_multiplier);
//            density_multiplier_pre_vect_simple = density_multiplier_pre_vect_simple + src.block(SolutionBlocks::density);
//            dst.block(SolutionBlocks::unfiltered_density_multiplier) = inverse_operator(op_simple_big_density_multiplier,other_bicgstab,PreconditionIdentity()) * density_multiplier_pre_vect_simple;

            Vector<double> density_pre_vect;
            density_pre_vect.reinit(src.block(SolutionBlocks::density));
            density_pre_vect = -1 * op_f_chunk * linear_operator(d_m_inv_direct) * src.block(SolutionBlocks::density)
                              + src.block(SolutionBlocks::unfiltered_density_multiplier);
            auto op_big_density =
                    -1 * op_f_chunk * linear_operator(d_m_inv_direct)
                    * op_b_chunk - linear_operator(d_m_mat);
            dst.block(SolutionBlocks::density) = inverse_operator(op_big_density,other_bicgstab,PreconditionIdentity()) * density_pre_vect;


//            Vector<double> density_pre_vect_simple;
//            density_pre_vect_simple = d_sum_inv_d_m_inv_density + src.block(SolutionBlocks::unfiltered_density_multiplier);
//            auto op_big_density_simple =
//            diag_cg.solve(d_m_mat,d_m_inv_density,src.block(SolutionBlocks::density),PreconditionIdentity());
//            d_sum_inv_d_m_inv_density = inverse_operator(linear_operator(d_1_mat) + linear_operator(d_2_mat),diag_cg,PreconditionIdentity()) * d_m_inv_density;
//                    inverse_operator(-1 * linear_operator(d_1_mat) - linear_operator(d_2_mat),diag_cg,PreconditionIdentity())
//                    * inverse_operator(linear_operator(d_m_mat),diag_cg,PreconditionIdentity())
//                    * linear_operator(weighted_mass_matrix.block(SolutionBlocks::unfiltered_density,SolutionBlocks::unfiltered_density))
//                    - linear_operator(d_m_mat);
//            dst.block(SolutionBlocks::unfiltered_density_multiplier) = inverse_operator(op_big_density_simple,other_bicgstab,PreconditionIdentity()) * density_pre_vect_simple;


        }
        std::cout << "inverse 4" << std::endl;
        {
            dst.block(SolutionBlocks::total_volume_multiplier) = src.block(SolutionBlocks::total_volume_multiplier);
        }
        std::cout << "inverse 5" << std::endl;
    }

    template<int dim>
    void TopOptSchurPreconditioner<dim>::assemble_mass_matrix(const BlockVector<double> &state,
                                                              const hp::FECollection<dim> &fe_collection,
                                                              const DoFHandler<dim> &dof_handler,
                                                              const AffineConstraints<double> &constraints,
                                                              const BlockSparsityPattern &bsp) {
        timer.reset();

        weighted_mass_matrix.reinit(bsp);

        std::cout << weighted_mass_matrix.n() << std::endl;

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

                    const double unfiltered_density_phi_i = fe_values[unfiltered_densities].value(i,
                                                                                                  q_point);
                    const double unfiltered_density_multiplier_phi_i = fe_values[unfiltered_density_multipliers].value(
                            i, q_point);

                    for (unsigned int j = 0; j < dofs_per_cell; ++j) {

                        const double unfiltered_density_phi_j = fe_values[unfiltered_densities].value(j,
                                                                                                      q_point);
                        const double unfiltered_density_multiplier_phi_j = fe_values[unfiltered_density_multipliers].value(
                                j, q_point);


                        //Equation 0
                        cell_matrix(i, j) +=
                                fe_values.JxW(q_point) *
                                (
                                        -1 * Input::density_penalty_exponent * (Input::density_penalty_exponent + 1)
                                        *
                                        std::pow(old_density_values[q_point],
                                                 Input::density_penalty_exponent - 2)
                                        *
                                        (old_displacement_divs[q_point] * old_displacement_multiplier_divs[q_point]
                                         * lambda_values[q_point]
                                         +
                                         2 * mu_values[q_point] * (old_displacement_symmgrads[q_point] *
                                                                   old_displacement_multiplier_symmgrads[q_point]))
                                        * unfiltered_density_phi_i * unfiltered_density_phi_j
                                );
//                        //Equation 1
//
//                        cell_matrix(i, i) +=
//                                fe_values.JxW(q_point) *
//                                (
//                                        -1 * Input::density_penalty_exponent * (Input::density_penalty_exponent + 1)
//                                        *
//                                        std::pow(old_density_values[q_point],
//                                                 Input::density_penalty_exponent - 2)
//                                        *
//                                        (old_displacement_divs[q_point] * old_displacement_multiplier_divs[q_point]
//                                         * lambda_values[q_point]
//                                         +
//                                         2 * mu_values[q_point] * (old_displacement_symmgrads[q_point] *
//                                                                   old_displacement_multiplier_symmgrads[q_point]))
//                                        * unfiltered_density_multiplier_phi_i * unfiltered_density_multiplier_phi_j
//                                );
                    }

                }

            }

            constraints.distribute_local_to_global(cell_matrix, local_dof_indices, weighted_mass_matrix);
        }

    }


    template<int dim>
    void TopOptSchurPreconditioner<dim>::print_stuff(const BlockSparseMatrix<double> &matrix) {

        const unsigned int vec_size = matrix.n();
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
            transformed_unit_vector_mass = linear_operator(weighted_mass_matrix.block(SolutionBlocks::unfiltered_density,SolutionBlocks::unfiltered_density))* unit_vector;

            for (unsigned int i = 0; i < vec_size; i++) {
                orig_mat(i,j) = transformed_unit_vector_orig[i];
                est_mass_mat(i,j) = transformed_unit_vector_mass[i];
            }
        }

        for (unsigned int block_row = 1; block_row < system_matrix.n_block_rows(); block_row++)
        {
            for (unsigned int block_col = 0; block_col < system_matrix.n_block_cols(); block_col++)
            {

                std::ofstream OGMat("original_matrix"+std::to_string(block_row)+"_"+std::to_string(block_col) +".csv");
                std::ofstream MassMat("mass_estimated"+std::to_string(block_row)+"_"+std::to_string(block_col) +".csv");

                for (unsigned int i = system_matrix.get_row_indices().block_start(block_row); i < system_matrix.get_row_indices().block_start(block_row+1); i++)
                {
                    OGMat << orig_mat(i, 0);
                    MassMat << est_mass_mat(i, 0);
                    for (unsigned int j = system_matrix.get_column_indices().block_start(block_col); j < system_matrix.get_column_indices().block_start(block_col+1); j++)
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
    }
}
template class SAND::TopOptSchurPreconditioner<2>;
template class SAND::TopOptSchurPreconditioner<3>;