//
// Created by justin on 7/3/21.
//

#ifndef SAND_MY_TOOLS_H
#define SAND_MY_TOOLS_H


namespace SAND{
    using namespace dealii;

    void build_matrix_element_by_element (const auto &X,
                                          FullMatrix<double>   &X_matrix)
    {

        Threads::TaskGroup<void> tasks;
        for (unsigned int j=0; j<X_matrix.n(); ++j)
            tasks += Threads::new_task ([&X, &X_matrix, j]()
                                        {
                                            Vector<double> e_j (X_matrix.m());
                                            Vector<double> r_j (X_matrix.n());

                                            e_j = 0;
                                            e_j(j) = 1;

                                            X.vmult (r_j, e_j);

                                            for (unsigned int i=0; i<X_matrix.m(); ++i)
                                                X_matrix(i,j) = r_j(i);
                                        });

        tasks.join_all();
    }

//    void print_matrix (std::string &name, FullMatrix<double>   &X_matrix)
//    {
////        const unsigned int n = X_matrix.n();
////        const unsigned int m = X_matrix.m();
////        std::ofstream Xmat(name);
////        for (unsigned int i = 0; i < m; i++)
////        {
////            Xmat << X_matrix(i, 0);
////            for (unsigned int j = 1; j < n; j++)
////            {
////                Xmat << "," << X_matrix(i, j);
////            }
////            Xmat << "\n";
////        }
////        Xmat.close();
//    }

//    void print_matrix (std::string &name, SparseMatrix<double>   &X_matrix)
//    {
////        const unsigned int n = X_matrix.n();
////        const unsigned int m = X_matrix.m();
////        std::ofstream Xmat(name);
////        for (unsigned int i = 0; i < m; i++)
////        {
////            Xmat << X_matrix.el(i, 0);
////            for (unsigned int j = 1; j < n; j++)
////            {
////                Xmat << "," << X_matrix.el(i, j);
////            }
////            Xmat << "\n";
////        }
////        Xmat.close();
//    }

}


#endif //SAND_MY_TOOLS_H
