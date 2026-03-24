#ifndef EXACTCONSTRAINTSATISFACTION_HH
#define EXACTCONSTRAINTSATISFACTION_HH

#include <CoMISo/Config/config.hh>
#include <CoMISo/Config/CoMISoDefines.hh>

#include <CoMISo/NSolver/NProblemInterface.hh>
#include <vector>
#include <time.h>

namespace COMISO {

class COMISODLLEXPORT ExactConstraintSatisfaction
{
public:
    ExactConstraintSatisfaction();

    typedef Eigen::SparseVector<int>::InnerIterator iteratorV;
    typedef Eigen::SparseVector<int> sparseVec;
    typedef Eigen::SparseMatrix<int, Eigen::ColMajor> SP_Matrix_C;
    typedef Eigen::SparseMatrix<int, Eigen::RowMajor> SP_Matrix_R;

    int    gcd(const int a, const int b);
    int    gcd_row(const SP_Matrix_R& A, int row, const int b);

    int    lcm(const int a, const int b);
    int    lcm(const std::vector<int>& D);

    void   swap_rows(SP_Matrix_R& mat,  int row1, int row2);
    void   eliminate_row(SP_Matrix_R& mat, Eigen::VectorXi& b, int row1, int row2, int pivot_column);
    void   largest_exponent(const Eigen::VectorXd& x);
    void   set_largest_exponent(int _exponent);
    static int compute_largest_exponent(const Eigen::VectorXd& x);
    int    index_pivot(const sparseVec& row);
    double F_delta(double x);
    double get_delta();

    //--------------------matrix transformation-------------------------------

    void   IREF_Gaussian(SP_Matrix_R& A, Eigen::VectorXi& b, const Eigen::VectorXd& x);
    void   IRREF_Jordan(SP_Matrix_R& A, Eigen::VectorXi& b);

    //-------------------Evaluation--------------------------------------------

    void   evaluation(SP_Matrix_R& _A, Eigen::VectorXi& b, Eigen::VectorXd& x);
    double makeDiv(const std::vector<int>& D, double x);
    double safeDot(const std::vector<std::pair<int, double> >& S);
    void   evaluate(const SP_Matrix_R& _A, const Eigen::VectorXi& b, Eigen::VectorXd& x);

private:

    // for IREF
    int get_pivot_col_student(SP_Matrix_R& _A, int k, int& col_index);
    int get_pivot_col_new(SP_Matrix_R& _A, int k, int& col_index);

    // for evaluation

    int get_pivot_row_student(const SP_Matrix_C& A, int col);
    int get_pivot_row_new(const SP_Matrix_C& A, const SP_Matrix_R& _A, int col);

    std::vector<int> get_divisors_student(const SP_Matrix_C& A, int col);
    std::vector<int> get_divisors_new(const SP_Matrix_C& A, const SP_Matrix_R& _A, int col);

    std::vector<std::pair<int, double>> get_dot_product_elements_student(const SP_Matrix_C& A, const Eigen::VectorXd& x, int k, int pivot_row);
    std::vector<std::pair<int, double>> get_dot_product_elements_new(const SP_Matrix_R& A, const Eigen::VectorXd& x, int pivot_row);

    //-----------------------helpfull variables-------------------------------

    int    number_pivots_ = 0; //number of rows with a pivot;
    int    largest_exponent_ = 0;
    double delta_ = 0;
};

}

#endif // EXACTCONSTRAINTSATISFACTION_HH
