#include "ExactConstraintSatisfaction.hh"

#include <CoMISo/Config/config.hh>

#include <CoMISo/NSolver/NProblemInterface.hh>
#include <CoMISo/Utils/CoMISoError.hh>
#include <Base/Debug/DebOut.hh>
#include <vector>

namespace COMISO {

ExactConstraintSatisfaction::ExactConstraintSatisfaction()
  :
    largest_exponent_(std::numeric_limits<int>::min())
{

}


// ------------------------Helpfull Methods----------------------------------------

//row1 belongs to vector b, row2 to a row in the matrix
void ExactConstraintSatisfaction::swap_rows(SP_Matrix_R& mat, int row1, int row2){

  Eigen::SparseVector<int> row_2 = mat.row(row2);
  Eigen::SparseVector<int> row_1 = mat.row(row1);
  mat.row(row2) = row_1;
  mat.row(row1) = row_2;

//  mat.prune(0.0, 0);
//  mat.makeCompressed();
//  mat.finalize();

}


//We want to eliminate row1 in mat with the row corresponding to row2 in mat
//the row_2 has a pivot in (row2, col_p)
void ExactConstraintSatisfaction::eliminate_row(SP_Matrix_R& mat, Eigen::VectorXi& b, int row1, int row2, int pivot_column)
{
  if(pivot_column < 0)
  {
    DEB_error("Pivot column is negative");
    return;
  }

  int pivot_row1 = mat.coeff(row1, pivot_column);     //the element under the pivot

  if(pivot_row1 == 0)
    return;

  int pivot_row2 = mat.coeff(row2, pivot_column);     //the pivot

  b.coeffRef(row1) *= pivot_row2;
  b.coeffRef(row1) -= pivot_row1 * b.coeffRef(row2);

  mat.row(row1) *= pivot_row2;
  mat.row(row1) -= pivot_row1 * mat.row(row2);
  mat.row(row1) = mat.row(row1).pruned(0,0);

  int gcdValue = gcd_row(mat, row1, b.coeff(row1));
  mat.row(row1) /= gcdValue;
  b.coeffRef(row1) /= gcdValue;

}

int ExactConstraintSatisfaction::gcd(const int a, const int b)
{
  if(b == 0)
    return std::abs(a);
  return gcd(std::abs(b) , std::abs(a) % std::abs(b));
}

int ExactConstraintSatisfaction::gcd_row(const SP_Matrix_R& A, int row, const int b)
{

  int gcdValue = b;
  bool first = true;
  bool is_negativ = 0;

  for(SP_Matrix_R::InnerIterator it(A, row); it; ++it)
  {
    if(it.value() != 0 && first)
    {
      first = false;
      is_negativ = it.value() < 0;
    }
    gcdValue = gcd(gcdValue, it.value());
  }
  if(gcdValue == 0)
    return 1;
  if(is_negativ)
    gcdValue = std::abs(gcdValue) * -1;
  return gcdValue;
}

void ExactConstraintSatisfaction::largest_exponent(const Eigen::VectorXd& x)
{
  // only compute largest component if it was not set from the outside
  if (largest_exponent_ == std::numeric_limits<int>::min())
    set_largest_exponent(std::max(compute_largest_exponent(x)+2, -65));
}

void ExactConstraintSatisfaction::set_largest_exponent(int _exponent)
{
  largest_exponent_ = _exponent;
  delta_ = std::pow(2, largest_exponent_);
}

int ExactConstraintSatisfaction::compute_largest_exponent(const Eigen::VectorXd& x)
{
  double max_elem = std::max(x.maxCoeff(), -x.minCoeff());
  auto expo = static_cast<int>(std::ceil(std::log2(max_elem)));
  return expo;
}

double ExactConstraintSatisfaction::F_delta(double x)
{
  int sign = -1;
  if(x >= 0)
    sign = 1;
  double x_of_F = (x + sign * delta_) - sign * delta_;
  return x_of_F;
}

int ExactConstraintSatisfaction::lcm(const int a, const int b)
{
  if(gcd(a,b) == 0)
    return 0;
  return (a/gcd(a,b)) * b;
}

int ExactConstraintSatisfaction::lcm(const std::vector<int>& D)
{
  int lcm_D = 1;
  for(int d : D)
  {
    lcm_D = lcm(lcm_D, d);
  }
  return lcm_D;
}

int ExactConstraintSatisfaction::index_pivot(const sparseVec& row)
{

  for(sparseVec::InnerIterator it(row); it; ++it)
  {
    if(it.value() != 0)
      return it.index();
  }
  return -1;
}

double ExactConstraintSatisfaction::get_delta()
{
  return delta_;
}

// ----------------------Matrix transformation-----------------------------------
void ExactConstraintSatisfaction::IREF_Gaussian(SP_Matrix_R& A, Eigen::VectorXi& b, const Eigen::VectorXd& x){

  number_pivots_ = 0;
  int rows = static_cast<int>(A.rows());        //number of rows
  int cols = static_cast<int>(A.cols());        //number of columns
  int col_index  = 0;         //save the next column where we search for a pivot

  for (int k = 0; k < rows; k++)
  {                                        //order Matrix after pivot
    if(A.coeff(k, col_index) == 0)
    {
      int pivot_row = get_pivot_col_new(A, k, col_index);

      if(pivot_row != -1)
        if(k != pivot_row)
          swap_rows(A, k, pivot_row);

      if(col_index == cols)
        break;

      if(pivot_row == -1)
        continue;
    }
    else
    {
      col_index++;//col_index = k +1;
    }
    number_pivots_++;

    if (number_pivots_ == 1)
    {
      // first row, divide by gcd to keep numbers small. All other rows will be divided by ther gcd in eliminate row
      int gcdValue = gcd_row(A, 0, b.coeffRef(0));                     //compute the gcd to make the values as small as possible
      if(gcdValue != 1)
      {
        A.row(0) /= gcdValue;
        b.coeffRef(0) /= gcdValue;
      }
    }

    int col_p = col_index -1;

    for(int i = k+1; i < rows; ++i)
    {
      SP_Matrix_R::InnerIterator row_iter_i(A,i);
      if (row_iter_i.index() > col_p)
        continue; // first element is right of i, so A(i,col_p) is already 0
      COMISO_THROW_TODO_if(row_iter_i.index() < col_p, "there should be now non zero values lower left of k,col_p");

      eliminate_row(A, b, i, k, col_p);
    }
  }
  A.prune(0, 0);
}

void ExactConstraintSatisfaction::IRREF_Jordan(SP_Matrix_R& A, Eigen::VectorXi& b)
{
  SP_Matrix_C A_colmaj = A; // for more efficient tests which rows need to be eliminated

  for(int k = number_pivots_ - 1; k > 0; k--)
  {
    int pivot_col = -1;
    for(SP_Matrix_R::InnerIterator it(A, k); it; ++it)
    {
      if(it.value() != 0)
      {
        pivot_col = it.index();
        break;
      }
    }
    COMISO_THROW_TODO_if(pivot_col == -1, "Could not find a pivot column");

    for (SP_Matrix_C::InnerIterator it(A_colmaj, pivot_col); it; ++it)
    {
      if (it.row() == k)
        break;
      if (it.value() == 0)
        continue;
      eliminate_row(A, b, static_cast<int>(it.row()), k, pivot_col);
    }

  }

//  A.makeCompressed();
//  A.finalize();
  A.prune(0, 0);
}

//-------------------------------------Evaluation--------------------------------------------------------------------

void ExactConstraintSatisfaction::evaluation(SP_Matrix_R& _A, Eigen::VectorXi& b, Eigen::VectorXd& x)
{
  IREF_Gaussian(_A, b, x);

  IRREF_Jordan(_A, b);

  largest_exponent(x);

  evaluate(_A, b, x);
}

double ExactConstraintSatisfaction::makeDiv(const std::vector<int>& D, double x)
{
  if(D.empty())
    return F_delta(x);

  int d = lcm(D);
  double result = F_delta(x/d) * d;
  return result;
}

double ExactConstraintSatisfaction::safeDot(const std::vector<std::pair<int, double> >& S)
{
  if (S.empty()) //case all Cij are zero after the pivot
    return 0;
  std::vector<std::pair<int, double>> P;
  std::vector<std::pair<int, double>> N;

  int k = 0;
  double result = 0;

  for(auto element : S)
  {
    if(element.first * element.second > 0)
    {
      element.first = std::abs(element.first);
      element.second = std::abs(element.second);
      P.push_back(element);
    }
    else if(element.first * element.second < 0)
    {
      element.first = std::abs(element.first);
      element.second = - std::abs(element.second);
      N.push_back(element);
    }
  }

  int safebreak = 9999999;
  while((!P.empty() || !N.empty()))
  {
    if(!P.empty() && (result < 0 || N.empty()))
    {
      const std::pair<int, double> element = P.back();
      P.pop_back();
      double test_value = element.second;
      if(test_value < 0.00000001)
      { //to prevent overflow through the dividing
        k = element.first;
      }
      else
      {
        k = std::min(element.first, static_cast<int>(std::floor((delta_ - result)/element.second)));
      }
      result = result + k * element.second;
      if(k < element.first)
        P.push_back({element.first - k, element.second});
    }
    else
    {
      const std::pair<int, double> element = N.back();
      N.pop_back();
      double test_value = element.second;
      if(std::abs(test_value) < 0.00000001) //to prevent overflow through the dividing
      {
        k = element.first;
      }
      else
      {
        k = std::min(element.first, static_cast<int>(std::floor((-delta_ - result)/element.second)));
      }
      result = result + k * element.second;
      if(k < element.first)
        N.push_back({element.first - k, element.second});
    }

    if(k == 0)
    {
      DEB_error("The representable range of double values (delta) is to small");
      return -99999999;
    }

    COMISO_THROW_TODO_if(--safebreak == 0, "Infinite loop detected");
  }

  return result;
}

void ExactConstraintSatisfaction::evaluate(const ExactConstraintSatisfaction::SP_Matrix_R& _A, const Eigen::VectorXi& b, Eigen::VectorXd& x)
{
  DEB_enter_func;

  SP_Matrix_C A = _A;         //change the matrix type to allow easier iteration

  int cols = static_cast<int>(A.cols());
  for(int k = cols -1; k >= 0; k--)
  {
    auto pivot_row = get_pivot_row_new(A, _A, k);

    if(pivot_row == -1)
    {
      //there is no pivot in this column
      auto D = get_divisors_new(A, _A, k);
      x.coeffRef(k) = makeDiv(D, x.coeffRef(k));            //fix free variables so they are in F_delta
    }
    else
    {
      auto S = get_dot_product_elements_new(_A, x, pivot_row);

      double divided_B = b.coeffRef(pivot_row);
      divided_B = F_delta(divided_B / A.coeffRef(pivot_row, k));
      DEB_warning_if( divided_B * A.coeffRef(pivot_row, k) !=  static_cast<double>(b.coeffRef(pivot_row)), 2,
                     "Can't handle the right hand side perfectly");
      x.coeffRef(k) = divided_B - safeDot(S);
    }
  }

}

int ExactConstraintSatisfaction::get_pivot_col_student(SP_Matrix_R& _A, int k, int& col_index)
{
  int pivot_row = -1;
  int cols = static_cast<int>(_A.cols());
  for(; col_index < cols; col_index++)
  {

    Eigen::SparseVector<int> col = _A.col(col_index);
    for(Eigen::SparseVector<int>::InnerIterator it(col); it; ++it)
    {
      if(it.value() != 0 && it.index() >= k)
      {
        pivot_row = it.index();
        break;
      }
    }
    if(pivot_row != -1)
    {
      col_index++;
      break;
    }
  }

  return pivot_row;
}

int ExactConstraintSatisfaction::get_pivot_col_new(ExactConstraintSatisfaction::SP_Matrix_R& _A, int k, int& col_index)
{
  int cols = static_cast<int>(_A.cols());
  int rows = static_cast<int>(_A.rows());
  for(; col_index < cols; col_index++)
  {
    for (int i = k; i < rows; ++i)
      if (_A.row(i).coeff(col_index) != 0)
      {
        ++col_index;
        return i;
      }
  }

  return -1;
}

int ExactConstraintSatisfaction::get_pivot_row_student(const SP_Matrix_C& A, int col)
{
  int pivot_row = -1;
  for(SP_Matrix_C::InnerIterator it(A, col); it; ++it)
  {
    if(it.value() != 0)
    {
      int index = index_pivot(A.row(it.index()));
      if(index == col)
      {
        pivot_row = it.index();
        break;
      }
    }
    else
    {
      COMISO_THROW_TODO("There should be no non zero values in the matrix");
    }
  }
  return pivot_row;
}

int ExactConstraintSatisfaction::get_pivot_row_new(const SP_Matrix_C& A, const SP_Matrix_R& _A, int col)
{
  auto collumn = A.col(col);
  if (collumn.nonZeros() != 1) // a pivot is allways the only entry in a column
    return -1;

  auto row = SP_Matrix_C::InnerIterator(A, col).index();

  // check if col is the first element in row
  auto first_index = SP_Matrix_R::InnerIterator(_A, row).index();

  if (first_index == col)
    return row;

  return -1;
}

std::vector<int> ExactConstraintSatisfaction::get_divisors_student(const ExactConstraintSatisfaction::SP_Matrix_C& A, int col)
{
  std::vector<int> D;
  for(SP_Matrix_C::InnerIterator it(A, col); it; ++it)
  {
    COMISO_THROW_TODO_if(it.value() == 0, "There should be no zeros left in the matrix");
    if(it.value() != 0 && it.index() <= col && it.index() < number_pivots_)
    {
      int pivot_col = index_pivot(A.row(it.index()));
      D.push_back(A.coeff(it.index(), pivot_col));
    }
  }

  return D;
}

std::vector<int> ExactConstraintSatisfaction::get_divisors_new(const SP_Matrix_C& A, const SP_Matrix_R& _A, int col)
{
  std::vector<int> D;
  for(SP_Matrix_C::InnerIterator it(A, col); it; ++it)
  {
    COMISO_THROW_TODO_if(it.value() == 0, "There should be no zeros left in the matrix");
    COMISO_THROW_TODO_if(it.index() >= number_pivots_, "The matrix should only contain number_pivots non empty rows");
    COMISO_THROW_TODO_if(it.index() > col,             "The matrix should not contain elements below the diagonal");

    D.push_back(SP_Matrix_R::InnerIterator(_A, it.index()).value());
  }
  return D;
}

std::vector<std::pair<int, double> > ExactConstraintSatisfaction::get_dot_product_elements_student(const SP_Matrix_C& A, const Eigen::VectorXd& x,  int k, int pivot_row)
{
  DEB_enter_func;
  std::vector<std::pair<int, double>> S;
  int cols = static_cast<int>(A.cols());
  for(int i = k+1; i < cols; i++)
  {                      //construct the list S to do the dot Product
    std::pair<int, double> pair;
    pair.first = A.coeff(pivot_row,i);
    DEB_only(double test = x.coeff(i) / A.coeff(pivot_row,k));
    DEB_warning_if(x.coeff(i) != ( test * A.coeff(pivot_row,k)), 2,
                   "can't devide" << " in row : " << i);
    pair.second = x.coeff(i) / A.coeff(pivot_row,k);
    if(pair.first != 0)
      S.push_back(pair);
  }

  return S;
}

std::vector<std::pair<int, double> > ExactConstraintSatisfaction::get_dot_product_elements_new(const SP_Matrix_R& A, const Eigen::VectorXd& x, int pivot_row)
{
  std::vector<std::pair<int, double>> S;

  SP_Matrix_R::InnerIterator it(A, pivot_row);
  auto pivot_val = it.value();

  while (++it)
  {
    std::pair<int, double> pair;
    pair.first = it.value();
    auto tmp = x.coeff(it.index());
    COMISO_THROW_TODO_if((tmp / pivot_val) * pivot_val != tmp, "element in x is not divisible by pivot element");
    pair.second = tmp / pivot_val;
    S.push_back(pair);
  }
  return S;
}

}

