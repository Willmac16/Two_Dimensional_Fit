// cppimport
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <iostream>

namespace py = pybind11;

py::array_t<double> rref_py(py::array_t<double> mat)
{
    py::buffer_info buf = mat.request();

    if (buf.ndim != 2)
        throw std::runtime_error("Number of dimensions must be two");

    py::array result = py::array_t<double>(buf.size);

    py::buffer_info out_buf = result.request();

    double *in_ptr = (double *) buf.ptr,
           *out_ptr = (double *) out_buf.ptr;

    size_t X = buf.shape[0];
    size_t Y = buf.shape[1];

    // Copy input to output
    for (size_t idx = 0; idx < X; idx++)
    {
        for (size_t idy = 0; idy < Y; idy++)
        {
            out_ptr[idx + idy * X] = in_ptr[idx + idy * X];
        }
    }

    double scale_factor, temp;

    // RREF the output
    for (size_t rc = 0; rc < Y; rc++)
    {
        bool lead = false;

        for (size_t row = 0; row < Y; row++)
        {
            if (!lead && out_ptr[rc + row*X] != 0.0)
            {
                lead = true;

                scale_factor = out_ptr[rc + row*X];

                // Normalize and Pivot
                for (size_t col = rc; col < X; col++)
                {

                    temp = out_ptr[col + row*X] / scale_factor;
                    out_ptr[col + row*X] = out_ptr[col + rc*X];

                    out_ptr[col + rc*X] = temp;
                }
            }
        }

        if (lead)
        {
            for (size_t row = 0; row < Y; row++)
            {
                if (row != rc)
                {
                    scale_factor = out_ptr[rc + row * X];

                    for (size_t col = rc; col < X; col++)
                    {
                        out_ptr[col + row*X] -= out_ptr[col + rc*X] * scale_factor;
                    }
                }
            }
        }
    }

    result.resize({X,Y});

    return result;
}

double * rrefInPlace(double *result, int rows, int cols)
{
    double scale_factor, temp;

    // RREF the output
    for (int rc = 0; rc < rows; rc++)
    {
        bool lead = false;

        for (int row = rc; row < rows; row++)
        {
            if (!lead && result[rc + row*cols] != 0.0)
            {
                lead = true;

                scale_factor = result[rc + row*cols];

                // Normalize and Pivot
                for (int col = rc; col < cols; col++)
                {
                    temp = result[col + row*cols] / scale_factor;
                    result[col + row*cols] = result[col + rc*cols];

                    result[col + rc*cols] = temp;
                }
            }
        }

        if (lead)
        {
            for (int row = 0; row < rows; row++)
            {
                if (row != rc)
                {
                    scale_factor = result[rc + row * cols];

                    for (int col = rc; col < cols; col++)
                    {
                        result[col + row*cols] -= result[col + rc*cols] * scale_factor;
                    }
                }
            }
        }
    }

    return result;
}

double * rref(double *mat, int rows, int cols)
{
    double *result = new double[rows*cols];

    // Copy input to output
    for (int idx = 0; idx < cols; idx++)
    {
        for (int idy = 0; idy < rows; idy++)
        {
            result[idx + idy * cols] = mat[idx + idy * cols];
        }
    }

    return rrefInPlace(result, rows, cols);
}

double * solve(double *A, double *Z, int rows)
{
    double *aug = new double[rows*(rows+1)];

    for (int row = 0; row < rows; row++)
    {
        for (int col = 0; col < rows; col++)
        {
            aug[col + row*(rows+1)] = A[col + row*rows];
        }

        aug[row*(rows+1) + rows] = Z[row];
    }

    rrefInPlace(aug, rows, rows+1);

    double *cs = new double[rows];

    for (int rc = 0; rc < rows; rc++)
    {
        if (aug[rc*(2 + rows)] == 1.0)
            cs[rc] = aug[rc * (rows + 1) + rows];
        else
            cs[rc] = 0.0;
    }

    delete[] aug;

    return cs;
}

double sigma(py::array_t<double> points, int x_deg, int y_deg)
{
    py::buffer_info buf = points.request();

    if (buf.ndim != 2)
        throw std::runtime_error("Number of dimensions must be two");

    if (buf.shape[1] < 2)
        throw std::runtime_error("Points must be at least 2 dimensional");

    size_t num_points = buf.shape[0], point_deg = buf.shape[1];

    double sum = 0.0;

    double *ptr = (double *) buf.ptr;

    for (int ind = 0; ind < num_points; ind++)
    {
        sum += pow(ptr[ind * point_deg + 0], x_deg) * pow(ptr[ind * point_deg + 1], y_deg);
    }

    return sum;
}

double sigma(py::array_t<double> points, int x_deg, int y_deg, int z_deg)
{
    py::buffer_info buf = points.request();

    if (buf.ndim != 2)
        throw std::runtime_error("Number of dimensions must be two");

    if (buf.shape[1] < 3)
        throw std::runtime_error("Points must be at least 3 dimensional");

    size_t num_points = buf.shape[0], point_deg = buf.shape[1];

    double sum = 0.0;

    double *ptr = (double *) buf.ptr;

    for (int ind = 0; ind < num_points; ind++)
    {
        sum += pow(ptr[ind * point_deg + 0], x_deg) * pow(ptr[ind * point_deg + 1], y_deg) * pow(ptr[ind * point_deg + 2], z_deg);
    }

    return sum;
}

double sigma(py::array_t<double> points, int x_deg, int y_deg, int z_deg, int q_deg)
{
    py::buffer_info buf = points.request();

    if (buf.ndim != 2)
        throw std::runtime_error("Number of dimensions must be two");

    if (buf.shape[1] < 4)
        throw std::runtime_error("Points must be at least 4 dimensional");

    size_t num_points = buf.shape[0], point_deg = buf.shape[1];

    double sum = 0.0;

    double *ptr = (double *) buf.ptr;

    for (int ind = 0; ind < num_points; ind++)
    {
        sum += pow(ptr[ind * point_deg + 0], x_deg) * pow(ptr[ind * point_deg + 1], y_deg) * pow(ptr[ind * point_deg + 2], z_deg) * pow(ptr[ind * point_deg + 3], q_deg);
    }

    return sum;
}

py::array_t<double> twoDimPolyFit(py::array_t<double> points, int x_degree, int y_degree)
{
    int x_coeffs = x_degree + 1, y_coeffs = y_degree + 1;
    int combs = x_coeffs * y_coeffs;
    double *A = new double[combs*combs];

    int x_row, x_col, y_row, y_col;

    for (int r = 0; r < combs; r++)
    {
        x_row = r / y_coeffs;
        y_row = r % y_coeffs;
        for (int c = 0; c < combs; c++)
        {
            x_col = c / y_coeffs;
            y_col = c % y_coeffs;

            A[c + r*combs] = sigma(points, x_row+x_col, y_row+y_col);
        }
    }

    double *Z = new double[combs];
    int x_tow, y_tow;

    for (int t = 0; t < combs; t++)
    {
        x_tow = t / y_coeffs;
        y_tow = t % y_coeffs;

        Z[t] = sigma(points, x_tow, y_tow, 1);
    }

    double *cs = solve(A, Z, combs);
    delete[] A;
    delete[] Z;



    // Convert double array into py::array
    // Reshape into coeff array
    py::array_t<double> coeffs = py::array_t<double>(combs);

    double *coes = (double *) coeffs.request().ptr;

    for (int i = 0; i < combs; i++)
    {
        coes[i] = cs[i];
        std::cout << cs[i] << std::endl;
    }

    coeffs.resize({x_coeffs, y_coeffs});

    delete[] cs;
    return coeffs;
}

void py_print_array(py::array_t<double> points)
{
    py::buffer_info buf = points.request();

    if (buf.ndim != 2)
        throw std::runtime_error("Number of dimensions must be two");

    if (buf.shape[1] < 2)
        throw std::runtime_error("Points must be at least 2 dimensional");

    size_t num_points = buf.shape[0], point_deg = buf.shape[1];

    double *ptr = (double *) buf.ptr;

    for (int ind = 0; ind < num_points; ind++)
    {
        for (int c = 0; c < point_deg; c++)
        {
            std::cout << ptr[c + ind*point_deg] << ", ";
        }

        std::cout << std::endl;
    }
}

py::array_t<double> threeDimPolyFit(py::array_t<double> points, int x_degree, int y_degree, int z_degree)
{
    int x_coeffs = x_degree + 1, y_coeffs = y_degree + 1, z_coeffs = z_degree + 1;
    int combs = x_coeffs * y_coeffs * z_degree;
    double *A = new double[combs*combs];

    int x_row, x_col, y_row, y_col, z_row, z_col;

    for (int r = 0; r < combs; r++)
    {
        x_row = r / y_coeffs / z_coeffs;
        y_row = r / z_coeffs % y_coeffs;
        z_row = r % z_coeffs;
        for (int c = 0; c < combs; c++)
        {
            x_col = c / y_coeffs / z_coeffs;
            y_col = c / z_coeffs % y_coeffs;
            z_col = c % z_coeffs;

            A[c + r*combs] = sigma(points, x_row*x_col, y_row*y_col);
        }
    }

    double *Z = new double[combs];
    int x_tow, y_tow;

    for (int t = 0; t < combs; t++)
    {
        x_tow = t / y_coeffs;
        y_tow = t % y_coeffs;

        Z[t] = sigma(points, x_tow, y_tow, 1);
    }

    double *cs = solve(A, Z, combs);
    delete[] A;
    delete[] Z;

    // Convert double array into py::array
    // Reshape into coeff array
    py::array_t<double> coeffs = py::array(combs, cs);

    coeffs.resize({x_coeffs, y_coeffs, z_coeffs});

    delete[] cs;
    return coeffs;
}

PYBIND11_MODULE(cpp_tdf, m) {
    m.doc() = "This is a Python binding of a C++ Polynomial Least Squares Fitting library";

    m.def("rref", &rref_py, "Executes Gauss-Jordan Elimination on a matrix");
    m.def("twoDpolyFit", &twoDimPolyFit, "Least Squares Two Dimensional Poly Fit");
    m.def("threeDpolyFit", &threeDimPolyFit, "Least Squares Three Dimensional Poly Fit");
    m.def("print_array", &py_print_array, "Print a Numpy Array");
}

/*
<%
setup_pybind11(cfg)
%>
*/
