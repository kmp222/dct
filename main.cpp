#include <iostream>
#include "fftw3.h"
#include <cstdlib>

const double pi = 3.14159265359;
using namespace std;

// 1D FUNCTIONS
/* void print_vector(int n, double* vec) {

    for (int i = 0; i < n; i++)
        cout << vec[i] << " ";
    printf("\n");

}

void normalize_vector(int n, double* vec) {

    double f = sqrt(1.0 / 2.0 / n);

    vec[0] *= sqrt(1.0 / 4.0 / n);

    for (int i = 1; i < n; i++)
        vec[i] *= f;

}

void library_dct2_1d() {

    // create vector
    double a[] = { 231, 32, 233, 161, 24, 71, 140, 245 };
    int vec_size = sizeof(a) / sizeof(double);
   
    // print vector
    printf("input vector: ");
    print_vector(vec_size, a);   

    // dct
    fftw_plan plan = fftw_plan_r2r_1d(vec_size, a, a, FFTW_REDFT10, FFTW_ESTIMATE);
    fftw_execute(plan);

    // normalize vector
    normalize_vector(vec_size, a);

    // print 
    printf("DCT2 on input vector: ");
    print_vector(vec_size, a);

    // free memory
    fftw_destroy_plan(plan);
    fftw_cleanup();

}
*/

// 2D FUNCTIONS

template <int rows, int column>
/* void print_matrix_as_ref(double(&a)[rows][column])
{
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < column; j++) {
            cout << a[i][j] << " ";
        }
        printf("\n");
    }
}
*/

void print_matrix(double** a, int size)
{
    for (int i = 0; i < size; i++) {
        for (int j = 0; j < size; j++) {
            cout << a[i][j] << " ";
        }
        printf("\n");
    }
}

double** generate_matrix(int size) {
    
    double** matrix = 0;
    matrix = new double* [size];

    for (int i = 0; i < size; i++) {
        
        matrix[i] = new double[size];

        for (int j = 0; j < size; j++) {
            matrix[i][j] = rand() % 256;
        }

    }

    return matrix;

}

double** library_dct2_2d(double** a, int matrix_size) {

    // init result matrix
    double** dct = new double* [matrix_size];
    for (int i = 0; i < matrix_size; i++) {
        dct[i] = new double[matrix_size]; 
        for (int j = 0; j < matrix_size; j++) {
            dct[i][j] = 0;
        }
    }

    // dct
    fftw_plan plan = fftw_plan_r2r_2d(matrix_size, matrix_size, *a, *dct, FFTW_REDFT10, FFTW_REDFT10, FFTW_ESTIMATE);
    fftw_execute(plan);

    // normalize matrix
    double c = 1.0 / (matrix_size * 2.0);
    double c0 = 1.0 / sqrt(2.0);

    for (int i = 0; i < matrix_size; i++) {
        for (int j = 0; j < matrix_size; j++) {

            if (i == 0 && j == 0)
                dct[i][j] *= c * c0 * c0;
            else if (i == 0 || j == 0)
                dct[i][j] *= c * c0;
            else
                dct[i][j] *= c;
            
        }
    } 
    
    // free memory
    fftw_destroy_plan(plan);
    fftw_cleanup();

    return dct;

}

double** my_dct2_2d(double** a, int matrix_size) {

    // init result matrix
    double** dct = new double* [matrix_size];
    for (int i = 0; i < matrix_size; i++) {
        dct[i] = new double[matrix_size];
    }

    // dct + normalization
    double ci, cj, val_dct, sum;

    // double tmp_dct[matrix_size][matrix_size]{};
    double** tmp_dct = new double* [matrix_size];
    for (int i = 0; i < matrix_size; i++) {
        tmp_dct[i] = new double[matrix_size];
    }

    // iterating on rows
    for (int i = 0; i < matrix_size; i++) {
        for (int j = 0; j < matrix_size; j++) {

            sum = 0;

            for (int col = 0; col < matrix_size; col++) {

                val_dct = a[i][col] *
                    cos((2.0 * col + 1.0) * j * pi / (2.0 * matrix_size));

                sum = sum + val_dct;

            }

            if (j == 0)
                cj = 1.0 / sqrt(matrix_size);
            else
                cj = sqrt(2.0) / sqrt(matrix_size);

            tmp_dct[i][j] = cj * sum;

        }
    }

    // iterating on columns
    for (int j = 0; j < matrix_size; j++) {
        for (int i = 0; i < matrix_size; i++) {

            sum = 0;

            for (int row = 0; row < matrix_size; row++) {

                val_dct = tmp_dct[row][j] *
                    cos((2.0 * row + 1.0) * i * pi / (2.0 * matrix_size));

                sum = sum + val_dct;

            }

            if (i == 0)
                ci = 1.0 / sqrt(matrix_size);
            else
                ci = sqrt(2.0) / sqrt(matrix_size);

            dct[i][j] = (sum * ci);

        }
    }

    return dct;

}

// MAIN

int main() {

    int sizes[28] = { 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000 };

    for (int i = 0; i < sizeof(sizes) / sizeof(int); i++) {

        double** a = generate_matrix(sizes[i]);

        // TIMER
        for (int j = 0; j < 100; j++) {

            my_dct2_2d(a, sizes[i]);

        }
        // END TIMER

        // int media tempo / 100
        // salva in file (size, tempo)

        // TIMER2
        /* for (i = 0; i < 100; i++) {

            library_dct2_2d(a, sizes[i]);

        } */
        // TIMER2

        // int media tempo / 100
        // salva in file2 (size, tempo)

        delete a;

    }

    return 0;

}