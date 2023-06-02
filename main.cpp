#include <iostream>
#include "fftw3.h"
#include <cstdlib>
#include <chrono>
#include <fstream>

const double pi = 3.14159265359;
using namespace std;
using namespace chrono;

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

void delete_matrix(double** M, int rowSize, int colSize) {
    for (int row = 0; row < rowSize; row++) {
        delete[] M[row];
        M[row] = NULL;
    }
    delete[] M;
    M = NULL;
}

double* library_dct2_2d(double* a, int matrix_size) {

    // init result matrix
    double* dct = new double[matrix_size*matrix_size];

    // dct
    fftw_plan plan = fftw_plan_r2r_2d(matrix_size, matrix_size, a, dct, FFTW_REDFT10, FFTW_REDFT10, FFTW_ESTIMATE);
    fftw_execute(plan);

    // normalize matrix
    double c = 1.0 / (matrix_size * 2.0);
    double c0 = 1.0 / sqrt(2.0);

    dct[0] *= c * c0 * c0;

    for (int i = 1; i < matrix_size; i++) {
        dct[i] *= c * c0;
    }
    for (int i = matrix_size; i < matrix_size * matrix_size; i++) {
        if (i % matrix_size == 0)
            dct[i] *= c * c0;
        else
            dct[i] *= c;
    }    

    // free memory
    fftw_destroy_plan(plan);
    fftw_free;
    fftw_cleanup();

    return dct;

}

/* double** library_dct2_2d_old(double** a, int matrix_size) {

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
    fftw_free;
    fftw_cleanup();

    return dct;

} */

double** my_dct2_2d(double** a, int matrix_size) {

    // init result matrix
    double** dct = new double* [matrix_size];
    for (int i = 0; i < matrix_size; i++) {
        dct[i] = new double[matrix_size];
    }

    // dct + normalization
    double ci, cj, val_dct, sum;

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

    delete_matrix(tmp_dct, matrix_size, matrix_size);

    return dct;

}

// MAIN

int main() {

    // init output file to save performances
    ofstream file;

    // matrix sizes for testing
    int sizes[28] = { 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000 };

    // test implementations on each size
    for (int i = 0; i < sizeof(sizes)/sizeof(int); i++) {

        double** a = generate_matrix(sizes[i]); // generate matrix of n-size

        auto start = high_resolution_clock::now();

        int nIter = 100; // set number of iterations on both implementations

        for (int j = 0; j < nIter; j++) {

            my_dct2_2d(a, sizes[i]); // apply my dct on matrix of n-size

        }

        auto end = high_resolution_clock::now();

        float avg_time = duration_cast<microseconds>(end - start).count() / 1000000.0 / (float) nIter; // average time in microseconds of a single iteration on my dct

        // print performances to console
        cout << "MY DCT : " << endl;
        cout << "SIZE: " << sizes[i] << " TEMPO MEDIO: " << avg_time << endl;
        
        // save performance to txt file as (size, time)
        file.open("my_dct_performance.txt", std::ofstream::out | std::ofstream::app);
        file << sizes[i] << " " << avg_time << "\n";
        file.close();

        // create matrix stored in row major order starting from original matrix (fftw requirement)
        double* b = new double[sizes[i] * sizes[i]];
        for (int k = 0; k < sizes[i]; k++) {
            b[k] = a[0][k];
        }

        int row = 1;
        int counter = sizes[i];

        while (counter < sizes[i] * sizes[i]) {

            for (int column = 0; column < sizes[i]; column++) {
                b[counter] = a[row][column];
                counter++;
            }
            row++;
        }

        start = high_resolution_clock::now();

        for (int k = 0; k < nIter; k++) {

            library_dct2_2d(b, sizes[i]); // apply fftw dct on matrix in row major order of n-size

        }

        end = high_resolution_clock::now();

        avg_time = duration_cast<microseconds>(end - start).count() / 1000000.0 / (float) nIter; // average time in microseconds of a single iteration on fftw dct

        // print performances to console
        cout << "LIBRARY DCT : " << endl;
        cout << "SIZE: " << sizes[i] << " TEMPO MEDIO: " << avg_time << endl;

        // save performance to txt file as (size, time)
        file.open("library_performance.txt", std::ofstream::out | std::ofstream::app);
        file << sizes[i] << " " << avg_time << "\n";
        file.close();

        // free memory
        delete_matrix(a, sizes[i], sizes[i]);
        delete[] b;
        b = NULL;

    }

    return 0;

}

// MAIN FOR TESTING
/* int main() {

    // init input
    int size = 4;
    double** a = generate_matrix(size);

    // print input
    printf("INPUT MATRIX:\n");
    print_matrix(a, size);

    // my dct
    double** r = my_dct2_2d(a, size);

    // print my dct
    printf("\nMY DCT:\n");
    print_matrix(r, size);

    // library dct
    // double** r2 = library_dct2_2d_old(a, size);

    // print library dct
    // printf("\nLIBRARY DCT:\n");
    // print_matrix(r2, size);

    // library2 dct

    // create matrix stored in row major order starting from original matrix (fftw requirement)
    double* b = new double[size*size];
    for (int i = 0; i < size; i++) {
        b[i] = a[0][i];
    }

    int row = 1;
    int i = size;

    while (i < size*size) {

        for (int column = 0; column < size; column++) {
            b[i] = a[row][column];
            i++;
        }
        row++;
    }

    double* r3 = library_dct2_2d(b, size);

    // print library2 dct
    printf("\nLIBRARY2 DCT:\n");
    for (int i = 0; i < size*size; i++) {
        cout << r3[i] << " ";
    }

    delete a;
    delete r;
    // delete r2;
    delete r3;

    return 0;
 
} */
