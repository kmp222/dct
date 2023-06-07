// Test DCT2 implementation and FFTW DCT2 on same inputs

// TODO separate functions and main in different files

#include <iostream>
#include "fftw3.h"
#include <cstdlib>
#include <fstream>
#include <time.h>

const double pi = 3.14159265359;
using namespace std;

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

void library_dct2_2d(double* a, double* r, int matrix_size) {

    // dct
    fftw_plan plan = fftw_plan_r2r_2d(matrix_size, matrix_size, a, r, FFTW_REDFT10, FFTW_REDFT10, FFTW_ESTIMATE);
    fftw_execute(plan);

    // normalize matrix
    double c = 1.0 / (matrix_size * 2.0);
    double c0 = 1.0 / sqrt(2.0);
    double c_scaled = c * c0;

    r[0] *= c_scaled * c0;

    for (int i = 1; i < matrix_size; i++) {
        r[i] *= c_scaled;
    }
    for (int i = matrix_size; i < matrix_size * matrix_size; i++) {
        r[i] *= (i % matrix_size == 0) ? c_scaled : c;
    }

    // free memory
    fftw_destroy_plan(plan);
    fftw_cleanup();
    fftw_free;

}

// TODO array instead of matrix as parameter
void my_dct2_2d(double** a, double** r, int matrix_size) {

    // dct + normalization
    double ci, cj, val_dct, sum;

    // temp result
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

            r[i][j] = (sum * ci);

        }
    }

    // free memory
    delete_matrix(tmp_dct, matrix_size, matrix_size);

}

int main() {

    // init output file to save performances
    ofstream file;

    // matrix sizes for testing
    int sizes[28] = { 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 150, 200, 250, 300, 350, 400, 450, 500, 550, 600, 650, 700, 750, 800, 850, 900, 950, 1000 };

    // set number of iterations on both implementations
    int nIter = 100; 

    // test implementations on each size
    for (int i = 0; i < sizeof(sizes) / sizeof(int); i++) {

        double** input_matrix = generate_matrix(sizes[i]); // generate matrix of n-size

        if (sizes[i] >= 100) {
            nIter = 50;
        }

        // ---------------------------------------------------------------------------------------- MY DCT TEST
        
        // create result matrix
        double** my_res = new double* [sizes[i]];
        for (int k = 0; k < sizes[i]; k++) {
            my_res[k] = new double[sizes[i]];
        }

        // measure time for benchmark
        auto start = (double) clock();

        for (int j = 0; j < nIter; j++) {

            my_dct2_2d(input_matrix, my_res, sizes[i]); // apply my dct on matrix of n-size

        }

        auto end = (double) clock() - start;

        float avg_time = ((float)end / (float)CLOCKS_PER_SEC) / (float)nIter; // average time in seconds of a single iteration on my dct

        // print performances to console
        cout << "MY DCT: " << endl;
        cout << "SIZE: " << sizes[i] << " TEMPO MEDIO: " << avg_time << endl;
        
        // save performance to txt file as (size, time)
        file.open("my_dct_performance.txt", std::ofstream::out | std::ofstream::app);
        file << sizes[i] << " " << avg_time << "\n";
        file.close();

        // free memory
        delete_matrix(my_res, sizes[i], sizes[i]);

        // ---------------------------------------------------------------------------------------- END MY DCT TEST

        // ---------------------------------------------------------------------------------------- LIBRARY DCT TEST

        // create matrix stored in row major order starting from original matrix (fftw requirement)
        // TODO AS FUNCTION
        double* lib_input_array = new double[sizes[i] * sizes[i]];

        for (int k = 0; k < sizes[i]; k++) {
            lib_input_array[k] = input_matrix[0][k];
        }

        int row = 1;
        int counter = sizes[i];

        while (counter < sizes[i] * sizes[i]) {

            for (int column = 0; column < sizes[i]; column++) {
                lib_input_array[counter] = input_matrix[row][column];
                counter++;
            }
            row++;
        }

        // free memory
        delete_matrix(input_matrix, sizes[i], sizes[i]);      
        
        // create array for fftw dct result
        double* lib_res = new double[sizes[i] * sizes[i]];

        // measure time for benchmark
        start = (double) clock();

        for (int k = 0; k < nIter; k++) {

            library_dct2_2d(lib_input_array, lib_res, sizes[i]); // apply fftw dct on matrix in row major order of n-size

        }

        end = (double) clock() - start;

        avg_time = ( (float) end / (float) CLOCKS_PER_SEC) / (float) nIter; // average time in seconds of a single iteration on fftw dct

        // print performances to console
        cout << "LIBRARY DCT: " << endl;
        cout << "SIZE: " << sizes[i] << " TEMPO MEDIO: " << avg_time << endl;

        // save performance to txt file as (size, time)
        file.open("library_performance.txt", std::ofstream::out | std::ofstream::app);
        file << sizes[i] << " " << avg_time << "\n";
        file.close();

        // free memory
        delete[] lib_input_array;
        lib_input_array = NULL;
        delete[] lib_res;
        lib_res = NULL;

        // ---------------------------------------------------------------------------------------- END LIBRARY DCT TEST

    }

    return 0;

}

// MAIN FOR TESTING
/* int main() {

    // init input
    int size = 5;
    double** a = generate_matrix(size);

    // print input
    printf("INPUT MATRIX:\n");
    print_matrix(a, size);

    // create result matrix
    double** res = new double*[size];
    for (int i = 0; i < size; i++) {
        res[i] = new double[size];
    }

    // my dct
    my_dct2_2d(a, res, size);

    // print my dct
    printf("\nMY DCT:\n");
    print_matrix(res, size);

    // free memory
    delete_matrix(res, size, size);

    // library dct

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

    // free memory
    delete_matrix(a, size, size);

    // create array for fftw dct result
    double* result = new double[size*size];

    library_dct2_2d(b, result, size);

    // print library dct
    // TODO define function to print vector as matrix in row major order
    printf("\nLIBRARY DCT:\n");
    for (int i = 0; i < size * size; i++) {
        cout << result[i] << " ";
    }

    // free memory
    delete[] b;
    delete[] result;
    b = NULL;
    result = NULL;

    return 0;
 
} */
