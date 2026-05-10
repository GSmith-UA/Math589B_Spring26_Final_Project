#include "lqr.h"
#include <cmath>
#include <Eigen/Dense>
#include <Eigen/Eigenvalues>

Matrix2 solveLQRRiccati(double alpha) {
    // Build the 4x4 Hamiltonian matrix H = [A, -S; -Q, -A^T]
    // A = [0 1; 1 -alpha],  S = B*R^{-1}*B^T = [0 0; 0 1],  Q = I_2
    //
    // This is identical to the Jacobian of the full Hamiltonian system at the
    // origin, which is not a coincidence: the linearized optimal problem IS LQR.
    Eigen::Matrix4d H;
    H <<  0,      1,    0,     0,
          1,  -alpha,   0,    -1,
         -1,      0,    0,    -1,
          0,     -1,   -1,  alpha;

    Eigen::EigenSolver<Eigen::Matrix4d> es(H);

    // Collect the 2 stable eigenvectors (Re(eigenvalue) < 0)
    Eigen::MatrixXcd Vs(4, 2);
    int col = 0;
    for (int i = 0; i < 4 && col < 2; ++i) {
        if (es.eigenvalues()[i].real() < -1e-10)
            Vs.col(col++) = es.eigenvectors().col(i);
    }

    // P = X2 * X1^{-1}, where X1 = top 2 rows, X2 = bottom 2 rows of Vs
    Eigen::Matrix2cd X1 = Vs.topRows(2);
    Eigen::Matrix2cd X2 = Vs.bottomRows(2);
    Eigen::Matrix2d P = (X2 * X1.inverse()).real();

    // Symmetrize to kill floating-point asymmetry from complex arithmetic
    P = 0.5 * (P + P.transpose());

    Matrix2 result;
    for (int i = 0; i < 2; ++i)
        for (int j = 0; j < 2; ++j)
            result[i][j] = P(i, j);

    return result;
}

Vector2 estimateCostateViaLQR(double theta, double phi, const Matrix2& P) {
    return { P[0][0]*theta + P[0][1]*phi,
             P[1][0]*theta + P[1][1]*phi };
}

bool isSmallAngle(double theta, double phi, double threshold) {
    return std::sqrt(theta*theta + phi*phi) < threshold;
}
