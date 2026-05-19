using LinearAlgebra: I

function quantecon_kalman_ared_problem()
    A_kalman = [0.95 0.0; 0.0 0.95]
    Q_kalman = Matrix(I, 2, 2) .* 0.5
    G_kalman = Matrix(I, 2, 2) .* 0.5
    R_kalman = Matrix(I, 2, 2) .* 0.2
    return Matrix(A_kalman'), Matrix(G_kalman'), R_kalman, Q_kalman
end

function random_symmetric_direction(rng, X, scale = 0.1)
    dX = scale .* randn(rng, size(X))
    return 0.5 .* (dX + dX')
end
