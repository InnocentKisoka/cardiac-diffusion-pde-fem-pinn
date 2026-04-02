function F = assembleReaction(nvx, nvy, hx, hy, U, a, fr, ft, fd)
    nv = nvx * nvy;
    ne = (nvx - 1) * (nvy - 1);
    F = zeros(nv, 1);

    id = reshape(1:nvx*nvy, nvy, nvx)';
    
    % One-point quadrature on [0, 1]^2 (midpoint rule)
    xi = 0.5; eta = 0.5;  % midpoint
    w = 1;                % weight

    % Shape functions at midpoint (same for all elems)
    N = [(1 - xi)*(1 - eta);
          xi    *(1 - eta);
          xi    *eta;
         (1 - xi)*eta];

    for j = 1:nvy-1
        for i = 1:nvx-1
            nodes = [id(i,j); id(i+1,j); id(i+1,j+1); id(i,j+1)];
            u_local = U(nodes);
            u_val = N' * u_local;

            % Cubic reaction at quadrature point
            f_val = a * (u_val - fr) * (u_val - ft) * (u_val - fd);

            % Element contribution to global F
            F(nodes) = F(nodes) + f_val * N * w * hx * hy;
        end
    end
end
