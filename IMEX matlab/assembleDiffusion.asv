function A = assembleDiffusion(nvx, nvy, hx, hy, Sigma)
% nvx, nvy: number of vertices along x, y
% hx, hy: mesh size along x, y
% Sigma: vector of diffusivities for each element (size ne)
Aref = [1 -1; -1 1]; % Stiffness reference matrix
Mref = [1/3 1/6; 1/6 1/3]; % Mass reference matrix
Ax = 1/hx * Aref;
Ay = 1/hy * Aref;
Mx = hx * Mref;
My = hy * Mref;
Aloc_base = kron(My, Ax) + kron(Ay, Mx); % Local stiffness matrix
nv = nvx * nvy;
ne = (nvx-1)*(nvy-1);
id = reshape(1:nv, nvx, nvy);
% Connectivity: bottom-left, bottom-right, top-left, top-right
a = id(1:end-1, 1:end-1); a = a(:)';
b = id(2:end, 1:end-1);   b = b(:)';
c = id(1:end-1, 2:end);   c = c(:)';
d = id(2:end, 2:end);     d = d(:)';
conn = [a; b; c; d];
% Assemble sparse matrix
I = zeros(16*ne, 1);
J = zeros(16*ne, 1);
V = zeros(16*ne, 1);
k = 1;
for e = 1:ne
    for i = 1:4
        for j = 1:4
            I(k) = conn(i, e);
            J(k) = conn(j, e);
            V(k) = Sigma(e) * Aloc_base(i, j);
            k = k + 1;
        end
    end
end
A = sparse(I, J, V, nv, nv);
% Verify matrix properties
offdiag = full(A(~eye(nv)));
if any(offdiag > 1e-8)
    warning('Positive off-diagonal entries in A: max = %e', max(offdiag));
end
if any(sum(A, 2) < -1e-8)
    warning('Negative row sums in A: min = %e', min(sum(A, 2)));
end
end