function A = assembleMass(nvx, nvy, hx, hy)
    Mloc = (hx * hy / 36) * ...
        [4 2 1 2;
         2 4 2 1;
         1 2 4 2;
         2 1 2 4];

    nv = nvx * nvy;
    ne = (nvx - 1) * (nvy - 1);

    id = reshape(1:nv, nvy, nvx)';  % <- FIXED orientation

    a = id(1:end-1, 1:end-1); a = a(:)';
    b = id(2:end,   1:end-1); b = b(:)';
    c = id(1:end-1, 2:end);   c = c(:)';
    d = id(2:end,   2:end);   d = d(:)';
    conn = [a; b; d; c];

    ii = repmat((1:4)', 1, 4);
    jj = ii';
    I = conn(ii(:), :);
    J = conn(jj(:), :);

    V = repmat(Mloc(:), 1, ne);
    A = sparse(I(:), J(:), V(:), nv, nv);
end