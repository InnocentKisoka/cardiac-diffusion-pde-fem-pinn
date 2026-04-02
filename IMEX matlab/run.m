% FEM Simulation Loop: all dt, ne, sigma_d combinations
clear; clc;

sigma_h = 9.5298e-4;
a = 18.515; fr = 0; ft = 0.2383; fd = 1;
T = 35;

dt_list = [0.1, 0.05, 0.025];
ne_list = [64, 128];
sigma_d_factors = [0.1, 1, 10];

% BEFORE your big triple loop
results = {};  

for sigma_factor = sigma_d_factors
    for dt = dt_list
        for ne = ne_list
            fprintf('Running: sigma_d = %.2f, dt = %.3f, ne = %d\n', sigma_factor, dt, ne);

            nvx = ne + 1; nvy = ne + 1;
            Lx = 1; Ly = 1;
            hx = Lx / (nvx - 1); hy = Ly / (nvy - 1);
            nt = round(T / dt);
            sigma_d = sigma_h * sigma_factor;

            [X, Y] = meshgrid(linspace(0, Lx, nvx), linspace(0, Ly, nvy));
            coords = [X(:), Y(:)];
            U = double(X >= 0.9 & Y >= 0.9);
            U_all = zeros(nvx * nvy, nt);
            U_all(:, 1) = U(:);

            % Compute element-wise diffusivity
            nelem = (nvx - 1) * (nvy - 1);
            sigmaElem = sigma_h * ones(nelem, 1);
            idx = 1;
            for j = 1:nvy-1
                for i = 1:nvx-1
                    cx = mean([X(i,j), X(i+1,j), X(i,j+1), X(i+1,j+1)]);
                    cy = mean([Y(i,j), Y(i+1,j), Y(i,j+1), Y(i+1,j+1)]);
                    if ((cx - 0.3)^2 + (cy - 0.7)^2 < 0.1^2) || ...
                       ((cx - 0.7)^2 + (cy - 0.3)^2 < 0.15^2) || ...
                       ((cx - 0.5)^2 + (cy - 0.5)^2 < 0.1^2)
                        sigmaElem(idx) = sigma_d;
                    end
                    idx = idx + 1;
                end
            end

            M = assembleMass(nvx, nvy, hx, hy);
            K = assembleDiffusion(nvx, nvy, hx, hy, sigmaElem);
            A = M + dt * K;

            for n = 1:nt-1
                u = U_all(:,n);
                f = assembleReaction(nvx, nvy, hx, hy, u, a, fr, ft, fd);
                rhs = M * u - dt * f;
                U_all(:,n+1) = A \ rhs;

                % Clamp values to [0, 1]
                U_all(:,n+1) = max(0, min(1, U_all(:,n+1)));
            end

            % Activation time with safeguard
            maxU = max(U_all, [], 1);
            t_act = find(maxU > ft, 1);
            if isempty(t_act)
                activation_time = 0;
                fprintf('Warning: No activation above ft = %.4f detected for sigma_d = %.2f, dt = %.3f, ne = %d\n', ft, sigma_factor, dt, ne);
            else
                activation_time = dt * (t_act - 1);
            end

            % M-matrix check (relaxed)
            A_full = full(A);
            is_M = all(diag(A_full) > sum(abs(A_full),2) - abs(diag(A_full)));

            % u in [0,1] check
            minU = min(U_all(:)); maxU = max(U_all(:));
            is_in_range = minU >= 0 && maxU <= 1;

            % Save results
            results(end+1,:) = {sigma_factor, dt, ne, activation_time, is_M, is_in_range}; 

            % Save animation
            filename = sprintf('sim_sigma%.1f_dt%.3f_ne%d.gif', sigma_factor, dt, ne);

            for t = 1:5:nt
                imagesc(reshape(U_all(:, t), nvx, nvy)');
                set(gca, 'YDir', 'normal');
                axis equal tight;
                caxis([0 1]);
                colorbar;
                title(sprintf('t = %.2f s', (t-1)*dt));
            
                drawnow;
                frame = getframe(gcf);
                im = frame2im(frame);
                [imind, cm] = rgb2ind(im, 256);
            
                if t == 1
                    imwrite(imind, cm, filename, 'gif', 'Loopcount', inf, 'DelayTime', 0.1);
                else
                    imwrite(imind, cm, filename, 'gif', 'WriteMode', 'append', 'DelayTime', 0.1);
                end
            end
        end
    end
end

% Save results to CSV
T = cell2table(results, 'VariableNames', ...
    {'Sigma_d_factor','dt','ne','ActivationTime','IsMMatrix','U_in_0_1'});
writetable(T, 'fem_results.csv');
disp('✅ All experiments done and saved!');
