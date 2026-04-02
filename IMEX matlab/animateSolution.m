function animateSolution(U_all, nvx, nvy, dt)
    figure;
    step = 1;
    
    for t = 1:size(U_all, 2)
        imagesc(reshape(U_all(:, t), nvx, nvy));  % reshape 1D → 2D
        set(gca, 'YDir', 'normal');               % ✅ flip vertically
        title(sprintf("Time: %.2f", (t-1)*dt));
        colorbar;
        axis equal tight;
        caxis([0 1]);  % 🔥 consistent color scale
        drawnow;
    end
end
