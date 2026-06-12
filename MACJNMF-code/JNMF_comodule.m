function [W, H1, H2, H3, H4] = JNMF_comodule(X1, X2, X3, X4, L4, H1, H2, H3, H4, W, Q, A, B, lambda1, lambda2, lambda3, alpha, beta, beta_param, K, max_iter)

modalities = {X1, X2, X3, X4};
for i = 1:4
    idx = find(sum(modalities{i}, 1) == 0);
    if ~isempty(idx)
        modalities{i}(:, idx) = modalities{i}(:, idx) + eps;
    end
end
X1 = modalities{1}; X2 = modalities{2}; X3 = modalities{3}; X4 = modalities{4};

n_runs = 5;
best_total_obj = Inf;
bestW = W; bestH1 = H1; bestH2 = H2; bestH3 = H3; bestH4 = H4;

for run = 1:n_runs
    fprintf('---- Run %d/%d ----\n', run, n_runs);
    
    W = max(0, rand(size(X1,1), K));
    H1 = max(0, rand(K, size(X1,2)));
    H2 = max(0, rand(K, size(X2,2)));
    H3 = max(0, rand(K, size(X3,2)));
    H4 = max(0, rand(K, size(X4,2)));
    
    [W, H1, H2, H3, H4, obj_history] = ...
        JNMF(X1, X2, X3, X4, L4, H1, H2, H3, H4, W, Q, A, B, ...
             lambda1, lambda2, lambda3, alpha, beta, beta_param, max_iter);

    total_obj = obj_history(end);
    if total_obj < best_total_obj
        bestW = W;
        bestH1 = H1; bestH2 = H2; bestH3 = H3; bestH4 = H4;
        best_total_obj = total_obj;
    end
end
fprintf('---- Best Final Objective (obj_history(end)): %.4f ----\n', best_total_obj); % <-- 添加这一行
W = bestW;
H1 = bestH1; H2 = bestH2; H3 = bestH3; H4 = bestH4;


    % writematrix(W, 'W_matrix.csv');
    % writematrix(H1, 'H1_matrix.csv');
    % writematrix(H2, 'H2_matrix.csv');
    % writematrix(H3, 'H3_matrix.csv');
    % writematrix(H4, 'H4_matrix.csv');
    % 
  
    % X1_recon = W * H1;
    % X2_recon = W * H2;
    % X3_recon = W * H3;
    % X4_recon = W * H4;
    % 
    % writematrix(X1_recon, 'X1_reconstructed.csv');
    % writematrix(X2_recon, 'X2_reconstructed.csv');
    % writematrix(X3_recon, 'X3_reconstructed.csv');
    % writematrix(X4_recon, 'X4_reconstructed.csv');
end
