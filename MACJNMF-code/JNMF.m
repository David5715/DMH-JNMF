function [W, H1, H2, H3, H4, obj_history] = JNMF(X1, X2, X3, X4, L4, H1, H2, H3, H4, W, Q, A, B, lambda1, lambda2, lambda3, alpha, beta, beta_param, max_iter)

H = {H1, H2, H3, H4};
obj_history = zeros(max_iter, 1);
output_idx = 1;
max_outputs = ceil(max_iter / 100);
modalities = {X1, X2, X3, X4};  
hsic_history = zeros(max_outputs, 3);
label_error_history = zeros(max_outputs, 3);
laplacian_history = zeros(max_outputs, 1);
corr_median = zeros(max_outputs, 4);
for iter = 1:max_iter
    % ----- Updating the H1-H3 -----
    H_cell = {H1, H2, H3};
    for m = 1:3
        Zm = W * H_cell{m};
        Z4 = W * H4;
        if any(isnan(Zm(:))) || any(isinf(Zm(:))) || any(isnan(Z4(:))) || any(isinf(Z4(:)))
            warning('NaN/Inf detected in Zm or Z4');
            continue;
        end
        [grad_neg, grad_pos] = compute_HSIC_grad(Zm, Z4, beta);

        numerator = W' * eval(['X' num2str(m)]) + lambda1 * W' * grad_neg + 2 * lambda3 * A' * Q * B{m}';
        denominator = W' * W * H_cell{m} + lambda1 * W' * grad_pos + 2 * lambda3 * A' * A * H_cell{m} * (B{m} * B{m}');
        H_cell{m} = H_cell{m} .* (numerator ./ max(denominator, 1e-8));
        H_cell{m} = max(min(H_cell{m}, 1e2), 1e-3);
    end
    H1 = H_cell{1}; H2 = H_cell{2}; H3 = H_cell{3};

    % ----- Updating H4 -----
    grad_HSIC_H4_neg = 0;
    grad_HSIC_H4_pos = 0;
    for m = 1:3
        Zm = W * eval(['H' num2str(m)]);
        Z4 = W * H4;
        [grad_neg, grad_pos] = compute_HSIC_grad(Z4, Zm, beta);
        grad_HSIC_H4_neg = grad_HSIC_H4_neg + grad_neg;
        grad_HSIC_H4_pos = grad_HSIC_H4_pos + grad_pos;
    end

    numerator = W' * X4 + lambda2 * H4 * L4' + lambda1 * W' * grad_HSIC_H4_neg;
    denominator = W' * W * H4 + lambda2 * H4 * diag(sum(L4, 2)) + lambda1 * W' * grad_HSIC_H4_pos;
    H4 = H4 .* (numerator ./ max(denominator, 1e-8));
    H4 = max(min(H4, 1e2), 1e-3);

    % ----- Updating W -----
       grad_HSIC_W_neg = 0;
    grad_HSIC_W_pos = 0;
    Z4 = W * H4;
    for m = 1:3
        Hm = eval(['H' num2str(m)]);
        Zm = W * Hm;
        [grad_neg, grad_pos] = compute_HSIC_grad(Zm, Z4, beta);
        grad_HSIC_W_neg = grad_HSIC_W_neg + grad_neg * Hm';
        grad_HSIC_W_pos = grad_HSIC_W_pos + grad_pos * Hm';
    end
    numerator = X1 * H1' + X2 * H2' + X3 * H3' + X4 * H4' + lambda1 * grad_HSIC_W_neg;
    denominator = W * (H1 * H1' + H2 * H2' + H3 * H3' + H4 * H4') + lambda1 * grad_HSIC_W_pos;
    W = W .* (numerator ./ max(denominator, 1e-8));
    W = max(min(W, 1e2), 1e-3);

    % ----- Updating A & B -----
    for m = 1:3
        AHmB = A * H_cell{m} * B{m};
        grad_A = 2 * lambda3 * (AHmB - Q) * B{m}' * H_cell{m}';
       A = max(A - alpha * grad_A, 0);

        B{m} = B{m} .* (H_cell{m}' * A' * Q ./ max(H_cell{m}' * A' * A * H_cell{m} * B{m} + beta_param * B{m}, 1e-8));
        B{m} = max(min(B{m}, 1e2), 1e-3);
    end

    % ----- Calculate the objective function -----
    obj = 0;
    for m = 1:4
        X_recon = W * eval(['H' num2str(m)]);
        obj = obj + 0.5 * norm(eval(['X' num2str(m)]) - X_recon, 'fro')^2;
    end

    Z4 = W * H4;
    for m = 1:3
        Zm = W * eval(['H' num2str(m)]);
        obj = obj + lambda1 * HSIC(Zm, Z4, beta);
        AHmB = A * eval(['H' num2str(m)]) * B{m};
        obj = obj + lambda3 * norm(Q - AHmB, 'fro')^2;
    end

    obj = obj + lambda2 * trace(H4 * L4 * H4') + alpha * sum(abs(A(:))) + beta_param * sum(cellfun(@(x) sum(x.^2, 'all'), B));
    obj_history(iter) = obj;

    if mod(iter,100) == 0 || iter == max_iter
         fprintf('\nIteration %d:\n', iter);
    Z4 = W * H4;
    hsic_values = zeros(1,3);
    label_errors = zeros(1,3);
    total_rel_error = 0;      
    total_abs_error = 0;      
    modal_corr = zeros(1,4);  
    for m = 1:3
        Zm = W * H_cell{m};
        hsic_values(m) = HSIC(Zm, Z4, beta);
        AHmB = A * H_cell{m} * B{m};
        label_errors(m) = norm(Q - AHmB, 'fro');
    end
    laplacian_val = trace(H4 * L4 * H4');

    fprintf('  HSIC(W*H{m}, W*H4): %.4f %.4f %.4f\n', hsic_values);
    fprintf('  Label alignment error ‖Q - AHB‖: %.4f %.4f %.4f\n', label_errors);
    fprintf('  Laplacian term trace(H4*L*H4^T): %.4f\n', laplacian_val);

    for m = 1:4
        X_orig = modalities{m};
        Hm = {H1, H2, H3, H4};
        X_recon = W * Hm{m};

        rel_error = norm(X_orig - X_recon, 'fro') / norm(X_orig, 'fro');
        abs_error = norm(X_orig - X_recon, 'fro');
        total_rel_error = total_rel_error + rel_error;
        total_abs_error = total_abs_error + abs_error;

        corr_values = arrayfun(@(col) corr(X_orig(:,col), X_recon(:,col)), 1:size(X_orig,2));
        corr_values(isnan(corr_values)) = 0;
        modal_corr(m) = median(corr_values);
    end

    fprintf('  Total Relative Error: %.4f\n', total_rel_error);
    fprintf('  Total Absolute Error: %.4f\n', total_abs_error);
    fprintf('  Median Correlation per Modality:\n');
    for m = 1:4
        fprintf('    Modality %d: %.4f\n', m, modal_corr(m));
    end
    fprintf('----------------------------------\n');

    hsic_history(output_idx, :) = hsic_values;
    label_error_history(output_idx, :) = label_errors;
    laplacian_history(output_idx) = laplacian_val;
    corr_median(output_idx, :) = modal_corr;
    output_idx = output_idx + 1;
end

    if isnan(obj) || isinf(obj)
        warning('Objective became NaN/Inf. Terminating early.');
        break;
    end

    % if iter > 1 && abs(obj_history(iter) - obj_history(iter-1)) < 1e-5
    %     break;
    % end
end

end
% ----- Helper Function -----
function [grad_neg, grad_pos] = compute_HSIC_grad(Z1, Z2, beta)
    n = size(Z1, 1);
    H = eye(n) - ones(n)/n;

    K1 = rbf_kernel(Z1, beta);
    K2 = rbf_kernel(Z2, beta);
    sigma = max(beta * sqrt(2 * var(Z1(:))), 1e-8);

    HK2H = H * K2 * H;
    G = (K1 .* HK2H) * Z1 - sum(K1 .* HK2H, 2) .* Z1;
    grad_Z1 = -2 / (n^2 * sigma^2) * G;

    norm_grad = norm(grad_Z1, 'fro');
    if norm_grad > 10
        grad_Z1 = grad_Z1 * (10 / norm_grad);
    end

    grad_neg = min(grad_Z1, 0);
    grad_pos = max(grad_Z1, 0);
end

function K = rbf_kernel(Z, beta)
    n = size(Z,1);
    sum_Z = sum(Z.^2, 2);
    sq_dist = sum_Z + sum_Z' - 2 * Z * Z';
    sigma = max(beta * sqrt(mean(sq_dist(:))), 1e-8);
    K = exp(-sq_dist / (2 * sigma^2));
end

function h = HSIC(Z1, Z2, beta)
    if any(isnan(Z1(:))) || any(isinf(Z1(:))) || any(isnan(Z2(:))) || any(isinf(Z2(:)))
        h = NaN;
        return;
    end
    K1 = rbf_kernel(Z1, beta);
    K2 = rbf_kernel(Z2, beta);
    n = size(Z1,1);
    H = eye(n) - ones(n)/n;
    h = trace(K1 * H * K2 * H) / n^2;
end
