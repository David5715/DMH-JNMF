clear; clc;

%% Load data
X1 = readmatrix('full_MRi_recon.csv');   % 196 x 166
X2 = readmatrix('full_FDG_recon.csv');   % 196 x 166
X3 =readmatrix('full_AV45_recon.csv');   % 196 x 166
X4 = readmatrix('full_gene_recon.csv');  % 196 x 166
 X1 = normalize(X1, 'minmax');
 X2 = normalize(X2, 'minmax');
 X3 = normalize(X3, 'minmax');
 X4 = normalize(X4, 'minmax');
roi_names = readcell('ROI_names.xlsx');   % 
roi_names = roi_names(:,1);
fid = fopen('ROI_names.csv');
roi_names = textscan(fid, '%s', 'Delimiter', '\n');
roi_names = roi_names{1};
fclose(fid);
gene_names = readcell('Gene_names.csv'); % 
gene_names = gene_names(:,1);

labels = readmatrix('labels.csv');    % 196 x 1 (类别)

[n, ~] = size(X1);
d = [size(X1,2), size(X2,2), size(X3,2), size(X4,2)];

%% Constructing the label matrix Q (K x n)
K = 8;  % Number of hidden factors
Q = create_label_matrix(labels, K);   % (12 x 196)

%% a Laplace matrix (Gene)
[D4, S4] = get_connectivity(X4, 2);  % 
D4_inv = diag(1 ./ sqrt(diag(D4) + 1e-8));
L4 = eye(size(D4)) - D4_inv * S4 * D4_inv;
%% Initialisation A, B
A = eye(K);
B = cell(1,3);
for m = 1:3
    B{m} = max(0, rand(d(m), n));
end

%% SVD Initialisatial W, H
X_all = [X1, X2, X3, X4];             % 196 x 950
[U, S, V] = svd(X_all, 'econ');
W = max(0, U(:,1:K) * sqrt(S(1:K,1:K)));      % 196 x 12
H_all = sqrt(S(1:K,1:K)) * max(V(:,1:K)', 0); % 12 x 950

cum_d = [0, cumsum(d)];
H1 = H_all(:, cum_d(1)+1 : cum_d(2));
H2 = H_all(:, cum_d(2)+1 : cum_d(3));
H3 = H_all(:, cum_d(3)+1 : cum_d(4));
H4 = H_all(:, cum_d(4)+1 : cum_d(5));

%% Parameterisation
lambda1 = 1;
lambda2 = 0;
lambda3 = 0;
alpha = 0.0001;
beta = 0.1;
beta_param = 0.0001;
max_iter = 1000;

%% Perform joint non-negative matrix factorisation
[W, H1, H2, H3, H4] = JNMF_comodule(X1, X2, X3, X4, L4, ...
    H1, H2, H3, H4, W, Q, A, B, ...
    lambda1, lambda2, lambda3, alpha, beta, beta_param, K, max_iter);

% %% Submodule
% tt = [1.5, 1.5, 1.5, 1.5];   %
% Co_module = Comodule_selection(W, {H1, H2, H3, H4}, tt);
% %% 
% K = size(Co_module, 1);
% max_len = 1000;  % 
% MRI_names_cell  = cell(max_len, K);
% FDG_names_cell  = cell(max_len, K);
% AV45_names_cell = cell(max_len, K);
% Gene_names_cell = cell(max_len, K);
% 
% for i = 1:K
%     idx1 = Co_module{i,1};  % MRI (H1)
%     idx2 = Co_module{i,2};  % FDG (H2)
%     idx3 = Co_module{i,3};  % AV45 (H3)
%     idx4 = Co_module{i,4};  % Gene (H4)
% 
%     MRI_names_cell(1:length(idx1), i)  = roi_names(idx1);
%     FDG_names_cell(1:length(idx2), i)  = roi_names(idx2);
%     AV45_names_cell(1:length(idx3), i) = roi_names(idx3);
%     Gene_names_cell(1:length(idx4), i) = gene_names(idx4);
% end
% 
% xlswrite('Module_MRI_Names_tt1.5.xlsx', MRI_names_cell);
% xlswrite('Module_FDG_Names_tt1.5.xlsx', FDG_names_cell);
% xlswrite('Module_AV45_Names_tt1.5.xlsx', AV45_names_cell);
% xlswrite('Module_Gene_Names_tt1.5.xlsx', Gene_names_cell);
%% 
% save_modules(Co_module, {'MRI', 'FDG', 'AV45', 'Gene'});
% %% ========= 3. 绘制重建相关性散点图 (已修改配色) =========
% % 此处代码使用已存在于工作区的变量 W, H1-H4, X1-X4
% 
% fprintf('Generating reconstruction correlation plots...\n');
% X_data = {X1, X2, X3, X4};
% H_data = {H1, H2, H3, H4};
% modality_names = {'MRI', 'FDG-PET', 'AV45-PET', 'Gene'};
% 
% % --- 1. 定义您的学术配色方案 ---
% % (基于 MATLAB R2019b+ 默认色板，非常专业)
% academic_colors = {
%     [0, 0.4470, 0.7410],    % 1. 蓝色 (用于 MRI)
%     [0.8500, 0.3250, 0.0980],   % 2. 橙红色 (用于 FDG-PET)
%     [0.4660, 0.6740, 0.1880],   % 3. 绿色 (用于 AV45-PET)
%     [0.4940, 0.1840, 0.5560]    % 4. 紫色 (用于 Gene)
% };
% % 为 1:1 对角线和注释框定义中性灰色
% line_gray = [0.5 0.5 0.5];       % 中灰色
% border_gray = [0.8 0.8 0.8];     % 浅灰色 (或者用 'none' 来移除边框)
% % -----------------------------------
% 
% for i = 1:4
%     X = X_data{i};
%     H = H_data{i};
%     X_hat = W * H;
% 
%     x = X(:);
%     y = X_hat(:);
% 
%     valid_idx = ~(isnan(x) | isnan(y));
%     x = x(valid_idx);
%     y = y(valid_idx);
% 
%     r = corr(x, y);
% 
%     figure('Position', [200+i*50, 200-i*30, 600, 500], 'Color', 'w');
% 
%     % --- 2. 应用配色方案 ---
%     % a) 为散点图应用第 i 种模态颜色
%     scatter(x, y, 100, academic_colors{i}, 'filled', 'MarkerFaceAlpha', 0.5);
% 
%     hold on;
%     min_val = min([x; y]);
%     max_val = max([x; y]);
% 
%     % b) 为 1:1 对角线应用中灰色
%     plot([min_val, max_val], [min_val, max_val], '--', 'Color', line_gray, 'LineWidth', 1.5);
%     hold off;
% 
%     % 坐标轴标签 + 加粗放大
%     xlabel(sprintf('X%d (Original Input)', i), 'FontSize', 24, 'FontWeight', 'bold');
%     ylabel(sprintf('W*H%d (Reconstructed)', i), 'FontSize', 24, 'FontWeight', 'bold');
% 
%     % 标题
%     title(modality_names{i}, 'FontSize', 32, 'FontWeight', 'bold');
% 
%     annotation_str = sprintf('Corr=%.4f', r); 
% 
%     % Corr 注释框 (调整了位置，使其在左上角)
%     ax = gca;
%     ax_pos = get(ax, 'Position'); % [left bottom width height]
%     x_rel = ax_pos(1) + 0.05 * ax_pos(3);  % 靠左
%     y_rel = ax_pos(2) + 0.85 * ax_pos(4);  % 靠上
% 
%     % c) 为注释框应用灰色边框 (或 'none')
%     annotation('textbox', [x_rel, y_rel, 0.15, 0.06], ...
%         'String', annotation_str, ...
%         'FontSize', 24, 'FontWeight', 'bold', ...
%         'BackgroundColor', 'w', 'EdgeColor', border_gray, ... % <-- 修改于此
%         'Margin', 4, 'Interpreter', 'none');
% 
%     % 只保留左+下边框
%     set(gca, 'Box', 'off');
%     set(gca, 'TickDir', 'out');
%     set(gca, 'XAxisLocation', 'bottom');
%     set(gca, 'YAxisLocation', 'left');
% 
%     % 坐标刻度字体放大
%     set(gca, 'FontSize', 24);
%     axis tight;
% 
%     % (推荐) 保存图像
%     saveas(gcf, sprintf('Correlation_Plot_%s.png', modality_names{i}));
% end
%% --- Helper Function ---
function Q = create_label_matrix(labels, K)
    num_classes = 4;
    K_per_class = floor(K / num_classes);
    Q = zeros(K, length(labels));
    for i = 1:length(labels)
        class_id = labels(i);
        start_idx = (class_id-1)*K_per_class + 1;
        Q(start_idx:start_idx+K_per_class-1, i) = 1;
    end
end

function save_modules(Co_module, modal_names)
    for m = 1:4
        max_feat = max(cellfun(@length, Co_module(:,m)));
        B = NaN(size(Co_module,1), max_feat);
        for i = 1:size(Co_module,1)
            feat = cell2mat(Co_module(i,m));
            B(i,1:length(feat)) = feat;
        end
        writematrix(B, sprintf('Co_tt1.5_module_%s.xlsx', modal_names{m}));
    end
end
