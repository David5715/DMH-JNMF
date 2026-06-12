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
