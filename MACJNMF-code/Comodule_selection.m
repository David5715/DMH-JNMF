function Co_module = Comodule_selection(W, H, tt)
    % 输入:
    % W: n×K
    % H: cell数组 {H1,H2,H3,H4}
    % tt: 1×4 数组, 每个模态的Z-score阈值
    % 输出:
    % Co_module: K×4 cell数组

    K = size(W,2);
    Co_module = cell(K,4);

    for k = 1:K
        for m = 1:4
            scores = H{m}(k,:);
            mu = mean(scores);
            sigma = std(scores);
            thr = mu + tt(m)*sigma;
            Co_module{k,m} = find(scores > thr);
        end
    end
end
