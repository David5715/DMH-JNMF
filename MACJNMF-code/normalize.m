function Y = normalize(X, type)


if nargin == 1
    type = 'std';
end

switch type
    case 'center'
        Y = X - mean(X);
    case 'minmax'
        Y = (X - min(X)) ./ (max(X) - min(X) + eps);
    case 'norm'
        X0 = X - mean(X);
        Y = X0 ./ (sqrt(sum(X0 .^ 2)) + eps);
    case 'std'
        Y = zscore(X);
    otherwise
        error('Error type.');
end
