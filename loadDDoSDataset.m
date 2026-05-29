function [X, y, featureNames] = loadDDoSDataset(dataPath, labelColumn)
%LOADDDOSDATASET Load CSV dataset and separate features/label.
% Numeric features are kept. Non-numeric feature columns are converted when possible.

T = readtable(dataPath, 'VariableNamingRule','preserve');

if strlength(labelColumn) == 0
    labelIdx = width(T);
else
    labelIdx = find(strcmp(string(T.Properties.VariableNames), string(labelColumn)), 1);
    if isempty(labelIdx)
        error("Label column '%s' not found.", labelColumn);
    end
end

y = T{:, labelIdx};
featureTable = T;
featureTable(:, labelIdx) = [];

featureNames = string(featureTable.Properties.VariableNames);

% Convert to numeric matrix
X = zeros(height(featureTable), width(featureTable));
for j = 1:width(featureTable)
    col = featureTable{:, j};
    if isnumeric(col) || islogical(col)
        X(:,j) = double(col);
    elseif iscellstr(col) || isstring(col) || iscategorical(col)
        X(:,j) = double(grp2idx(categorical(col)));
    else
        error("Unsupported feature type in column %s", featureNames(j));
    end
end

% Remove inf/nan rows
bad = any(~isfinite(X), 2) | isundefined(categorical(y));
X(bad,:) = [];
y(bad,:) = [];

% Remove constant features
stdVals = std(X, 0, 1);
keep = stdVals > 0;
X = X(:, keep);
featureNames = featureNames(keep);
end
