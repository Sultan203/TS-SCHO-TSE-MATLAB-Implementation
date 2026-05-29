function metrics = stratifiedKFoldEval(X, y, sol, opts)
%STRATIFIEDKFOLDEVAL Evaluate weighted ensemble using stratified K-fold CV.

cv = cvpartition(y, 'KFold', opts.numFolds, 'Stratify', true);

allTrue = [];
allPred = [];
allScores = [];

classes = categories(y);
if numel(classes) ~= 2
    warning("This implementation reports binary-style AUC/F1. Multiclass labels detected.");
end

for fold = 1:opts.numFolds
    tr = training(cv, fold);
    te = test(cv, fold);

    % Leakage prevention: normalization fitted on training only
    Xtr = X(tr,:);
    Xte = X(te,:);
    ytr = y(tr);
    yte = y(te);

    minVals = min(Xtr, [], 1);
    maxVals = max(Xtr, [], 1);
    denom = maxVals - minVals;
    denom(denom == 0) = 1;

    Xtr = (Xtr - minVals) ./ denom;
    Xte = (Xte - minVals) ./ denom;

    model = train_tse_ensemble(Xtr, ytr, sol);
    [pred, score] = predict_tse_ensemble(model, Xte, sol, classes);

    allTrue = [allTrue; yte];
    allPred = [allPred; pred];

    if ~isempty(score)
        allScores = [allScores; score];
    end
end

metrics = classificationMetrics(allTrue, allPred, allScores);
end
