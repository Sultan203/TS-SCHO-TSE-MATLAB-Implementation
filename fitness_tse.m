function [fit, sol, metrics] = fitness_tse(vec, X, y, opts, hpBounds)
%FITNESS_TSE Decode candidate, evaluate by stratified CV, and compute fitness.

sol = decode_solution(vec, size(X,2), opts.learners, hpBounds);

% Avoid empty feature subset
if ~any(sol.featureMask)
    sol.featureMask(randi(size(X,2))) = true;
end

% Avoid empty ensemble
if ~any(sol.activeLearners)
    sol.activeLearners(randi(numel(opts.learners))) = true;
end

% Normalize active learner weights
w = sol.weights;
w(~sol.activeLearners) = 0;
if sum(w) <= 0
    w(sol.activeLearners) = 1 / sum(sol.activeLearners);
else
    w = w / sum(w);
end
sol.weights = w;

metrics = stratifiedKFoldEval(X(:, sol.featureMask), y, sol, opts);

featureRatio = sum(sol.featureMask) / size(X,2);
activeRatio = sum(sol.activeLearners) / numel(opts.learners);

% approximate normalized computational cost
costTerm = 0.5 * featureRatio + 0.5 * activeRatio;

% Minimize: lower is better
lossAcc = 1 - metrics.Accuracy;
lossF1 = 1 - metrics.F1;

fit = opts.wAcc*lossAcc + opts.wF1*lossF1 + opts.wFeat*featureRatio + opts.wCost*costTerm;
end
