function sol = decode_solution(vec, D, learners, hpBounds)
%DECODE_SOLUTION Decode normalized vector into feature mask, learners, weights, hyperparameters.

L = numel(learners);
idx = 1;

featureScores = vec(idx:idx+D-1); idx = idx + D;
featureMask = featureScores >= 0.5;

actScores = vec(idx:idx+L-1); idx = idx + L;
activeLearners = actScores >= 0.5;

weightScores = vec(idx:idx+L-1); idx = idx + L;
weights = max(weightScores, 0);
if sum(weights) > 0
    weights = weights / sum(weights);
else
    weights = ones(1,L)/L;
end

hpRaw = vec(idx:idx+5);

hp.rfTrees = round(scale01(hpRaw(1), hpBounds.rfTrees));
hp.knnK = round(scale01(hpRaw(2), hpBounds.knnK));
hp.svmC = scale01(hpRaw(3), hpBounds.svmC);
hp.svmGamma = scale01(hpRaw(4), hpBounds.svmGamma);
hp.mlpHidden = round(scale01(hpRaw(5), hpBounds.mlpHidden));
hp.nbSmooth = scale01(hpRaw(6), hpBounds.nbSmooth);

sol.featureMask = featureMask;
sol.activeLearners = activeLearners;
sol.weights = weights;
sol.hp = hp;
sol.learners = learners;
end

function val = scale01(z, bounds)
val = bounds(1) + z * (bounds(2)-bounds(1));
end
