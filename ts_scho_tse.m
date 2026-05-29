function results = ts_scho_tse(X, y, opts)
%TS_SCHO_TSE Main optimizer for adaptive feature selection and weighted ensemble.
%
% Candidate vector contains:
% [feature scores D] + [learner activation L] + [learner weights L] + hyperparameters
%
% Hyperparameter vector:
% RF trees, KNN k, SVM C, SVM gamma, MLP hidden neurons, NB smoothing

rng(opts.randomSeed);

[Nsamples, D] = size(X);
learners = opts.learners;
L = numel(learners);
H = 6;
dim = D + L + L + H;

lb = zeros(1, dim);
ub = ones(1, dim);

% Hyperparameter bounds stored separately and decoded from [0,1]
hpBounds = struct();
hpBounds.rfTrees = [10, 200];
hpBounds.knnK = [1, 15];
hpBounds.svmC = [0.1, 100];
hpBounds.svmGamma = [0.001, 1];
hpBounds.mlpHidden = [10, 100];
hpBounds.nbSmooth = [1e-9, 1];

% Initialize population
pop = rand(opts.populationSize, dim);
fitness = inf(opts.populationSize,1);
solutions = cell(opts.populationSize,1);
metricsList = cell(opts.populationSize,1);

fprintf("Initial fitness evaluation...\n");
for i = 1:opts.populationSize
    [fitness(i), solutions{i}, metricsList{i}] = fitness_tse(pop(i,:), X, y, opts, hpBounds);
end

[bestFit, idx] = min(fitness);
bestVec = pop(idx,:);
bestSol = solutions{idx};
bestMetrics = metricsList{idx};

convergence = zeros(opts.maxIterations,1);
Ts = round(opts.stageSwitchRatio * opts.maxIterations);

fprintf("Starting TS-SCHO-TSE optimization...\n");

for t = 1:opts.maxIterations
    a = 2 * (1 - t/opts.maxIterations);      % shrinking search coefficient
    rScale = 1 - t/opts.maxIterations;       % local radius

    if t <= Ts
        %% Stage I: Sinh-Cosh Opposition-Based Exploration
        for i = 1:opts.populationSize
            x = pop(i,:);

            % Opposition solution in [0,1]
            xOpp = lb + ub - x;

            % Hyperbolic warping toward global best
            r = -1 + 2*rand(1, dim);
            warped = xOpp + a .* sinh(r) .* (bestVec - xOpp);

            % Boundary correction
            warped = min(max(warped, lb), ub);

            [newFit, newSol, newMetrics] = fitness_tse(warped, X, y, opts, hpBounds);

            if newFit < fitness(i)
                pop(i,:) = warped;
                fitness(i) = newFit;
                solutions{i} = newSol;
                metricsList{i} = newMetrics;
            end
        end

    else
        %% Stage II: Clustered Sinh-Cosh Memetic Refinement
        eliteCount = max(2, round(opts.eliteRatio * opts.populationSize));
        [~, order] = sort(fitness);
        eliteIdx = order(1:eliteCount);
        elites = pop(eliteIdx,:);

        K = min(max(2, round(sqrt(eliteCount))), eliteCount);

        try
            clusterID = kmeans(elites, K, 'Replicates', 3, 'MaxIter', 50, 'Display','off');
        catch
            clusterID = ones(eliteCount,1);
            K = 1;
        end

        for k = 1:K
            members = find(clusterID == k);
            if isempty(members), continue; end

            memberGlobalIdx = eliteIdx(members);
            [~, localBestPos] = min(fitness(memberGlobalIdx));
            center = pop(memberGlobalIdx(localBestPos),:);

            % Local hyperbolic perturbation
            perturb = rScale .* sinh(-1 + 2*rand(1,dim)) .* (rand(1,dim)-0.5);
            refined = center + perturb;
            refined = min(max(refined, lb), ub);

            [newFit, newSol, newMetrics] = fitness_tse(refined, X, y, opts, hpBounds);

            % Replace worst individual if improvement
            [worstFit, worstIdx] = max(fitness);
            if newFit < worstFit
                pop(worstIdx,:) = refined;
                fitness(worstIdx) = newFit;
                solutions{worstIdx} = newSol;
                metricsList{worstIdx} = newMetrics;
            end
        end
    end

    [iterBestFit, idx] = min(fitness);
    if iterBestFit < bestFit
        bestFit = iterBestFit;
        bestVec = pop(idx,:);
        bestSol = solutions{idx};
        bestMetrics = metricsList{idx};
    end

    convergence(t) = bestFit;
    fprintf("Iter %3d/%3d | Best Fitness = %.6f | Acc = %.4f | F1 = %.4f | Features = %d\n", ...
        t, opts.maxIterations, bestFit, bestMetrics.Accuracy, bestMetrics.F1, sum(bestSol.featureMask));
end

results.bestFitness = bestFit;
results.bestVector = bestVec;
results.bestSolution = bestSol;
results.bestMetrics = bestMetrics;
results.convergence = convergence;
results.options = opts;
end
