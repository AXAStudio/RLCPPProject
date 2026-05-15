#pragma once

#include "raylib.h"
#include "raymath.h"
#include "Box.h"
#include "Player.h"
#include <algorithm>
#include <array>
#include <cmath>
#include <random>
#include <string>
#include <vector>

struct RLDiscreteAction {
    const char* name;
    float angleDegrees;
    float strength;
    float pitchDegrees;
    bool dash;
    bool jump;
    bool slide;
};

struct RLReplayFrame {
    Vector3 position;
    Camera3D camera;
    float time;
    int action = -1;
};

struct RLExperience {
    Player::RLState state = {};
    Player::RLState nextState = {};
    int action = 0;
    float reward = 0.0f;
    float priority = 1.0f;
    bool done = false;
};

struct RLPolicySnapshot {
    bool valid = false;
    int rawFeatureCount = 0;
    int featureCount = 0;
    float epsilon = 0.0f;
    std::vector<std::vector<float>> weights;
    std::vector<float> actionBias;
};

struct RLEpisodeSummary {
    int episode = 0;
    float time = 0.0f;
    float reward = 0.0f;
    float distanceToGoal = 0.0f;
    float collisionCost = 0.0f;
    int wallHits = 0;
    int decisions = 0;
    bool success = false;
    bool timeout = false;
    bool stuck = false;
    std::vector<int> actionCounts;
};

inline const std::vector<RLDiscreteAction>& RLActions() {
    static const std::vector<RLDiscreteAction> actions = {
        { "GOAL",        0.0f,  1.0f,   0.0f, false, false, false },
        { "TIGHT L",   -18.0f,  1.0f,   0.0f, false, false, false },
        { "TIGHT R",    18.0f,  1.0f,   0.0f, false, false, false },
        { "ARC L",     -35.0f,  1.0f,   0.0f, false, false, false },
        { "ARC R",      35.0f,  1.0f,   0.0f, false, false, false },
        { "WIDE L",    -75.0f,  1.0f,   0.0f, false, false, false },
        { "WIDE R",     75.0f,  1.0f,   0.0f, false, false, false },
        { "CUT L",    -115.0f,  0.85f,  0.0f, false, false, false },
        { "CUT R",     115.0f,  0.85f,  0.0f, false, false, false },
        { "DASH",        0.0f,  1.0f,   0.0f, true,  false, false },
        { "DASH UP",     0.0f,  1.0f, -22.0f, true,  false, false },
        { "DASH DOWN",   0.0f,  1.0f,  18.0f, true,  false, false },
        { "DASH UP L", -35.0f,  1.0f, -22.0f, true,  false, false },
        { "DASH UP R",  35.0f,  1.0f, -22.0f, true,  false, false },
        { "DASH LOW L",-35.0f,  1.0f,  16.0f, true,  false, false },
        { "DASH LOW R", 35.0f,  1.0f,  16.0f, true,  false, false },
        { "DASH T L",  -18.0f,  1.0f,   0.0f, true,  false, false },
        { "DASH T R",   18.0f,  1.0f,   0.0f, true,  false, false },
        { "DASH L",    -35.0f,  1.0f,   0.0f, true,  false, false },
        { "DASH R",     35.0f,  1.0f,   0.0f, true,  false, false },
        { "DASH W L",  -75.0f,  1.0f, -16.0f, true,  false, false },
        { "DASH W R",   75.0f,  1.0f, -16.0f, true,  false, false },
        { "JUMP",        0.0f,  1.0f, -12.0f, false, true,  false },
        { "JUMP L",    -35.0f,  1.0f, -10.0f, false, true,  false },
        { "JUMP R",     35.0f,  1.0f, -10.0f, false, true,  false },
        { "JUMP W L",  -75.0f,  1.0f, -10.0f, false, true,  false },
        { "JUMP W R",   75.0f,  1.0f, -10.0f, false, true,  false },
        { "JUMP DASH",   0.0f,  1.0f, -18.0f, true,  true,  false },
        { "JUMP DASH L",-35.0f,  1.0f, -18.0f, true,  true,  false },
        { "JUMP DASH R", 35.0f,  1.0f, -18.0f, true,  true,  false },
        { "SLIDE",       0.0f,  1.0f,   8.0f, false, false, true  }
    };
    return actions;
}

inline float RLWrapDegrees(float angle) {
    while (angle > 180.0f) angle -= 360.0f;
    while (angle < -180.0f) angle += 360.0f;
    return angle;
}

inline float RLYawFromDirection(const Vector3& dir) {
    return atan2f(dir.x, dir.z) * RAD2DEG;
}

inline int RLActionIndexByName(const char* name) {
    const std::vector<RLDiscreteAction>& actions = RLActions();
    for (int i = 0; i < (int)actions.size(); ++i) {
        if (std::string(actions[i].name) == name) return i;
    }
    return -1;
}

inline void RLAddActionBias(std::vector<float>& actionBias, const char* name, float amount) {
    int index = RLActionIndexByName(name);
    if (index >= 0 && index < (int)actionBias.size()) actionBias[index] += amount;
}

struct RLLinearQModel {
    static constexpr int ENCODED_FEATURE_CAPACITY = Player::RL_STATE_SIZE + 24 + 10;
    using FeatureArray = std::array<float, ENCODED_FEATURE_CAPACITY>;

    std::string name;
    Color color = WHITE;
    float learningRate = 0.06f;
    float discount = 0.96f;
    float epsilon = 0.22f;
    float minEpsilon = 0.03f;
    float epsilonDecay = 0.9993f;
    float turnRate = 360.0f;
    float clearanceGuidance = 0.18f;
    float lastTD = 0.0f;
    int rawFeatureCount = 0;
    int featureCount = 0;
    int memoryCursor = 0;
    int eliteMemoryCursor = 0;
    int memoryCapacity = 24000;
    int eliteMemoryCapacity = 4000;
    int replayBatchSize = 1;
    std::vector<std::vector<float>> weights;
    std::vector<float> actionBias;
    std::vector<float> explorationWeight;
    std::vector<RLExperience> memory;
    std::vector<RLExperience> eliteMemory;
    std::mt19937 rng;

    RLLinearQModel() : rng(1) {}

    RLLinearQModel(
        const std::string& modelName,
        Color modelColor,
        unsigned int seed,
        float alpha,
        float gamma,
        float startEpsilon,
        float floorEpsilon,
        float decay
    ) :
        name(modelName),
        color(modelColor),
        learningRate(alpha),
        discount(gamma),
        epsilon(startEpsilon),
        minEpsilon(floorEpsilon),
        epsilonDecay(decay),
        rng(seed)
    {
        actionBias.assign(RLActions().size(), 0.0f);
        explorationWeight.assign(RLActions().size(), 1.0f);
    }

    void EnsureFeatureCount(int count) {
        EnsureActionTuning();
        if (rawFeatureCount == count && !weights.empty()) return;

        rawFeatureCount = count;
        featureCount = EncodedFeatureCount(count);
        memory.clear();
        eliteMemory.clear();
        memoryCursor = 0;
        eliteMemoryCursor = 0;
        if (memory.capacity() < (size_t)memoryCapacity) memory.reserve(memoryCapacity);
        if (eliteMemory.capacity() < (size_t)eliteMemoryCapacity) eliteMemory.reserve(eliteMemoryCapacity);
        std::uniform_real_distribution<float> init(-0.025f, 0.025f);
        weights.assign(RLActions().size(), std::vector<float>(featureCount + 1, 0.0f));

        for (auto& actionWeights : weights) {
            for (float& weight : actionWeights) {
                weight = init(rng);
            }
        }

        EnsureActionTuning();
    }

    void EnsureActionTuning() {
        int actionCount = (int)RLActions().size();
        if ((int)actionBias.size() != actionCount) {
            actionBias.assign(actionCount, 0.0f);
        }
        if ((int)explorationWeight.size() != actionCount) {
            explorationWeight.assign(actionCount, 1.0f);
        }
    }

    void DiscourageAction(const char* name, float scoreBias, float explorationMultiplier) {
        EnsureActionTuning();
        int index = RLActionIndexByName(name);
        if (index < 0 || index >= (int)RLActions().size()) return;

        actionBias[index] += scoreBias;
        explorationWeight[index] = fminf(
            explorationWeight[index],
            Clamp(explorationMultiplier, 0.05f, 1.0f)
        );
    }

    void ProtectActionFromLowUsagePenalty(int action) {
        EnsureActionTuning();
        if (action < 0 || action >= (int)RLActions().size()) return;

        if (actionBias[action] < -0.02f) actionBias[action] = -0.02f;
        if (explorationWeight[action] < 0.75f) explorationWeight[action] = 0.75f;
    }

    RLPolicySnapshot MakePolicySnapshot() const {
        RLPolicySnapshot snapshot;
        snapshot.valid = !weights.empty();
        snapshot.rawFeatureCount = rawFeatureCount;
        snapshot.featureCount = featureCount;
        snapshot.epsilon = epsilon;
        snapshot.weights = weights;
        snapshot.actionBias = actionBias;
        return snapshot;
    }

    bool CanUsePolicySnapshot(const RLPolicySnapshot& snapshot) const {
        if (!snapshot.valid ||
            snapshot.rawFeatureCount != rawFeatureCount ||
            snapshot.featureCount != featureCount ||
            snapshot.weights.size() != weights.size() ||
            snapshot.actionBias.size() != actionBias.size()) {
            return false;
        }

        for (int i = 0; i < (int)weights.size(); ++i) {
            if (snapshot.weights[i].size() != weights[i].size()) return false;
        }

        return true;
    }

    void BlendTowardPolicySnapshot(const RLPolicySnapshot& snapshot, float amount) {
        if (!CanUsePolicySnapshot(snapshot)) return;

        float t = Clamp(amount, 0.0f, 1.0f);
        for (int action = 0; action < (int)weights.size(); ++action) {
            for (int i = 0; i < (int)weights[action].size(); ++i) {
                weights[action][i] = Lerp(weights[action][i], snapshot.weights[action][i], t);
            }
            actionBias[action] = Lerp(actionBias[action], snapshot.actionBias[action], t);
        }
    }

    int EncodedFeatureCount(int count) const {
        return count + std::min(count, 24) + (count >= 25 ? 10 : 0);
    }

    FeatureArray EncodeState(const Player::RLState& state) const {
        FeatureArray features = {};
        int write = 0;

        for (float value : state) {
            features[write++] = Clamp(value, -2.5f, 2.5f);
        }

        int squaredCount = std::min((int)state.size(), 24);
        for (int i = 0; i < squaredCount; ++i) {
            float value = Clamp(state[i], -2.0f, 2.0f);
            features[write++] = value * value;
        }

        if (state.size() >= 25) {
            float completion = Clamp(state[3], 0.0f, 1.0f);
            float speed = Clamp(state[7], -1.5f, 1.5f);
            float timeUsed = Clamp(state[13], 0.0f, 1.5f);
            float timeLeft = Clamp(state[14], 0.0f, 1.0f);
            float areaPressure = Clamp(state[15], 0.0f, 2.0f);
            float noProgress = Clamp(state[16], 0.0f, 2.0f);
            float progress = Clamp(state[17], -1.0f, 1.0f);
            float forwardClear = Clamp(state[18], 0.0f, 1.0f);
            float minClear = Clamp(state[23], 0.0f, 1.0f);
            float sideDiff = Clamp(state[25], -1.0f, 1.0f);

            features[write++] = completion * timeLeft;
            features[write++] = completion * speed;
            features[write++] = progress * timeLeft;
            features[write++] = (1.0f - forwardClear) * (1.0f - completion);
            features[write++] = areaPressure * noProgress;
            features[write++] = timeUsed * (1.0f - completion);
            features[write++] = minClear * speed;
            features[write++] = sideDiff * (1.0f - forwardClear);
            features[write++] = noProgress * (1.0f - completion);
            features[write++] = areaPressure * timeUsed;
        }

        return features;
    }

    float ActionClearanceBias(const FeatureArray& features, int action) const {
        if (rawFeatureCount < 25 || action < 0 || action >= (int)RLActions().size()) return 0.0f;

        const RLDiscreteAction& candidate = RLActions()[action];
        float completion = Clamp(features[3], 0.0f, 1.0f);
        float forwardClear = features[18];
        float chosenClearance = ChosenClearanceForAction(features, action);
        float minClear = features[23];
        bool finalClimbDash = completion > 0.82f && candidate.dash && (candidate.pitchDegrees < -8.0f || candidate.jump);
        float lowChosenClearance = Clamp((0.56f - chosenClearance) / 0.56f, 0.0f, 1.0f);
        float lowForwardClearance = Clamp((0.40f - forwardClear) / 0.40f, 0.0f, 1.0f);

        float bias = (chosenClearance - 0.48f) * (clearanceGuidance * 1.15f);
        if (fabsf(candidate.angleDegrees) <= 20.0f && forwardClear < 0.36f && !finalClimbDash) {
            bias -= 0.14f + lowForwardClearance * 0.07f;
        }
        if (candidate.dash && chosenClearance < 0.52f && !finalClimbDash) {
            bias -= 0.12f + lowChosenClearance * 0.18f;
        }
        if (candidate.dash && minClear < 0.38f && !finalClimbDash) bias -= 0.08f;
        if (candidate.dash && candidate.jump && chosenClearance < 0.60f && !finalClimbDash) {
            bias -= 0.10f + lowChosenClearance * 0.08f;
        }
        if (candidate.jump && forwardClear < 0.34f && chosenClearance < 0.52f) {
            bias -= 0.06f;
        }
        if (finalClimbDash && forwardClear < 0.46f) {
            bias += 0.10f + Clamp((0.46f - forwardClear) / 0.46f, 0.0f, 1.0f) * 0.08f;
        }
        if (candidate.dash && chosenClearance > 0.76f && minClear > 0.50f) bias += 0.025f;
        if (candidate.jump && forwardClear > 0.58f && minClear > 0.45f) bias += 0.015f;
        return bias;
    }

    float ChosenClearanceForAction(const FeatureArray& features, int action) const {
        if (rawFeatureCount < 25 || action < 0 || action >= (int)RLActions().size()) return 1.0f;

        const RLDiscreteAction& candidate = RLActions()[action];
        float forwardClear = features[18];
        float leftClear = features[19];
        float rightClear = features[20];
        float hardLeftClear = features[21];
        float hardRightClear = features[22];

        if (candidate.angleDegrees <= -55.0f) return hardLeftClear;
        if (candidate.angleDegrees < -10.0f) return leftClear;
        if (candidate.angleDegrees >= 55.0f) return hardRightClear;
        if (candidate.angleDegrees > 10.0f) return rightClear;
        return forwardClear;
    }

    bool ActionAllowedByMask(const FeatureArray& features, int action) const {
        if (rawFeatureCount < 25 || action < 0 || action >= (int)RLActions().size()) return true;

        const RLDiscreteAction& candidate = RLActions()[action];
        float completion = Clamp(features[3], 0.0f, 1.0f);
        float dashCooldown = Clamp(features[8], 0.0f, 1.0f);
        float stamina = Clamp(features[9], 0.0f, 1.0f);
        float onGround = features[10];
        float wallRunning = features[11];
        float alreadyDashing = features[12];
        float forwardClear = features[18];
        float chosenClearance = ChosenClearanceForAction(features, action);
        float minClear = features[23];
        bool finalClimbDash = completion > 0.82f && candidate.dash && (candidate.pitchDegrees < -8.0f || candidate.jump);

        if (candidate.dash) {
            if (dashCooldown > 0.08f || alreadyDashing > 0.5f) return false;
            if (chosenClearance < 0.22f && !finalClimbDash) return false;
            if (minClear < 0.16f && completion < 0.96f && !finalClimbDash) return false;
            if (fabsf(candidate.angleDegrees) <= 20.0f && forwardClear < 0.24f && !finalClimbDash) return false;
        }

        if (candidate.jump && onGround < 0.5f && wallRunning < 0.5f) {
            return false;
        }

        if (candidate.slide && (onGround < 0.5f || stamina < 0.08f)) {
            return false;
        }

        if (!candidate.dash && fabsf(candidate.angleDegrees) >= 55.0f && chosenClearance < 0.07f && forwardClear > 0.20f) {
            return false;
        }

        return true;
    }

    float ScoreFeatures(const FeatureArray& features, int action) const {
        const std::vector<float>& w = weights[action];
        float baseBias = (action >= 0 && action < (int)actionBias.size()) ? actionBias[action] : 0.0f;
        float q = w[0] + baseBias + ActionClearanceBias(features, action);
        for (int i = 0; i < featureCount && i < (int)features.size(); ++i) {
            q += w[i + 1] * features[i];
        }
        return q;
    }

    float Score(const Player::RLState& state, int action) const {
        return ScoreFeatures(EncodeState(state), action);
    }

    int BestActionFromFeatures(const FeatureArray& features) {
        int best = -1;
        float bestScore = 0.0f;
        std::uniform_real_distribution<float> tieJitter(-0.0005f, 0.0005f);

        for (int i = 0; i < (int)RLActions().size(); ++i) {
            if (!ActionAllowedByMask(features, i)) continue;
            float q = ScoreFeatures(features, i) + tieJitter(rng);
            if (best < 0 || q > bestScore) {
                bestScore = q;
                best = i;
            }
        }

        if (best >= 0) return best;

        best = 0;
        bestScore = ScoreFeatures(features, 0);
        for (int i = 1; i < (int)RLActions().size(); ++i) {
            float q = ScoreFeatures(features, i) + tieJitter(rng);
            if (q > bestScore) {
                bestScore = q;
                best = i;
            }
        }

        return best;
    }

    int BestAction(const Player::RLState& state) {
        EnsureFeatureCount((int)state.size());
        return BestActionFromFeatures(EncodeState(state));
    }

    float ExplorationWeightForAction(const FeatureArray& features, int action) const {
        if (action < 0 || action >= (int)RLActions().size()) return 0.0f;
        if (!ActionAllowedByMask(features, action)) return 0.0f;

        float weight = (action < (int)explorationWeight.size()) ? explorationWeight[action] : 1.0f;
        if (rawFeatureCount >= 25) {
            const RLDiscreteAction& candidate = RLActions()[action];
            float completion = Clamp(features[3], 0.0f, 1.0f);
            float forwardClear = features[18];
            float chosenClearance = ChosenClearanceForAction(features, action);
            float minClear = features[23];
            bool finalClimbDash = completion > 0.82f && candidate.dash && (candidate.pitchDegrees < -8.0f || candidate.jump);

            if (candidate.dash && chosenClearance < 0.60f && !finalClimbDash) {
                weight *= Lerp(0.05f, 0.80f, Clamp(chosenClearance / 0.60f, 0.0f, 1.0f));
            }
            if (candidate.dash && candidate.jump && chosenClearance < 0.58f && !finalClimbDash) {
                weight *= Lerp(0.25f, 0.90f, Clamp(chosenClearance / 0.58f, 0.0f, 1.0f));
            }
            if (candidate.dash && minClear < 0.45f && !finalClimbDash) {
                weight *= Lerp(0.25f, 0.90f, Clamp(minClear / 0.45f, 0.0f, 1.0f));
            }
            if (finalClimbDash) {
                weight *= 1.18f;
            }
            if (candidate.jump && forwardClear < 0.34f && chosenClearance < 0.50f) {
                weight *= 0.45f;
            }
            if (fabsf(candidate.angleDegrees) >= 55.0f && chosenClearance < 0.30f) {
                weight *= Lerp(0.20f, 0.70f, Clamp(chosenClearance / 0.30f, 0.0f, 1.0f));
            }
        }

        return Clamp(weight, 0.02f, 1.25f);
    }

    int ChooseExplorationAction(const FeatureArray& features) {
        float totalWeight = 0.0f;
        for (int i = 0; i < (int)RLActions().size(); ++i) {
            totalWeight += ExplorationWeightForAction(features, i);
        }

        if (totalWeight <= 0.0f) return BestActionFromFeatures(features);

        std::uniform_real_distribution<float> pick(0.0f, totalWeight);
        float cursor = pick(rng);
        for (int i = 0; i < (int)RLActions().size(); ++i) {
            cursor -= ExplorationWeightForAction(features, i);
            if (cursor <= 0.0f) return i;
        }

        return (int)RLActions().size() - 1;
    }

    int ChooseAction(const Player::RLState& state) {
        EnsureFeatureCount((int)state.size());
        FeatureArray features = EncodeState(state);

        std::uniform_real_distribution<float> unit(0.0f, 1.0f);
        if (unit(rng) < epsilon) {
            return ChooseExplorationAction(features);
        }

        return BestActionFromFeatures(features);
    }

    float Learn(const Player::RLState& state, int action, float reward, const Player::RLState& nextState, bool done, float learningScale = 1.0f) {
        if (action < 0 || action >= (int)RLActions().size()) return 0.0f;

        EnsureFeatureCount((int)state.size());
        FeatureArray features = EncodeState(state);
        float nextValue = 0.0f;
        if (!done) {
            FeatureArray nextFeatures = EncodeState(nextState);
            nextValue = ScoreFeatures(nextFeatures, BestActionFromFeatures(nextFeatures));
        }
        float target = reward + discount * nextValue;
        float td = Clamp(target - ScoreFeatures(features, action), -55.0f, 55.0f);
        lastTD = td;

        std::vector<float>& w = weights[action];
        float alpha = learningRate * learningScale;
        w[0] += alpha * td;
        for (int i = 0; i < featureCount && i < (int)features.size(); ++i) {
            w[i + 1] += alpha * td * features[i];
        }

        if (learningScale >= 0.99f) {
            epsilon = fmaxf(minEpsilon, epsilon * epsilonDecay);
        }

        return td;
    }

    float ImitateAction(const Player::RLState& state, int action, float margin = 0.55f, float learningScale = 0.18f) {
        if (action < 0 || action >= (int)RLActions().size()) return 0.0f;

        EnsureFeatureCount((int)state.size());
        FeatureArray features = EncodeState(state);
        if (!ActionAllowedByMask(features, action)) return 0.0f;

        float chosenScore = ScoreFeatures(features, action);
        float totalCorrection = 0.0f;
        float alpha = learningRate * learningScale;
        std::vector<float>& chosenWeights = weights[action];

        for (int other = 0; other < (int)RLActions().size(); ++other) {
            if (other == action || !ActionAllowedByMask(features, other)) continue;

            float otherScore = ScoreFeatures(features, other);
            float correction = Clamp(otherScore + margin - chosenScore, 0.0f, 6.0f);
            if (correction <= 0.0f) continue;

            chosenWeights[0] += alpha * correction;
            for (int i = 0; i < featureCount && i < (int)features.size(); ++i) {
                chosenWeights[i + 1] += alpha * correction * features[i];
            }

            std::vector<float>& otherWeights = weights[other];
            otherWeights[0] -= alpha * correction * 0.22f;
            for (int i = 0; i < featureCount && i < (int)features.size(); ++i) {
                otherWeights[i + 1] -= alpha * correction * 0.22f * features[i];
            }

            totalCorrection += correction;
            chosenScore += correction * 0.20f;
        }

        return totalCorrection;
    }

    void RememberExperience(const RLExperience& experience) {
        if (memory.capacity() < (size_t)memoryCapacity) memory.reserve(memoryCapacity);

        if ((int)memory.size() < memoryCapacity) {
            memory.push_back(experience);
        } else {
            memory[memoryCursor] = experience;
            memoryCursor = (memoryCursor + 1) % memoryCapacity;
        }
    }

    void RememberEliteExperience(RLExperience experience) {
        if (eliteMemory.capacity() < (size_t)eliteMemoryCapacity) eliteMemory.reserve(eliteMemoryCapacity);
        experience.priority += 8.0f;

        if ((int)eliteMemory.size() < eliteMemoryCapacity) {
            eliteMemory.push_back(experience);
        } else {
            eliteMemory[eliteMemoryCursor] = experience;
            eliteMemoryCursor = (eliteMemoryCursor + 1) % eliteMemoryCapacity;
        }
    }

    RLExperience MakeExperience(const Player::RLState& state, int action, float reward, const Player::RLState& nextState, bool done) const {
        RLExperience experience;
        experience.state = state;
        experience.nextState = nextState;
        experience.action = action;
        experience.reward = reward;
        experience.done = done;
        experience.priority = 1.0f + fabsf(reward) * 0.08f + (done ? 3.0f : 0.0f);
        return experience;
    }

    void Remember(const Player::RLState& state, int action, float reward, const Player::RLState& nextState, bool done) {
        RememberExperience(MakeExperience(state, action, reward, nextState, done));
    }

    int PickReplayIndex(const std::vector<RLExperience>& source) {
        std::uniform_int_distribution<int> pick(0, (int)source.size() - 1);
        int best = pick(rng);
        float bestPriority = source[best].priority;

        for (int i = 0; i < 2; ++i) {
            int candidate = pick(rng);
            if (source[candidate].priority > bestPriority) {
                best = candidate;
                bestPriority = source[candidate].priority;
            }
        }

        return best;
    }

    void ReplayFrom(std::vector<RLExperience>& source, int batches, float learningScale) {
        if (source.empty()) return;

        int count = std::min(batches, (int)source.size());

        for (int i = 0; i < count; ++i) {
            int index = PickReplayIndex(source);
            const RLExperience& experience = source[index];
            float td = Learn(
                experience.state,
                experience.action,
                experience.reward,
                experience.nextState,
                experience.done,
                learningScale
            );
            source[index].priority = 0.985f * source[index].priority + 0.015f * (1.0f + fabsf(td));
        }
    }

    void ReplayEliteMemory(int batches) {
        ReplayFrom(eliteMemory, batches, 0.55f);
    }

    void ReplayMemory(int batches = -1) {
        int count = batches > 0 ? batches : replayBatchSize;
        ReplayFrom(memory, count, 0.35f);

        if (!eliteMemory.empty()) {
            ReplayEliteMemory(std::max(1, count / 3));
        }
    }

    int MemorySize() const {
        return (int)memory.size();
    }

    int EliteMemorySize() const {
        return (int)eliteMemory.size();
    }
};

struct RLRunner {
    Player player;
    RLLinearQModel model;
    int episode = 0;
    int successes = 0;
    int timeouts = 0;
    int stucks = 0;
    int episodesSinceBest = 0;
    int manualGuidanceReplays = 0;
    int manualGuidanceExperiences = 0;
    int bestRunRehearsals = 0;
    int currentAction = 0;
    float decisionTimer = 0.0f;
    float decisionInterval = 0.09f;
    float actionReward = 0.0f;
    float resetTimer = 0.0f;
    float resetDelay = 0.05f;
    float bestTime = 9999.0f;
    float bestReward = -9999.0f;
    float bestSuccessReward = -9999.0f;
    float lastEpisodeReward = 0.0f;
    float lastEpisodeTime = 0.0f;
    float lastDistance = 0.0f;
    float lastFrameRecordTime = 0.0f;
    float lastBestPathReward = 0.0f;
    float episodeBestPathReward = 0.0f;
    float totalBestPathReward = 0.0f;
    Player::RLState currentState = {};
    bool hasCurrentState = false;
    std::vector<Vector3> trail;
    std::vector<Vector3> episodeTrail;
    std::vector<Vector3> bestTrail;
    std::vector<RLReplayFrame> episodeFrames;
    std::vector<RLReplayFrame> bestFrames;
    std::vector<RLExperience> episodeExperiences;
    std::vector<RLExperience> bestExperiences;
    std::vector<RLExperience> manualGuidanceBest;
    std::vector<int> episodeActionCounts;
    std::vector<int> lifetimeActionCounts;
    std::vector<RLEpisodeSummary> recentEpisodes;
    RLPolicySnapshot bestPolicy;
    float manualGuidanceBestTime = 9999.0f;
    int manualGuidanceRehearsals = 0;

    RLRunner() = default;

    RLRunner(const RLLinearQModel& qModel, float interval) :
        model(qModel),
        decisionInterval(interval)
    {}

    void EnsureActionTelemetry() {
        int actionCount = (int)RLActions().size();
        if ((int)episodeActionCounts.size() != actionCount) episodeActionCounts.assign(actionCount, 0);
        if ((int)lifetimeActionCounts.size() != actionCount) lifetimeActionCounts.assign(actionCount, 0);
    }

    void CountAction(int action) {
        EnsureActionTelemetry();
        if (action < 0 || action >= (int)RLActions().size()) return;
        episodeActionCounts[action] += 1;
        lifetimeActionCounts[action] += 1;
    }

    int EpisodeDecisionCount() const {
        int total = 0;
        for (int count : episodeActionCounts) total += count;
        return total;
    }

    void SaveEpisodeSummary() {
        RLEpisodeSummary summary;
        summary.episode = episode;
        summary.time = lastEpisodeTime;
        summary.reward = lastEpisodeReward;
        summary.distanceToGoal = lastDistance;
        summary.collisionCost = player.rlCollisionPenaltyTotal;
        summary.wallHits = player.rlWallHits;
        summary.decisions = EpisodeDecisionCount();
        summary.success = player.rlSucceeded;
        summary.timeout = player.rlTimedOut;
        summary.stuck = player.rlStuck;
        summary.actionCounts = episodeActionCounts;

        recentEpisodes.push_back(summary);
        if (recentEpisodes.size() > 160) {
            recentEpisodes.erase(recentEpisodes.begin());
        }
    }

    void ProtectBestReplayActionsFromPruning() {
        std::vector<bool> used(RLActions().size(), false);
        for (const RLReplayFrame& frame : bestFrames) {
            if (frame.action >= 0 && frame.action < (int)used.size()) {
                used[frame.action] = true;
            }
        }

        for (int action = 0; action < (int)used.size(); ++action) {
            if (used[action]) model.ProtectActionFromLowUsagePenalty(action);
        }
    }

    void RehearseBestRun(int samples, float tdScale, float imitationScale) {
        if (bestExperiences.empty() || samples <= 0) return;

        int count = std::min(samples, (int)bestExperiences.size());
        for (int i = 0; i < count; ++i) {
            int index = ((bestRunRehearsals + i) * 7) % (int)bestExperiences.size();
            const RLExperience& source = bestExperiences[index];
            if (source.action < 0 || source.action >= (int)RLActions().size()) continue;

            float phase = (index + 1.0f) / fmaxf(1.0f, (float)bestExperiences.size());
            float anchoredReward = source.reward + 2.5f + 10.0f * phase + (source.done ? 80.0f : 0.0f);
            model.Learn(source.state, source.action, anchoredReward, source.nextState, source.done, tdScale);
            model.ImitateAction(source.state, source.action, 0.50f + 0.20f * phase, imitationScale);
            model.ProtectActionFromLowUsagePenalty(source.action);
        }

        bestRunRehearsals += count;
    }

    void RehearseManualGuidance(int samples, float tdScale, float imitationScale) {
        if (manualGuidanceBest.empty() || samples <= 0) return;

        int count = std::min(samples, (int)manualGuidanceBest.size());
        for (int i = 0; i < count; ++i) {
            int index = ((manualGuidanceRehearsals + i) * 5) % (int)manualGuidanceBest.size();
            const RLExperience& source = manualGuidanceBest[index];
            if (source.action < 0 || source.action >= (int)RLActions().size()) continue;

            float phase = (index + 1.0f) / fmaxf(1.0f, (float)manualGuidanceBest.size());
            float timeEdge = Clamp((bestTime - manualGuidanceBestTime) / 0.50f, 0.0f, 1.0f);
            float guidedReward = source.reward + 3.0f + 8.0f * phase + 3.0f * timeEdge + (source.done ? 60.0f : 0.0f);
            model.Learn(source.state, source.action, guidedReward, source.nextState, source.done, tdScale);
            model.ImitateAction(source.state, source.action, 0.46f + 0.12f * phase + 0.05f * timeEdge, imitationScale);
            model.ProtectActionFromLowUsagePenalty(source.action);
        }

        manualGuidanceRehearsals += count;
    }

    int ActionFromExperienceRunNearState(const std::vector<RLExperience>& run, const Player::RLState& state, float& confidence) {
        confidence = 0.0f;
        if (run.empty()) return -1;

        model.EnsureFeatureCount((int)state.size());
        RLLinearQModel::FeatureArray features = model.EncodeState(state);

        int bestIndex = -1;
        float bestMatch = 9999.0f;
        for (int i = 0; i < (int)run.size(); ++i) {
            const RLExperience& experience = run[i];
            if (experience.action < 0 || experience.action >= (int)RLActions().size()) continue;
            if (!model.ActionAllowedByMask(features, experience.action)) continue;

            float completionDiff = fabsf(state[3] - experience.state[3]);
            float directionDiff = fabsf(state[0] - experience.state[0]) + fabsf(state[1] - experience.state[1]);
            float distanceDiff = fabsf(state[2] - experience.state[2]);
            float clearanceDiff =
                fabsf(state[18] - experience.state[18]) * 0.35f +
                fabsf(state[23] - experience.state[23]) * 0.25f;
            float match = completionDiff + directionDiff * 0.055f + distanceDiff * 0.18f + clearanceDiff;

            if (match < bestMatch) {
                bestMatch = match;
                bestIndex = i;
            }
        }

        if (bestIndex < 0 || bestMatch > 0.13f) return -1;

        confidence = Clamp((0.13f - bestMatch) / 0.13f, 0.0f, 1.0f);
        return run[bestIndex].action;
    }

    int BestRunActionNearState(const Player::RLState& state, float& confidence) {
        return ActionFromExperienceRunNearState(bestExperiences, state, confidence);
    }

    int ChooseAnchoredAction(const Player::RLState& state) {
        if (!manualGuidanceBest.empty() && manualGuidanceBestTime < bestTime - 0.004f) {
            float manualConfidence = 0.0f;
            int manualAction = ActionFromExperienceRunNearState(manualGuidanceBest, state, manualConfidence);
            if (manualAction >= 0 && manualConfidence >= 0.68f) {
                float completion = Clamp(state[3], 0.0f, 1.0f);
                float manualChance = completion > 0.82f ? 0.16f : 0.09f;
                if (episodesSinceBest > 80) manualChance += 0.03f;
                if (episodesSinceBest > 240) manualChance += 0.03f;
                manualChance = Clamp(manualChance * manualConfidence, 0.0f, 0.22f);

                std::uniform_real_distribution<float> unit(0.0f, 1.0f);
                if (unit(model.rng) < manualChance) return manualAction;
            }
        }

        float confidence = 0.0f;
        int bestAction = BestRunActionNearState(state, confidence);
        if (bestAction >= 0) {
            float completion = Clamp(state[3], 0.0f, 1.0f);
            float anchorChance = completion > 0.82f ? 0.66f : 0.38f;
            if (episodesSinceBest > 80) anchorChance += 0.14f;
            if (episodesSinceBest > 240) anchorChance += 0.10f;
            anchorChance = Clamp(anchorChance * confidence, 0.0f, 0.82f);

            std::uniform_real_distribution<float> unit(0.0f, 1.0f);
            if (unit(model.rng) < anchorChance) return bestAction;
        }

        return model.ChooseAction(state);
    }

    float BestPathGuidanceReward(float dt, const Vector3& goal) {
        lastBestPathReward = 0.0f;
        if (!HasBestRun() || bestFrames.size() < 2 || player.rlDone) return 0.0f;

        float currentDistance = RLHorizontalDistance(player.position, goal);
        int bestIndex = 0;
        float bestDistanceDiff = fabsf(RLHorizontalDistance(bestFrames[0].position, goal) - currentDistance);

        for (int i = 1; i < (int)bestFrames.size(); ++i) {
            float distanceDiff = fabsf(RLHorizontalDistance(bestFrames[i].position, goal) - currentDistance);
            if (distanceDiff < bestDistanceDiff) {
                bestDistanceDiff = distanceDiff;
                bestIndex = i;
            }
        }

        const RLReplayFrame& anchor = bestFrames[bestIndex];
        float dx = player.position.x - anchor.position.x;
        float dz = player.position.z - anchor.position.z;
        float dy = player.position.y - anchor.position.y;
        float routeError = sqrtf(dx * dx + dz * dz + dy * dy * 0.35f);
        float completion = 1.0f - Clamp(currentDistance / fmaxf(1.0f, player.rlStartTargetDistance), 0.0f, 1.0f);
        float finalApproach = Clamp((completion - 0.78f) / 0.20f, 0.0f, 1.0f);
        float closeness = Clamp((7.0f - routeError) / 7.0f, 0.0f, 1.0f);
        float offRoute = Clamp((routeError - 4.0f) / 12.0f, 0.0f, 1.0f);
        float timeDelta = player.rlEpisodeTime - anchor.time;

        float reward =
            closeness * (1.00f + completion * 0.40f + finalApproach * 0.55f) * dt -
            offRoute * (1.10f + completion * 0.45f) * dt;

        if (timeDelta > 0.30f) {
            reward -= Clamp((timeDelta - 0.30f) / 2.25f, 0.0f, 1.0f) * 0.70f * dt;
        } else {
            reward += Clamp((-timeDelta) / 1.15f, 0.0f, 1.0f) * 0.22f * dt;
        }

        if (routeError < 2.5f && player.rlLastProgress >= -0.01f) {
            reward += (0.22f + finalApproach * 0.18f) * dt;
        }
        if (routeError > 10.0f && completion > 0.10f) {
            reward -= 0.55f * dt;
        }

        reward = Clamp(reward, -0.045f, 0.040f);
        lastBestPathReward = reward;
        episodeBestPathReward += reward;
        totalBestPathReward += reward;
        return reward;
    }

    void StartEpisode(const Vector3& start, const Vector3& goal, const std::vector<Box>& obstacles) {
        player.ResetRL(start, goal);
        currentState = player.GetRLState(obstacles);
        hasCurrentState = true;
        if (!bestExperiences.empty()) {
            model.minEpsilon = fminf(model.minEpsilon, 0.006f);
            model.epsilon = fmaxf(0.004f, fminf(model.epsilon, 0.018f));
            if ((episode % 12) == 0) RehearseBestRun(4, 0.24f, 0.06f);
        }
        if (!manualGuidanceBest.empty()) {
            bool manualIsFaster = manualGuidanceBestTime < bestTime - 0.004f;
            int interval = manualIsFaster ? 18 : 40;
            if ((episode % interval) == 0) {
                RehearseManualGuidance(manualIsFaster ? 2 : 1, manualIsFaster ? 0.18f : 0.12f, manualIsFaster ? 0.035f : 0.02f);
            }
        }
        currentAction = ChooseAnchoredAction(currentState);
        EnsureActionTelemetry();
        std::fill(episodeActionCounts.begin(), episodeActionCounts.end(), 0);
        CountAction(currentAction);
        decisionTimer = decisionInterval;
        actionReward = 0.0f;
        resetTimer = 0.0f;
        lastDistance = RLHorizontalDistance(player.position, goal);
        lastBestPathReward = 0.0f;
        episodeBestPathReward = 0.0f;
        if (trail.capacity() < 160) trail.reserve(160);
        if (episodeTrail.capacity() < 360) episodeTrail.reserve(360);
        if (episodeFrames.capacity() < 520) episodeFrames.reserve(520);
        if (episodeExperiences.capacity() < 260) episodeExperiences.reserve(260);
        trail.clear();
        episodeTrail.clear();
        episodeFrames.clear();
        episodeExperiences.clear();
        trail.push_back(player.position);
        episodeTrail.push_back(player.position);
        episodeFrames.push_back({ player.position, player.camera, 0.0f, currentAction });
        lastFrameRecordTime = 0.0f;
        episode += 1;
    }

    RLControl BuildControl(float dt) const {
        const RLDiscreteAction& action = RLActions()[currentAction];
        Vector3 goalDir = RLFlatDirection(player.position, player.rlTarget);
        Vector3 moveDir = RLRotateFlat(goalDir, action.angleDegrees);

        RLControl control;
        control.moveInput = Vector3Scale(moveDir, action.strength);
        control.dash = action.dash;
        control.jump = action.jump;
        control.slide = action.slide;
        control.aimPitch = action.pitchDegrees;

        Vector3 lookDir = (Vector3Length(control.moveInput) > 0.1f) ? control.moveInput : goalDir;
        float desiredYaw = RLYawFromDirection(lookDir);
        float yawError = RLWrapDegrees(player.yaw - desiredYaw);
        float maxTurn = model.turnRate * dt;
        control.yawDelta = Clamp(yawError, -maxTurn, maxTurn);
        float pitchError = player.pitch - action.pitchDegrees;
        control.pitchDelta = Clamp(pitchError, -maxTurn * 0.35f, maxTurn * 0.35f);

        return control;
    }

    void Update(float dt, const Vector3& start, const Vector3& goal, const std::vector<Box>& obstacles) {
        if (resetTimer > 0.0f) {
            resetTimer -= dt;
            if (resetTimer <= 0.0f) {
                StartEpisode(start, goal, obstacles);
            }
            return;
        }

        if (!hasCurrentState) {
            StartEpisode(start, goal, obstacles);
        }

        decisionTimer -= dt;
        int previousAction = currentAction;

        RLControl control = BuildControl(dt);
        player.Update(dt, obstacles, false, false, 0.0f, true, control);
        float bestPathReward = BestPathGuidanceReward(dt, goal);
        if (bestPathReward != 0.0f) {
            player.rlReward += bestPathReward;
            player.rlEpisodeReward += bestPathReward;
        }
        actionReward += player.rlReward;
        if (episodeFrames.empty() || player.rlDone || player.rlEpisodeTime - lastFrameRecordTime >= 1.0f / 30.0f) {
            episodeFrames.push_back({ player.position, player.camera, player.rlEpisodeTime, currentAction });
            lastFrameRecordTime = player.rlEpisodeTime;
        }
        lastDistance = RLHorizontalDistance(player.position, goal);

        if (player.rlDone || decisionTimer <= 0.0f) {
            Player::RLState nextState = player.GetRLState(obstacles);
            model.Learn(currentState, previousAction, actionReward, nextState, player.rlDone);
            RLExperience experience = model.MakeExperience(currentState, previousAction, actionReward, nextState, player.rlDone);
            model.RememberExperience(experience);
            episodeExperiences.push_back(experience);
            if (player.rlDone) {
                model.ReplayMemory(4);
            }

            currentState = nextState;
            actionReward = 0.0f;

            if (!player.rlDone) {
                currentAction = ChooseAnchoredAction(currentState);
                CountAction(currentAction);
                decisionTimer += decisionInterval;
                if (decisionTimer <= 0.0f) decisionTimer = decisionInterval;
            }
        }

        if (trail.empty() || RLHorizontalDistance(trail.back(), player.position) > 1.0f) {
            trail.push_back(player.position);
            if (trail.size() > 140) trail.erase(trail.begin());
        }

        if (episodeTrail.empty() || RLHorizontalDistance(episodeTrail.back(), player.position) > 0.75f) {
            episodeTrail.push_back(player.position);
            if (episodeTrail.size() > 320) episodeTrail.erase(episodeTrail.begin());
        }

        if (player.rlDone) {
            lastEpisodeReward = player.rlEpisodeReward;
            lastEpisodeTime = player.rlEpisodeTime;
            bestReward = fmaxf(bestReward, lastEpisodeReward);
            bool improvedBest = false;

            if (player.rlSucceeded) {
                successes += 1;
                bool isBestSuccess =
                    bestTrail.empty() ||
                    player.rlEpisodeTime < bestTime ||
                    (fabsf(player.rlEpisodeTime - bestTime) < 0.001f && lastEpisodeReward > bestSuccessReward);

                if (isBestSuccess) {
                    bestTime = player.rlEpisodeTime;
                    bestSuccessReward = lastEpisodeReward;
                    bestTrail = episodeTrail;
                    bestFrames = episodeFrames;
                    bestExperiences = episodeExperiences;
                    ProtectBestReplayActionsFromPruning();
                    improvedBest = true;
                }

                bool cleanFastSuccess =
                    lastEpisodeTime <= 6.35f &&
                    player.rlWallHits <= 10 &&
                    player.rlCollisionPenaltyTotal <= 20.0f;
                float speedPolish = Clamp((6.65f - lastEpisodeTime) / 1.15f, 0.0f, 1.0f);
                for (int i = 0; i < (int)episodeExperiences.size(); ++i) {
                    RLExperience elite = episodeExperiences[i];
                    float phase = (i + 1.0f) / fmaxf(1.0f, (float)episodeExperiences.size());
                    elite.reward += 4.0f + 14.0f * phase + speedPolish * (3.0f + 10.0f * phase);
                    elite.priority += 18.0f + 28.0f * phase + speedPolish * (10.0f + 18.0f * phase);
                    model.RememberExperience(elite);
                    model.RememberEliteExperience(elite);
                }

                model.ReplayMemory(improvedBest ? 14 : 8);
                model.ReplayEliteMemory(improvedBest ? 22 : 12);
                if (cleanFastSuccess) {
                    int polishCount = std::min((int)episodeExperiences.size(), 80);
                    for (int i = 0; i < polishCount; ++i) {
                        const RLExperience& experience = episodeExperiences[i];
                        float phase = (i + 1.0f) / fmaxf(1.0f, (float)polishCount);
                        model.ImitateAction(experience.state, experience.action, 0.64f + 0.22f * phase, 0.12f);
                        model.ProtectActionFromLowUsagePenalty(experience.action);
                    }
                    model.ReplayEliteMemory(18);
                }
                if (improvedBest) {
                    RehearseBestRun(std::min(80, std::max(20, (int)bestExperiences.size())), 0.55f, 0.16f);
                    bestPolicy = model.MakePolicySnapshot();
                    model.BlendTowardPolicySnapshot(bestPolicy, 0.24f);
                    model.minEpsilon = fminf(model.minEpsilon, 0.006f);
                    model.epsilon = fmaxf(0.004f, fminf(model.epsilon, 0.014f));
                } else if (!bestExperiences.empty()) {
                    RehearseBestRun(6, 0.30f, 0.07f);
                }
                model.epsilon = fmaxf(model.minEpsilon, model.epsilon * 0.975f);
            } else if (player.rlTimedOut) {
                timeouts += 1;
            } else if (player.rlStuck) {
                stucks += 1;
            }

            if (improvedBest) {
                episodesSinceBest = 0;
            } else {
                episodesSinceBest += 1;
                if (!bestExperiences.empty() && (episodesSinceBest % 4) == 0) {
                    RehearseBestRun(player.rlSucceeded ? 4 : 7, 0.28f, 0.06f);
                }
                if (!manualGuidanceBest.empty() && manualGuidanceBestTime < bestTime - 0.004f && (episodesSinceBest % 20) == 0) {
                    RehearseManualGuidance(player.rlSucceeded ? 2 : 3, 0.16f, 0.035f);
                }
                if (bestPolicy.valid && episodesSinceBest > 0 && episodesSinceBest % 120 == 0) {
                    model.BlendTowardPolicySnapshot(bestPolicy, 0.18f);
                    model.epsilon = fmaxf(0.004f, fminf(model.epsilon, 0.018f));
                } else if (bestPolicy.valid && episodesSinceBest > 0 && episodesSinceBest % 24 == 0) {
                    model.BlendTowardPolicySnapshot(bestPolicy, 0.05f);
                }
                if (episodesSinceBest >= 800) {
                    model.epsilon = fmaxf(0.004f, fminf(model.epsilon, 0.026f));
                    if (bestPolicy.valid) model.BlendTowardPolicySnapshot(bestPolicy, 0.28f);
                    RehearseBestRun(24, 0.40f, 0.10f);
                    if (!manualGuidanceBest.empty()) RehearseManualGuidance(6, 0.18f, 0.04f);
                    model.ReplayEliteMemory(18);
                    episodesSinceBest = 200;
                }
            }

            SaveEpisodeSummary();
            resetTimer = resetDelay;
        }
    }

    const char* CurrentActionName() const {
        return RLActions()[currentAction].name;
    }

    int FinishedEpisodes() const {
        return successes + timeouts + stucks;
    }

    float RecentSuccessRate(int maxEpisodes = 80) const {
        if (recentEpisodes.empty()) return 0.0f;

        int start = std::max(0, (int)recentEpisodes.size() - maxEpisodes);
        int count = 0;
        int recentSuccesses = 0;
        for (int i = start; i < (int)recentEpisodes.size(); ++i) {
            count += 1;
            recentSuccesses += recentEpisodes[i].success ? 1 : 0;
        }

        return count > 0 ? (float)recentSuccesses / (float)count : 0.0f;
    }

    float ReliableSelectionScore(int maxEpisodes = 80) const {
        int finished = FinishedEpisodes();
        float lifetimeSuccessRate = finished > 0 ? (float)successes / (float)finished : 0.0f;

        int start = std::max(0, (int)recentEpisodes.size() - maxEpisodes);
        int recentCount = 0;
        int recentSuccesses = 0;
        int recentTimeouts = 0;
        int recentStucks = 0;
        int recentWalls = 0;
        float recentDistance = 0.0f;
        float recentCollision = 0.0f;
        float recentReward = 0.0f;
        float recentSuccessTime = 0.0f;
        float bestRecentSuccessTime = 9999.0f;

        for (int i = start; i < (int)recentEpisodes.size(); ++i) {
            const RLEpisodeSummary& episodeSummary = recentEpisodes[i];
            recentCount += 1;
            recentSuccesses += episodeSummary.success ? 1 : 0;
            recentTimeouts += episodeSummary.timeout ? 1 : 0;
            recentStucks += episodeSummary.stuck ? 1 : 0;
            recentWalls += episodeSummary.wallHits;
            recentDistance += episodeSummary.distanceToGoal;
            recentCollision += episodeSummary.collisionCost;
            recentReward += episodeSummary.reward;

            if (episodeSummary.success) {
                recentSuccessTime += episodeSummary.time;
                bestRecentSuccessTime = fminf(bestRecentSuccessTime, episodeSummary.time);
            }
        }

        float sampleTrust = Clamp((float)recentCount / 40.0f, 0.20f, 1.0f);
        float recentSuccessRateValue = recentCount > 0 ? (float)recentSuccesses / (float)recentCount : 0.0f;
        float recentTimeoutRate = recentCount > 0 ? (float)recentTimeouts / (float)recentCount : 0.0f;
        float recentStuckRate = recentCount > 0 ? (float)recentStucks / (float)recentCount : 0.0f;
        float averageDistance = recentCount > 0 ? recentDistance / (float)recentCount : lastDistance;
        float averageWalls = recentCount > 0 ? (float)recentWalls / (float)recentCount : 0.0f;
        float averageCollision = recentCount > 0 ? recentCollision / (float)recentCount : 0.0f;

        float score = lifetimeSuccessRate * 140.0f + recentSuccessRateValue * 620.0f * sampleTrust;
        score += (float)successes * 0.80f;

        if (recentSuccesses > 0) {
            float averageSuccessTime = recentSuccessTime / (float)recentSuccesses;
            score += (RL_MAX_EPISODE_TIME - averageSuccessTime) * 18.0f * sampleTrust;
            score += (RL_MAX_EPISODE_TIME - bestRecentSuccessTime) * 7.0f;
        } else {
            score += Clamp(bestReward, -160.0f, 160.0f) * 0.18f;
            score -= averageDistance * 4.0f * sampleTrust;
            score += Clamp(recentReward / fmaxf(1.0f, (float)recentCount), -80.0f, 80.0f) * 0.20f;
        }

        if (HasBestRun()) {
            score += 60.0f + (RL_MAX_EPISODE_TIME - bestTime) * 9.0f;
        }

        score -= recentTimeoutRate * 110.0f * sampleTrust;
        score -= recentStuckRate * 145.0f * sampleTrust;
        score -= averageWalls * 3.20f * sampleTrust;
        score -= averageCollision * 1.70f * sampleTrust;

        if (recentCount >= 20 && recentSuccessRateValue < lifetimeSuccessRate * 0.45f && successes > 5) {
            score -= 90.0f;
        }
        if (HasBestRun()) {
            score -= Clamp((float)episodesSinceBest / 500.0f, 0.0f, 1.0f) * 55.0f;
        }
        if (model.eliteMemoryCapacity > 0) {
            score += Clamp((float)model.EliteMemorySize() / (float)model.eliteMemoryCapacity, 0.0f, 1.0f) * 14.0f;
        }

        return score;
    }

    void LearnFromEliteReplay(const std::vector<RLExperience>& replay, float finishTime, bool manualGuidance) {
        if (replay.empty()) return;

        if (manualGuidance && finishTime > 0.0f && finishTime < manualGuidanceBestTime) {
            manualGuidanceBestTime = finishTime;
            manualGuidanceBest = replay;
            manualGuidanceRehearsals = 0;
        }

        int learned = 0;
        float manualTimeEdge = manualGuidance ? Clamp((bestTime - finishTime) / 0.60f, 0.0f, 1.0f) : 0.0f;
        for (int i = 0; i < (int)replay.size(); ++i) {
            const RLExperience& source = replay[i];
            if (source.action < 0 || source.action >= (int)RLActions().size()) continue;

            model.EnsureFeatureCount((int)source.state.size());
            model.ProtectActionFromLowUsagePenalty(source.action);

            RLExperience elite = source;
            float phase = (i + 1.0f) / fmaxf(1.0f, (float)replay.size());
            if (manualGuidance) {
                elite.reward += 5.0f + 14.0f * phase + 5.0f * manualTimeEdge;
                elite.priority += 24.0f + 45.0f * phase + 14.0f * manualTimeEdge;
            } else {
                elite.reward += 6.0f + 24.0f * phase;
                elite.priority += 30.0f + 58.0f * phase;
            }

            if (elite.done) {
                float timeBonus = fmaxf(0.0f, RL_MAX_EPISODE_TIME - finishTime) * (manualGuidance ? 8.0f : 7.0f);
                elite.reward += (manualGuidance ? 110.0f : 120.0f) + timeBonus;
                elite.priority += manualGuidance ? 80.0f : 90.0f;
            }

            model.RememberExperience(elite);
            model.RememberEliteExperience(elite);
            if ((manualGuidance && ((i % 3) == 0 || elite.done)) || (!manualGuidance && ((i % 2) == 0 || elite.done))) {
                model.Learn(elite.state, elite.action, elite.reward, elite.nextState, elite.done, manualGuidance ? 0.42f : 0.58f);
            }
            if (manualGuidance) {
                model.ImitateAction(elite.state, elite.action, 0.50f + 0.12f * phase + 0.05f * manualTimeEdge, 0.08f);
            }
            learned += 1;
        }

        if (learned <= 0) return;

        model.ReplayEliteMemory(std::min(manualGuidance ? 70 : 90, std::max(18, learned / (manualGuidance ? 3 : 2))));
        model.ReplayMemory(std::min(manualGuidance ? 32 : 42, std::max(8, learned / (manualGuidance ? 6 : 4))));
        if (manualGuidance) {
            RehearseManualGuidance(std::min(32, std::max(8, learned / 12)), 0.18f, 0.04f);
        }
        model.epsilon = fmaxf(model.minEpsilon, fminf(model.epsilon, manualGuidance ? 0.08f : 0.10f));

        if (manualGuidance) {
            manualGuidanceReplays += 1;
            manualGuidanceExperiences += learned;
        }
    }

    bool HasBestRun() const {
        return !bestTrail.empty() && !bestFrames.empty();
    }

    float BestRunDuration() const {
        return HasBestRun() ? bestFrames.back().time : 0.0f;
    }

    RLReplayFrame BestRunFrame(float replayTime) const {
        if (!HasBestRun()) return { player.position, player.camera, 0.0f, -1 };
        if (bestFrames.size() == 1 || replayTime <= bestFrames.front().time) return bestFrames.front();
        if (replayTime >= bestFrames.back().time) return bestFrames.back();

        int nextIndex = 1;
        while (nextIndex < (int)bestFrames.size() && bestFrames[nextIndex].time < replayTime) {
            nextIndex += 1;
        }

        const RLReplayFrame& a = bestFrames[nextIndex - 1];
        const RLReplayFrame& b = bestFrames[nextIndex];
        float span = fmaxf(0.001f, b.time - a.time);
        float t = Clamp((replayTime - a.time) / span, 0.0f, 1.0f);

        RLReplayFrame frame = b;
        frame.position = Vector3Lerp(a.position, b.position, t);
        frame.camera.position = Vector3Lerp(a.camera.position, b.camera.position, t);
        frame.camera.target = Vector3Lerp(a.camera.target, b.camera.target, t);
        frame.camera.up = Vector3Normalize(Vector3Lerp(a.camera.up, b.camera.up, t));
        frame.camera.fovy = Lerp(a.camera.fovy, b.camera.fovy, t);
        frame.time = Lerp(a.time, b.time, t);
        frame.action = (t < 0.5f) ? a.action : b.action;
        return frame;
    }

    Camera3D BestRunCamera(float replayTime) const {
        return BestRunFrame(replayTime).camera;
    }
};

struct RLTrainer {
    Vector3 start = { 0, 5.0f, 52 };
    Vector3 goal = { 0, 4.85f, -52 };
    std::vector<RLRunner> runners;
    int selectedRunner = 0;

    RLTrainer() = default;

    RLTrainer(const Vector3& startPoint, const Vector3& goalPoint) :
        start(startPoint),
        goal(goalPoint)
    {
        RLLinearQModel balanced("Balanced", SKYBLUE, 11, 0.055f, 0.96f, 0.24f, 0.015f, 0.9992f);
        balanced.clearanceGuidance = 0.24f;
        balanced.DiscourageAction("WIDE L", -0.05f, 0.45f);
        balanced.DiscourageAction("WIDE R", -0.07f, 0.35f);
        balanced.DiscourageAction("CUT L", -0.07f, 0.35f);
        balanced.DiscourageAction("ARC L", -0.025f, 0.65f);
        balanced.DiscourageAction("ARC R", -0.025f, 0.65f);
        balanced.DiscourageAction("DASH UP L", -0.025f, 0.65f);
        balanced.DiscourageAction("DASH UP R", -0.025f, 0.65f);
        balanced.DiscourageAction("DASH T R", -0.06f, 0.35f);
        balanced.DiscourageAction("DASH R", -0.025f, 0.65f);
        balanced.DiscourageAction("DASH W L", -0.10f, 0.20f);
        balanced.DiscourageAction("DASH W R", -0.04f, 0.55f);

        RLLinearQModel explorer("Explorer", ORANGE, 23, 0.050f, 0.95f, 0.48f, 0.05f, 0.9995f);

        RLLinearQModel sprinter("Sprinter", LIME, 37, 0.065f, 0.95f, 0.30f, 0.012f, 0.9991f);
        RLAddActionBias(sprinter.actionBias, "DASH", 0.14f);
        RLAddActionBias(sprinter.actionBias, "DASH UP", 0.06f);
        RLAddActionBias(sprinter.actionBias, "DASH UP L", 0.055f);
        RLAddActionBias(sprinter.actionBias, "DASH UP R", 0.055f);
        RLAddActionBias(sprinter.actionBias, "JUMP DASH", 0.045f);
        RLAddActionBias(sprinter.actionBias, "DASH T L", 0.09f);
        RLAddActionBias(sprinter.actionBias, "DASH T R", 0.09f);
        RLAddActionBias(sprinter.actionBias, "DASH L", 0.08f);
        RLAddActionBias(sprinter.actionBias, "DASH R", 0.08f);
        RLAddActionBias(sprinter.actionBias, "DASH W L", 0.05f);
        RLAddActionBias(sprinter.actionBias, "DASH W R", 0.05f);

        RLLinearQModel router("Side Route", VIOLET, 51, 0.055f, 0.97f, 0.34f, 0.02f, 0.99925f);
        RLAddActionBias(router.actionBias, "ARC L", 0.07f);
        RLAddActionBias(router.actionBias, "ARC R", 0.07f);
        RLAddActionBias(router.actionBias, "WIDE L", 0.12f);
        RLAddActionBias(router.actionBias, "WIDE R", 0.12f);
        RLAddActionBias(router.actionBias, "CUT L", 0.08f);
        RLAddActionBias(router.actionBias, "CUT R", 0.08f);

        RLLinearQModel steady("Steady", GOLD, 71, 0.045f, 0.98f, 0.18f, 0.010f, 0.9990f);
        RLAddActionBias(steady.actionBias, "GOAL", 0.08f);
        RLAddActionBias(steady.actionBias, "DASH", -0.04f);
        RLAddActionBias(steady.actionBias, "DASH UP", -0.04f);
        RLAddActionBias(steady.actionBias, "DASH DOWN", -0.04f);
        RLAddActionBias(steady.actionBias, "DASH UP L", -0.04f);
        RLAddActionBias(steady.actionBias, "DASH UP R", -0.04f);
        RLAddActionBias(steady.actionBias, "DASH LOW L", -0.04f);
        RLAddActionBias(steady.actionBias, "DASH LOW R", -0.04f);
        RLAddActionBias(steady.actionBias, "DASH T L", -0.04f);
        RLAddActionBias(steady.actionBias, "DASH T R", -0.04f);
        RLAddActionBias(steady.actionBias, "DASH L", -0.04f);
        RLAddActionBias(steady.actionBias, "DASH R", -0.04f);
        RLAddActionBias(steady.actionBias, "DASH W L", -0.04f);
        RLAddActionBias(steady.actionBias, "DASH W R", -0.04f);
        RLAddActionBias(steady.actionBias, "JUMP DASH", -0.04f);
        RLAddActionBias(steady.actionBias, "JUMP DASH L", -0.04f);
        RLAddActionBias(steady.actionBias, "JUMP DASH R", -0.04f);

        runners.push_back(RLRunner(balanced, 0.12f));
        runners.push_back(RLRunner(explorer, 0.11f));
        runners.push_back(RLRunner(sprinter, 0.10f));
        runners.push_back(RLRunner(router, 0.12f));
        runners.push_back(RLRunner(steady, 0.13f));
        selectedRunner = 0;
    }

    void ResetEpisodes(const std::vector<Box>& obstacles) {
        for (auto& runner : runners) {
            runner.StartEpisode(start, goal, obstacles);
        }
    }

    void Update(float dt, const std::vector<Box>& obstacles, bool selectedOnly = false) {
        if (selectedOnly) {
            if (!runners.empty()) {
                runners[selectedRunner].Update(dt, start, goal, obstacles);
            }
            return;
        }

        for (auto& runner : runners) {
            runner.Update(dt, start, goal, obstacles);
        }
    }

    void LearnFromManualReplay(const std::vector<RLExperience>& replay, float finishTime) {
        if (replay.empty()) return;

        for (auto& runner : runners) {
            runner.LearnFromEliteReplay(replay, finishTime, true);
        }
    }

    void SelectNext() {
        if (runners.empty()) return;
        selectedRunner = (selectedRunner + 1) % (int)runners.size();
    }

    void SelectPrevious() {
        if (runners.empty()) return;
        selectedRunner = (selectedRunner - 1 + (int)runners.size()) % (int)runners.size();
    }

    int BestRunnerIndex() const {
        if (runners.empty()) return 0;

        int best = 0;
        float bestScore = runners[0].ReliableSelectionScore();
        for (int i = 1; i < (int)runners.size(); ++i) {
            const RLRunner& a = runners[i];
            const RLRunner& b = runners[best];
            float score = a.ReliableSelectionScore();
            bool tieBreak =
                fabsf(score - bestScore) <= 0.001f &&
                (a.successes > b.successes ||
                (a.successes == b.successes && a.successes > 0 && a.bestTime < b.bestTime) ||
                (a.successes == b.successes && a.successes == 0 && a.bestReward > b.bestReward));

            if (score > bestScore || tieBreak) {
                best = i;
                bestScore = score;
            }
        }

        return best;
    }

    RLRunner& ActiveRunner() {
        return runners[selectedRunner];
    }

    const RLRunner& ActiveRunner() const {
        return runners[selectedRunner];
    }

    void Draw3D() const {
        for (int i = 0; i < (int)runners.size(); ++i) {
            const RLRunner& runner = runners[i];
            Color runnerColor = runner.model.color;

            for (int t = 1; t < (int)runner.trail.size(); ++t) {
                DrawLine3D(runner.trail[t - 1], runner.trail[t], Fade(runnerColor, 0.45f));
            }

            Vector3 p = runner.player.position;
            DrawCube(p, 0.85f, PLAYER_HEIGHT, 0.85f, runnerColor);
            DrawCubeWires(p, 0.85f, PLAYER_HEIGHT, 0.85f, Fade(BLACK, 0.55f));

            Vector3 forward = {
                sinf(DEG2RAD * runner.player.yaw),
                0.0f,
                cosf(DEG2RAD * runner.player.yaw)
            };
            DrawLine3D(
                { p.x, p.y + 1.05f, p.z },
                { p.x + forward.x * 1.7f, p.y + 1.05f, p.z + forward.z * 1.7f },
                WHITE
            );

            if (i == selectedRunner) {
                DrawSphere({ p.x, p.y + 1.35f, p.z }, 0.32f, WHITE);
            }
        }
    }

    void DrawBestRuns3D() const {
        for (int i = 0; i < (int)runners.size(); ++i) {
            const RLRunner& runner = runners[i];
            if (!runner.HasBestRun()) continue;

            Color runnerColor = runner.model.color;

            for (int t = 1; t < (int)runner.bestTrail.size(); ++t) {
                DrawLine3D(runner.bestTrail[t - 1], runner.bestTrail[t], runnerColor);
            }

            for (int t = 0; t < (int)runner.bestTrail.size(); t += 10) {
                DrawSphere(runner.bestTrail[t], 0.22f, Fade(runnerColor, 0.85f));
            }

            Vector3 end = runner.bestTrail.back();
            DrawCube({ end.x, end.y + 0.45f, end.z }, 0.8f, 0.9f, 0.8f, runnerColor);
            DrawCubeWires({ end.x, end.y + 0.45f, end.z }, 0.8f, 0.9f, 0.8f, WHITE);

            if (i == selectedRunner) {
                DrawSphere({ end.x, end.y + 1.15f, end.z }, 0.35f, WHITE);
            }
        }
    }
};
