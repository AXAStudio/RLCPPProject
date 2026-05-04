#pragma once

#include "raylib.h"
#include "raymath.h"
#include "Box.h"
#include "Player.h"
#include <algorithm>
#include <cmath>
#include <random>
#include <string>
#include <vector>

struct RLDiscreteAction {
    const char* name;
    float angleDegrees;
    float strength;
    bool dash;
    bool jump;
    bool slide;
};

struct RLReplayFrame {
    Vector3 position;
    Camera3D camera;
    float time;
};

struct RLExperience {
    std::vector<float> state;
    std::vector<float> nextState;
    int action = 0;
    float reward = 0.0f;
    float priority = 1.0f;
    bool done = false;
};

inline const std::vector<RLDiscreteAction>& RLActions() {
    static const std::vector<RLDiscreteAction> actions = {
        { "GOAL",        0.0f,  1.0f, false, false, false },
        { "TIGHT L",   -18.0f,  1.0f, false, false, false },
        { "TIGHT R",    18.0f,  1.0f, false, false, false },
        { "ARC L",     -35.0f,  1.0f, false, false, false },
        { "ARC R",      35.0f,  1.0f, false, false, false },
        { "WIDE L",    -75.0f,  1.0f, false, false, false },
        { "WIDE R",     75.0f,  1.0f, false, false, false },
        { "CUT L",    -115.0f,  0.85f, false, false, false },
        { "CUT R",     115.0f,  0.85f, false, false, false },
        { "DASH",        0.0f,  1.0f, true,  false, false },
        { "DASH T L",  -18.0f,  1.0f, true,  false, false },
        { "DASH T R",   18.0f,  1.0f, true,  false, false },
        { "DASH L",    -35.0f,  1.0f, true,  false, false },
        { "DASH R",     35.0f,  1.0f, true,  false, false },
        { "DASH W L",  -75.0f,  1.0f, true,  false, false },
        { "DASH W R",   75.0f,  1.0f, true,  false, false },
        { "JUMP",        0.0f,  1.0f, false, true,  false },
        { "JUMP L",    -35.0f,  1.0f, false, true,  false },
        { "JUMP R",     35.0f,  1.0f, false, true,  false },
        { "JUMP W L",  -75.0f,  1.0f, false, true,  false },
        { "JUMP W R",   75.0f,  1.0f, false, true,  false },
        { "SLIDE",       0.0f,  1.0f, false, false, true  }
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

struct RLLinearQModel {
    std::string name;
    Color color = WHITE;
    float learningRate = 0.06f;
    float discount = 0.96f;
    float epsilon = 0.22f;
    float minEpsilon = 0.03f;
    float epsilonDecay = 0.9993f;
    float turnRate = 360.0f;
    float lastTD = 0.0f;
    int rawFeatureCount = 0;
    int featureCount = 0;
    int memoryCursor = 0;
    int eliteMemoryCursor = 0;
    int memoryCapacity = 60000;
    int eliteMemoryCapacity = 6000;
    int replayBatchSize = 8;
    std::vector<std::vector<float>> weights;
    std::vector<float> actionBias;
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
    }

    void EnsureFeatureCount(int count) {
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

        if (actionBias.size() != RLActions().size()) {
            actionBias.assign(RLActions().size(), 0.0f);
        }
    }

    int EncodedFeatureCount(int count) const {
        return count + std::min(count, 24) + (count >= 25 ? 10 : 0);
    }

    std::vector<float> EncodeState(const std::vector<float>& state) const {
        std::vector<float> features;
        features.reserve(EncodedFeatureCount((int)state.size()));

        for (float value : state) {
            features.push_back(Clamp(value, -2.5f, 2.5f));
        }

        int squaredCount = std::min((int)state.size(), 24);
        for (int i = 0; i < squaredCount; ++i) {
            float value = Clamp(state[i], -2.0f, 2.0f);
            features.push_back(value * value);
        }

        if (state.size() >= 25) {
            float completion = Clamp(state[3], 0.0f, 1.0f);
            float speed = Clamp(state[6], -1.5f, 1.5f);
            float timeUsed = Clamp(state[12], 0.0f, 1.5f);
            float timeLeft = Clamp(state[13], 0.0f, 1.0f);
            float areaPressure = Clamp(state[14], 0.0f, 2.0f);
            float noProgress = Clamp(state[15], 0.0f, 2.0f);
            float progress = Clamp(state[16], -1.0f, 1.0f);
            float forwardClear = Clamp(state[17], 0.0f, 1.0f);
            float minClear = Clamp(state[22], 0.0f, 1.0f);
            float sideDiff = Clamp(state[24], -1.0f, 1.0f);

            features.push_back(completion * timeLeft);
            features.push_back(completion * speed);
            features.push_back(progress * timeLeft);
            features.push_back((1.0f - forwardClear) * (1.0f - completion));
            features.push_back(areaPressure * noProgress);
            features.push_back(timeUsed * (1.0f - completion));
            features.push_back(minClear * speed);
            features.push_back(sideDiff * (1.0f - forwardClear));
            features.push_back(noProgress * (1.0f - completion));
            features.push_back(areaPressure * timeUsed);
        }

        return features;
    }

    float ScoreFeatures(const std::vector<float>& features, int action) const {
        const std::vector<float>& w = weights[action];
        float q = w[0] + actionBias[action];
        for (int i = 0; i < featureCount && i < (int)features.size(); ++i) {
            q += w[i + 1] * features[i];
        }
        return q;
    }

    float Score(const std::vector<float>& state, int action) const {
        return ScoreFeatures(EncodeState(state), action);
    }

    int BestActionFromFeatures(const std::vector<float>& features) {
        int best = 0;
        float bestScore = ScoreFeatures(features, 0);
        std::uniform_real_distribution<float> tieJitter(-0.0005f, 0.0005f);

        for (int i = 1; i < (int)RLActions().size(); ++i) {
            float q = ScoreFeatures(features, i) + tieJitter(rng);
            if (q > bestScore) {
                bestScore = q;
                best = i;
            }
        }

        return best;
    }

    int BestAction(const std::vector<float>& state) {
        EnsureFeatureCount((int)state.size());
        return BestActionFromFeatures(EncodeState(state));
    }

    int ChooseAction(const std::vector<float>& state) {
        EnsureFeatureCount((int)state.size());

        std::uniform_real_distribution<float> unit(0.0f, 1.0f);
        if (unit(rng) < epsilon) {
            std::uniform_int_distribution<int> actionPick(0, (int)RLActions().size() - 1);
            return actionPick(rng);
        }

        return BestActionFromFeatures(EncodeState(state));
    }

    float Learn(const std::vector<float>& state, int action, float reward, const std::vector<float>& nextState, bool done, float learningScale = 1.0f) {
        if (action < 0 || action >= (int)RLActions().size()) return 0.0f;

        EnsureFeatureCount((int)state.size());
        std::vector<float> features = EncodeState(state);
        float nextValue = 0.0f;
        if (!done) {
            std::vector<float> nextFeatures = EncodeState(nextState);
            nextValue = ScoreFeatures(nextFeatures, BestActionFromFeatures(nextFeatures));
        }
        float target = reward + discount * nextValue;
        float td = Clamp(target - ScoreFeatures(features, action), -35.0f, 35.0f);
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

    RLExperience MakeExperience(const std::vector<float>& state, int action, float reward, const std::vector<float>& nextState, bool done) const {
        RLExperience experience;
        experience.state = state;
        experience.nextState = nextState;
        experience.action = action;
        experience.reward = reward;
        experience.done = done;
        experience.priority = 1.0f + fabsf(reward) * 0.08f + (done ? 3.0f : 0.0f);
        return experience;
    }

    void Remember(const std::vector<float>& state, int action, float reward, const std::vector<float>& nextState, bool done) {
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
            RLExperience experience = source[index];
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
    int currentAction = 0;
    float decisionTimer = 0.0f;
    float decisionInterval = 0.09f;
    float actionReward = 0.0f;
    float resetTimer = 0.0f;
    float resetDelay = 0.45f;
    float bestTime = 9999.0f;
    float bestReward = -9999.0f;
    float bestSuccessReward = -9999.0f;
    float lastEpisodeReward = 0.0f;
    float lastEpisodeTime = 0.0f;
    float lastDistance = 0.0f;
    std::vector<float> currentState;
    std::vector<Vector3> trail;
    std::vector<Vector3> episodeTrail;
    std::vector<Vector3> bestTrail;
    std::vector<RLReplayFrame> episodeFrames;
    std::vector<RLReplayFrame> bestFrames;
    std::vector<RLExperience> episodeExperiences;

    RLRunner() = default;

    RLRunner(const RLLinearQModel& qModel, float interval) :
        model(qModel),
        decisionInterval(interval)
    {}

    void StartEpisode(const Vector3& start, const Vector3& goal, const std::vector<Box>& obstacles) {
        player.ResetRL(start, goal);
        currentState = player.GetRLState(obstacles);
        currentAction = model.ChooseAction(currentState);
        decisionTimer = decisionInterval;
        actionReward = 0.0f;
        resetTimer = 0.0f;
        lastDistance = RLHorizontalDistance(player.position, goal);
        if (trail.capacity() < 160) trail.reserve(160);
        if (episodeTrail.capacity() < 360) episodeTrail.reserve(360);
        if (episodeFrames.capacity() < 1200) episodeFrames.reserve(1200);
        if (episodeExperiences.capacity() < 260) episodeExperiences.reserve(260);
        trail.clear();
        episodeTrail.clear();
        episodeFrames.clear();
        episodeExperiences.clear();
        trail.push_back(player.position);
        episodeTrail.push_back(player.position);
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

        Vector3 lookDir = (Vector3Length(control.moveInput) > 0.1f) ? control.moveInput : goalDir;
        float desiredYaw = RLYawFromDirection(lookDir);
        float yawError = RLWrapDegrees(player.yaw - desiredYaw);
        float maxTurn = model.turnRate * dt;
        control.yawDelta = Clamp(yawError, -maxTurn, maxTurn);
        control.pitchDelta = Clamp(player.pitch, -maxTurn * 0.25f, maxTurn * 0.25f);

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

        if (currentState.empty()) {
            StartEpisode(start, goal, obstacles);
        }

        decisionTimer -= dt;
        int previousAction = currentAction;

        RLControl control = BuildControl(dt);
        player.Update(dt, obstacles, false, false, 0.0f, true, control);
        actionReward += player.rlReward;
        episodeFrames.push_back({ player.position, player.camera, player.rlEpisodeTime });
        lastDistance = RLHorizontalDistance(player.position, goal);

        if (player.rlDone || decisionTimer <= 0.0f) {
            std::vector<float> nextState = player.GetRLState(obstacles);
            model.Learn(currentState, previousAction, actionReward, nextState, player.rlDone);
            RLExperience experience = model.MakeExperience(currentState, previousAction, actionReward, nextState, player.rlDone);
            model.RememberExperience(experience);
            episodeExperiences.push_back(experience);
            model.ReplayMemory(player.rlDone ? 12 : model.replayBatchSize);

            currentState = nextState;
            actionReward = 0.0f;

            if (!player.rlDone) {
                currentAction = model.ChooseAction(currentState);
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
                    improvedBest = true;
                }

                for (int i = 0; i < (int)episodeExperiences.size(); ++i) {
                    RLExperience elite = episodeExperiences[i];
                    float phase = (i + 1.0f) / fmaxf(1.0f, (float)episodeExperiences.size());
                    elite.reward += 2.0f + 8.0f * phase;
                    elite.priority += 12.0f + 16.0f * phase;
                    model.RememberExperience(elite);
                    model.RememberEliteExperience(elite);
                }

                model.ReplayMemory(24);
                model.ReplayEliteMemory(36);
            } else if (player.rlTimedOut) {
                timeouts += 1;
            } else if (player.rlStuck) {
                stucks += 1;
            }

            if (improvedBest) {
                episodesSinceBest = 0;
            } else {
                episodesSinceBest += 1;
                if (episodesSinceBest >= 800) {
                    model.epsilon = fmaxf(model.epsilon, fminf(0.24f, model.minEpsilon + 0.16f));
                    model.ReplayEliteMemory(48);
                    episodesSinceBest = 0;
                }
            }

            resetTimer = resetDelay;
        }
    }

    const char* CurrentActionName() const {
        return RLActions()[currentAction].name;
    }

    bool HasBestRun() const {
        return !bestTrail.empty() && !bestFrames.empty();
    }

    float BestRunDuration() const {
        return HasBestRun() ? bestFrames.back().time : 0.0f;
    }

    Camera3D BestRunCamera(float replayTime) const {
        if (!HasBestRun()) return player.camera;
        if (bestFrames.size() == 1 || replayTime <= bestFrames.front().time) return bestFrames.front().camera;
        if (replayTime >= bestFrames.back().time) return bestFrames.back().camera;

        int nextIndex = 1;
        while (nextIndex < (int)bestFrames.size() && bestFrames[nextIndex].time < replayTime) {
            nextIndex += 1;
        }

        const RLReplayFrame& a = bestFrames[nextIndex - 1];
        const RLReplayFrame& b = bestFrames[nextIndex];
        float span = fmaxf(0.001f, b.time - a.time);
        float t = Clamp((replayTime - a.time) / span, 0.0f, 1.0f);

        Camera3D camera = b.camera;
        camera.position = Vector3Lerp(a.camera.position, b.camera.position, t);
        camera.target = Vector3Lerp(a.camera.target, b.camera.target, t);
        camera.up = Vector3Normalize(Vector3Lerp(a.camera.up, b.camera.up, t));
        camera.fovy = Lerp(a.camera.fovy, b.camera.fovy, t);
        return camera;
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
        RLLinearQModel balanced("Balanced", SKYBLUE, 11, 0.055f, 0.96f, 0.24f, 0.035f, 0.9992f);

        RLLinearQModel explorer("Explorer", ORANGE, 23, 0.050f, 0.95f, 0.48f, 0.08f, 0.9995f);

        RLLinearQModel sprinter("Sprinter", LIME, 37, 0.065f, 0.95f, 0.30f, 0.04f, 0.9991f);
        sprinter.actionBias[9] = 0.14f;
        sprinter.actionBias[10] = 0.09f;
        sprinter.actionBias[11] = 0.09f;
        sprinter.actionBias[12] = 0.08f;
        sprinter.actionBias[13] = 0.08f;
        sprinter.actionBias[14] = 0.05f;
        sprinter.actionBias[15] = 0.05f;

        RLLinearQModel router("Side Route", VIOLET, 51, 0.055f, 0.97f, 0.34f, 0.05f, 0.99925f);
        router.actionBias[3] = 0.07f;
        router.actionBias[4] = 0.07f;
        router.actionBias[5] = 0.12f;
        router.actionBias[6] = 0.12f;
        router.actionBias[7] = 0.08f;
        router.actionBias[8] = 0.08f;

        RLLinearQModel steady("Steady", GOLD, 71, 0.045f, 0.98f, 0.18f, 0.025f, 0.9990f);
        steady.actionBias[0] = 0.08f;
        steady.actionBias[9] = -0.04f;
        steady.actionBias[10] = -0.04f;
        steady.actionBias[11] = -0.04f;
        steady.actionBias[12] = -0.04f;
        steady.actionBias[13] = -0.04f;
        steady.actionBias[14] = -0.04f;
        steady.actionBias[15] = -0.04f;

        runners.push_back(RLRunner(balanced, 0.09f));
        runners.push_back(RLRunner(explorer, 0.08f));
        runners.push_back(RLRunner(sprinter, 0.075f));
        runners.push_back(RLRunner(router, 0.09f));
        runners.push_back(RLRunner(steady, 0.10f));
    }

    void ResetEpisodes(const std::vector<Box>& obstacles) {
        for (auto& runner : runners) {
            runner.StartEpisode(start, goal, obstacles);
        }
    }

    void Update(float dt, const std::vector<Box>& obstacles) {
        for (auto& runner : runners) {
            runner.Update(dt, start, goal, obstacles);
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
        for (int i = 1; i < (int)runners.size(); ++i) {
            const RLRunner& a = runners[i];
            const RLRunner& b = runners[best];
            if (a.successes > b.successes ||
                (a.successes == b.successes && a.successes > 0 && a.bestTime < b.bestTime) ||
                (a.successes == b.successes && a.successes == 0 && a.bestReward > b.bestReward)) {
                best = i;
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
