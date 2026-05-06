#include "raylib.h"
#include "raymath.h"
#include "rlgl.h"
#include <algorithm>
#include <ctime>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <string>
#include <vector>
#include <cmath>
#include "structs/Box.h"
#include "structs/HitMarker.h"
#include "structs/SlashParticle3D.h"
#include "structs/KatanaFX.h"
#include "structs/Player.h"
#include "structs/RLAgent.h"
#include "structs/World.h"

const char* postProcessCode = R"(
#version 330
in vec2 fragTexCoord;
out vec4 finalColor;
uniform sampler2D texture0;
void main() {
    vec2 uv = fragTexCoord;
    float offset = 0.0015;
    float r = texture(texture0, uv + vec2(offset, 0.0)).r;
    float g = texture(texture0, uv).g;
    float b = texture(texture0, uv - vec2(offset, 0.0)).b;
    vec3 col = vec3(r, g, b);
    float dist = distance(uv, vec2(0.5));
    col *= smoothstep(0.8, 0.2, dist);
    col = mix(vec3(0.5), col, 1.15);
    finalColor = vec4(col, 1.0);
}
)";

void DrawHitmarker(int cx, int cy, float timer, float maxTime) {
    float t = 1.0f - (timer / maxTime);
    float gap = 2.0f + t * 8.0f;
    float len = 6.0f;
    unsigned char alpha = (unsigned char)((1.0f - t) * 255.0f);
    Color c = { 255, 255, 255, alpha };

    DrawLineEx({(float)cx - gap, (float)cy - gap}, {(float)cx - gap - len, (float)cy - gap - len}, 2.0f, c);
    DrawLineEx({(float)cx + gap, (float)cy - gap}, {(float)cx + gap + len, (float)cy - gap - len}, 2.0f, c);
    DrawLineEx({(float)cx - gap, (float)cy + gap}, {(float)cx - gap - len, (float)cy + gap + len}, 2.0f, c);
    DrawLineEx({(float)cx + gap, (float)cy + gap}, {(float)cx + gap + len, (float)cy + gap + len}, 2.0f, c);
}

void DrawCrosshair(int cx, int cy, float spread) {
    int gap = (int)(5 + spread);
    DrawRectangle(cx - gap - 8, cy - 1, 8, 2, WHITE);
    DrawRectangle(cx + gap,     cy - 1, 8, 2, WHITE);
    DrawRectangle(cx - 1, cy - gap - 8, 2, 8, WHITE);
    DrawRectangle(cx - 1, cy + gap,     2, 8, WHITE);
    DrawRectangle(cx - 1, cy - 1,       2, 2, RED);
}

void DrawStatusBar(float x, float y, float width, float height, float value, float maxValue, Color fill, const char* label) {
    float ratio = Clamp(value / maxValue, 0.0f, 1.0f);
    Rectangle outer = { x, y, width, height };
    Rectangle fillRect = { x + 1, y + 1, (width - 2) * ratio, height - 2 };

    DrawRectangleRounded(outer, 0.25f, 6, (Color){18, 18, 18, 220});
    if (fillRect.width > 0) DrawRectangleRounded(fillRect, 0.25f, 6, fill);
    DrawRectangleRoundedLines(outer, 0.25f, 6, Fade(WHITE, 0.18f));
    DrawText(label, (int)(x + 10), (int)(y + (height - 16) * 0.5f), 16, WHITE);
}

bool BoxWithinMeleeRange(const Camera3D& cam, const Box& box) {
    Vector3 fwd = Vector3Normalize(Vector3Subtract(cam.target, cam.position));
    Vector3 toBox = Vector3Subtract(box.center, cam.position);
    float forwardDist = Vector3DotProduct(toBox, fwd);

    if (forwardDist < 0.2f || forwardDist > KATANA_RANGE) return false;

    Vector3 projected = Vector3Add(cam.position, Vector3Scale(fwd, forwardDist));
    float radialDist = Vector3Distance(projected, box.center);

    float approxBoxRadius = fmaxf(box.half.x, fmaxf(box.half.y, box.half.z));
    return radialDist <= (KATANA_RADIUS + approxBoxRadius);
}

Camera3D MakeRLTrainingCamera() {
    Camera3D camera = {};
    camera.position = { 0.0f, 92.0f, 86.0f };
    camera.target = { 0.0f, 0.0f, 0.0f };
    camera.up = { 0.0f, 1.0f, 0.0f };
    camera.fovy = 45.0f;
    camera.projection = CAMERA_PERSPECTIVE;
    return camera;
}

const int TRAINING_SPEED_OPTION_COUNT = 11;
const float TRAINING_SPEED_OPTIONS[TRAINING_SPEED_OPTION_COUNT] = { 1.0f, 10.0f, 100.0f, 1000.0f, 2500.0f, 5000.0f, 10000.0f, 25000.0f, 50000.0f, 100000.0f, 250000.0f };
const char* TRAINING_SPEED_LABELS[TRAINING_SPEED_OPTION_COUNT] = { "1x", "10x", "100x", "1000x", "2500x", "5000x", "10000x", "25000x", "50000x", "100000x", "250000x" };

Rectangle TrainingSpeedButtonRect(int index) {
    int col = index % 4;
    int row = index / 4;
    return {
        24.0f + col * 86.0f,
        292.0f + row * 36.0f,
        index >= 9 ? 90.0f : (index >= 3 ? 78.0f : 66.0f),
        30.0f
    };
}

Rectangle BestRunButtonRect() {
    return { 24.0f, 430.0f, 154.0f, 30.0f };
}

int HandleTrainingSpeedButtons(int currentIndex) {
    Vector2 mouse = GetMousePosition();

    for (int i = 0; i < TRAINING_SPEED_OPTION_COUNT; ++i) {
        if (CheckCollisionPointRec(mouse, TrainingSpeedButtonRect(i)) && IsMouseButtonPressed(MOUSE_LEFT_BUTTON)) {
            return i;
        }
    }

    if (IsKeyPressed(KEY_ONE)) return 0;
    if (IsKeyPressed(KEY_TWO)) return 1;
    if (IsKeyPressed(KEY_THREE)) return 2;
    if (IsKeyPressed(KEY_FOUR)) return 3;
    if (IsKeyPressed(KEY_FIVE)) return 4;
    if (IsKeyPressed(KEY_SIX)) return 5;
    if (IsKeyPressed(KEY_SEVEN)) return 6;
    if (IsKeyPressed(KEY_EIGHT)) return 7;
    if (IsKeyPressed(KEY_NINE)) return 8;
    if (IsKeyPressed(KEY_ZERO)) return 9;
    if (IsKeyPressed(KEY_MINUS)) return 10;

    return currentIndex;
}

bool HandleBestRunButton(bool currentValue) {
    if (CheckCollisionPointRec(GetMousePosition(), BestRunButtonRect()) && IsMouseButtonPressed(MOUSE_LEFT_BUTTON)) {
        return !currentValue;
    }

    if (IsKeyPressed(KEY_B)) return !currentValue;

    return currentValue;
}

bool HandleSoloTrainingButton(bool currentValue) {
    if (IsKeyPressed(KEY_M)) return !currentValue;
    return currentValue;
}

double TrainingWallBudgetSeconds(float speedScale, bool turboTraining) {
    if (turboTraining) {
        if (speedScale >= 250000.0f) return 0.050;
        if (speedScale >= 100000.0f) return 0.042;
        if (speedScale >= 50000.0f) return 0.034;
        if (speedScale >= 25000.0f) return 0.026;
        if (speedScale >= 10000.0f) return 0.020;
        if (speedScale >= 1000.0f) return 0.014;
        return 0.010;
    }

    if (speedScale >= 250000.0f) return 0.015;
    if (speedScale >= 100000.0f) return 0.013;
    if (speedScale >= 50000.0f) return 0.011;
    if (speedScale >= 10000.0f) return 0.009;
    if (speedScale >= 1000.0f) return 0.0075;
    return 0.005;
}

int AdvanceRLTraining(
    RLTrainer& trainer,
    float frameDt,
    const std::vector<Box>& obstacles,
    float speedScale,
    bool turboTraining,
    bool soloTraining,
    float& pendingSimSeconds,
    float& simulatedSeconds
) {
    const float fixedStep = 1.0f / 60.0f;
    int maxStepsPerFrame = turboTraining ? 5000 : 2200;
    if (speedScale >= 250000.0f) maxStepsPerFrame = turboTraining ? 90000 : 22000;
    else if (speedScale >= 100000.0f) maxStepsPerFrame = turboTraining ? 60000 : 18000;
    else if (speedScale >= 50000.0f) maxStepsPerFrame = turboTraining ? 36000 : 14000;
    else if (speedScale >= 25000.0f) maxStepsPerFrame = turboTraining ? 22000 : 10000;
    else if (speedScale >= 10000.0f) maxStepsPerFrame = turboTraining ? 14000 : 7500;
    else if (speedScale >= 5000.0f) maxStepsPerFrame = turboTraining ? 9000 : 5200;
    else if (speedScale >= 2500.0f) maxStepsPerFrame = turboTraining ? 6500 : 3800;

    pendingSimSeconds += Clamp(frameDt, 0.0f, 0.25f) * speedScale;
    const float maxBufferedSimSeconds = turboTraining ? 1800.0f : 300.0f;
    if (pendingSimSeconds > maxBufferedSimSeconds) pendingSimSeconds = maxBufferedSimSeconds;

    const double wallBudgetSeconds = TrainingWallBudgetSeconds(speedScale, turboTraining);
    const double startTime = GetTime();

    int steps = 0;
    simulatedSeconds = 0.0f;

    while (pendingSimSeconds >= fixedStep && steps < maxStepsPerFrame) {
        trainer.Update(fixedStep, obstacles, soloTraining);
        pendingSimSeconds -= fixedStep;
        simulatedSeconds += fixedStep;
        steps += 1;

        if ((GetTime() - startTime) >= wallBudgetSeconds) break;
    }

    return steps;
}

void DrawRLRunnerLabels(const RLTrainer& trainer, const Camera3D& camera, bool bestRunView) {
    for (int i = 0; i < (int)trainer.runners.size(); ++i) {
        const RLRunner& runner = trainer.runners[i];
        if (bestRunView && !runner.HasBestRun()) continue;

        Vector3 anchor = bestRunView ? runner.bestTrail.back() : runner.player.position;
        Vector3 labelPos = {
            anchor.x,
            anchor.y + 2.2f,
            anchor.z
        };
        Vector2 screen = GetWorldToScreen(labelPos, camera);

        if (screen.x < -80 || screen.x > SCREEN_WIDTH + 80 || screen.y < -30 || screen.y > SCREEN_HEIGHT + 30) {
            continue;
        }

        DrawText(
            TextFormat("%s%s", i == trainer.selectedRunner ? "> " : "", runner.model.name.c_str()),
            (int)screen.x - 34,
            (int)screen.y,
            14,
            runner.model.color
        );
    }
}

void DrawRLTrainingOverlay(const RLTrainer& trainer, int speedIndex, int trainingSteps, float simulatedSeconds, float pendingSimSeconds, float achievedSpeedScale, bool bestRunView, bool turboTraining, bool soloTraining) {
    const RLRunner& active = trainer.ActiveRunner();
    const int best = trainer.BestRunnerIndex();

    DrawRectangle(12, 12, 390, 476, (Color){ 8, 10, 12, 210 });
    DrawRectangleLines(12, 12, 390, 476, Fade(WHITE, 0.24f));
    DrawText(bestRunView ? "BEST RUN VIEW" : "RL TRAINING VIEW", 24, 24, 20, GREEN);
    DrawText("T manual  V camera  B best  H turbo  M solo", 24, 50, 14, LIGHTGRAY);
    DrawText("TAB/arrows select  1-0/- speed  P report", 24, 68, 14, LIGHTGRAY);
    DrawText(TextFormat("ACTIVE: %s  ACTION: %s", active.model.name.c_str(), active.CurrentActionName()), 24, 76, 16, WHITE);
    DrawText(TextFormat("EPISODE: %d  TIME: %.1f / %.1f", active.episode, active.player.rlEpisodeTime, RL_MAX_EPISODE_TIME), 24, 104, 16, WHITE);
    DrawText(TextFormat("REWARD: %.2f  TD: %.2f  EPS: %.2f", active.player.rlEpisodeReward, active.model.lastTD, active.model.epsilon), 24, 128, 16, YELLOW);
    DrawText(TextFormat("DIST: %.1f  BEST MODEL: %s", active.lastDistance, trainer.runners[best].model.name.c_str()), 24, 152, 16, SKYBLUE);
    DrawText(TextFormat("END: %s%s%s",
        active.player.rlSucceeded ? "SUCCESS" : "",
        active.player.rlTimedOut ? "TIMEOUT" : "",
        active.player.rlStuck ? "STUCK" : ""),
        24,
        174,
        14,
        active.player.rlSucceeded ? GREEN : (active.player.rlDone ? ORANGE : LIGHTGRAY)
    );
    DrawText(TextFormat("WALLS: %d  COLLISION COST: %.1f", active.player.rlWallHits, active.player.rlCollisionPenaltyTotal), 24, 266, 14, active.player.rlWallHits > 0 ? ORANGE : LIGHTGRAY);
    DrawText(TextFormat("TARGET: %s  ACTUAL: %.0fx", TRAINING_SPEED_LABELS[speedIndex], achievedSpeedScale), 24, 194, 14, SKYBLUE);
    DrawText(TextFormat("STEPS: %d  SIM: %.2fs/frame", trainingSteps, simulatedSeconds), 24, 212, 14, LIGHTGRAY);
    DrawText(TextFormat("TURBO: %s", turboTraining ? "ON" : "OFF"), 24, 230, 14, turboTraining ? GOLD : LIGHTGRAY);
    DrawText(TextFormat("SOLO: %s  CHAMPION: %s", soloTraining ? "SELECTED" : "ALL", active.bestPolicy.valid ? "SAVED" : "NONE"), 24, 248, 14, soloTraining ? GOLD : LIGHTGRAY);
    DrawText(TextFormat("SIM QUEUE: %.1fs", pendingSimSeconds), 24, 402, 14, pendingSimSeconds > 1.0f ? ORANGE : LIGHTGRAY);

    Vector2 mouse = GetMousePosition();
    for (int i = 0; i < TRAINING_SPEED_OPTION_COUNT; ++i) {
        Rectangle rect = TrainingSpeedButtonRect(i);
        bool activeSpeed = i == speedIndex;
        bool hot = CheckCollisionPointRec(mouse, rect);
        Color fill = activeSpeed ? GREEN : (hot ? (Color){ 72, 78, 84, 235 } : (Color){ 34, 38, 44, 235 });
        Color text = activeSpeed ? BLACK : WHITE;

        DrawRectangleRounded(rect, 0.18f, 6, fill);
        DrawRectangleRoundedLines(rect, 0.18f, 6, Fade(WHITE, activeSpeed ? 0.45f : 0.22f));
        DrawText(TRAINING_SPEED_LABELS[i], (int)(rect.x + (i >= 9 ? 6 : (i >= 4 ? 10 : 16))), (int)(rect.y + 7), 16, text);
    }

    Rectangle bestButton = BestRunButtonRect();
    bool hotBest = CheckCollisionPointRec(mouse, bestButton);
    Color bestFill = bestRunView ? GOLD : (hotBest ? (Color){ 72, 78, 84, 235 } : (Color){ 34, 38, 44, 235 });
    DrawRectangleRounded(bestButton, 0.18f, 6, bestFill);
    DrawRectangleRoundedLines(bestButton, 0.18f, 6, Fade(WHITE, bestRunView ? 0.45f : 0.22f));
    DrawText(bestRunView ? "LIVE RUNS" : "BEST RUNS", (int)bestButton.x + 16, (int)bestButton.y + 7, 16, bestRunView ? BLACK : WHITE);

    if (bestRunView) {
        const char* bestInfo = active.HasBestRun()
            ? TextFormat("SELECTED BEST: %.1fs reward %.1f", active.bestTime, active.bestSuccessReward)
            : "SELECTED BEST: none yet";
        DrawText(bestInfo, 194, 436, 14, active.HasBestRun() ? GOLD : LIGHTGRAY);
    }

    int panelX = SCREEN_WIDTH - 420;
    DrawRectangle(panelX, 12, 408, 30 + (int)trainer.runners.size() * 54, (Color){ 8, 10, 12, 210 });
    DrawRectangleLines(panelX, 12, 408, 30 + (int)trainer.runners.size() * 54, Fade(WHITE, 0.24f));
    DrawText("MODELS", panelX + 14, 24, 18, WHITE);

    for (int i = 0; i < (int)trainer.runners.size(); ++i) {
        const RLRunner& runner = trainer.runners[i];
        int y = 54 + i * 54;
        Color rowColor = (i == trainer.selectedRunner) ? Fade(WHITE, 0.12f) : Fade(WHITE, 0.04f);
        DrawRectangle(panelX + 10, y - 4, 388, 46, rowColor);
        DrawRectangle(panelX + 18, y + 4, 12, 12, runner.model.color);
        DrawText(runner.model.name.c_str(), panelX + 38, y, 16, runner.model.color);
        DrawText(
            TextFormat("E%d S%d T%d X%d M%d L%d e%.2f",
                runner.episode,
                runner.successes,
                runner.timeouts,
                runner.stucks,
                runner.model.MemorySize(),
                runner.model.EliteMemorySize(),
                runner.model.epsilon
            ),
            panelX + 38,
            y + 20,
            13,
            LIGHTGRAY
        );

        const char* bestText = (runner.bestTime < 9998.0f) ? TextFormat("best %.1fs", runner.bestTime) : "best --";
        DrawText(bestText, panelX + 310, y, 14, GOLD);
        DrawText(TextFormat("%.1fm", runner.lastDistance), panelX + 310, y + 20, 14, SKYBLUE);
    }
}

void DrawManualBestGhost(const RLRunner& runner, const Vector3& playerPosition, const Vector3& startPoint, float manualTime) {
    if (!runner.HasBestRun()) return;

    Color ghostColor = Fade(runner.model.color, 0.65f);
    for (int i = 1; i < (int)runner.bestTrail.size(); ++i) {
        DrawLine3D(runner.bestTrail[i - 1], runner.bestTrail[i], Fade(runner.model.color, 0.35f));
    }

    RLReplayFrame ghost = runner.BestRunFrame(manualTime);
    Vector3 p = ghost.position;
    if (manualTime <= 0.0001f) p = startPoint;
    if (RLHorizontalDistance(playerPosition, p) <= 0.1f) return;

    DrawCube({ p.x, p.y + 0.35f, p.z }, 0.75f, 0.9f, 0.75f, ghostColor);
    DrawCubeWires({ p.x, p.y + 0.35f, p.z }, 0.75f, 0.9f, 0.75f, WHITE);
    DrawSphere({ p.x, p.y + 1.05f, p.z }, 0.24f, runner.model.color);
}

struct ManualRunReplay {
    std::vector<RLReplayFrame> frames;
    std::vector<Vector3> trail;
    std::vector<RLExperience> experiences;
    bool finished = false;
    float finishTime = 0.0f;
    float lastFrameTime = 0.0f;

    void Reset() {
        frames.clear();
        trail.clear();
        experiences.clear();
        finished = false;
        finishTime = 0.0f;
        lastFrameTime = 0.0f;
    }

    void Record(
        const Player& player,
        const Player::RLState& state,
        int action,
        float reward,
        const Player::RLState& nextState
    ) {
        if (finished) return;
        if (player.rlDone && !player.rlSucceeded) return;

        if (frames.empty() || player.rlDone || player.rlEpisodeTime - lastFrameTime >= 1.0f / 30.0f) {
            frames.push_back({ player.position, player.camera, player.rlEpisodeTime, -1 });
            lastFrameTime = player.rlEpisodeTime;
        }

        if (trail.empty() || RLHorizontalDistance(trail.back(), player.position) > 0.75f) {
            trail.push_back(player.position);
        }

        if (action >= 0 && action < (int)RLActions().size()) {
            RLExperience experience;
            experience.state = state;
            experience.nextState = nextState;
            experience.action = action;
            experience.reward = reward;
            experience.done = player.rlDone;
            experience.priority = 2.0f + fabsf(reward) * 0.12f + (player.rlDone ? 10.0f : 0.0f);
            experiences.push_back(experience);
        }

        if (player.rlSucceeded) {
            finished = true;
            finishTime = player.rlEpisodeTime;
        }
    }

    bool HasRun() const {
        return finished && !frames.empty() && !trail.empty();
    }

    bool IsBetterThan(const ManualRunReplay& other) const {
        return HasRun() && (!other.HasRun() || finishTime < other.finishTime);
    }

    RLReplayFrame FrameAt(float replayTime) const {
        if (frames.empty()) return {};
        if (frames.size() == 1 || replayTime <= frames.front().time) return frames.front();
        if (replayTime >= frames.back().time) return frames.back();

        int nextIndex = 1;
        while (nextIndex < (int)frames.size() && frames[nextIndex].time < replayTime) {
            nextIndex += 1;
        }

        const RLReplayFrame& a = frames[nextIndex - 1];
        const RLReplayFrame& b = frames[nextIndex];
        float span = fmaxf(0.001f, b.time - a.time);
        float t = Clamp((replayTime - a.time) / span, 0.0f, 1.0f);

        RLReplayFrame frame = b;
        frame.position = Vector3Lerp(a.position, b.position, t);
        frame.camera.position = Vector3Lerp(a.camera.position, b.camera.position, t);
        frame.camera.target = Vector3Lerp(a.camera.target, b.camera.target, t);
        frame.camera.up = Vector3Normalize(Vector3Lerp(a.camera.up, b.camera.up, t));
        frame.camera.fovy = Lerp(a.camera.fovy, b.camera.fovy, t);
        frame.time = Lerp(a.time, b.time, t);
        return frame;
    }
};

int ClosestRLActionForManualInput(float relativeAngle, bool dash, bool jump, bool slide) {
    const std::vector<RLDiscreteAction>& actions = RLActions();

    auto choose = [&](bool exactAbilities) {
        int best = -1;
        float bestCost = 9999.0f;

        for (int i = 0; i < (int)actions.size(); ++i) {
            const RLDiscreteAction& action = actions[i];
            if (action.slide != slide) continue;
            if (exactAbilities && (action.dash != dash || action.jump != jump)) continue;
            if (!exactAbilities && slide && !action.slide) continue;

            float cost = fabsf(RLWrapDegrees(relativeAngle - action.angleDegrees));
            if (!exactAbilities) {
                if (action.dash != dash) cost += dash ? 22.0f : 14.0f;
                if (action.jump != jump) cost += jump ? 18.0f : 10.0f;
            }

            if (cost < bestCost) {
                bestCost = cost;
                best = i;
            }
        }

        return best;
    };

    int exact = choose(true);
    if (exact >= 0) return exact;

    int fallback = choose(false);
    if (fallback >= 0) return fallback;

    return RLActionIndexByName("GOAL");
}

int InferManualRLAction(
    const Player& player,
    bool keyW,
    bool keyA,
    bool keyS,
    bool keyD,
    bool dash,
    bool jump,
    bool slide
) {
    if (slide) return RLActionIndexByName("SLIDE");

    Vector3 forward = { sinf(DEG2RAD * player.yaw), 0.0f, cosf(DEG2RAD * player.yaw) };
    Vector3 right = { -forward.z, 0.0f, forward.x };
    Vector3 move = { 0.0f, 0.0f, 0.0f };

    if (keyW) move = Vector3Add(move, forward);
    if (keyS) move = Vector3Subtract(move, forward);
    if (keyD) move = Vector3Add(move, right);
    if (keyA) move = Vector3Subtract(move, right);
    if (Vector3Length(move) <= 0.1f) move = player.lastMoveDir;
    if (Vector3Length(move) <= 0.1f) move = RLFlatDirection(player.position, player.rlTarget);
    move = Vector3Normalize(move);

    Vector3 goalDir = RLFlatDirection(player.position, player.rlTarget);
    float relativeAngle = RLWrapDegrees(RLYawFromDirection(move) - RLYawFromDirection(goalDir));
    return ClosestRLActionForManualInput(relativeAngle, dash, jump, false);
}

void DrawManualCompletedGhost(const ManualRunReplay& replay, const Vector3& playerPosition, float replayTime) {
    if (!replay.HasRun()) return;

    Color manualColor = { 255, 245, 180, 255 };
    for (int i = 1; i < (int)replay.trail.size(); ++i) {
        DrawLine3D(replay.trail[i - 1], replay.trail[i], Fade(manualColor, 0.45f));
    }

    RLReplayFrame ghost = replay.FrameAt(replayTime);
    Vector3 p = ghost.position;
    if (RLHorizontalDistance(playerPosition, p) <= 0.1f) return;

    DrawCube({ p.x, p.y + 0.35f, p.z }, 0.72f, 0.86f, 0.72f, Fade(manualColor, 0.70f));
    DrawCubeWires({ p.x, p.y + 0.35f, p.z }, 0.72f, 0.86f, 0.72f, manualColor);
    DrawSphere({ p.x, p.y + 1.05f, p.z }, 0.22f, manualColor);
}

void DrawKeyboardKey(Rectangle rect, const char* label, bool active) {
    Color fill = active ? GREEN : (Color){ 28, 32, 38, 225 };
    Color outline = active ? WHITE : Fade(WHITE, 0.28f);
    Color text = active ? BLACK : WHITE;

    DrawRectangleRounded(rect, 0.16f, 6, fill);
    DrawRectangleRoundedLines(rect, 0.16f, 6, outline);
    int textWidth = MeasureText(label, 18);
    DrawText(label, (int)(rect.x + (rect.width - textWidth) * 0.5f), (int)(rect.y + 10), 18, text);
}

void DrawAIInputOverlay(const RLRunner& runner, float replayTime) {
    if (!runner.HasBestRun()) return;

    RLReplayFrame frame = runner.BestRunFrame(replayTime);
    bool w = false;
    bool a = false;
    bool s = false;
    bool d = false;
    bool space = false;
    bool e = false;

    if (frame.action >= 0 && frame.action < (int)RLActions().size()) {
        const RLDiscreteAction& action = RLActions()[frame.action];
        w = action.strength > 0.1f && fabsf(action.angleDegrees) < 105.0f;
        a = action.angleDegrees < -8.0f;
        d = action.angleDegrees > 8.0f;
        s = fabsf(action.angleDegrees) > 105.0f;
        space = action.jump;
        e = action.dash;
    }

    float x = SCREEN_WIDTH - 238.0f;
    float y = SCREEN_HEIGHT - 168.0f;
    DrawRectangle((int)x - 12, (int)y - 12, 226, 150, (Color){ 8, 10, 12, 190 });
    DrawRectangleLines((int)x - 12, (int)y - 12, 226, 150, Fade(WHITE, 0.22f));
    DrawText("AI INPUT", (int)x, (int)y - 4, 16, runner.model.color);

    DrawKeyboardKey({ x + 46, y + 22, 42, 42 }, "W", w);
    DrawKeyboardKey({ x + 0, y + 68, 42, 42 }, "A", a);
    DrawKeyboardKey({ x + 46, y + 68, 42, 42 }, "S", s);
    DrawKeyboardKey({ x + 92, y + 68, 42, 42 }, "D", d);
    DrawKeyboardKey({ x + 148, y + 22, 48, 42 }, "E", e);
    DrawKeyboardKey({ x + 0, y + 114, 134, 30 }, "SPACE", space);

    Vector3 look = Vector3Subtract(frame.camera.target, frame.camera.position);
    float lookYaw = atan2f(look.x, look.z) * RAD2DEG;
    float lookPitch = asinf(Clamp(Vector3Normalize(look).y, -1.0f, 1.0f)) * RAD2DEG;
    float turn = RLWrapDegrees(lookYaw - RLYawFromDirection(RLFlatDirection(frame.position, runner.player.rlTarget)));
    float cx = x + 172.0f;
    float cy = y + 104.0f;
    DrawCircleLines((int)cx, (int)cy, 24.0f, Fade(WHITE, 0.45f));
    DrawLine((int)cx, (int)cy, (int)(cx + sinf(DEG2RAD * turn) * 22.0f), (int)(cy - cosf(DEG2RAD * turn) * 22.0f), runner.model.color);
    DrawText(TextFormat("CAM %.0f", lookPitch), (int)cx - 27, (int)cy + 31, 14, LIGHTGRAY);
}

struct RLRecentStats {
    int count = 0;
    int successes = 0;
    int timeouts = 0;
    int stucks = 0;
    int wallHits = 0;
    int decisions = 0;
    float rewardTotal = 0.0f;
    float timeTotal = 0.0f;
    float successTimeTotal = 0.0f;
    float distanceTotal = 0.0f;
    float collisionTotal = 0.0f;
    float bestSuccessTime = 9999.0f;
    float bestReward = -9999.0f;
    std::vector<int> actionCounts;
};

int SumActionCounts(const std::vector<int>& counts) {
    int total = 0;
    for (int count : counts) total += count;
    return total;
}

int TopActionIndex(const std::vector<int>& counts) {
    if (counts.empty()) return -1;

    int best = 0;
    for (int i = 1; i < (int)counts.size(); ++i) {
        if (counts[i] > counts[best]) best = i;
    }

    return counts[best] > 0 ? best : -1;
}

float SafePercent(int part, int total) {
    return total > 0 ? (100.0f * (float)part / (float)total) : 0.0f;
}

float SafeAverage(float total, int count) {
    return count > 0 ? total / (float)count : 0.0f;
}

const char* EpisodeOutcomeName(const RLEpisodeSummary& episode) {
    if (episode.success) return "SUCCESS";
    if (episode.timeout) return "TIMEOUT";
    if (episode.stuck) return "STUCK";
    return "ENDED";
}

std::string FormatVector3(Vector3 value) {
    std::ostringstream text;
    text << std::fixed << std::setprecision(2)
         << "(" << value.x << ", " << value.y << ", " << value.z << ")";
    return text.str();
}

std::string ActionFlags(const RLDiscreteAction& action) {
    std::string flags;
    if (action.dash) flags += "dash ";
    if (action.jump) flags += "jump ";
    if (action.slide) flags += "slide ";
    if (flags.empty()) flags = "move";
    return flags;
}

RLRecentStats BuildRecentStats(const RLRunner& runner, int maxEpisodes) {
    RLRecentStats stats;
    const std::vector<RLDiscreteAction>& actions = RLActions();
    stats.actionCounts.assign(actions.size(), 0);

    int start = std::max(0, (int)runner.recentEpisodes.size() - maxEpisodes);
    for (int i = start; i < (int)runner.recentEpisodes.size(); ++i) {
        const RLEpisodeSummary& episode = runner.recentEpisodes[i];
        stats.count += 1;
        stats.successes += episode.success ? 1 : 0;
        stats.timeouts += episode.timeout ? 1 : 0;
        stats.stucks += episode.stuck ? 1 : 0;
        stats.wallHits += episode.wallHits;
        stats.decisions += episode.decisions;
        stats.rewardTotal += episode.reward;
        stats.timeTotal += episode.time;
        stats.distanceTotal += episode.distanceToGoal;
        stats.collisionTotal += episode.collisionCost;
        stats.bestReward = fmaxf(stats.bestReward, episode.reward);

        if (episode.success) {
            stats.successTimeTotal += episode.time;
            stats.bestSuccessTime = fminf(stats.bestSuccessTime, episode.time);
        }

        for (int action = 0; action < (int)episode.actionCounts.size() && action < (int)stats.actionCounts.size(); ++action) {
            stats.actionCounts[action] += episode.actionCounts[action];
        }
    }

    return stats;
}

void WriteActionUsage(std::ofstream& out, const std::vector<int>& actionCounts) {
    const std::vector<RLDiscreteAction>& actions = RLActions();
    int total = SumActionCounts(actionCounts);

    out << "  total decisions: " << total << "\n";
    out << "  action table:\n";
    out << "    idx  action          count    pct    yaw    pitch   flags\n";
    out << "    -------------------------------------------------------------\n";
    for (int i = 0; i < (int)actions.size(); ++i) {
        int count = (i < (int)actionCounts.size()) ? actionCounts[i] : 0;
        out << "    "
            << std::setw(3) << i << "  "
            << std::left << std::setw(13) << actions[i].name << std::right
            << std::setw(7) << count << "  "
            << std::setw(5) << std::fixed << std::setprecision(1) << SafePercent(count, total) << "%  "
            << std::setw(6) << std::setprecision(1) << actions[i].angleDegrees << "  "
            << std::setw(7) << std::setprecision(1) << actions[i].pitchDegrees << "   "
            << ActionFlags(actions[i]) << "\n";
    }
}

void WriteModelParameterSnapshot(std::ofstream& out, const RLLinearQModel& model) {
    const std::vector<RLDiscreteAction>& actions = RLActions();

    out << "learned action parameter snapshot:\n";
    out << "  idx  action           bias     weights   avg |w|   max |w|   last flag\n";
    out << "  -----------------------------------------------------------------------\n";
    for (int action = 0; action < (int)actions.size(); ++action) {
        float bias = (action < (int)model.actionBias.size()) ? model.actionBias[action] : 0.0f;
        float absTotal = 0.0f;
        float absMax = 0.0f;
        int weightCount = 0;

        if (action < (int)model.weights.size()) {
            weightCount = (int)model.weights[action].size();
            for (float weight : model.weights[action]) {
                float value = fabsf(weight);
                absTotal += value;
                absMax = fmaxf(absMax, value);
            }
        }

        out << "  "
            << std::setw(3) << action << "  "
            << std::left << std::setw(13) << actions[action].name << std::right
            << std::setw(8) << std::fixed << std::setprecision(3) << bias
            << std::setw(10) << weightCount
            << std::setw(10) << std::setprecision(4) << SafeAverage(absTotal, weightCount)
            << std::setw(10) << std::setprecision(4) << absMax << "   "
            << ActionFlags(actions[action]) << "\n";
    }
}

void WriteRecentEpisodeLog(std::ofstream& out, const RLRunner& runner) {
    int start = std::max(0, (int)runner.recentEpisodes.size() - 160);
    out << "  recent episode log (last " << ((int)runner.recentEpisodes.size() - start) << " stored episodes):\n";
    out << "    ep      outcome   time    reward    dist    walls   coll    decisions   top action\n";
    out << "    ----------------------------------------------------------------------------------\n";

    for (int i = start; i < (int)runner.recentEpisodes.size(); ++i) {
        const RLEpisodeSummary& episode = runner.recentEpisodes[i];
        int topAction = TopActionIndex(episode.actionCounts);
        const char* topName = topAction >= 0 ? RLActions()[topAction].name : "none";
        out << "    "
            << std::setw(6) << episode.episode << "  "
            << std::left << std::setw(8) << EpisodeOutcomeName(episode) << std::right << "  "
            << std::setw(5) << std::fixed << std::setprecision(2) << episode.time << "  "
            << std::setw(8) << std::setprecision(2) << episode.reward << "  "
            << std::setw(6) << std::setprecision(2) << episode.distanceToGoal << "  "
            << std::setw(5) << episode.wallHits << "  "
            << std::setw(6) << std::setprecision(2) << episode.collisionCost << "  "
            << std::setw(9) << episode.decisions << "   "
            << topName << "\n";
    }
}

void WriteModelRecommendations(
    std::ofstream& out,
    const RLRunner& runner,
    const RLRecentStats& stats,
    float targetSpeedScale,
    float achievedSpeedScale
) {
    int finished = runner.successes + runner.timeouts + runner.stucks;
    float lifetimeSuccessRate = SafePercent(runner.successes, finished);
    float recentSuccessRate = SafePercent(stats.successes, stats.count);
    float recentTimeoutRate = SafePercent(stats.timeouts, stats.count);
    float recentStuckRate = SafePercent(stats.stucks, stats.count);
    float avgWalls = SafeAverage((float)stats.wallHits, stats.count);
    float avgCollision = SafeAverage(stats.collisionTotal, stats.count);
    float avgDistance = SafeAverage(stats.distanceTotal, stats.count);
    float avgReward = SafeAverage(stats.rewardTotal, stats.count);
    int recentDecisionTotal = SumActionCounts(stats.actionCounts);
    int topRecentAction = TopActionIndex(stats.actionCounts);
    float topRecentActionPct = topRecentAction >= 0 ? SafePercent(stats.actionCounts[topRecentAction], recentDecisionTotal) : 0.0f;

    out << "  readout and recommendations:\n";
    if (stats.count < 12) {
        out << "    - Not enough finished episodes for a stable diagnosis yet. Train at least 50-100 completed episodes before trusting rates.\n";
    }

    if (finished > 0 && lifetimeSuccessRate < 8.0f && runner.episode > 80) {
        out << "    - Lifetime success rate is very low. Keep progress reward, but make failure penalties sharper for timeout/stuck and reward safe clearance more consistently.\n";
    }

    if (stats.count > 0 && recentSuccessRate < lifetimeSuccessRate * 0.55f && runner.successes > 5) {
        out << "    - Recent success rate is much worse than lifetime. This model may be drifting; replay elite memory more often or blend toward the saved best policy sooner.\n";
    }

    if (stats.successes > 0 && runner.bestTime < 9998.0f) {
        float avgSuccessTime = SafeAverage(stats.successTimeTotal, stats.successes);
        if (avgSuccessTime > runner.bestTime * 1.30f) {
            out << "    - Best run is much faster than average recent successes. Treat the best replay as a rare peak, not the model's normal performance.\n";
        }
    }

    if (stats.count > 0 && avgReward > 0.0f && recentSuccessRate < 5.0f) {
        out << "    - Reward may still be hackable: recent average reward is positive but almost no episodes succeed. Increase terminal success weight relative to partial progress.\n";
    }

    if (recentTimeoutRate > 35.0f) {
        out << "    - Timeout rate is high. Add stronger reward for reaching route milestones earlier, and increase time-pressure once the model is not stuck on walls.\n";
    }

    if (recentStuckRate > 20.0f) {
        out << "    - Stuck rate is high. Penalize low speed/no-progress harder after wall contact and consider reducing dash-heavy action bias for this model.\n";
    }

    if (avgWalls > 2.0f || avgCollision > 6.0f) {
        out << "    - Wall contact is expensive for learning quality. RL treats wallruns as a fallback except for productive final-cube climbs; keep rewarding clear forward progress and down-rank dash actions when clearance is low.\n";
    }

    if (avgDistance > 20.0f && recentSuccessRate < 15.0f && stats.count >= 20) {
        out << "    - Episodes usually end far from the goal. The model needs better route-shaping, not just more speed; add waypoint/side-lane progress terms.\n";
    }

    if (topRecentActionPct > 42.0f && recentDecisionTotal > 80) {
        out << "    - Action usage is dominated by " << RLActions()[topRecentAction].name << " (" << std::fixed << std::setprecision(1) << topRecentActionPct << "%). Biases may be too strong or exploration too low.\n";
    }

    if (runner.model.epsilon > 0.18f && runner.episode > 300) {
        out << "    - Epsilon is still high after many episodes. Lower decay/floor if behavior looks random instead of exploratory.\n";
    }

    if (runner.model.MemorySize() > runner.model.memoryCapacity * 9 / 10) {
        out << "    - Replay memory is near capacity. If training lags, lower memory capacity or replay batch counts before adding more actions/features.\n";
    }

    if (targetSpeedScale >= 1000.0f && achievedSpeedScale > 0.0f && achievedSpeedScale < targetSpeedScale * 0.10f) {
        out << "    - Program is CPU-budget limited at this speed setting. The label is not real sim speed; use actual speed, solo mode, and fewer active runners when benchmarking.\n";
    }

    if (runner.episodesSinceBest > 500 && runner.bestPolicy.valid) {
        out << "    - Model has gone many episodes without improving. Increase elite replay/blend frequency or briefly raise epsilon to escape the local policy.\n";
    }

    if (stats.count >= 12 && recentSuccessRate >= 30.0f && avgWalls < 1.0f) {
        out << "    - Model is stable enough for speed optimization. Increase time bonus and compare average successful time, not only best time.\n";
    }
}

bool WriteTrainingReport(
    const char* path,
    const RLTrainer& trainer,
    const World& world,
    int speedIndex,
    float achievedSpeedScale,
    float pendingSimSeconds,
    int trainingStepsLastFrame,
    float trainingSimSecondsLastFrame,
    bool turboTraining,
    bool soloTraining,
    const ManualRunReplay& manualBestReplay
) {
    std::ofstream out(path);
    if (!out.is_open()) return false;

    std::time_t now = std::time(nullptr);
    float targetSpeedScale = TRAINING_SPEED_OPTIONS[speedIndex];
    int activeModelCount = soloTraining ? 1 : (int)trainer.runners.size();
    float runnerUpdatesPerSecond = achievedSpeedScale * 60.0f * fmaxf(1.0f, (float)activeModelCount);
    int bestIndex = trainer.BestRunnerIndex();

    out << "RL MOVEMENT TRAINING REPORT\n";
    out << "Generated: " << std::put_time(std::localtime(&now), "%Y-%m-%d %H:%M:%S") << "\n";
    out << "File: " << path << "\n\n";

    out << "HOW TO READ THIS FILE\n";
    out << "  - Success rate and average successful time matter more than one lucky best run.\n";
    out << "  - Positive reward with bad outcomes means the reward function can still be exploited.\n";
    out << "  - Target speed is the requested multiplier. Actual speed is what the executable really achieved.\n";
    out << "  - Wall hits, collision cost, timeout rate, and stuck rate explain why a high reward episode can still look bad.\n\n";

    out << "RUN CONFIGURATION\n";
    out << "  selected model: " << trainer.ActiveRunner().model.name << "\n";
    out << "  best model by trainer ranking: " << trainer.runners[bestIndex].model.name << "\n";
    out << "  target training speed: " << TRAINING_SPEED_LABELS[speedIndex] << " (" << std::fixed << std::setprecision(0) << targetSpeedScale << "x)\n";
    out << "  achieved training speed: " << std::setprecision(1) << achievedSpeedScale << "x\n";
    out << "  achieved/target ratio: " << std::setprecision(2) << (targetSpeedScale > 0.0f ? achievedSpeedScale / targetSpeedScale : 0.0f) << "\n";
    out << "  runner updates/sec estimate: " << std::setprecision(0) << runnerUpdatesPerSecond << "\n";
    out << "  active model count: " << activeModelCount << "\n";
    out << "  turbo mode: " << (turboTraining ? "on" : "off") << "\n";
    out << "  solo mode: " << (soloTraining ? "selected model only" : "all models") << "\n";
    out << "  fixed training steps last frame: " << trainingStepsLastFrame << "\n";
    out << "  simulated seconds last frame: " << std::setprecision(4) << trainingSimSecondsLastFrame << "\n";
    out << "  pending sim queue: " << std::setprecision(2) << pendingSimSeconds << "s\n\n";

    out << "WORLD AND OBSERVATION CONFIGURATION\n";
    out << "  start point: " << FormatVector3(world.startPoint) << "\n";
    out << "  goal point: " << FormatVector3(world.goalPoint) << "\n";
    out << "  start-to-goal horizontal distance: " << std::setprecision(2) << RLHorizontalDistance(world.startPoint, world.goalPoint) << "m\n";
    out << "  obstacle count: " << world.obstacles.size() << "\n";
    out << "  observation size: " << Player::RL_STATE_SIZE << "\n";
    out << "  discrete action count: " << RLActions().size() << "\n";
    out << "  manual personal best replay: " << (manualBestReplay.HasRun() ? "saved" : "none") << "\n";
    if (manualBestReplay.HasRun()) {
        out << "  manual best finish time: " << std::setprecision(2) << manualBestReplay.finishTime << "s\n";
        out << "  manual best frames/trail/experiences: " << manualBestReplay.frames.size() << " frames, " << manualBestReplay.trail.size() << " trail points, " << manualBestReplay.experiences.size() << " elite samples\n";
    }
    out << "  obstacle table:\n";
    out << "    idx   center                  half-size\n";
    out << "    ------------------------------------------------------\n";
    for (int i = 0; i < (int)world.obstacles.size(); ++i) {
        out << "    "
            << std::setw(3) << i << "   "
            << std::left << std::setw(23) << FormatVector3(world.obstacles[i].center) << std::right
            << FormatVector3(world.obstacles[i].half) << "\n";
    }
    out << "\n";

    out << "GLOBAL PERFORMANCE DIAGNOSIS\n";
    if (targetSpeedScale >= 1000.0f && achievedSpeedScale > 0.0f && achievedSpeedScale < targetSpeedScale * 0.10f) {
        out << "  - The executable is CPU-budget limited at the selected multiplier. Clicking 250000x cannot force 250000x if the frame budget can only simulate about "
            << std::setprecision(0) << achievedSpeedScale << "x.\n";
    } else if (achievedSpeedScale > 0.0f) {
        out << "  - Actual training speed is tracking the requested setting well enough for current diagnostics.\n";
    } else {
        out << "  - Training is not currently running, so actual speed is zero.\n";
    }

    if (!soloTraining && trainer.runners.size() > 1) {
        out << "  - All-model training multiplies CPU work by " << trainer.runners.size() << ". Use solo mode (M) when tuning one model or measuring true speed.\n";
    }
    if (RLActions().size() > 24) {
        out << "  - Action space is broad. Remove actions that stay below 1-2% usage after training, unless they appear in the best successful replay.\n";
    }
    out << "  - Biggest CPU costs are fixed-step runner updates, obstacle collision/sensor checks, replay learning bursts after episode end, and rendering if lightweight mode is off.\n";
    out << "  - Useful next optimizations: profile runner updates/sec, cache obstacle clearance checks per frame, lower replay bursts for weak models, and benchmark with rendering skipped.\n\n";

    out << "MODEL SUMMARY TABLE\n";
    out << "  model          ep    done   succ   succ%   timeout   stuck   best time   best reward   eps    recent succ%   rank\n";
    out << "  ----------------------------------------------------------------------------------------------------------------------\n";
    for (const RLRunner& runner : trainer.runners) {
        int finished = runner.successes + runner.timeouts + runner.stucks;
        RLRecentStats stats = BuildRecentStats(runner, 80);
        out << "  "
            << std::left << std::setw(13) << runner.model.name << std::right
            << std::setw(6) << runner.episode
            << std::setw(8) << finished
            << std::setw(7) << runner.successes
            << std::setw(8) << std::fixed << std::setprecision(1) << SafePercent(runner.successes, finished)
            << std::setw(10) << runner.timeouts
            << std::setw(8) << runner.stucks
            << std::setw(12) << (runner.HasBestRun() ? runner.bestTime : 0.0f)
            << std::setw(14) << runner.bestReward
            << std::setw(7) << runner.model.epsilon
            << std::setw(12) << SafePercent(stats.successes, stats.count)
            << std::setw(8) << runner.ReliableSelectionScore()
            << "\n";
    }
    out << "\n";

    for (int modelIndex = 0; modelIndex < (int)trainer.runners.size(); ++modelIndex) {
        const RLRunner& runner = trainer.runners[modelIndex];
        RLRecentStats stats = BuildRecentStats(runner, 80);
        int finished = runner.successes + runner.timeouts + runner.stucks;

        out << "================================================================================\n";
        out << "MODEL " << modelIndex << ": " << runner.model.name << "\n";
        out << "================================================================================\n";
        out << "identity:\n";
        out << "  selected: " << (modelIndex == trainer.selectedRunner ? "yes" : "no") << "\n";
        out << "  trainer-ranked best: " << (modelIndex == bestIndex ? "yes" : "no") << "\n";
        out << "  reliable selection score: " << std::fixed << std::setprecision(2) << runner.ReliableSelectionScore() << "\n";
        out << "  color rgba: (" << (int)runner.model.color.r << ", " << (int)runner.model.color.g << ", " << (int)runner.model.color.b << ", " << (int)runner.model.color.a << ")\n\n";

        out << "lifetime outcomes:\n";
        out << "  episodes started: " << runner.episode << "\n";
        out << "  finished episodes: " << finished << "\n";
        out << "  successes: " << runner.successes << " (" << std::fixed << std::setprecision(1) << SafePercent(runner.successes, finished) << "%)\n";
        out << "  timeouts: " << runner.timeouts << " (" << SafePercent(runner.timeouts, finished) << "%)\n";
        out << "  stucks: " << runner.stucks << " (" << SafePercent(runner.stucks, finished) << "%)\n";
        out << "  episodes since best improvement: " << runner.episodesSinceBest << "\n";
        out << "  best reward seen: " << std::setprecision(3) << runner.bestReward << "\n";
        out << "  best success reward: " << runner.bestSuccessReward << "\n";
        if (runner.HasBestRun()) {
            out << "  best success time: " << std::setprecision(3) << runner.bestTime << "s\n";
            out << "  best replay duration: " << runner.BestRunDuration() << "s\n";
            out << "  best replay frames/trail: " << runner.bestFrames.size() << " frames, " << runner.bestTrail.size() << " trail points\n";
        } else {
            out << "  best success time: none\n";
        }
        out << "\n";

        out << "current live episode:\n";
        out << "  current action: " << runner.CurrentActionName() << "\n";
        out << "  episode time: " << std::setprecision(3) << runner.player.rlEpisodeTime << "s / " << RL_MAX_EPISODE_TIME << "s\n";
        out << "  episode reward: " << runner.player.rlEpisodeReward << "\n";
        out << "  last step reward: " << runner.player.rlReward << "\n";
        out << "  distance to goal: " << runner.lastDistance << "m\n";
        out << "  wall hits: " << runner.player.rlWallHits << "\n";
        out << "  collision cost: " << runner.player.rlCollisionPenaltyTotal << "\n";
        out << "  position: " << FormatVector3(runner.player.position) << "\n";
        out << "  velocity: " << FormatVector3(runner.player.velocity) << "\n\n";

        out << "learning parameters:\n";
        out << "  learning rate: " << runner.model.learningRate << "\n";
        out << "  discount: " << runner.model.discount << "\n";
        out << "  epsilon/current floor decay: " << runner.model.epsilon << " / " << runner.model.minEpsilon << " / " << runner.model.epsilonDecay << "\n";
        out << "  decision interval: " << runner.decisionInterval << "s\n";
        out << "  turn rate: " << runner.model.turnRate << " deg/s\n";
        out << "  clearance guidance: " << runner.model.clearanceGuidance << "\n";
        out << "  raw feature count: " << runner.model.rawFeatureCount << "\n";
        out << "  encoded feature count: " << runner.model.featureCount << "\n";
        out << "  replay memory: " << runner.model.MemorySize() << " / " << runner.model.memoryCapacity << "\n";
        out << "  elite memory: " << runner.model.EliteMemorySize() << " / " << runner.model.eliteMemoryCapacity << "\n";
        out << "  manual guidance: " << runner.manualGuidanceReplays << " replays, " << runner.manualGuidanceExperiences << " samples\n";
        out << "  best-run anchor: " << runner.bestExperiences.size() << " samples, " << runner.bestRunRehearsals << " rehearsed updates\n";
        out << "  best-path reward: current episode " << runner.episodeBestPathReward << ", lifetime " << runner.totalBestPathReward << ", last step " << runner.lastBestPathReward << "\n";
        out << "  best policy saved: " << (runner.bestPolicy.valid ? "yes" : "no") << "\n";
        out << "  last TD error: " << runner.model.lastTD << "\n\n";

        out << "recent " << stats.count << " episode aggregate:\n";
        out << "  successes: " << stats.successes << " (" << std::setprecision(1) << SafePercent(stats.successes, stats.count) << "%)\n";
        out << "  timeouts: " << stats.timeouts << " (" << SafePercent(stats.timeouts, stats.count) << "%)\n";
        out << "  stucks: " << stats.stucks << " (" << SafePercent(stats.stucks, stats.count) << "%)\n";
        out << "  average reward: " << std::setprecision(3) << SafeAverage(stats.rewardTotal, stats.count) << "\n";
        out << "  average episode time: " << SafeAverage(stats.timeTotal, stats.count) << "s\n";
        out << "  average successful time: " << SafeAverage(stats.successTimeTotal, stats.successes) << "s\n";
        out << "  best recent success time: " << (stats.bestSuccessTime < 9998.0f ? stats.bestSuccessTime : 0.0f) << "s\n";
        out << "  average distance remaining: " << SafeAverage(stats.distanceTotal, stats.count) << "m\n";
        out << "  average wall hits: " << SafeAverage((float)stats.wallHits, stats.count) << "\n";
        out << "  average collision cost: " << SafeAverage(stats.collisionTotal, stats.count) << "\n";
        out << "  average decisions: " << SafeAverage((float)stats.decisions, stats.count) << "\n\n";

        WriteModelRecommendations(out, runner, stats, targetSpeedScale, achievedSpeedScale);
        out << "\n";

        WriteModelParameterSnapshot(out, runner.model);
        out << "\n";

        out << "lifetime action usage:\n";
        WriteActionUsage(out, runner.lifetimeActionCounts);
        out << "\nrecent action usage:\n";
        WriteActionUsage(out, stats.actionCounts);
        out << "\n";

        WriteRecentEpisodeLog(out, runner);
        out << "\n\n";
    }

    out << "REPORT-LEVEL NEXT STEPS\n";
    out << "  - Compare best time, recent average successful time, and manual personal best. If best time is much lower than recent average, the model is not reliably that fast.\n";
    out << "  - If bad-looking runs keep high reward, reduce shaping reward that can be farmed without finishing and make terminal success dominate the score.\n";
    out << "  - If 250000x reports only a few hundred actual x, the bottleneck is CPU simulation, not the rest of the computer. Keep the UI responsive by trusting actual speed and reducing active model work.\n";
    out << "  - Use this report after each tuning pass. Look for lower wall/collision averages, higher recent success rate, and a smaller gap between best run and average success time.\n";

    return true;
}

int main() {
    SetConfigFlags(FLAG_MSAA_4X_HINT);
    InitWindow(SCREEN_WIDTH, SCREEN_HEIGHT, "PRO FPS FEEL - RL COURSE TRAINER");
    SetTargetFPS(144);
    DisableCursor();

    Shader ppShader = LoadShaderFromMemory(0, postProcessCode);
    RenderTexture2D renderTarget = LoadRenderTexture(SCREEN_WIDTH, SCREEN_HEIGHT);

    Player player;
    HitMarker hit;
    KatanaFX katana;
    World world;
    RLTrainer trainer(world.startPoint, world.goalPoint);
    Camera3D trainingCamera = MakeRLTrainingCamera();
    ManualRunReplay manualReplay;
    ManualRunReplay manualBestReplay;

    bool rlAutoplay = true;
    bool rlTrainingView = true;
    bool bestRunView = false;
    bool turboTraining = false;
    bool soloTraining = false;
    bool manualRaceStarted = false;
    bool uiCursorEnabled = false;
    int trainingSpeedIndex = 0;
    int trainingStepsLastFrame = 0;
    float trainingSimSecondsLastFrame = 0.0f;
    float trainingPendingSimSeconds = 0.0f;
    float trainingActualSpeedScale = 0.0f;
    float trainingActualSimWindow = 0.0f;
    float trainingActualWallWindow = 0.0f;
    float bestReplayTimer = 0.0f;
    float reportFlashTimer = 0.0f;
    std::string reportMessage;

    player.ResetRL(world.startPoint, world.goalPoint);
    trainer.ResetEpisodes(world.obstacles);

    float spread = 0.0f;

    while (!WindowShouldClose()) {
        SetTargetFPS((rlAutoplay && rlTrainingView) ? 120 : 144);
        float dt = GetFrameTime();

        if (IsKeyPressed(KEY_T)) {
            rlAutoplay = !rlAutoplay;
            katana.active = false;
            if (!rlAutoplay) {
                player.ResetRL(world.startPoint, world.goalPoint);
                manualReplay.Reset();
                manualRaceStarted = false;
                spread = 0.0f;
            }
        }

        if (IsKeyPressed(KEY_V)) {
            rlTrainingView = !rlTrainingView;
        }

        if (IsKeyPressed(KEY_TAB) || IsKeyPressed(KEY_RIGHT)) {
            trainer.SelectNext();
            bestReplayTimer = 0.0f;
        }

        if (IsKeyPressed(KEY_LEFT)) {
            trainer.SelectPrevious();
            bestReplayTimer = 0.0f;
        }

        bool wantsUICursor = rlAutoplay && rlTrainingView;
        if (wantsUICursor != uiCursorEnabled) {
            if (wantsUICursor) EnableCursor();
            else DisableCursor();
            uiCursorEnabled = wantsUICursor;
        }

        if (rlAutoplay) {
            int previousSpeedIndex = trainingSpeedIndex;
            trainingSpeedIndex = HandleTrainingSpeedButtons(trainingSpeedIndex);
            if (trainingSpeedIndex != previousSpeedIndex) {
                trainingPendingSimSeconds = 0.0f;
                trainingActualSimWindow = 0.0f;
                trainingActualWallWindow = 0.0f;
            }

            bool previousBestRunView = bestRunView;
            bestRunView = HandleBestRunButton(bestRunView);
            if (bestRunView != previousBestRunView) bestReplayTimer = 0.0f;
            if (IsKeyPressed(KEY_H)) {
                turboTraining = !turboTraining;
                trainingPendingSimSeconds = 0.0f;
            }
            soloTraining = HandleSoloTrainingButton(soloTraining);
        } else {
            bestRunView = false;
            turboTraining = false;
            soloTraining = false;
            bestReplayTimer = 0.0f;
        }

        if (IsKeyPressed(KEY_R)) {
            if (rlAutoplay) {
                trainer.ResetEpisodes(world.obstacles);
                trainingPendingSimSeconds = 0.0f;
                trainingActualSimWindow = 0.0f;
                trainingActualWallWindow = 0.0f;
            } else {
                player.ResetRL(world.startPoint, world.goalPoint);
                manualReplay.Reset();
                manualRaceStarted = false;
            }
        }

        if (rlAutoplay) {
            trainingStepsLastFrame = AdvanceRLTraining(
                trainer,
                dt,
                world.obstacles,
                TRAINING_SPEED_OPTIONS[trainingSpeedIndex],
                turboTraining,
                soloTraining,
                trainingPendingSimSeconds,
                trainingSimSecondsLastFrame
            );
            trainingActualSimWindow += trainingSimSecondsLastFrame;
            trainingActualWallWindow += dt;
            if (trainingActualWallWindow >= 0.35f) {
                trainingActualSpeedScale = trainingActualSimWindow / fmaxf(0.001f, trainingActualWallWindow);
                trainingActualSimWindow = 0.0f;
                trainingActualWallWindow = 0.0f;
            }
        } else {
            trainingStepsLastFrame = 0;
            trainingSimSecondsLastFrame = 0.0f;
            trainingPendingSimSeconds = 0.0f;
            trainingActualSpeedScale = 0.0f;
            trainingActualSimWindow = 0.0f;
            trainingActualWallWindow = 0.0f;

            if (IsKeyPressed(KEY_Q) && katana.cooldown <= 0.0f) {
                katana.Trigger();
            }

            bool manualW = IsKeyDown(KEY_W);
            bool manualA = IsKeyDown(KEY_A);
            bool manualS = IsKeyDown(KEY_S);
            bool manualD = IsKeyDown(KEY_D);
            bool manualJumpPressed = IsKeyPressed(KEY_SPACE);
            bool manualDashPressed = IsKeyPressed(KEY_E);
            bool manualSlide = IsKeyDown(KEY_LEFT_CONTROL) || IsKeyDown(KEY_LEFT_SUPER) || IsKeyDown(KEY_C);
            bool manualStartInput =
                manualW ||
                manualA ||
                manualS ||
                manualD ||
                manualJumpPressed ||
                manualDashPressed ||
                IsKeyDown(KEY_LEFT_SHIFT);

            if (!manualRaceStarted && manualStartInput) {
                manualRaceStarted = true;
            }

            float playerTimeBeforeUpdate = player.rlEpisodeTime;
            Player::RLState manualStateBefore = player.GetRLState(world.obstacles);
            int manualAction = InferManualRLAction(
                player,
                manualW,
                manualA,
                manualS,
                manualD,
                manualDashPressed && !katana.active,
                manualJumpPressed,
                manualSlide
            );
            player.Update(
                dt,
                world.obstacles,
                katana.active,
                katana.active,
                katana.Normalized(),
                false
            );
            if (!manualRaceStarted) {
                player.rlEpisodeTime = 0.0f;
                player.rlReward = 0.0f;
                player.rlEpisodeReward = 0.0f;
                player.rlPreviousTargetDistance = player.rlStartTargetDistance;
                player.rlBestTargetDistance = player.rlStartTargetDistance;
                player.rlNoProgressTimer = 0.0f;
                player.rlAreaTimer = 0.0f;
                player.rlAreaAnchor = player.position;
                player.rlDone = false;
                player.rlSucceeded = false;
                player.rlTimedOut = false;
                player.rlStuck = false;
            } else {
                if (playerTimeBeforeUpdate <= 0.0f && player.rlEpisodeTime > 0.0f) {
                    manualReplay.Reset();
                }
                Player::RLState manualStateAfter = player.GetRLState(world.obstacles);
                manualReplay.Record(player, manualStateBefore, manualAction, player.rlReward, manualStateAfter);
                if (manualReplay.IsBetterThan(manualBestReplay)) {
                    manualBestReplay = manualReplay;
                    trainer.LearnFromManualReplay(manualBestReplay.experiences, manualBestReplay.finishTime);
                    reportMessage = TextFormat("MANUAL BEST FED TO AI: %.2fs", manualBestReplay.finishTime);
                    reportFlashTimer = 2.5f;
                }
            }

            if (katana.active && !katana.didHitCheck && katana.Normalized() >= 0.25f) {
                katana.didHitCheck = true;
                for (const auto& box : world.obstacles) {
                    if (BoxWithinMeleeRange(player.camera, box)) {
                        hit.Trigger();
                        break;
                    }
                }
            }

            if (IsMouseButtonPressed(MOUSE_LEFT_BUTTON)) {
                spread += 12.0f;

                Vector3 shotDir = Vector3Normalize(Vector3Subtract(player.camera.target, player.camera.position));
                Ray ray = { player.camera.position, shotDir };

                for (const auto& box : world.obstacles) {
                    BoundingBox bb = {
                        Vector3Subtract(box.center, box.half),
                        Vector3Add(box.center, box.half)
                    };

                    if (GetRayCollisionBox(ray, bb).hit) {
                        hit.Trigger();
                        break;
                    }
                }
            }
        }

        if (IsKeyPressed(KEY_P)) {
            bool savedReport = WriteTrainingReport(
                "rl_training_report.txt",
                trainer,
                world,
                trainingSpeedIndex,
                trainingActualSpeedScale,
                trainingPendingSimSeconds,
                trainingStepsLastFrame,
                trainingSimSecondsLastFrame,
                turboTraining,
                soloTraining,
                manualBestReplay
            );
            reportMessage = savedReport
                ? "REPORT SAVED: rl_training_report.txt"
                : "REPORT FAILED: rl_training_report.txt";
            reportFlashTimer = 2.5f;
        }

        if (reportFlashTimer > 0.0f) {
            reportFlashTimer = fmaxf(0.0f, reportFlashTimer - dt);
        }

        hit.Update(dt);
        katana.Update(dt);

        const Player& viewPlayer = rlAutoplay ? trainer.ActiveRunner().player : player;
        const RLRunner& activeRunner = trainer.ActiveRunner();
        bool bestRunCameraView = rlAutoplay && bestRunView && !rlTrainingView && activeRunner.HasBestRun();

        if (bestRunCameraView) {
            bestReplayTimer += dt;
            float duration = activeRunner.BestRunDuration();
            if (duration > 0.001f) {
                while (bestReplayTimer > duration) bestReplayTimer -= duration;
            }
        } else if (!bestRunView) {
            bestReplayTimer = 0.0f;
        }

        Camera3D replayCamera = bestRunCameraView ? activeRunner.BestRunCamera(bestReplayTimer) : viewPlayer.camera;

        float hSpeed = viewPlayer.GetHorizontalSpeed();
        float targetSpread = hSpeed * 1.35f + (!viewPlayer.onGround ? 12.0f : 0.0f);
        if (viewPlayer.isDashing) targetSpread += 6.0f;
        if (!rlAutoplay && katana.active) targetSpread += 4.0f;
        spread = Lerp(spread, targetSpread, dt * 15.0f);

        bool lightweightTrainingView = rlAutoplay && rlTrainingView && (turboTraining || TRAINING_SPEED_OPTIONS[trainingSpeedIndex] >= 1000.0f);
        bool skipWorldRender = lightweightTrainingView;

        if (!skipWorldRender) {
            BeginTextureMode(renderTarget);
                ClearBackground((Color){20, 25, 30, 255});
                if (rlAutoplay && rlTrainingView) {
                    BeginMode3D(trainingCamera);
                        world.Draw(trainingCamera);
                        DrawLine3D(
                            { world.startPoint.x, 0.12f, world.startPoint.z },
                            { world.goalPoint.x, 0.12f, world.goalPoint.z },
                            Fade(GOLD, 0.75f)
                        );
                        DrawSphere({ world.startPoint.x, 0.4f, world.startPoint.z }, 1.1f, GREEN);
                        if (bestRunView) trainer.DrawBestRuns3D();
                        else trainer.Draw3D();
                    EndMode3D();
                } else {
                    BeginMode3D(replayCamera);
                        world.Draw(replayCamera);
                        if (!rlAutoplay) {
                            float manualRaceTime = manualRaceStarted ? player.rlEpisodeTime : 0.0f;
                            DrawManualBestGhost(activeRunner, player.position, world.startPoint, manualRaceTime);
                            DrawManualCompletedGhost(manualBestReplay, player.position, manualRaceTime);
                            katana.Draw3D(replayCamera, player.currentBob);
                        }
                    EndMode3D();
                }
            EndTextureMode();
        }

        BeginDrawing();
            ClearBackground(skipWorldRender ? (Color){10, 12, 16, 255} : BLACK);

            if (!skipWorldRender) {
                BeginShaderMode(ppShader);
                    DrawTextureRec(
                        renderTarget.texture,
                        (Rectangle){0, 0, (float)renderTarget.texture.width, (float)-renderTarget.texture.height},
                        (Vector2){0, 0},
                        WHITE
                    );
                EndShaderMode();
            }

            int cx = SCREEN_WIDTH / 2;
            int cy = SCREEN_HEIGHT / 2;

            if (rlAutoplay && rlTrainingView) {
                if (!skipWorldRender) DrawRLRunnerLabels(trainer, trainingCamera, bestRunView);
                DrawRLTrainingOverlay(trainer, trainingSpeedIndex, trainingStepsLastFrame, trainingSimSecondsLastFrame, trainingPendingSimSeconds, trainingActualSpeedScale, bestRunView, turboTraining, soloTraining);
            } else {
                DrawCrosshair(cx, cy, spread);
                if (!rlAutoplay && hit.active) DrawHitmarker(cx, cy, hit.timer, hit.maxTime);
            }

            if ((!rlAutoplay || !rlTrainingView) && viewPlayer.dashVisualTimer > 0.0f) {
                float a = viewPlayer.dashVisualTimer / DASH_TIME;
                DrawRectangle(0, 0, SCREEN_WIDTH, SCREEN_HEIGHT, Fade(WHITE, a * 0.14f));
            }

            if (!rlAutoplay && katana.active) {
                float slashFlash = 1.0f - katana.Normalized();
                DrawRectangle(0, 0, SCREEN_WIDTH, SCREEN_HEIGHT, Fade((Color){180, 220, 255, 255}, slashFlash * 0.05f));
            }

            if (!rlAutoplay || !rlTrainingView) {
                DrawStatusBar(20, SCREEN_HEIGHT - 70, 300, 25, viewPlayer.health, MAX_HEALTH, RED, "HEALTH");

                float dashStatus = 1.0f - Clamp(viewPlayer.dashCooldown / DASH_COOLDOWN_TIME, 0.0f, 1.0f);
                DrawStatusBar(
                    20,
                    SCREEN_HEIGHT - 40,
                    300,
                    20,
                    dashStatus,
                    1.0f,
                    (viewPlayer.dashCooldown <= 0.0f ? GREEN : GOLD),
                    rlAutoplay ? "DASH (RL)" : "DASH (E)"
                );

                DrawText(TextFormat("MODE: %s (T to toggle)", rlAutoplay ? "RL SELECTED RUNNER" : "MANUAL"), 20, 20, 20, rlAutoplay ? GREEN : LIGHTGRAY);
                DrawText(TextFormat("SPEED: %02.0f", hSpeed), 20, 46, 20, WHITE);
                DrawText(TextFormat("RL STEP REWARD: %.3f", viewPlayer.rlReward), 20, 72, 20, YELLOW);
                DrawText(TextFormat("EPISODE REWARD: %.3f", viewPlayer.rlEpisodeReward), 20, 98, 20, YELLOW);
                DrawText(TextFormat("EPISODE TIME: %.1f / %.1f", viewPlayer.rlEpisodeTime, RL_MAX_EPISODE_TIME), 20, 124, 20, YELLOW);
                DrawText(TextFormat("DIST TO GOAL: %.1f", RLHorizontalDistance(viewPlayer.position, world.goalPoint)), 20, 150, 20, WHITE);
                DrawText(TextFormat("OBS SIZE: %d", Player::RL_STATE_SIZE), 20, 176, 20, SKYBLUE);

                if (rlAutoplay) {
                    const RLRunner& active = trainer.ActiveRunner();
                    if (bestRunCameraView) {
                        DrawText(TextFormat("BEST RUN CAMERA: %s  %.1f / %.1fs", active.model.name.c_str(), bestReplayTimer, active.BestRunDuration()), 20, 202, 16, active.model.color);
                        DrawText("V = overview  B = live/best  TAB/arrows = select", 20, 224, 16, LIGHTGRAY);
                        DrawAIInputOverlay(active, bestReplayTimer);
                    } else if (bestRunView && !active.HasBestRun()) {
                        DrawText(TextFormat("BEST RUN CAMERA: %s has no successful run yet", active.model.name.c_str()), 20, 202, 16, ORANGE);
                        DrawText("V = overview  B = live/best  keep training for a saved run", 20, 224, 16, LIGHTGRAY);
                    } else {
                        DrawText(TextFormat("MODEL: %s  ACTION: %s", active.model.name.c_str(), active.CurrentActionName()), 20, 202, 16, active.model.color);
                        DrawText("V = overview  TAB/arrows = select  R = reset runs", 20, 224, 16, LIGHTGRAY);
                    }
                } else {
                    DrawText("Manual: WASD + Mouse + Space + Shift + E + Q", 20, 202, 16, LIGHTGRAY);
                    if (activeRunner.HasBestRun()) {
                        float manualRaceTime = manualRaceStarted ? player.rlEpisodeTime : 0.0f;
                        RLReplayFrame ghost = manualRaceStarted ? activeRunner.BestRunFrame(manualRaceTime) : (RLReplayFrame){ world.startPoint, player.camera, 0.0f, -1 };
                        float playerDist = RLHorizontalDistance(player.position, world.goalPoint);
                        float ghostDist = RLHorizontalDistance(ghost.position, world.goalPoint);
                        float ghostLead = playerDist - ghostDist;
                        const char* leadLabel = ghostLead >= 0.0f ? "GHOST AHEAD" : "PLAYER AHEAD";
                        DrawText(
                            TextFormat("GHOST: %s best %.1fs  %s %.1fm", activeRunner.model.name.c_str(), activeRunner.BestRunDuration(), leadLabel, fabsf(ghostLead)),
                            20,
                            224,
                            16,
                            activeRunner.model.color
                        );
                        DrawText(manualRaceStarted ? "TAB/arrows = switch ghost  R = restart race" : "Race starts when you move", 20, 246, 16, manualRaceStarted ? LIGHTGRAY : GOLD);
                        if (manualBestReplay.HasRun()) {
                            DrawText(TextFormat("YOUR BEST: %.1fs  yellow ghost saved", manualBestReplay.finishTime), 20, 268, 16, GOLD);
                        }
                    } else {
                        DrawText(TextFormat("GHOST: %s has no best run yet", activeRunner.model.name.c_str()), 20, 224, 16, ORANGE);
                        DrawText("Train a successful run first, then switch back to manual", 20, 246, 16, LIGHTGRAY);
                    }
                }

                if (viewPlayer.isWallRunning) DrawText("WALLRUNNING", cx - 60, cy + 60, 20, GREEN);
                if (viewPlayer.isDashing) DrawText("DASH", cx - 25, cy + 84, 20, SKYBLUE);
                if (!rlAutoplay && katana.active) DrawText("SLASH", cx - 28, cy + 108, 20, WHITE);
                if (rlAutoplay) DrawText("RL CONTROL ACTIVE", cx - 82, cy + 132, 20, YELLOW);

                if (rlAutoplay && viewPlayer.rlDone) {
                    DrawText("EPISODE COMPLETE - RESETTING", cx - 140, cy - 110, 20, GOLD);
                }
            }

            if (reportFlashTimer > 0.0f && !reportMessage.empty()) {
                Color reportColor = reportMessage.find("FAILED") == std::string::npos ? GREEN : ORANGE;
                int reportWidth = MeasureText(reportMessage.c_str(), 18);
                DrawRectangle(
                    SCREEN_WIDTH / 2 - reportWidth / 2 - 14,
                    18,
                    reportWidth + 28,
                    32,
                    (Color){ 8, 10, 12, 215 }
                );
                DrawRectangleLines(
                    SCREEN_WIDTH / 2 - reportWidth / 2 - 14,
                    18,
                    reportWidth + 28,
                    32,
                    Fade(reportColor, 0.65f)
                );
                DrawText(reportMessage.c_str(), SCREEN_WIDTH / 2 - reportWidth / 2, 25, 18, reportColor);
            }

        EndDrawing();
    }

    WriteTrainingReport(
        "rl_training_report.txt",
        trainer,
        world,
        trainingSpeedIndex,
        trainingActualSpeedScale,
        trainingPendingSimSeconds,
        trainingStepsLastFrame,
        trainingSimSecondsLastFrame,
        turboTraining,
        soloTraining,
        manualBestReplay
    );

    UnloadShader(ppShader);
    UnloadRenderTexture(renderTarget);
    CloseWindow();
    return 0;
}
