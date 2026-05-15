#include "raylib.h"
#include "raymath.h"
#include "rlgl.h"
#include <algorithm>
#include <cctype>
#include <ctime>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <string>
#include <system_error>
#include <vector>
#include <cmath>
#include <cstdlib>
#include "structs/Box.h"
#include "structs/HitMarker.h"
#include "structs/SlashParticle3D.h"
#include "structs/KatanaFX.h"
#include "structs/Player.h"
#include "structs/RLAgent.h"
#include "structs/World.h"

namespace fs = std::filesystem;

const char* RL_MODEL_DIRECTORY = "rl_models";
const char* RL_MODEL_EXTENSION = ".rlmodel";
const char* RL_MODEL_MAGIC = "RL_MODEL_V1";
const float RL_AUTOSAVE_INTERVAL = 3.0f;
const char* RL_TRAINING_REPORT_FILE = "rl_training_report.txt";
const float RL_REPORT_AUTOSAVE_INTERVAL = 60.0f;
const char* MANUAL_RUN_DIRECTORY = "manual_runs";
const char* MANUAL_RUN_EXTENSION = ".manualrun";
const char* MANUAL_RUN_MAGIC = "MANUAL_RUN_V1";
const int MANUAL_RUN_KEEP_COUNT = 2;

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

Camera3D MakeSpectatorRaceCamera(const Vector3& start, const Vector3& goal) {
    Camera3D camera = {};
    Vector3 center = Vector3Lerp(start, goal, 0.5f);
    camera.position = { center.x - 24.0f, center.y + 22.0f, center.z + 34.0f };
    camera.target = { center.x, center.y + 1.5f, center.z };
    camera.up = { 0.0f, 1.0f, 0.0f };
    camera.fovy = 58.0f;
    camera.projection = CAMERA_PERSPECTIVE;
    return camera;
}

void UpdateSpectatorFreeCamera(Camera3D& camera, float dt) {
    Vector3 look = Vector3Subtract(camera.target, camera.position);
    if (Vector3Length(look) <= 0.001f) look = { 0.0f, 0.0f, -1.0f };
    Vector3 lookDir = Vector3Normalize(look);

    float yaw = atan2f(lookDir.x, lookDir.z) * RAD2DEG;
    float pitch = asinf(Clamp(lookDir.y, -1.0f, 1.0f)) * RAD2DEG;
    Vector2 mouse = GetMouseDelta();
    yaw -= mouse.x * MOUSE_SENSITIVITY;
    pitch = Clamp(pitch - mouse.y * MOUSE_SENSITIVITY, -86.0f, 86.0f);

    float yawRad = yaw * DEG2RAD;
    float pitchRad = pitch * DEG2RAD;
    Vector3 forward = {
        sinf(yawRad) * cosf(pitchRad),
        sinf(pitchRad),
        cosf(yawRad) * cosf(pitchRad)
    };
    forward = Vector3Normalize(forward);

    Vector3 flatForward = { forward.x, 0.0f, forward.z };
    if (Vector3Length(flatForward) <= 0.001f) flatForward = { 0.0f, 0.0f, -1.0f };
    flatForward = Vector3Normalize(flatForward);
    Vector3 right = { -flatForward.z, 0.0f, flatForward.x };

    Vector3 move = { 0.0f, 0.0f, 0.0f };
    if (IsKeyDown(KEY_W)) move = Vector3Add(move, flatForward);
    if (IsKeyDown(KEY_S)) move = Vector3Subtract(move, flatForward);
    if (IsKeyDown(KEY_D)) move = Vector3Add(move, right);
    if (IsKeyDown(KEY_A)) move = Vector3Subtract(move, right);
    if (IsKeyDown(KEY_SPACE)) move.y += 1.0f;
    if (IsKeyDown(KEY_LEFT_CONTROL) || IsKeyDown(KEY_LEFT_SUPER)) move.y -= 1.0f;

    if (Vector3Length(move) > 0.001f) {
        float speed = IsKeyDown(KEY_LEFT_SHIFT) ? 42.0f : 18.0f;
        Vector3 delta = Vector3Scale(Vector3Normalize(move), speed * dt);
        camera.position = Vector3Add(camera.position, delta);
    }

    camera.target = Vector3Add(camera.position, forward);
}

void CameraLookAngles(const Camera3D& camera, float& yaw, float& pitch) {
    Vector3 look = Vector3Subtract(camera.target, camera.position);
    if (Vector3Length(look) <= 0.001f) look = { 0.0f, 0.0f, -1.0f };
    look = Vector3Normalize(look);
    yaw = atan2f(look.x, look.z) * RAD2DEG;
    pitch = asinf(Clamp(look.y, -1.0f, 1.0f)) * RAD2DEG;
}

Vector3 DirectionFromLookAngles(float yaw, float pitch) {
    float yawRad = yaw * DEG2RAD;
    float pitchRad = pitch * DEG2RAD;
    return Vector3Normalize({
        sinf(yawRad) * cosf(pitchRad),
        sinf(pitchRad),
        cosf(yawRad) * cosf(pitchRad)
    });
}

void UpdateSpectatorPovFreeLook(float& yaw, float& pitch) {
    Vector2 mouse = GetMouseDelta();
    yaw -= mouse.x * MOUSE_SENSITIVITY;
    pitch = Clamp(pitch - mouse.y * MOUSE_SENSITIVITY, -86.0f, 86.0f);
}

Camera3D SpectatorPovFreeLookCamera(Camera3D baseCamera, float yaw, float pitch) {
    baseCamera.target = Vector3Add(baseCamera.position, DirectionFromLookAngles(yaw, pitch));
    baseCamera.up = { 0.0f, 1.0f, 0.0f };
    return baseCamera;
}

const char* SpectatorRaceCameraModeName(int cameraMode) {
    if (cameraMode == 1) return "PLAYER POV";
    if (cameraMode == 2) return "AI POV";
    return "FREECAM";
}

const int TRAINING_SPEED_OPTION_COUNT = 11;
const float TRAINING_SPEED_OPTIONS[TRAINING_SPEED_OPTION_COUNT] = { 1.0f, 10.0f, 100.0f, 1000.0f, 2500.0f, 5000.0f, 10000.0f, 25000.0f, 50000.0f, 100000.0f, 250000.0f };
const char* TRAINING_SPEED_LABELS[TRAINING_SPEED_OPTION_COUNT] = { "1x", "10x", "100x", "1000x", "2500x", "5000x", "10000x", "25000x", "50000x", "100000x", "250000x" };
const int SPECTATOR_RACE_SPEED_OPTION_COUNT = 5;
const float SPECTATOR_RACE_SPEED_OPTIONS[SPECTATOR_RACE_SPEED_OPTION_COUNT] = { 0.25f, 0.5f, 1.0f, 2.0f, 4.0f };

struct RLModelLibrary {
    std::vector<std::string> files;
    int selectedFile = 0;
    std::string status = "MODEL AUTOSAVE READY";

    bool HasFiles() const {
        return !files.empty();
    }

    std::string SelectedPath() const {
        if (files.empty()) return "";
        int index = std::max(0, std::min(selectedFile, (int)files.size() - 1));
        return files[index];
    }

    std::string SelectedFileName() const {
        std::string path = SelectedPath();
        return path.empty() ? "no saved model files yet" : fs::path(path).filename().string();
    }
};

struct RLPersistenceState {
    float autosaveTimer = 0.0f;
    int savedStartedEpisodes = -1;
    int savedFinishedEpisodes = -1;
    int savedBestFrameCount = -1;
    int savedManualGuidance = -1;
};

enum class RLModelFileAction {
    None,
    Previous,
    Next,
    Load,
    SaveSnapshot
};

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

Rectangle ModelFilePrevButtonRect() {
    return { 24.0f, 520.0f, 34.0f, 30.0f };
}

Rectangle ModelFileLoadButtonRect() {
    return { 64.0f, 520.0f, 70.0f, 30.0f };
}

Rectangle ModelFileSaveButtonRect() {
    return { 140.0f, 520.0f, 92.0f, 30.0f };
}

Rectangle ModelFileNextButtonRect() {
    return { 238.0f, 520.0f, 34.0f, 30.0f };
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

RLModelFileAction HandleModelFileControls() {
    Vector2 mouse = GetMousePosition();

    if (CheckCollisionPointRec(mouse, ModelFilePrevButtonRect()) && IsMouseButtonPressed(MOUSE_LEFT_BUTTON)) {
        return RLModelFileAction::Previous;
    }
    if (CheckCollisionPointRec(mouse, ModelFileNextButtonRect()) && IsMouseButtonPressed(MOUSE_LEFT_BUTTON)) {
        return RLModelFileAction::Next;
    }
    if (CheckCollisionPointRec(mouse, ModelFileLoadButtonRect()) && IsMouseButtonPressed(MOUSE_LEFT_BUTTON)) {
        return RLModelFileAction::Load;
    }
    if (CheckCollisionPointRec(mouse, ModelFileSaveButtonRect()) && IsMouseButtonPressed(MOUSE_LEFT_BUTTON)) {
        return RLModelFileAction::SaveSnapshot;
    }

    if (IsKeyPressed(KEY_LEFT_BRACKET)) return RLModelFileAction::Previous;
    if (IsKeyPressed(KEY_RIGHT_BRACKET)) return RLModelFileAction::Next;
    if (IsKeyPressed(KEY_L)) return RLModelFileAction::Load;
    if (IsKeyPressed(KEY_S)) return RLModelFileAction::SaveSnapshot;

    return RLModelFileAction::None;
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

std::string TruncateTextToWidth(const std::string& text, int maxWidth, int fontSize) {
    if (MeasureText(text.c_str(), fontSize) <= maxWidth) return text;

    std::string suffix = text;
    while (!suffix.empty() && MeasureText(("..." + suffix).c_str(), fontSize) > maxWidth) {
        suffix.erase(suffix.begin());
    }

    return "..." + suffix;
}

void DrawRLTrainingOverlay(
    const RLTrainer& trainer,
    int speedIndex,
    int trainingSteps,
    float simulatedSeconds,
    float pendingSimSeconds,
    float achievedSpeedScale,
    bool bestRunView,
    bool turboTraining,
    bool soloTraining,
    const RLModelLibrary& modelLibrary
) {
    const RLRunner& active = trainer.ActiveRunner();
    const int best = trainer.BestRunnerIndex();

    DrawRectangle(12, 12, 390, 556, (Color){ 8, 10, 12, 210 });
    DrawRectangleLines(12, 12, 390, 556, Fade(WHITE, 0.24f));
    DrawText(bestRunView ? "BEST RUN VIEW" : "RL TRAINING VIEW", 24, 24, 20, GREEN);
    DrawText("T manual  V camera  B best  H turbo  M solo", 24, 50, 14, LIGHTGRAY);
    DrawText("TAB/arrows select  1-0/- speed  P report (auto 60s)", 24, 68, 14, LIGHTGRAY);
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

    DrawText("MODEL FILES: [ ] browse  L load  S snapshot", 24, 472, 13, LIGHTGRAY);
    Color statusColor = modelLibrary.status.find("FAILED") == std::string::npos ? GREEN : ORANGE;
    std::string statusText = TruncateTextToWidth(modelLibrary.status, 344, 13);
    DrawText(statusText.c_str(), 24, 490, 13, statusColor);
    std::string selectedFile = TruncateTextToWidth(modelLibrary.SelectedFileName(), 344, 13);
    DrawText(selectedFile.c_str(), 24, 506, 13, SKYBLUE);

    Rectangle prevButton = ModelFilePrevButtonRect();
    Rectangle loadButton = ModelFileLoadButtonRect();
    Rectangle saveButton = ModelFileSaveButtonRect();
    Rectangle nextButton = ModelFileNextButtonRect();
    Vector2 modelMouse = GetMousePosition();

    auto drawModelButton = [&](Rectangle rect, const char* label, bool enabled) {
        bool hot = enabled && CheckCollisionPointRec(modelMouse, rect);
        Color fill = enabled
            ? (hot ? (Color){ 72, 78, 84, 235 } : (Color){ 34, 38, 44, 235 })
            : (Color){ 24, 26, 30, 180 };
        Color text = enabled ? WHITE : Fade(WHITE, 0.38f);
        DrawRectangleRounded(rect, 0.18f, 6, fill);
        DrawRectangleRoundedLines(rect, 0.18f, 6, Fade(WHITE, enabled ? 0.22f : 0.10f));
        int textWidth = MeasureText(label, 15);
        DrawText(label, (int)(rect.x + (rect.width - textWidth) * 0.5f), (int)(rect.y + 8), 15, text);
    };

    drawModelButton(prevButton, "<", modelLibrary.HasFiles());
    drawModelButton(loadButton, "LOAD", modelLibrary.HasFiles());
    drawModelButton(saveButton, "SNAPSHOT", true);
    drawModelButton(nextButton, ">", modelLibrary.HasFiles());

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

        const char* bestText = (runner.bestTime < 9998.0f) ? TextFormat("best %.2fs", runner.bestTime) : "best --";
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

struct ManualInputFrame {
    float time = 0.0f;
    bool w = false;
    bool a = false;
    bool s = false;
    bool d = false;
    bool jump = false;
    bool dash = false;
    bool slide = false;
    bool slash = false;
    bool isDashing = false;
    bool isWallRunning = false;
    float dashCooldown = 0.0f;
    float dashVisualTimer = 0.0f;
};

ManualInputFrame ManualInputFromReplayFrame(const RLReplayFrame& frame) {
    ManualInputFrame input;
    input.time = frame.time;

    if (frame.action >= 0 && frame.action < (int)RLActions().size()) {
        const RLDiscreteAction& action = RLActions()[frame.action];
        input.w = action.strength > 0.1f && fabsf(action.angleDegrees) < 105.0f;
        input.a = action.angleDegrees < -8.0f;
        input.d = action.angleDegrees > 8.0f;
        input.s = fabsf(action.angleDegrees) > 105.0f;
        input.jump = action.jump;
        input.dash = action.dash;
        input.slide = action.slide;
        input.isDashing = action.dash;
    }

    return input;
}

struct ManualRunReplay {
    std::vector<RLReplayFrame> frames;
    std::vector<Vector3> trail;
    std::vector<RLExperience> experiences;
    std::vector<ManualInputFrame> inputs;
    bool finished = false;
    bool savedToLibrary = false;
    float finishTime = 0.0f;
    float lastFrameTime = 0.0f;
    std::string sourcePath;
    std::string label;

    void Reset() {
        frames.clear();
        trail.clear();
        experiences.clear();
        inputs.clear();
        finished = false;
        savedToLibrary = false;
        finishTime = 0.0f;
        lastFrameTime = 0.0f;
        sourcePath.clear();
        label.clear();
    }

    void Record(
        const Player& player,
        const Player::RLState& state,
        int action,
        float reward,
        const Player::RLState& nextState,
        const ManualInputFrame& input
    ) {
        if (finished) return;
        if (player.rlDone && !player.rlSucceeded) return;

        ManualInputFrame savedInput = input;
        savedInput.time = player.rlEpisodeTime;
        inputs.push_back(savedInput);

        if (frames.empty() || player.rlDone || player.rlEpisodeTime - lastFrameTime >= 1.0f / 30.0f) {
            frames.push_back({ player.position, player.camera, player.rlEpisodeTime, action });
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

    void SeedStartFrame(
        const Vector3& startPosition,
        const Camera3D& startCamera,
        int action,
        const ManualInputFrame& input
    ) {
        if (!frames.empty() || !trail.empty() || !inputs.empty()) return;

        frames.push_back({ startPosition, startCamera, 0.0f, action });
        trail.push_back(startPosition);

        ManualInputFrame startInput = input;
        startInput.time = 0.0f;
        startInput.isDashing = false;
        startInput.isWallRunning = false;
        startInput.dashCooldown = 0.0f;
        startInput.dashVisualTimer = 0.0f;
        inputs.push_back(startInput);
        lastFrameTime = 0.0f;
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

    ManualInputFrame ManualInputAt(float replayTime) const {
        if (inputs.empty()) return ManualInputFromReplayFrame(FrameAt(replayTime));
        if (inputs.size() == 1 || replayTime <= inputs.front().time) return inputs.front();
        if (replayTime >= inputs.back().time) return inputs.back();

        int nextIndex = 1;
        while (nextIndex < (int)inputs.size() && inputs[nextIndex].time < replayTime) {
            nextIndex += 1;
        }

        const ManualInputFrame& before = inputs[nextIndex - 1];
        const ManualInputFrame& after = inputs[nextIndex];
        return (replayTime - before.time <= after.time - replayTime) ? before : after;
    }

    void BackfillInputsFromReplayData() {
        if (!inputs.empty() || frames.empty()) return;

        inputs.reserve(frames.size());
        for (const RLReplayFrame& frame : frames) {
            RLReplayFrame actionFrame = frame;
            if ((actionFrame.action < 0 || actionFrame.action >= (int)RLActions().size()) && !experiences.empty()) {
                float ratio = finishTime > 0.001f ? Clamp(frame.time / finishTime, 0.0f, 1.0f) : 0.0f;
                int experienceIndex = std::min((int)experiences.size() - 1, (int)roundf(ratio * (float)(experiences.size() - 1)));
                actionFrame.action = experiences[experienceIndex].action;
            }
            inputs.push_back(ManualInputFromReplayFrame(actionFrame));
        }
    }
};

struct ManualRunLibrary {
    std::vector<ManualRunReplay> runs;
    int selectedRun = 0;
    std::string status = "MANUAL RUN SAVES READY";

    bool HasRuns() const {
        return !runs.empty();
    }

    const ManualRunReplay& ActiveRun() const {
        int index = std::max(0, std::min(selectedRun, (int)runs.size() - 1));
        return runs[index];
    }

    ManualRunReplay& ActiveRun() {
        int index = std::max(0, std::min(selectedRun, (int)runs.size() - 1));
        return runs[index];
    }

    std::string ActiveLabel() const {
        if (!HasRuns()) return "no saved manual runs yet";
        return ActiveRun().label.empty() ? fs::path(ActiveRun().sourcePath).filename().string() : ActiveRun().label;
    }
};

RLReplayFrame ManualRaceFrameAt(const ManualRunReplay& replay, float raceTime, const Vector3& start);

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

void DrawManualCompletedGhost(const ManualRunReplay& replay, const Vector3& playerPosition, const Vector3& startPoint, float replayTime) {
    if (!replay.HasRun()) return;

    Color manualColor = { 255, 245, 180, 255 };
    for (int i = 1; i < (int)replay.trail.size(); ++i) {
        DrawLine3D(replay.trail[i - 1], replay.trail[i], Fade(manualColor, 0.45f));
    }

    RLReplayFrame ghost = ManualRaceFrameAt(replay, replayTime, startPoint);
    Vector3 p = ghost.position;
    if (RLHorizontalDistance(playerPosition, p) <= 0.1f) return;

    DrawCube({ p.x, p.y + 0.35f, p.z }, 0.72f, 0.86f, 0.72f, Fade(manualColor, 0.70f));
    DrawCubeWires({ p.x, p.y + 0.35f, p.z }, 0.72f, 0.86f, 0.72f, manualColor);
    DrawSphere({ p.x, p.y + 1.05f, p.z }, 0.22f, manualColor);
}

void DrawManualPlayerGhostStats(
    const ManualRunReplay& replay,
    const Vector3& playerPosition,
    const Vector3& startPoint,
    const Vector3& goalPoint,
    float replayTime,
    int x,
    int y
) {
    if (!replay.HasRun()) return;

    RLReplayFrame ghost = ManualRaceFrameAt(replay, replayTime, startPoint);
    float playerDist = RLHorizontalDistance(playerPosition, goalPoint);
    float ghostDist = RLHorizontalDistance(ghost.position, goalPoint);
    float ghostLead = playerDist - ghostDist;
    const char* leadLabel = ghostLead >= 0.0f ? "PLAYER GHOST" : "PLAYER";

    DrawText(
        TextFormat("PLAYER GHOST: best %.1fs  %s %.1fm", replay.finishTime, leadLabel, fabsf(ghostLead)),
        x,
        y,
        16,
        GOLD
    );
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

void DrawManualInputOverlay(const ManualRunReplay& replay, const Vector3& goalPoint, float replayTime) {
    if (!replay.HasRun()) return;

    ManualInputFrame input = replay.ManualInputAt(replayTime);
    RLReplayFrame frame = replay.FrameAt(replayTime);
    bool dashActive = input.dash || input.isDashing || input.dashVisualTimer > 0.0f;
    Color manualColor = { 255, 224, 92, 255 };

    float x = SCREEN_WIDTH - 238.0f;
    float y = SCREEN_HEIGHT - 168.0f;
    DrawRectangle((int)x - 12, (int)y - 12, 226, 150, (Color){ 8, 10, 12, 190 });
    DrawRectangleLines((int)x - 12, (int)y - 12, 226, 150, Fade(WHITE, 0.22f));
    DrawText("MANUAL INPUT", (int)x, (int)y - 4, 16, manualColor);

    DrawKeyboardKey({ x + 46, y + 22, 42, 42 }, "W", input.w);
    DrawKeyboardKey({ x + 0, y + 68, 42, 42 }, "A", input.a);
    DrawKeyboardKey({ x + 46, y + 68, 42, 42 }, "S", input.s);
    DrawKeyboardKey({ x + 92, y + 68, 42, 42 }, "D", input.d);
    DrawKeyboardKey({ x + 148, y + 22, 48, 42 }, "E", dashActive);
    DrawKeyboardKey({ x + 0, y + 114, 134, 30 }, "SPACE", input.jump);

    Vector3 look = Vector3Subtract(frame.camera.target, frame.camera.position);
    float lookYaw = atan2f(look.x, look.z) * RAD2DEG;
    float lookPitch = asinf(Clamp(Vector3Normalize(look).y, -1.0f, 1.0f)) * RAD2DEG;
    float turn = RLWrapDegrees(lookYaw - RLYawFromDirection(RLFlatDirection(frame.position, goalPoint)));
    float cx = x + 172.0f;
    float cy = y + 104.0f;
    DrawCircleLines((int)cx, (int)cy, 24.0f, Fade(WHITE, 0.45f));
    DrawLine((int)cx, (int)cy, (int)(cx + sinf(DEG2RAD * turn) * 22.0f), (int)(cy - cosf(DEG2RAD * turn) * 22.0f), manualColor);
    DrawText(TextFormat("CAM %.0f", lookPitch), (int)cx - 27, (int)cy + 31, 14, LIGHTGRAY);
}

float RaceCourseLength(const Vector3& start, const Vector3& goal) {
    Vector3 delta = { goal.x - start.x, 0.0f, goal.z - start.z };
    return fmaxf(0.001f, Vector3Length(delta));
}

float RaceCourseProgress01(const Vector3& position, const Vector3& start, const Vector3& goal) {
    Vector3 delta = { goal.x - start.x, 0.0f, goal.z - start.z };
    float length = fmaxf(0.001f, Vector3Length(delta));
    Vector3 axis = Vector3Scale(delta, 1.0f / length);
    Vector3 relative = { position.x - start.x, 0.0f, position.z - start.z };
    return Clamp(Vector3DotProduct(relative, axis) / length, 0.0f, 1.0f);
}

Vector3 RaceCoursePointAt(const Vector3& start, const Vector3& goal, float progress01) {
    Vector3 point = Vector3Lerp(start, goal, Clamp(progress01, 0.0f, 1.0f));
    point.y = 0.32f;
    return point;
}

Vector3 RaceCourseAxis(const Vector3& start, const Vector3& goal) {
    Vector3 delta = { goal.x - start.x, 0.0f, goal.z - start.z };
    if (Vector3Length(delta) <= 0.001f) return { 0.0f, 0.0f, -1.0f };
    return Vector3Normalize(delta);
}

Vector3 RaceCourseSide(const Vector3& start, const Vector3& goal) {
    Vector3 axis = RaceCourseAxis(start, goal);
    return { -axis.z, 0.0f, axis.x };
}

float SpectatorRaceDuration(const ManualRunReplay& manualReplay, const RLRunner& aiRunner) {
    float duration = 0.0f;
    if (manualReplay.HasRun()) duration = fmaxf(duration, manualReplay.finishTime);
    if (aiRunner.HasBestRun()) duration = fmaxf(duration, aiRunner.BestRunDuration());
    return duration;
}

RLReplayFrame ManualRaceStartFrame(const ManualRunReplay& replay, const Vector3& start) {
    RLReplayFrame first = replay.FrameAt(0.0f);
    Vector3 cameraOffset = Vector3Subtract(first.camera.position, first.position);
    Vector3 look = Vector3Subtract(first.camera.target, first.camera.position);
    first.position = start;
    first.camera.position = Vector3Add(start, cameraOffset);
    first.camera.target = Vector3Add(first.camera.position, look);
    first.time = 0.0f;
    return first;
}

RLReplayFrame ManualRaceFrameAt(const ManualRunReplay& replay, float raceTime, const Vector3& start) {
    if (!replay.HasRun()) return { start, {}, 0.0f, -1 };
    const RLReplayFrame& firstSaved = replay.frames.front();
    bool needsSyntheticStart = firstSaved.time > 0.001f || RLHorizontalDistance(firstSaved.position, start) > 0.02f;
    if (!needsSyntheticStart) return replay.FrameAt(raceTime);

    RLReplayFrame startFrame = ManualRaceStartFrame(replay, start);
    if (raceTime <= 0.0f) return startFrame;
    if (raceTime >= firstSaved.time) return replay.FrameAt(raceTime);

    RLReplayFrame firstFrame = firstSaved;
    float t = Clamp(raceTime / fmaxf(0.001f, firstSaved.time), 0.0f, 1.0f);
    RLReplayFrame frame = firstFrame;
    frame.position = Vector3Lerp(startFrame.position, firstFrame.position, t);
    frame.camera.position = Vector3Lerp(startFrame.camera.position, firstFrame.camera.position, t);
    frame.camera.target = Vector3Lerp(startFrame.camera.target, firstFrame.camera.target, t);
    frame.camera.up = Vector3Normalize(Vector3Lerp(startFrame.camera.up, firstFrame.camera.up, t));
    frame.camera.fovy = Lerp(startFrame.camera.fovy, firstFrame.camera.fovy, t);
    frame.time = raceTime;
    frame.action = (t < 0.5f) ? startFrame.action : firstFrame.action;
    return frame;
}

void DrawRaceReplayTrail(const std::vector<Vector3>& trail, Color color, float alpha) {
    for (int i = 1; i < (int)trail.size(); ++i) {
        DrawLine3D(trail[i - 1], trail[i], Fade(color, alpha));
    }
}

void DrawRaceActor(const RLReplayFrame& frame, Color color, bool highlighted) {
    Vector3 p = frame.position;
    DrawCube({ p.x, p.y + 0.35f, p.z }, 0.82f, 0.94f, 0.82f, Fade(color, 0.86f));
    DrawCubeWires({ p.x, p.y + 0.35f, p.z }, 0.82f, 0.94f, 0.82f, WHITE);

    Vector3 look = Vector3Subtract(frame.camera.target, frame.camera.position);
    Vector3 forward = { look.x, 0.0f, look.z };
    if (Vector3Length(forward) <= 0.001f) forward = { 0.0f, 0.0f, -1.0f };
    forward = Vector3Normalize(forward);
    DrawLine3D(
        { p.x, p.y + 1.12f, p.z },
        { p.x + forward.x * 1.7f, p.y + 1.12f, p.z + forward.z * 1.7f },
        WHITE
    );

    if (highlighted) {
        DrawSphere({ p.x, p.y + 1.28f, p.z }, 0.30f, WHITE);
    }
}

void DrawRaceProgressRail3D(
    const Vector3& start,
    const Vector3& goal,
    const Vector3& manualPosition,
    const Vector3& aiPosition,
    Color manualColor,
    Color aiColor
) {
    Vector3 startFlat = { start.x, 0.30f, start.z };
    Vector3 goalFlat = { goal.x, 0.30f, goal.z };
    Vector3 side = RaceCourseSide(start, goal);

    DrawLine3D(startFlat, goalFlat, Fade(WHITE, 0.55f));
    for (int i = 0; i <= 10; ++i) {
        float t = (float)i / 10.0f;
        Vector3 tick = RaceCoursePointAt(start, goal, t);
        DrawLine3D(
            Vector3Subtract(tick, Vector3Scale(side, 0.75f)),
            Vector3Add(tick, Vector3Scale(side, 0.75f)),
            Fade(WHITE, i == 0 || i == 10 ? 0.75f : 0.34f)
        );
    }

    float manualProgress = RaceCourseProgress01(manualPosition, start, goal);
    float aiProgress = RaceCourseProgress01(aiPosition, start, goal);
    Vector3 manualGate = RaceCoursePointAt(start, goal, manualProgress);
    Vector3 aiGate = RaceCoursePointAt(start, goal, aiProgress);
    const float gateHalfWidth = 48.0f;

    manualGate.y = 10.8f;
    aiGate.y = 9.8f;
    DrawLine3D(
        Vector3Subtract(manualGate, Vector3Scale(side, gateHalfWidth)),
        Vector3Add(manualGate, Vector3Scale(side, gateHalfWidth)),
        manualColor
    );
    DrawLine3D(
        Vector3Subtract(aiGate, Vector3Scale(side, gateHalfWidth)),
        Vector3Add(aiGate, Vector3Scale(side, gateHalfWidth)),
        aiColor
    );
    DrawSphere(manualGate, 0.26f, manualColor);
    DrawSphere(aiGate, 0.26f, aiColor);
}

void DrawSpectatorRace3D(
    const ManualRunReplay& manualReplay,
    const RLRunner& aiRunner,
    float raceTime,
    int cameraMode,
    const Vector3& start,
    const Vector3& goal
) {
    const Color manualColor = { 255, 224, 92, 255 };
    bool hasManual = manualReplay.HasRun();
    bool hasAi = aiRunner.HasBestRun();
    if (!hasManual && !hasAi) return;

    RLReplayFrame manualFrame = hasManual ? ManualRaceFrameAt(manualReplay, raceTime, start) : RLReplayFrame{ start, {}, 0.0f, -1 };
    RLReplayFrame aiFrame = hasAi ? aiRunner.BestRunFrame(raceTime) : RLReplayFrame{ start, {}, 0.0f, -1 };

    if (hasManual) DrawRaceReplayTrail(manualReplay.trail, manualColor, 0.42f);
    if (hasAi) DrawRaceReplayTrail(aiRunner.bestTrail, aiRunner.model.color, 0.38f);

    DrawRaceProgressRail3D(
        start,
        goal,
        hasManual ? manualFrame.position : start,
        hasAi ? aiFrame.position : start,
        manualColor,
        hasAi ? aiRunner.model.color : SKYBLUE
    );

    if (hasManual && cameraMode != 1) DrawRaceActor(manualFrame, manualColor, cameraMode == 2);
    if (hasAi && cameraMode != 2) DrawRaceActor(aiFrame, aiRunner.model.color, cameraMode == 1);
}

void DrawSpectatorRaceOverlay(
    const ManualRunReplay& manualReplay,
    const RLRunner& aiRunner,
    float raceTime,
    bool raceRunning,
    float playbackScale,
    int cameraMode,
    bool povFreeLook,
    const Vector3& start,
    const Vector3& goal
) {
    const Color manualColor = { 255, 224, 92, 255 };
    bool hasManual = manualReplay.HasRun();
    bool hasAi = aiRunner.HasBestRun();
    float duration = SpectatorRaceDuration(manualReplay, aiRunner);
    const char* playbackState = raceRunning
        ? "PLAY"
        : ((duration > 0.001f && raceTime >= duration - 0.001f) ? "FINISH" : (raceTime > 0.001f ? "PAUSED" : "READY"));

    int panelHeight = hasManual && hasAi ? 164 : 144;
    DrawRectangle(12, 12, 540, panelHeight, (Color){ 8, 10, 12, 210 });
    DrawRectangleLines(12, 12, 540, panelHeight, Fade(WHITE, 0.24f));
    DrawText("SPECTATOR RACE", 24, 24, 20, GOLD);
    DrawText(
        TextFormat("CAM: %s  %s  %.1f / %.1fs  %.2fx", SpectatorRaceCameraModeName(cameraMode), playbackState, raceTime, duration, playbackScale),
        24,
        50,
        16,
        WHITE
    );
    DrawText("G exit  R restart  ENTER play/pause  J/L scrub  ,/. speed", 24, 72, 14, LIGHTGRAY);
    DrawText("C camera  TAB/arrows select AI", 24, 90, 14, LIGHTGRAY);
    if (cameraMode == 0) DrawText("FREECAM: WASD + mouse  SPACE up  CTRL down  SHIFT fast", 24, 108, 14, SKYBLUE);
    else DrawText(TextFormat("POV FREE LOOK: %s  F toggle", povFreeLook ? "ON" : "OFF"), 24, 108, 14, povFreeLook ? GREEN : SKYBLUE);

    if (!hasManual || !hasAi) {
        DrawText(
            !hasManual ? "Need a saved manual best run first" : "Selected AI has no successful best run yet",
            24,
            126,
            16,
            ORANGE
        );
        return;
    }

    RLReplayFrame manualFrame = ManualRaceFrameAt(manualReplay, raceTime, start);
    RLReplayFrame aiFrame = aiRunner.BestRunFrame(raceTime);
    float courseLength = RaceCourseLength(start, goal);
    float manualProgress = RaceCourseProgress01(manualFrame.position, start, goal);
    float aiProgress = RaceCourseProgress01(aiFrame.position, start, goal);
    float manualMeters = manualProgress * courseLength;
    float aiMeters = aiProgress * courseLength;
    float leadMeters = manualMeters - aiMeters;
    const char* leader = leadMeters >= 0.0f ? "PLAYER GHOST" : "AI";

    DrawText(TextFormat("PLAYER GHOST best %.2fs  course %.1fm", manualReplay.finishTime, manualMeters), 24, 126, 14, manualColor);
    DrawText(TextFormat("AI %s best %.2fs  course %.1fm", aiRunner.model.name.c_str(), aiRunner.BestRunDuration(), aiMeters), 24, 144, 14, aiRunner.model.color);

    int barX = 86;
    int barY = SCREEN_HEIGHT - 84;
    int barW = SCREEN_WIDTH - 172;
    DrawRectangle(barX, barY, barW, 4, Fade(WHITE, 0.36f));
    DrawText("START", barX - 56, barY - 8, 14, LIGHTGRAY);
    DrawText("GOAL", barX + barW + 12, barY - 8, 14, LIGHTGRAY);

    int manualX = barX + (int)(manualProgress * barW);
    int aiX = barX + (int)(aiProgress * barW);
    DrawRectangle(manualX - 2, barY - 30, 4, 64, manualColor);
    DrawRectangle(aiX - 2, barY - 30, 4, 64, aiRunner.model.color);
    DrawCircle(manualX, barY + 2, 5.0f, manualColor);
    DrawCircle(aiX, barY + 2, 5.0f, aiRunner.model.color);
    DrawText("PLAYER", manualX - 28, barY - 30, 13, manualColor);
    DrawText("AI", aiX - 7, barY + 16, 13, aiRunner.model.color);

    const char* leadText = TextFormat("%s ahead %.1fm", leader, fabsf(leadMeters));
    int leadWidth = MeasureText(leadText, 18);
    DrawRectangle(SCREEN_WIDTH / 2 - leadWidth / 2 - 12, barY - 48, leadWidth + 24, 30, (Color){ 8, 10, 12, 210 });
    DrawRectangleLines(SCREEN_WIDTH / 2 - leadWidth / 2 - 12, barY - 48, leadWidth + 24, 30, Fade(WHITE, 0.22f));
    DrawText(leadText, SCREEN_WIDTH / 2 - leadWidth / 2, barY - 42, 18, leadMeters >= 0.0f ? manualColor : aiRunner.model.color);
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

long long SumActionCounts(const std::vector<int>& counts) {
    long long total = 0;
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

float SafePercent(int part, long long total) {
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
    if (action.dash) flags += flags.empty() ? "dash" : " dash";
    if (action.jump) flags += flags.empty() ? "jump" : " jump";
    if (action.slide) flags += flags.empty() ? "slide" : " slide";
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
    long long total = SumActionCounts(actionCounts);

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
    long long recentDecisionTotal = SumActionCounts(stats.actionCounts);
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
    const std::string& path,
    const RLTrainer& trainer,
    const World& world,
    int speedIndex,
    float achievedSpeedScale,
    float pendingSimSeconds,
    int trainingStepsLastFrame,
    float trainingSimSecondsLastFrame,
    bool turboTraining,
    bool soloTraining,
    const ManualRunReplay& manualBestReplay,
    const ManualRunLibrary& manualRunLibrary,
    std::string* error = nullptr
) {
    std::error_code ec;
    fs::path outputPath(path);
    fs::path parentPath = outputPath.parent_path();
    if (!parentPath.empty()) {
        fs::create_directories(parentPath, ec);
        if (ec) {
            if (error) *error = "could not create report folder: " + ec.message();
            return false;
        }
    }

    std::ofstream out(path, std::ios::out | std::ios::trunc);
    if (!out.is_open()) {
        if (error) *error = "could not open " + path;
        return false;
    }

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
    out << "  achieved/target ratio: " << std::setprecision(4) << (targetSpeedScale > 0.0f ? achievedSpeedScale / targetSpeedScale : 0.0f) << "\n";
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
    out << "  saved manual run files: " << manualRunLibrary.runs.size() << " / " << MANUAL_RUN_KEEP_COUNT << " kept\n";
    if (manualRunLibrary.HasRuns()) {
        out << "  selected saved manual run: " << manualRunLibrary.ActiveLabel() << "\n";
    }
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
    out << "  model              ep       done       succ   succ%    timeout      stuck   best time   best reward    eps  recent succ%        rank\n";
    out << "  -------------------------------------------------------------------------------------------------------------------------------------\n";
    for (const RLRunner& runner : trainer.runners) {
        int finished = runner.successes + runner.timeouts + runner.stucks;
        RLRecentStats stats = BuildRecentStats(runner, 80);
        out << "  "
            << std::left << std::setw(13) << runner.model.name << std::right
            << std::setw(10) << runner.episode
            << std::setw(11) << finished
            << std::setw(11) << runner.successes
            << std::setw(8) << std::fixed << std::setprecision(1) << SafePercent(runner.successes, finished)
            << std::setw(11) << runner.timeouts
            << std::setw(11) << runner.stucks
            << std::setw(12) << (runner.HasBestRun() ? runner.bestTime : 0.0f)
            << std::setw(14) << runner.bestReward
            << std::setw(7) << runner.model.epsilon
            << std::setw(14) << SafePercent(stats.successes, stats.count)
            << std::setw(12) << runner.ReliableSelectionScore()
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
        if (!runner.manualGuidanceBest.empty()) {
            out << "  manual guidance anchor: " << runner.manualGuidanceBest.size() << " samples, best "
                << std::setprecision(3) << runner.manualGuidanceBestTime << "s, "
                << runner.manualGuidanceRehearsals << " rehearsed updates\n";
        }
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
    out << "  - Saved manual runs can now be reviewed from the player camera and reused as manual guidance evidence.\n";
    out << "  - If bad-looking runs keep high reward, reduce shaping reward that can be farmed without finishing and make terminal success dominate the score.\n";
    out << "  - If 250000x reports only a few hundred actual x, the bottleneck is CPU simulation, not the rest of the computer. Keep the UI responsive by trusting actual speed and reducing active model work.\n";
    out << "  - Use this report after each tuning pass. Look for lower wall/collision averages, higher recent success rate, and a smaller gap between best run and average success time.\n";

    out.close();
    if (!out) {
        if (error) *error = "write failed for " + path;
        return false;
    }

    return true;
}

std::string ModelSlug(const std::string& name) {
    std::string slug;
    bool lastWasSeparator = false;

    for (char c : name) {
        unsigned char value = (unsigned char)c;
        if (std::isalnum(value)) {
            slug.push_back((char)std::tolower(value));
            lastWasSeparator = false;
        } else if (!lastWasSeparator && !slug.empty()) {
            slug.push_back('_');
            lastWasSeparator = true;
        }
    }

    while (!slug.empty() && slug.back() == '_') slug.pop_back();
    return slug.empty() ? "model" : slug;
}

std::string TimestampForModelFile() {
    std::time_t now = std::time(nullptr);
    std::tm* local = std::localtime(&now);
    if (!local) return "unknown_time";

    std::ostringstream text;
    text << std::put_time(local, "%Y%m%d_%H%M%S");
    return text.str();
}

std::string TimestampForModelText() {
    std::time_t now = std::time(nullptr);
    std::tm* local = std::localtime(&now);
    if (!local) return "unknown time";

    std::ostringstream text;
    text << std::put_time(local, "%Y-%m-%d %H:%M:%S");
    return text.str();
}

std::string FileNameOnly(const std::string& path) {
    if (path.empty()) return "";
    return fs::path(path).filename().string();
}

bool IsMigrationBackupFile(const std::string& path) {
    std::string fileName = FileNameOnly(path);
    return fileName.find("_legacy_") != std::string::npos ||
           fileName.find("_pre_migration_") != std::string::npos;
}

std::string AppOutputPath(const char* fileName) {
    const char* appDirectory = GetApplicationDirectory();
    if (appDirectory && appDirectory[0] != '\0') {
        return (fs::path(appDirectory) / fileName).lexically_normal().string();
    }
    return fileName;
}

std::string ModelDirectoryPath() {
    return AppOutputPath(RL_MODEL_DIRECTORY);
}

std::string ManualRunDirectoryPath() {
    return AppOutputPath(MANUAL_RUN_DIRECTORY);
}

std::string TrainingReportPath() {
    return AppOutputPath(RL_TRAINING_REPORT_FILE);
}

bool SameFilesystemLocation(const fs::path& a, const fs::path& b) {
    std::error_code ec;
    if (fs::exists(a, ec) && fs::exists(b, ec)) {
        ec.clear();
        if (fs::equivalent(a, b, ec) && !ec) return true;
    }

    std::error_code aEc;
    std::error_code bEc;
    fs::path canonicalA = fs::weakly_canonical(a, aEc);
    fs::path canonicalB = fs::weakly_canonical(b, bEc);
    if (!aEc && !bEc) return canonicalA == canonicalB;

    return fs::absolute(a).lexically_normal() == fs::absolute(b).lexically_normal();
}

fs::path UniqueSiblingPath(const fs::path& path, const std::string& tag) {
    fs::path directory = path.parent_path();
    std::string stem = path.stem().string();
    std::string extension = path.extension().string();
    std::string timestamp = TimestampForModelFile();

    for (int i = 0; i < 1000; ++i) {
        std::ostringstream name;
        name << stem << "_" << tag << "_" << timestamp;
        if (i > 0) name << "_" << i;
        name << extension;

        fs::path candidate = directory / name.str();
        std::error_code ec;
        if (!fs::exists(candidate, ec)) return candidate;
    }

    return directory / (stem + "_" + tag + "_" + timestamp + extension);
}

int CopyPersistenceFilesFromDirectory(
    const fs::path& sourceDirectory,
    const fs::path& targetDirectory,
    const std::string& extension,
    std::string& firstError
) {
    std::error_code ec;
    if (sourceDirectory.empty() ||
        !fs::exists(sourceDirectory, ec) ||
        !fs::is_directory(sourceDirectory, ec) ||
        SameFilesystemLocation(sourceDirectory, targetDirectory)) {
        return 0;
    }

    fs::create_directories(targetDirectory, ec);
    if (ec) {
        if (firstError.empty()) firstError = "could not create " + targetDirectory.string() + ": " + ec.message();
        return 0;
    }

    int copied = 0;
    fs::directory_iterator it(sourceDirectory, ec);
    fs::directory_iterator end;
    if (ec) {
        if (firstError.empty()) firstError = "could not scan " + sourceDirectory.string() + ": " + ec.message();
        return 0;
    }

    while (it != end) {
        std::error_code entryError;
        const fs::directory_entry& entry = *it;
        if (entry.is_regular_file(entryError) && entry.path().extension() == extension) {
            fs::path targetPath = targetDirectory / entry.path().filename();
            bool targetExists = fs::exists(targetPath, entryError);
            fs::path copyTarget = targetPath;

            if (targetExists && !SameFilesystemLocation(entry.path(), targetPath)) {
                std::error_code timeError;
                auto sourceTime = fs::last_write_time(entry.path(), timeError);
                auto targetTime = fs::last_write_time(targetPath, timeError);
                bool sourceIsNewer = !timeError && sourceTime > targetTime;

                if (sourceIsNewer) {
                    fs::path backupPath = UniqueSiblingPath(targetPath, "pre_migration");
                    std::error_code renameError;
                    fs::rename(targetPath, backupPath, renameError);
                    if (renameError) {
                        copyTarget = UniqueSiblingPath(targetPath, "legacy");
                    }
                } else {
                    copyTarget = UniqueSiblingPath(targetPath, "legacy");
                }
            }

            std::error_code copyError;
            fs::copy_file(
                entry.path(),
                copyTarget,
                targetExists && copyTarget == targetPath ? fs::copy_options::overwrite_existing : fs::copy_options::none,
                copyError
            );

            if (!copyError) {
                copied += 1;
            } else if (firstError.empty()) {
                firstError = "could not migrate " + entry.path().string() + ": " + copyError.message();
            }
        }
        it.increment(ec);
        if (ec && firstError.empty()) {
            firstError = "could not continue scanning " + sourceDirectory.string() + ": " + ec.message();
        }
    }

    return copied;
}

int CopyReportIfNewer(const fs::path& sourcePath, const fs::path& targetPath, std::string& firstError) {
    std::error_code ec;
    if (sourcePath.empty() ||
        !fs::exists(sourcePath, ec) ||
        !fs::is_regular_file(sourcePath, ec) ||
        SameFilesystemLocation(sourcePath, targetPath)) {
        return 0;
    }

    fs::create_directories(targetPath.parent_path(), ec);
    if (ec) {
        if (firstError.empty()) firstError = "could not create report folder: " + ec.message();
        return 0;
    }

    bool shouldCopy = !fs::exists(targetPath, ec);
    if (!shouldCopy) {
        std::error_code timeError;
        auto sourceTime = fs::last_write_time(sourcePath, timeError);
        auto targetTime = fs::last_write_time(targetPath, timeError);
        shouldCopy = !timeError && sourceTime > targetTime;
    }

    if (!shouldCopy) return 0;

    if (fs::exists(targetPath, ec)) {
        fs::path backupPath = UniqueSiblingPath(targetPath, "pre_migration");
        std::error_code renameError;
        fs::rename(targetPath, backupPath, renameError);
        if (renameError && firstError.empty()) {
            firstError = "could not back up old report: " + renameError.message();
            return 0;
        }
    }

    std::error_code copyError;
    fs::copy_file(sourcePath, targetPath, fs::copy_options::overwrite_existing, copyError);
    if (copyError) {
        if (firstError.empty()) firstError = "could not migrate report: " + copyError.message();
        return 0;
    }

    return 1;
}

int MigrateLegacyPersistence(std::string& message) {
    fs::path targetModels(ModelDirectoryPath());
    fs::path targetManualRuns(ManualRunDirectoryPath());
    fs::path targetReport(TrainingReportPath());

    std::vector<fs::path> modelSources;
    std::vector<fs::path> manualSources;
    std::vector<fs::path> reportSources;

    std::error_code ec;
    fs::path cwd = fs::current_path(ec);
    if (!ec) {
        modelSources.push_back(cwd / RL_MODEL_DIRECTORY);
        manualSources.push_back(cwd / MANUAL_RUN_DIRECTORY);
        reportSources.push_back(cwd / RL_TRAINING_REPORT_FILE);
    }

    const char* home = std::getenv("HOME");
    if (home && home[0] != '\0') {
        fs::path homePath(home);
        modelSources.push_back(homePath / RL_MODEL_DIRECTORY);
        manualSources.push_back(homePath / MANUAL_RUN_DIRECTORY);
        reportSources.push_back(homePath / RL_TRAINING_REPORT_FILE);
    }

    int copied = 0;
    int reportCopied = 0;
    std::string firstError;
    for (const fs::path& source : modelSources) {
        copied += CopyPersistenceFilesFromDirectory(source, targetModels, RL_MODEL_EXTENSION, firstError);
    }
    for (const fs::path& source : manualSources) {
        copied += CopyPersistenceFilesFromDirectory(source, targetManualRuns, MANUAL_RUN_EXTENSION, firstError);
    }
    for (const fs::path& source : reportSources) {
        reportCopied += CopyReportIfNewer(source, targetReport, firstError);
    }

    if (!firstError.empty()) {
        message = "SAVE MIGRATION WARNING: " + firstError;
    } else if (copied > 0 || reportCopied > 0) {
        message = TextFormat("MIGRATED %d SAVE FILES + %d REPORTS", copied, reportCopied);
    } else {
        message.clear();
    }

    return copied + reportCopied;
}

std::string RunnerAutosavePath(const RLRunner& runner) {
    fs::path path = fs::path(ModelDirectoryPath()) / ("autosave_" + ModelSlug(runner.model.name) + RL_MODEL_EXTENSION);
    return path.string();
}

std::string RunnerSnapshotPath(const RLRunner& runner) {
    fs::path path = fs::path(ModelDirectoryPath()) / (ModelSlug(runner.model.name) + "_" + TimestampForModelFile() + RL_MODEL_EXTENSION);
    return path.string();
}

bool EnsureModelDirectory(std::string& error) {
    std::error_code ec;
    fs::path directory(ModelDirectoryPath());
    if (fs::exists(directory, ec)) {
        if (fs::is_directory(directory, ec)) return true;
        error = directory.string() + " exists but is not a directory";
        return false;
    }

    if (!fs::create_directories(directory, ec) || ec) {
        error = "could not create " + directory.string() + ": " + ec.message();
        return false;
    }

    return true;
}

void SetModelLibraryStatus(RLModelLibrary& library, const std::string& status) {
    library.status = status;
}

void RefreshModelLibrary(RLModelLibrary& library) {
    std::string error;
    if (!EnsureModelDirectory(error)) {
        SetModelLibraryStatus(library, "MODEL FOLDER FAILED: " + error);
        library.files.clear();
        library.selectedFile = 0;
        return;
    }

    std::string previousSelection = library.SelectedPath();
    std::vector<std::string> files;
    std::error_code ec;
    fs::directory_iterator it(ModelDirectoryPath(), ec);
    fs::directory_iterator end;

    while (!ec && it != end) {
        std::error_code entryError;
        const fs::directory_entry& entry = *it;
        if (entry.is_regular_file(entryError) &&
            entry.path().extension() == RL_MODEL_EXTENSION &&
            !IsMigrationBackupFile(entry.path().string())) {
            files.push_back(entry.path().string());
        }
        it.increment(ec);
    }

    std::sort(files.begin(), files.end(), [](const std::string& a, const std::string& b) {
        return FileNameOnly(a) < FileNameOnly(b);
    });

    library.files = files;
    library.selectedFile = 0;
    for (int i = 0; i < (int)library.files.size(); ++i) {
        if (library.files[i] == previousSelection) {
            library.selectedFile = i;
            break;
        }
    }

    if (library.files.empty() && library.status.find("FAILED") == std::string::npos) {
        SetModelLibraryStatus(library, "AUTOSAVE READY: rl_models/");
    }
}

void SelectPreviousModelFile(RLModelLibrary& library) {
    RefreshModelLibrary(library);
    if (library.files.empty()) {
        SetModelLibraryStatus(library, "NO MODEL FILES IN rl_models/");
        return;
    }

    library.selectedFile = (library.selectedFile - 1 + (int)library.files.size()) % (int)library.files.size();
    SetModelLibraryStatus(library, "SELECTED: " + FileNameOnly(library.SelectedPath()));
}

void SelectNextModelFile(RLModelLibrary& library) {
    RefreshModelLibrary(library);
    if (library.files.empty()) {
        SetModelLibraryStatus(library, "NO MODEL FILES IN rl_models/");
        return;
    }

    library.selectedFile = (library.selectedFile + 1) % (int)library.files.size();
    SetModelLibraryStatus(library, "SELECTED: " + FileNameOnly(library.SelectedPath()));
}

void WriteFloatVector(std::ofstream& out, const char* label, const std::vector<float>& values) {
    out << label << " " << values.size();
    for (float value : values) out << " " << value;
    out << "\n";
}

void WriteIntVector(std::ofstream& out, const char* label, const std::vector<int>& values) {
    out << label << " " << values.size();
    for (int value : values) out << " " << value;
    out << "\n";
}

void WriteFloatMatrix(std::ofstream& out, const char* label, const std::vector<std::vector<float>>& rows) {
    int rowCount = (int)rows.size();
    int columnCount = rowCount > 0 ? (int)rows[0].size() : 0;

    out << label << " " << rowCount << " " << columnCount << "\n";
    for (const auto& row : rows) {
        for (int i = 0; i < columnCount; ++i) {
            float value = i < (int)row.size() ? row[i] : 0.0f;
            out << (i == 0 ? "" : " ") << value;
        }
        out << "\n";
    }
}

void WriteVector3Values(std::ofstream& out, Vector3 value) {
    out << value.x << " " << value.y << " " << value.z;
}

void WriteStateValues(std::ofstream& out, const Player::RLState& state) {
    for (float value : state) out << " " << value;
}

void WritePolicySnapshot(std::ofstream& out, const RLPolicySnapshot& snapshot) {
    out << "best_policy "
        << (snapshot.valid ? 1 : 0) << " "
        << snapshot.rawFeatureCount << " "
        << snapshot.featureCount << " "
        << snapshot.epsilon << "\n";
    WriteFloatMatrix(out, "best_policy_weights", snapshot.valid ? snapshot.weights : std::vector<std::vector<float>>());
    WriteFloatVector(out, "best_policy_action_bias", snapshot.valid ? snapshot.actionBias : std::vector<float>());
}

void WriteVector3List(std::ofstream& out, const char* label, const std::vector<Vector3>& values) {
    out << label << " " << values.size() << "\n";
    for (Vector3 value : values) {
        WriteVector3Values(out, value);
        out << "\n";
    }
}

void WriteReplayFrames(std::ofstream& out, const char* label, const std::vector<RLReplayFrame>& frames) {
    out << label << " " << frames.size() << "\n";
    for (const RLReplayFrame& frame : frames) {
        WriteVector3Values(out, frame.position);
        out << " ";
        WriteVector3Values(out, frame.camera.position);
        out << " ";
        WriteVector3Values(out, frame.camera.target);
        out << " ";
        WriteVector3Values(out, frame.camera.up);
        out << " " << frame.camera.fovy
            << " " << frame.camera.projection
            << " " << frame.time
            << " " << frame.action << "\n";
    }
}

void WriteExperiences(std::ofstream& out, const char* label, const std::vector<RLExperience>& experiences) {
    out << label << " " << experiences.size() << "\n";
    for (const RLExperience& experience : experiences) {
        out << experience.action
            << " " << experience.reward
            << " " << experience.priority
            << " " << (experience.done ? 1 : 0);
        WriteStateValues(out, experience.state);
        WriteStateValues(out, experience.nextState);
        out << "\n";
    }
}

void WriteManualInputs(std::ofstream& out, const std::vector<ManualInputFrame>& inputs) {
    out << "inputs " << inputs.size() << "\n";
    for (const ManualInputFrame& input : inputs) {
        out << input.time
            << " " << (input.w ? 1 : 0)
            << " " << (input.a ? 1 : 0)
            << " " << (input.s ? 1 : 0)
            << " " << (input.d ? 1 : 0)
            << " " << (input.jump ? 1 : 0)
            << " " << (input.dash ? 1 : 0)
            << " " << (input.slide ? 1 : 0)
            << " " << (input.slash ? 1 : 0)
            << " " << (input.isDashing ? 1 : 0)
            << " " << (input.isWallRunning ? 1 : 0)
            << " " << input.dashCooldown
            << " " << input.dashVisualTimer << "\n";
    }
}

void WriteRecentEpisodes(std::ofstream& out, const std::vector<RLEpisodeSummary>& episodes) {
    out << "recent_episodes " << episodes.size() << "\n";
    for (const RLEpisodeSummary& episode : episodes) {
        out << episode.episode
            << " " << episode.time
            << " " << episode.reward
            << " " << episode.distanceToGoal
            << " " << episode.collisionCost
            << " " << episode.wallHits
            << " " << episode.decisions
            << " " << (episode.success ? 1 : 0)
            << " " << (episode.timeout ? 1 : 0)
            << " " << (episode.stuck ? 1 : 0)
            << " " << episode.actionCounts.size();
        for (int count : episode.actionCounts) out << " " << count;
        out << "\n";
    }
}

bool ReadExpected(std::istream& in, const char* expected, std::string& error) {
    std::string token;
    if (!(in >> token)) {
        error = std::string("missing token: ") + expected;
        return false;
    }
    if (token != expected) {
        error = "expected " + std::string(expected) + ", found " + token;
        return false;
    }
    return true;
}

bool ReadFloatVector(std::istream& in, const char* label, std::vector<float>& values, int maxCount, std::string& error) {
    if (!ReadExpected(in, label, error)) return false;

    int count = 0;
    if (!(in >> count) || count < 0 || count > maxCount) {
        error = std::string("bad vector size for ") + label;
        return false;
    }

    values.assign(count, 0.0f);
    for (float& value : values) {
        if (!(in >> value)) {
            error = std::string("bad vector value for ") + label;
            return false;
        }
    }

    return true;
}

bool ReadIntVector(std::istream& in, const char* label, std::vector<int>& values, int maxCount, std::string& error) {
    if (!ReadExpected(in, label, error)) return false;

    int count = 0;
    if (!(in >> count) || count < 0 || count > maxCount) {
        error = std::string("bad vector size for ") + label;
        return false;
    }

    values.assign(count, 0);
    for (int& value : values) {
        if (!(in >> value)) {
            error = std::string("bad vector value for ") + label;
            return false;
        }
    }

    return true;
}

bool ReadFloatMatrix(std::istream& in, const char* label, std::vector<std::vector<float>>& values, int maxRows, int maxColumns, std::string& error) {
    if (!ReadExpected(in, label, error)) return false;

    int rows = 0;
    int columns = 0;
    if (!(in >> rows >> columns) || rows < 0 || columns < 0 || rows > maxRows || columns > maxColumns) {
        error = std::string("bad matrix shape for ") + label;
        return false;
    }
    if ((rows == 0) != (columns == 0)) {
        error = std::string("empty matrix shape mismatch for ") + label;
        return false;
    }

    values.assign(rows, std::vector<float>(columns, 0.0f));
    for (int row = 0; row < rows; ++row) {
        for (int column = 0; column < columns; ++column) {
            if (!(in >> values[row][column])) {
                error = std::string("bad matrix value for ") + label;
                return false;
            }
        }
    }

    return true;
}

bool ReadVector3Values(std::istream& in, Vector3& value) {
    return (bool)(in >> value.x >> value.y >> value.z);
}

bool ReadStateValues(std::istream& in, Player::RLState& state) {
    for (float& value : state) {
        if (!(in >> value)) return false;
    }
    return true;
}

bool ReadPolicySnapshot(std::istream& in, RLPolicySnapshot& snapshot, std::string& error) {
    if (!ReadExpected(in, "best_policy", error)) return false;

    int valid = 0;
    if (!(in >> valid >> snapshot.rawFeatureCount >> snapshot.featureCount >> snapshot.epsilon)) {
        error = "bad best_policy header";
        return false;
    }

    snapshot.valid = valid != 0;
    if (!ReadFloatMatrix(in, "best_policy_weights", snapshot.weights, 128, 1024, error)) return false;
    if (!ReadFloatVector(in, "best_policy_action_bias", snapshot.actionBias, 128, error)) return false;
    return true;
}

bool ReadVector3List(std::istream& in, const char* label, std::vector<Vector3>& values, int maxCount, std::string& error) {
    if (!ReadExpected(in, label, error)) return false;

    int count = 0;
    if (!(in >> count) || count < 0 || count > maxCount) {
        error = std::string("bad vector3 list size for ") + label;
        return false;
    }

    values.assign(count, {});
    for (Vector3& value : values) {
        if (!ReadVector3Values(in, value)) {
            error = std::string("bad vector3 value for ") + label;
            return false;
        }
    }

    return true;
}

bool ReadReplayFrames(std::istream& in, const char* label, std::vector<RLReplayFrame>& frames, int maxCount, std::string& error) {
    if (!ReadExpected(in, label, error)) return false;

    int count = 0;
    if (!(in >> count) || count < 0 || count > maxCount) {
        error = std::string("bad replay frame count for ") + label;
        return false;
    }

    frames.assign(count, {});
    for (RLReplayFrame& frame : frames) {
        int projection = CAMERA_PERSPECTIVE;
        if (!ReadVector3Values(in, frame.position) ||
            !ReadVector3Values(in, frame.camera.position) ||
            !ReadVector3Values(in, frame.camera.target) ||
            !ReadVector3Values(in, frame.camera.up) ||
            !(in >> frame.camera.fovy >> projection >> frame.time >> frame.action)) {
            error = std::string("bad replay frame for ") + label;
            return false;
        }
        frame.camera.projection = projection;
    }

    return true;
}

bool ReadExperiences(std::istream& in, const char* label, std::vector<RLExperience>& experiences, int maxCount, std::string& error) {
    if (!ReadExpected(in, label, error)) return false;

    int count = 0;
    if (!(in >> count) || count < 0 || count > maxCount) {
        error = std::string("bad experience count for ") + label;
        return false;
    }

    experiences.assign(count, {});
    for (RLExperience& experience : experiences) {
        int done = 0;
        if (!(in >> experience.action >> experience.reward >> experience.priority >> done) ||
            !ReadStateValues(in, experience.state) ||
            !ReadStateValues(in, experience.nextState)) {
            error = std::string("bad experience for ") + label;
            return false;
        }
        experience.done = done != 0;
    }

    return true;
}

bool ReadManualInputsAfterLabel(std::istream& in, std::vector<ManualInputFrame>& inputs, int maxCount, std::string& error) {
    int count = 0;
    if (!(in >> count) || count < 0 || count > maxCount) {
        error = "bad manual input count";
        return false;
    }

    inputs.assign(count, {});
    for (ManualInputFrame& input : inputs) {
        int w = 0;
        int a = 0;
        int s = 0;
        int d = 0;
        int jump = 0;
        int dash = 0;
        int slide = 0;
        int slash = 0;
        int isDashing = 0;
        int isWallRunning = 0;

        if (!(in >> input.time
                 >> w
                 >> a
                 >> s
                 >> d
                 >> jump
                 >> dash
                 >> slide
                 >> slash
                 >> isDashing
                 >> isWallRunning
                 >> input.dashCooldown
                 >> input.dashVisualTimer)) {
            error = "bad manual input frame";
            return false;
        }

        input.w = w != 0;
        input.a = a != 0;
        input.s = s != 0;
        input.d = d != 0;
        input.jump = jump != 0;
        input.dash = dash != 0;
        input.slide = slide != 0;
        input.slash = slash != 0;
        input.isDashing = isDashing != 0;
        input.isWallRunning = isWallRunning != 0;
    }

    return true;
}

bool ReadRecentEpisodes(std::istream& in, std::vector<RLEpisodeSummary>& episodes, std::string& error) {
    if (!ReadExpected(in, "recent_episodes", error)) return false;

    int count = 0;
    if (!(in >> count) || count < 0 || count > 1000) {
        error = "bad recent episode count";
        return false;
    }

    episodes.assign(count, {});
    for (RLEpisodeSummary& episode : episodes) {
        int success = 0;
        int timeout = 0;
        int stuck = 0;
        int actionCount = 0;
        if (!(in >> episode.episode
                >> episode.time
                >> episode.reward
                >> episode.distanceToGoal
                >> episode.collisionCost
                >> episode.wallHits
                >> episode.decisions
                >> success
                >> timeout
                >> stuck
                >> actionCount) ||
            actionCount < 0 ||
            actionCount > 128) {
            error = "bad recent episode row";
            return false;
        }

        episode.success = success != 0;
        episode.timeout = timeout != 0;
        episode.stuck = stuck != 0;
        episode.actionCounts.assign(actionCount, 0);
        for (int& value : episode.actionCounts) {
            if (!(in >> value)) {
                error = "bad recent episode action count";
                return false;
            }
        }
    }

    return true;
}

bool ValidateLoadedRunner(RLRunner& runner, std::string& error) {
    int actionCount = (int)RLActions().size();
    int expectedFeatureCount = runner.model.EncodedFeatureCount(runner.model.rawFeatureCount);
    int expectedWeightsPerAction = expectedFeatureCount + 1;

    if (runner.model.rawFeatureCount != Player::RL_STATE_SIZE) {
        error = "feature size mismatch; saved model does not match this build";
        return false;
    }
    if (runner.model.featureCount != expectedFeatureCount) {
        error = "encoded feature size mismatch; saved model does not match this build";
        return false;
    }
    if ((int)runner.model.weights.size() != actionCount) {
        error = "action count mismatch; saved model does not match this build";
        return false;
    }
    for (const auto& row : runner.model.weights) {
        if ((int)row.size() != expectedWeightsPerAction) {
            error = "weight row size mismatch; saved model does not match this build";
            return false;
        }
    }
    if ((int)runner.model.actionBias.size() != actionCount) {
        error = "action bias count mismatch; saved model does not match this build";
        return false;
    }
    if ((int)runner.model.explorationWeight.size() != actionCount) {
        error = "exploration weight count mismatch; saved model does not match this build";
        return false;
    }

    if ((int)runner.lifetimeActionCounts.size() != actionCount) {
        runner.lifetimeActionCounts.assign(actionCount, 0);
    }
    runner.EnsureActionTelemetry();

    if (runner.bestPolicy.valid && !runner.model.CanUsePolicySnapshot(runner.bestPolicy)) {
        runner.bestPolicy.valid = false;
    }

    runner.episode = std::max(0, runner.episode);
    runner.successes = std::max(0, runner.successes);
    runner.timeouts = std::max(0, runner.timeouts);
    runner.stucks = std::max(0, runner.stucks);
    runner.episodesSinceBest = std::max(0, runner.episodesSinceBest);
    runner.bestTime = runner.HasBestRun() ? runner.bestTime : 9999.0f;
    runner.bestReward = fmaxf(runner.bestReward, -9999.0f);
    runner.bestSuccessReward = runner.HasBestRun() ? runner.bestSuccessReward : -9999.0f;
    runner.model.EnsureActionTuning();
    return true;
}

void SeedLoadedRunnerReplay(RLRunner& runner) {
    runner.model.memory.clear();
    runner.model.eliteMemory.clear();
    runner.model.memoryCursor = 0;
    runner.model.eliteMemoryCursor = 0;

    for (const RLExperience& experience : runner.bestExperiences) {
        runner.model.RememberExperience(experience);
        runner.model.RememberEliteExperience(experience);
    }

    runner.trail.clear();
    runner.episodeTrail.clear();
    runner.episodeFrames.clear();
    runner.episodeExperiences.clear();
    runner.hasCurrentState = false;
    runner.resetTimer = 0.0f;
}

bool WriteRunnerModel(std::ofstream& out, const RLRunner& runner) {
    out << std::fixed << std::setprecision(9);
    out << RL_MODEL_MAGIC << "\n";
    out << "saved_at " << std::quoted(TimestampForModelText()) << "\n";
    out << "name " << std::quoted(runner.model.name) << "\n";
    out << "color "
        << (int)runner.model.color.r << " "
        << (int)runner.model.color.g << " "
        << (int)runner.model.color.b << " "
        << (int)runner.model.color.a << "\n";
    out << "decision_interval " << runner.decisionInterval << "\n";
    out << "model_params "
        << runner.model.learningRate << " "
        << runner.model.discount << " "
        << runner.model.epsilon << " "
        << runner.model.minEpsilon << " "
        << runner.model.epsilonDecay << " "
        << runner.model.turnRate << " "
        << runner.model.clearanceGuidance << " "
        << runner.model.lastTD << " "
        << runner.model.rawFeatureCount << " "
        << runner.model.featureCount << " "
        << runner.model.memoryCapacity << " "
        << runner.model.eliteMemoryCapacity << " "
        << runner.model.replayBatchSize << "\n";

    WriteFloatMatrix(out, "weights", runner.model.weights);
    WriteFloatVector(out, "action_bias", runner.model.actionBias);
    WriteFloatVector(out, "exploration_weight", runner.model.explorationWeight);

    out << "runner_stats "
        << runner.episode << " "
        << runner.successes << " "
        << runner.timeouts << " "
        << runner.stucks << " "
        << runner.episodesSinceBest << " "
        << runner.manualGuidanceReplays << " "
        << runner.manualGuidanceExperiences << " "
        << runner.bestRunRehearsals << " "
        << runner.bestTime << " "
        << runner.bestReward << " "
        << runner.bestSuccessReward << " "
        << runner.lastEpisodeReward << " "
        << runner.lastEpisodeTime << " "
        << runner.lastDistance << " "
        << runner.totalBestPathReward << "\n";

    WriteIntVector(out, "lifetime_action_counts", runner.lifetimeActionCounts);
    WritePolicySnapshot(out, runner.bestPolicy);
    WriteVector3List(out, "best_trail", runner.bestTrail);
    WriteReplayFrames(out, "best_frames", runner.bestFrames);
    WriteExperiences(out, "best_experiences", runner.bestExperiences);
    WriteRecentEpisodes(out, runner.recentEpisodes);
    out << "end\n";

    return out.good();
}

bool SaveRunnerModelToFile(const std::string& path, const RLRunner& runner, std::string& error) {
    std::error_code ec;
    fs::path outputPath(path);
    fs::create_directories(outputPath.parent_path(), ec);
    if (ec) {
        error = "could not create model folder: " + ec.message();
        return false;
    }

    std::string tempPath = path + ".tmp";
    std::ofstream out(tempPath);
    if (!out.is_open()) {
        error = "could not open " + tempPath;
        return false;
    }

    if (!WriteRunnerModel(out, runner)) {
        error = "could not write " + tempPath;
        return false;
    }

    out.close();
    if (!out.good()) {
        error = "could not finish writing " + tempPath;
        return false;
    }

    fs::remove(outputPath, ec);
    ec.clear();
    fs::rename(tempPath, outputPath, ec);
    if (ec) {
        error = "could not replace " + path + ": " + ec.message();
        return false;
    }

    return true;
}

bool LoadRunnerModelFromFile(const std::string& path, RLRunner& runner, std::string& error) {
    std::ifstream in(path);
    if (!in.is_open()) {
        error = "could not open " + path;
        return false;
    }

    std::string magic;
    if (!(in >> magic) || magic != RL_MODEL_MAGIC) {
        error = "not an RL model file: " + FileNameOnly(path);
        return false;
    }

    RLRunner loaded = runner;
    std::string savedAt;

    if (!ReadExpected(in, "saved_at", error) || !(in >> std::quoted(savedAt))) return false;
    if (!ReadExpected(in, "name", error) || !(in >> std::quoted(loaded.model.name))) return false;

    int r = 255;
    int g = 255;
    int b = 255;
    int a = 255;
    if (!ReadExpected(in, "color", error) || !(in >> r >> g >> b >> a)) return false;
    loaded.model.color = {
        (unsigned char)Clamp((float)r, 0.0f, 255.0f),
        (unsigned char)Clamp((float)g, 0.0f, 255.0f),
        (unsigned char)Clamp((float)b, 0.0f, 255.0f),
        (unsigned char)Clamp((float)a, 0.0f, 255.0f)
    };

    if (!ReadExpected(in, "decision_interval", error) || !(in >> loaded.decisionInterval)) return false;
    if (!ReadExpected(in, "model_params", error) ||
        !(in >> loaded.model.learningRate
              >> loaded.model.discount
              >> loaded.model.epsilon
              >> loaded.model.minEpsilon
              >> loaded.model.epsilonDecay
              >> loaded.model.turnRate
              >> loaded.model.clearanceGuidance
              >> loaded.model.lastTD
              >> loaded.model.rawFeatureCount
              >> loaded.model.featureCount
              >> loaded.model.memoryCapacity
              >> loaded.model.eliteMemoryCapacity
              >> loaded.model.replayBatchSize)) {
        error = "bad model_params";
        return false;
    }

    if (!ReadFloatMatrix(in, "weights", loaded.model.weights, 128, 1024, error)) return false;
    if (!ReadFloatVector(in, "action_bias", loaded.model.actionBias, 128, error)) return false;
    if (!ReadFloatVector(in, "exploration_weight", loaded.model.explorationWeight, 128, error)) return false;

    if (!ReadExpected(in, "runner_stats", error) ||
        !(in >> loaded.episode
              >> loaded.successes
              >> loaded.timeouts
              >> loaded.stucks
              >> loaded.episodesSinceBest
              >> loaded.manualGuidanceReplays
              >> loaded.manualGuidanceExperiences
              >> loaded.bestRunRehearsals
              >> loaded.bestTime
              >> loaded.bestReward
              >> loaded.bestSuccessReward
              >> loaded.lastEpisodeReward
              >> loaded.lastEpisodeTime
              >> loaded.lastDistance
              >> loaded.totalBestPathReward)) {
        error = "bad runner_stats";
        return false;
    }

    if (!ReadIntVector(in, "lifetime_action_counts", loaded.lifetimeActionCounts, 128, error)) return false;
    if (!ReadPolicySnapshot(in, loaded.bestPolicy, error)) return false;
    if (!ReadVector3List(in, "best_trail", loaded.bestTrail, 5000, error)) return false;
    if (!ReadReplayFrames(in, "best_frames", loaded.bestFrames, 8000, error)) return false;
    if (!ReadExperiences(in, "best_experiences", loaded.bestExperiences, 12000, error)) return false;
    if (!ReadRecentEpisodes(in, loaded.recentEpisodes, error)) return false;
    if (!ReadExpected(in, "end", error)) return false;

    if (!ValidateLoadedRunner(loaded, error)) {
        error = FileNameOnly(path) + ": " + error;
        return false;
    }

    SeedLoadedRunnerReplay(loaded);
    runner = loaded;
    return true;
}

void TrainerPersistenceCounters(const RLTrainer& trainer, int& started, int& finished, int& bestFrames, int& manualGuidance) {
    started = 0;
    finished = 0;
    bestFrames = 0;
    manualGuidance = 0;

    for (const RLRunner& runner : trainer.runners) {
        started += runner.episode;
        finished += runner.successes + runner.timeouts + runner.stucks;
        bestFrames += (int)runner.bestFrames.size();
        manualGuidance += runner.manualGuidanceExperiences + runner.manualGuidanceReplays;
    }
}

void CapturePersistenceState(const RLTrainer& trainer, RLPersistenceState& state) {
    TrainerPersistenceCounters(
        trainer,
        state.savedStartedEpisodes,
        state.savedFinishedEpisodes,
        state.savedBestFrameCount,
        state.savedManualGuidance
    );
}

bool TrainerChangedSinceAutosave(const RLTrainer& trainer, const RLPersistenceState& state) {
    int started = 0;
    int finished = 0;
    int bestFrames = 0;
    int manualGuidance = 0;
    TrainerPersistenceCounters(trainer, started, finished, bestFrames, manualGuidance);

    return started != state.savedStartedEpisodes ||
           finished != state.savedFinishedEpisodes ||
           bestFrames != state.savedBestFrameCount ||
           manualGuidance != state.savedManualGuidance;
}

int SaveTrainerAutosaves(const RLTrainer& trainer, std::string& error) {
    std::string directoryError;
    if (!EnsureModelDirectory(directoryError)) {
        error = directoryError;
        return 0;
    }

    int saved = 0;
    for (const RLRunner& runner : trainer.runners) {
        std::string saveError;
        if (SaveRunnerModelToFile(RunnerAutosavePath(runner), runner, saveError)) {
            saved += 1;
        } else if (error.empty()) {
            error = saveError;
        }
    }

    return saved;
}

int LoadTrainerAutosaves(RLTrainer& trainer, std::string& error) {
    int loaded = 0;
    for (RLRunner& runner : trainer.runners) {
        std::string path = RunnerAutosavePath(runner);
        std::error_code ec;
        if (!fs::exists(path, ec)) continue;

        std::string loadError;
        if (LoadRunnerModelFromFile(path, runner, loadError)) {
            loaded += 1;
        } else if (error.empty()) {
            error = loadError;
        }
    }

    return loaded;
}

void AutoSaveTrainerModels(const RLTrainer& trainer, float dt, RLPersistenceState& persistence, RLModelLibrary& library) {
    persistence.autosaveTimer += dt;
    if (persistence.autosaveTimer < RL_AUTOSAVE_INTERVAL || !TrainerChangedSinceAutosave(trainer, persistence)) {
        return;
    }

    std::string error;
    int saved = SaveTrainerAutosaves(trainer, error);
    persistence.autosaveTimer = 0.0f;

    if (saved > 0) {
        CapturePersistenceState(trainer, persistence);
        RefreshModelLibrary(library);
        SetModelLibraryStatus(library, "AUTOSAVED " + std::to_string(saved) + " MODELS");
    } else {
        SetModelLibraryStatus(library, "AUTOSAVE FAILED: " + error);
    }
}

bool SaveSelectedModelSnapshot(const RLTrainer& trainer, RLModelLibrary& library, std::string& message) {
    if (trainer.runners.empty()) {
        message = "SNAPSHOT FAILED: no active model";
        SetModelLibraryStatus(library, message);
        return false;
    }

    const RLRunner& runner = trainer.ActiveRunner();
    std::string path = RunnerSnapshotPath(runner);
    std::string error;
    if (!SaveRunnerModelToFile(path, runner, error)) {
        message = "SNAPSHOT FAILED: " + error;
        SetModelLibraryStatus(library, message);
        return false;
    }

    RefreshModelLibrary(library);
    for (int i = 0; i < (int)library.files.size(); ++i) {
        if (library.files[i] == path) {
            library.selectedFile = i;
            break;
        }
    }

    message = "SNAPSHOT SAVED: " + FileNameOnly(path);
    SetModelLibraryStatus(library, message);
    return true;
}

bool LoadSelectedModelFile(RLTrainer& trainer, const std::vector<Box>& obstacles, RLModelLibrary& library, RLPersistenceState& persistence, std::string& message) {
    RefreshModelLibrary(library);
    if (!library.HasFiles()) {
        message = "LOAD FAILED: no model files in rl_models/";
        SetModelLibraryStatus(library, message);
        return false;
    }

    std::string path = library.SelectedPath();
    std::string error;
    RLRunner& runner = trainer.ActiveRunner();
    if (!LoadRunnerModelFromFile(path, runner, error)) {
        message = "LOAD FAILED: " + error;
        SetModelLibraryStatus(library, message);
        return false;
    }

    runner.StartEpisode(trainer.start, trainer.goal, obstacles);
    CapturePersistenceState(trainer, persistence);
    message = "LOADED: " + FileNameOnly(path);
    SetModelLibraryStatus(library, message);
    return true;
}

std::string ManualRunLabel(float finishTime, const std::string& fileName) {
    std::ostringstream text;
    text << std::fixed << std::setprecision(2) << finishTime << "s";
    if (!fileName.empty()) text << "  " << fileName;
    return text.str();
}

std::string ManualRunSavePath(const ManualRunReplay& replay) {
    int centiseconds = (int)roundf(replay.finishTime * 100.0f);
    std::ostringstream name;
    name << "manual_"
         << std::setw(5) << std::setfill('0') << centiseconds
         << "_" << TimestampForModelFile()
         << MANUAL_RUN_EXTENSION;
    return (fs::path(ManualRunDirectoryPath()) / name.str()).string();
}

bool EnsureManualRunDirectory(std::string& error) {
    std::error_code ec;
    fs::path directory(ManualRunDirectoryPath());
    if (fs::exists(directory, ec)) {
        if (fs::is_directory(directory, ec)) return true;
        error = directory.string() + " exists but is not a directory";
        return false;
    }

    if (!fs::create_directories(directory, ec) || ec) {
        error = "could not create " + directory.string() + ": " + ec.message();
        return false;
    }

    return true;
}

bool WriteManualRunToFile(const std::string& path, const ManualRunReplay& replay, std::string& error) {
    if (!replay.HasRun()) {
        error = "manual run is not finished";
        return false;
    }

    std::error_code ec;
    fs::path outputPath(path);
    fs::create_directories(outputPath.parent_path(), ec);
    if (ec) {
        error = "could not create manual run folder: " + ec.message();
        return false;
    }

    std::string tempPath = path + ".tmp";
    std::ofstream out(tempPath);
    if (!out.is_open()) {
        error = "could not open " + tempPath;
        return false;
    }

    out << std::fixed << std::setprecision(9);
    out << MANUAL_RUN_MAGIC << "\n";
    out << "saved_at " << std::quoted(TimestampForModelText()) << "\n";
    out << "finish_time " << replay.finishTime << "\n";
    WriteReplayFrames(out, "frames", replay.frames);
    WriteVector3List(out, "trail", replay.trail);
    WriteExperiences(out, "experiences", replay.experiences);
    WriteManualInputs(out, replay.inputs);
    out << "end\n";

    out.close();
    if (!out.good()) {
        error = "could not finish writing " + tempPath;
        return false;
    }

    fs::remove(outputPath, ec);
    ec.clear();
    fs::rename(tempPath, outputPath, ec);
    if (ec) {
        error = "could not replace " + path + ": " + ec.message();
        return false;
    }

    return true;
}

bool ReadManualRunFromFile(const std::string& path, ManualRunReplay& replay, std::string& error) {
    std::ifstream in(path);
    if (!in.is_open()) {
        error = "could not open " + path;
        return false;
    }

    std::string magic;
    if (!(in >> magic) || magic != MANUAL_RUN_MAGIC) {
        error = "not a manual run file: " + FileNameOnly(path);
        return false;
    }

    ManualRunReplay loaded;
    std::string savedAt;
    if (!ReadExpected(in, "saved_at", error) || !(in >> std::quoted(savedAt))) return false;
    if (!ReadExpected(in, "finish_time", error) || !(in >> loaded.finishTime)) {
        error = "bad finish_time";
        return false;
    }
    if (!ReadReplayFrames(in, "frames", loaded.frames, 12000, error)) return false;
    if (!ReadVector3List(in, "trail", loaded.trail, 8000, error)) return false;
    if (!ReadExperiences(in, "experiences", loaded.experiences, 16000, error)) return false;

    std::string token;
    if (!(in >> token)) {
        error = "missing token: inputs or end";
        return false;
    }
    if (token == "inputs") {
        if (!ReadManualInputsAfterLabel(in, loaded.inputs, 16000, error)) return false;
        if (!ReadExpected(in, "end", error)) return false;
    } else if (token != "end") {
        error = "expected inputs or end, found " + token;
        return false;
    }

    loaded.finished = !loaded.frames.empty() && !loaded.trail.empty() && loaded.finishTime > 0.0f;
    loaded.savedToLibrary = true;
    loaded.sourcePath = path;
    loaded.label = ManualRunLabel(loaded.finishTime, FileNameOnly(path));
    loaded.BackfillInputsFromReplayData();
    if (!loaded.HasRun()) {
        error = "manual run file has no complete replay data";
        return false;
    }

    replay = loaded;
    return true;
}

void SortManualRunLibrary(ManualRunLibrary& library) {
    std::sort(library.runs.begin(), library.runs.end(), [](const ManualRunReplay& a, const ManualRunReplay& b) {
        if (fabsf(a.finishTime - b.finishTime) > 0.001f) return a.finishTime < b.finishTime;
        return FileNameOnly(a.sourcePath) < FileNameOnly(b.sourcePath);
    });
    if (!library.runs.empty()) {
        library.selectedRun = std::max(0, std::min(library.selectedRun, (int)library.runs.size() - 1));
    } else {
        library.selectedRun = 0;
    }
}

bool ManualRunQualifiesForTopSaves(const ManualRunLibrary& library, const ManualRunReplay& replay) {
    if (!replay.HasRun()) return false;
    if ((int)library.runs.size() < MANUAL_RUN_KEEP_COUNT) return true;

    const ManualRunReplay& slowestSaved = library.runs.back();
    return replay.finishTime < slowestSaved.finishTime - 0.001f;
}

void PruneManualRunLibrary(ManualRunLibrary& library) {
    SortManualRunLibrary(library);

    while ((int)library.runs.size() > MANUAL_RUN_KEEP_COUNT) {
        ManualRunReplay removed = library.runs.back();
        library.runs.pop_back();

        if (!removed.sourcePath.empty()) {
            std::error_code ec;
            fs::remove(removed.sourcePath, ec);
        }
    }

    SortManualRunLibrary(library);
}

void RefreshManualRunLibrary(ManualRunLibrary& library) {
    std::string selectedPath = library.HasRuns() ? library.ActiveRun().sourcePath : "";
    std::string directoryError;
    if (!EnsureManualRunDirectory(directoryError)) {
        library.status = "MANUAL RUN FOLDER FAILED: " + directoryError;
        library.runs.clear();
        library.selectedRun = 0;
        return;
    }

    std::vector<ManualRunReplay> runs;
    std::string firstError;
    std::error_code ec;
    fs::directory_iterator it(ManualRunDirectoryPath(), ec);
    fs::directory_iterator end;

    while (!ec && it != end) {
        std::error_code entryError;
        const fs::directory_entry& entry = *it;
        if (entry.is_regular_file(entryError) &&
            entry.path().extension() == MANUAL_RUN_EXTENSION &&
            !IsMigrationBackupFile(entry.path().string())) {
            ManualRunReplay replay;
            std::string loadError;
            if (ReadManualRunFromFile(entry.path().string(), replay, loadError)) {
                runs.push_back(replay);
            } else if (firstError.empty()) {
                firstError = loadError;
            }
        }
        it.increment(ec);
    }

    library.runs = runs;
    PruneManualRunLibrary(library);
    library.selectedRun = 0;
    for (int i = 0; i < (int)library.runs.size(); ++i) {
        if (library.runs[i].sourcePath == selectedPath) {
            library.selectedRun = i;
            break;
        }
    }

    if (!firstError.empty()) {
        library.status = "MANUAL RUN LOAD FAILED: " + firstError;
    } else if (library.HasRuns()) {
        library.status = "LOADED TOP " + std::to_string(library.runs.size()) + " MANUAL RUNS";
    } else {
        library.status = "MANUAL RUN SAVES READY";
    }
}

bool AddSavedManualRun(ManualRunLibrary& library, ManualRunReplay& replay, std::string& message) {
    if (!replay.HasRun()) {
        message = "MANUAL RUN SAVE FAILED: run is not complete";
        library.status = message;
        return false;
    }

    SortManualRunLibrary(library);
    if (!ManualRunQualifiesForTopSaves(library, replay)) {
        replay.savedToLibrary = true;
        message = "MANUAL RUN NOT SAVED: top 2 are faster";
        library.status = message;
        return false;
    }

    std::string directoryError;
    if (!EnsureManualRunDirectory(directoryError)) {
        message = "MANUAL RUN SAVE FAILED: " + directoryError;
        library.status = message;
        return false;
    }

    std::string path = ManualRunSavePath(replay);
    std::string error;
    if (!WriteManualRunToFile(path, replay, error)) {
        message = "MANUAL RUN SAVE FAILED: " + error;
        library.status = message;
        return false;
    }

    replay.savedToLibrary = true;
    replay.sourcePath = path;
    replay.label = ManualRunLabel(replay.finishTime, FileNameOnly(path));

    library.runs.push_back(replay);
    PruneManualRunLibrary(library);
    library.selectedRun = 0;
    for (int i = 0; i < (int)library.runs.size(); ++i) {
        if (library.runs[i].sourcePath == path) {
            library.selectedRun = i;
            break;
        }
    }

    message = "TOP 2 MANUAL RUN SAVED: " + FileNameOnly(path);
    library.status = message;
    return true;
}

void SelectPreviousManualRun(ManualRunLibrary& library) {
    if (!library.HasRuns()) {
        library.status = "NO SAVED MANUAL RUNS";
        return;
    }

    library.selectedRun = (library.selectedRun - 1 + (int)library.runs.size()) % (int)library.runs.size();
    library.status = "SELECTED MANUAL RUN: " + library.ActiveLabel();
}

void SelectNextManualRun(ManualRunLibrary& library) {
    if (!library.HasRuns()) {
        library.status = "NO SAVED MANUAL RUNS";
        return;
    }

    library.selectedRun = (library.selectedRun + 1) % (int)library.runs.size();
    library.status = "SELECTED MANUAL RUN: " + library.ActiveLabel();
}

ManualRunReplay BestManualRunFromLibrary(const ManualRunLibrary& library) {
    if (!library.HasRuns()) return {};
    return library.runs.front();
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
    Camera3D spectatorRaceCamera = MakeSpectatorRaceCamera(world.startPoint, world.goalPoint);
    ManualRunReplay manualReplay;
    ManualRunReplay manualBestReplay;
    ManualRunLibrary manualRunLibrary;
    RLModelLibrary modelLibrary;
    RLPersistenceState persistenceState;

    bool rlAutoplay = true;
    bool rlTrainingView = true;
    bool bestRunView = false;
    bool turboTraining = false;
    bool soloTraining = false;
    bool manualRaceStarted = false;
    bool manualRunView = false;
    bool spectatorRaceView = false;
    bool spectatorRaceRunning = false;
    bool spectatorPovFreeLook = false;
    bool uiCursorEnabled = false;
    int spectatorRaceCameraMode = 0;
    int spectatorRaceSpeedIndex = 2;
    int trainingSpeedIndex = 0;
    int trainingStepsLastFrame = 0;
    float trainingSimSecondsLastFrame = 0.0f;
    float trainingPendingSimSeconds = 0.0f;
    float trainingActualSpeedScale = 0.0f;
    float trainingActualSimWindow = 0.0f;
    float trainingActualWallWindow = 0.0f;
    float bestReplayTimer = 0.0f;
    float manualRunReplayTimer = 0.0f;
    float spectatorRaceTimer = 0.0f;
    float spectatorPovYaw = 0.0f;
    float spectatorPovPitch = 0.0f;
    float reportFlashTimer = 0.0f;
    float reportAutosaveTimer = 0.0f;
    std::string reportMessage;

    std::string startupMigrationMessage;
    MigrateLegacyPersistence(startupMigrationMessage);

    RefreshModelLibrary(modelLibrary);
    RefreshManualRunLibrary(manualRunLibrary);
    manualBestReplay = BestManualRunFromLibrary(manualRunLibrary);
    std::string startupLoadError;
    int loadedModels = LoadTrainerAutosaves(trainer, startupLoadError);
    if (loadedModels > 0) {
        SetModelLibraryStatus(modelLibrary, "LOADED " + std::to_string(loadedModels) + " AUTOSAVED MODELS");
    } else if (!startupLoadError.empty()) {
        SetModelLibraryStatus(modelLibrary, "LOAD FAILED: " + startupLoadError);
    }

    int seededManualRuns = 0;
    for (const ManualRunReplay& savedRun : manualRunLibrary.runs) {
        if (!savedRun.experiences.empty()) {
            trainer.LearnFromManualReplay(savedRun.experiences, savedRun.finishTime);
            seededManualRuns += 1;
        }
    }

    player.ResetRL(world.startPoint, world.goalPoint);
    trainer.ResetEpisodes(world.obstacles);
    if (seededManualRuns > 0) {
        std::string seedAutosaveError;
        SaveTrainerAutosaves(trainer, seedAutosaveError);
    }
    CapturePersistenceState(trainer, persistenceState);
    RefreshModelLibrary(modelLibrary);
    if (!startupMigrationMessage.empty()) {
        reportMessage = startupMigrationMessage;
        reportFlashTimer = 4.0f;
    } else if (seededManualRuns > 0) {
        reportMessage = TextFormat("SEEDED AI FROM %d MANUAL RUNS", seededManualRuns);
        reportFlashTimer = 3.0f;
    }

    float spread = 0.0f;

    auto writeCurrentTrainingReport = [&](std::string& error) {
        return WriteTrainingReport(
            TrainingReportPath(),
            trainer,
            world,
            trainingSpeedIndex,
            trainingActualSpeedScale,
            trainingPendingSimSeconds,
            trainingStepsLastFrame,
            trainingSimSecondsLastFrame,
            turboTraining,
            soloTraining,
            manualBestReplay,
            manualRunLibrary,
            &error
        );
    };

    while (!WindowShouldClose()) {
        SetTargetFPS((rlAutoplay && rlTrainingView) ? 120 : 144);
        float dt = GetFrameTime();

        if (IsKeyPressed(KEY_T)) {
            rlAutoplay = !rlAutoplay;
            katana.active = false;
            manualRunView = false;
            spectatorRaceView = false;
            spectatorRaceRunning = false;
            spectatorPovFreeLook = false;
            manualRunReplayTimer = 0.0f;
            spectatorRaceTimer = 0.0f;
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

            RLModelFileAction modelAction = HandleModelFileControls();
            if (modelAction == RLModelFileAction::Previous) {
                SelectPreviousModelFile(modelLibrary);
            } else if (modelAction == RLModelFileAction::Next) {
                SelectNextModelFile(modelLibrary);
            } else if (modelAction == RLModelFileAction::Load) {
                std::string modelMessage;
                LoadSelectedModelFile(trainer, world.obstacles, modelLibrary, persistenceState, modelMessage);
                reportMessage = modelMessage;
                reportFlashTimer = 2.5f;
                bestReplayTimer = 0.0f;
                trainingPendingSimSeconds = 0.0f;
                trainingActualSimWindow = 0.0f;
                trainingActualWallWindow = 0.0f;
            } else if (modelAction == RLModelFileAction::SaveSnapshot) {
                std::string modelMessage;
                SaveSelectedModelSnapshot(trainer, modelLibrary, modelMessage);
                reportMessage = modelMessage;
                reportFlashTimer = 2.5f;
            }
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
                if (spectatorRaceView) {
                    spectatorRaceTimer = 0.0f;
                    spectatorRaceRunning = false;
                    spectatorPovFreeLook = false;
                } else if (manualRunView) {
                    manualRunReplayTimer = 0.0f;
                } else {
                    player.ResetRL(world.startPoint, world.goalPoint);
                    manualReplay.Reset();
                    manualRaceStarted = false;
                }
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

            if (IsKeyPressed(KEY_G)) {
                spectatorRaceView = !spectatorRaceView;
                spectatorRaceRunning = false;
                spectatorPovFreeLook = false;
                spectatorRaceTimer = 0.0f;
                manualRunView = false;
                manualRunReplayTimer = 0.0f;
                katana.active = false;
                hit.active = false;
                if (spectatorRaceView) {
                    spectatorRaceCamera = MakeSpectatorRaceCamera(world.startPoint, world.goalPoint);
                    spectatorRaceCameraMode = 0;
                    spectatorRaceSpeedIndex = 2;
                    reportMessage = "SPECTATOR RACE READY: ENTER TO PLAY";
                } else {
                    reportMessage = "SPECTATOR RACE OFF";
                }
                reportFlashTimer = 2.5f;
            }

            if (spectatorRaceView) {
                manualRunView = false;
                katana.active = false;
                hit.active = false;
                const RLRunner& raceRunner = trainer.ActiveRunner();
                float raceDuration = SpectatorRaceDuration(manualBestReplay, raceRunner);
                bool raceReady = manualBestReplay.HasRun() && raceRunner.HasBestRun() && raceDuration > 0.001f;

                if (IsKeyPressed(KEY_ENTER)) {
                    if (raceReady) {
                        if (spectatorRaceTimer >= raceDuration - 0.001f) {
                            spectatorRaceTimer = 0.0f;
                        }
                        spectatorRaceRunning = !spectatorRaceRunning;
                    } else {
                        reportMessage = "SPECTATOR RACE NEEDS MANUAL + AI BEST RUNS";
                        reportFlashTimer = 2.5f;
                    }
                }

                if (IsKeyPressed(KEY_C)) {
                    spectatorRaceCameraMode = (spectatorRaceCameraMode + 1) % 3;
                    spectatorPovFreeLook = false;
                }

                if (spectatorRaceCameraMode != 0 && IsKeyPressed(KEY_F)) {
                    if (!spectatorPovFreeLook) {
                        Camera3D basePovCamera = {};
                        bool hasBasePovCamera = false;
                        if (spectatorRaceCameraMode == 1 && manualBestReplay.HasRun()) {
                            basePovCamera = ManualRaceFrameAt(manualBestReplay, spectatorRaceTimer, world.startPoint).camera;
                            hasBasePovCamera = true;
                        } else if (spectatorRaceCameraMode == 2 && raceRunner.HasBestRun()) {
                            basePovCamera = raceRunner.BestRunCamera(spectatorRaceTimer);
                            hasBasePovCamera = true;
                        }

                        if (hasBasePovCamera) {
                            CameraLookAngles(basePovCamera, spectatorPovYaw, spectatorPovPitch);
                            spectatorPovFreeLook = true;
                        }
                    } else {
                        spectatorPovFreeLook = false;
                    }
                }

                if (IsKeyPressed(KEY_COMMA)) {
                    spectatorRaceSpeedIndex = std::max(0, spectatorRaceSpeedIndex - 1);
                }
                if (IsKeyPressed(KEY_PERIOD)) {
                    spectatorRaceSpeedIndex = std::min(SPECTATOR_RACE_SPEED_OPTION_COUNT - 1, spectatorRaceSpeedIndex + 1);
                }

                float scrubSpeed = IsKeyDown(KEY_LEFT_SHIFT) ? 10.0f : 3.0f;
                if (raceReady && IsKeyDown(KEY_J)) {
                    spectatorRaceTimer -= dt * scrubSpeed;
                    spectatorRaceRunning = false;
                }
                if (raceReady && IsKeyDown(KEY_L)) {
                    spectatorRaceTimer += dt * scrubSpeed;
                    spectatorRaceRunning = false;
                }

                if (raceReady && spectatorRaceRunning) {
                    spectatorRaceTimer += dt * SPECTATOR_RACE_SPEED_OPTIONS[spectatorRaceSpeedIndex];
                }

                if (raceReady) {
                    spectatorRaceTimer = Clamp(spectatorRaceTimer, 0.0f, raceDuration);
                    if (spectatorRaceTimer >= raceDuration - 0.001f) spectatorRaceRunning = false;
                } else {
                    spectatorRaceTimer = 0.0f;
                    spectatorRaceRunning = false;
                }

                if (spectatorRaceCameraMode == 0) {
                    UpdateSpectatorFreeCamera(spectatorRaceCamera, dt);
                    spectatorPovFreeLook = false;
                } else if (spectatorPovFreeLook) {
                    UpdateSpectatorPovFreeLook(spectatorPovYaw, spectatorPovPitch);
                }
            }

            if (!spectatorRaceView && IsKeyPressed(KEY_F)) {
                if (manualRunLibrary.HasRuns()) {
                    manualRunView = !manualRunView;
                    manualRunReplayTimer = 0.0f;
                    katana.active = false;
                    hit.active = false;
                    manualRunLibrary.status = manualRunView
                        ? "VIEWING MANUAL RUN: " + manualRunLibrary.ActiveLabel()
                        : "LIVE MANUAL MODE";
                } else {
                    manualRunView = false;
                    manualRunLibrary.status = "NO SAVED MANUAL RUNS";
                    reportMessage = "NO SAVED MANUAL RUNS";
                    reportFlashTimer = 2.5f;
                }
            }

            if (!spectatorRaceView && !manualRunView && manualRunLibrary.HasRuns()) {
                if (IsKeyPressed(KEY_LEFT_BRACKET)) SelectPreviousManualRun(manualRunLibrary);
                if (IsKeyPressed(KEY_RIGHT_BRACKET)) SelectNextManualRun(manualRunLibrary);
            }

            if (!spectatorRaceView && manualRunView && manualRunLibrary.HasRuns()) {
                if (IsKeyPressed(KEY_LEFT_BRACKET)) {
                    SelectPreviousManualRun(manualRunLibrary);
                    manualRunReplayTimer = 0.0f;
                }
                if (IsKeyPressed(KEY_RIGHT_BRACKET)) {
                    SelectNextManualRun(manualRunLibrary);
                    manualRunReplayTimer = 0.0f;
                }

                const ManualRunReplay& replay = manualRunLibrary.ActiveRun();
                manualRunReplayTimer += dt;
                if (replay.finishTime > 0.001f) {
                    while (manualRunReplayTimer > replay.finishTime) manualRunReplayTimer -= replay.finishTime;
                }
                katana.active = false;
            } else if (!spectatorRaceView) {
                manualRunView = false;

            if (IsKeyPressed(KEY_Q) && katana.cooldown <= 0.0f) {
                katana.Trigger();
            }

            bool manualW = IsKeyDown(KEY_W);
            bool manualA = IsKeyDown(KEY_A);
            bool manualS = IsKeyDown(KEY_S);
            bool manualD = IsKeyDown(KEY_D);
            bool manualJumpPressed = IsKeyPressed(KEY_SPACE);
            bool manualJumpHeld = IsKeyDown(KEY_SPACE);
            bool manualDashPressed = IsKeyPressed(KEY_E);
            bool manualSlide = IsKeyDown(KEY_LEFT_CONTROL) || IsKeyDown(KEY_LEFT_SUPER) || IsKeyDown(KEY_C);
            bool manualStartInput =
                manualW ||
                manualA ||
                manualS ||
                manualD ||
                manualJumpHeld ||
                manualDashPressed ||
                IsKeyDown(KEY_LEFT_SHIFT);

            if (!manualRaceStarted && manualStartInput) {
                manualRaceStarted = true;
            }

            float playerTimeBeforeUpdate = player.rlEpisodeTime;
            Vector3 playerPositionBeforeUpdate = player.position;
            Camera3D playerCameraBeforeUpdate = player.camera;
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
                ManualInputFrame manualInput;
                manualInput.w = manualW;
                manualInput.a = manualA;
                manualInput.s = manualS;
                manualInput.d = manualD;
                manualInput.jump = manualJumpHeld;
                manualInput.dash = manualDashPressed || player.isDashing || player.dashVisualTimer > 0.0f;
                manualInput.slide = manualSlide;
                manualInput.slash = katana.active;
                manualInput.isDashing = player.isDashing;
                manualInput.isWallRunning = player.isWallRunning;
                manualInput.dashCooldown = player.dashCooldown;
                manualInput.dashVisualTimer = player.dashVisualTimer;
                if (manualReplay.frames.empty() && playerTimeBeforeUpdate <= 0.0f) {
                    manualReplay.SeedStartFrame(playerPositionBeforeUpdate, playerCameraBeforeUpdate, manualAction, manualInput);
                }
                manualReplay.Record(player, manualStateBefore, manualAction, player.rlReward, manualStateAfter, manualInput);
                if (manualReplay.HasRun() && !manualReplay.savedToLibrary) {
                    std::string manualRunMessage;
                    AddSavedManualRun(manualRunLibrary, manualReplay, manualRunMessage);
                    reportMessage = manualRunMessage;
                    reportFlashTimer = 2.5f;
                }
                if (manualReplay.IsBetterThan(manualBestReplay)) {
                    manualBestReplay = manualReplay;
                    trainer.LearnFromManualReplay(manualBestReplay.experiences, manualBestReplay.finishTime);
                    std::string autosaveError;
                    int saved = SaveTrainerAutosaves(trainer, autosaveError);
                    if (saved > 0) {
                        CapturePersistenceState(trainer, persistenceState);
                        RefreshModelLibrary(modelLibrary);
                        SetModelLibraryStatus(modelLibrary, "AUTOSAVED MANUAL GUIDANCE");
                    } else {
                        SetModelLibraryStatus(modelLibrary, "AUTOSAVE FAILED: " + autosaveError);
                    }
                    reportMessage = TextFormat("MANUAL BEST SAVED + FED TO AI: %.2fs", manualBestReplay.finishTime);
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
        }

        if (IsKeyPressed(KEY_P)) {
            std::string reportError;
            bool savedReport = writeCurrentTrainingReport(reportError);
            reportMessage = savedReport
                ? "REPORT SAVED: " + TrainingReportPath()
                : "REPORT FAILED: " + reportError;
            reportFlashTimer = 2.5f;
        }

        AutoSaveTrainerModels(trainer, dt, persistenceState, modelLibrary);
        reportAutosaveTimer += dt;
        if (reportAutosaveTimer >= RL_REPORT_AUTOSAVE_INTERVAL) {
            reportAutosaveTimer = 0.0f;
            std::string reportError;
            bool savedReport = writeCurrentTrainingReport(reportError);
            if (!savedReport) {
                reportMessage = "AUTO REPORT FAILED: " + reportError;
                reportFlashTimer = 3.0f;
            }
        }

        if (reportFlashTimer > 0.0f) {
            reportFlashTimer = fmaxf(0.0f, reportFlashTimer - dt);
        }

        hit.Update(dt);
        katana.Update(dt);

        const Player& viewPlayer = rlAutoplay ? trainer.ActiveRunner().player : player;
        const RLRunner& activeRunner = trainer.ActiveRunner();
        bool bestRunCameraView = rlAutoplay && bestRunView && !rlTrainingView && activeRunner.HasBestRun();
        bool spectatorRaceCameraView = !rlAutoplay && spectatorRaceView;
        bool manualRunCameraView = !rlAutoplay && manualRunView && manualRunLibrary.HasRuns() && !spectatorRaceCameraView;
        ManualInputFrame manualReplayInput = manualRunCameraView
            ? manualRunLibrary.ActiveRun().ManualInputAt(manualRunReplayTimer)
            : (spectatorRaceCameraView && spectatorRaceCameraMode == 1 && manualBestReplay.HasRun()
                ? manualBestReplay.ManualInputAt(spectatorRaceTimer)
                : ManualInputFrame());
        RLReplayFrame spectatorAiFrame = (spectatorRaceCameraView && activeRunner.HasBestRun())
            ? activeRunner.BestRunFrame(spectatorRaceTimer)
            : RLReplayFrame{};
        bool spectatorAiDash = false;
        if (spectatorRaceCameraView && spectatorRaceCameraMode == 2 && spectatorAiFrame.action >= 0 && spectatorAiFrame.action < (int)RLActions().size()) {
            spectatorAiDash = RLActions()[spectatorAiFrame.action].dash;
        }
        bool spectatorManualDash = manualReplayInput.isDashing || manualReplayInput.dashVisualTimer > 0.0f;
        bool spectatorDashActive = spectatorRaceCameraMode == 1 ? spectatorManualDash : (spectatorRaceCameraMode == 2 && spectatorAiDash);
        Camera3D spectatorActiveCamera = spectatorRaceCamera;
        if (spectatorRaceCameraView && spectatorRaceCameraMode == 1 && manualBestReplay.HasRun()) {
            spectatorActiveCamera = ManualRaceFrameAt(manualBestReplay, spectatorRaceTimer, world.startPoint).camera;
        } else if (spectatorRaceCameraView && spectatorRaceCameraMode == 2 && activeRunner.HasBestRun()) {
            spectatorActiveCamera = activeRunner.BestRunCamera(spectatorRaceTimer);
        }
        if (spectatorRaceCameraView && spectatorRaceCameraMode != 0 && spectatorPovFreeLook) {
            spectatorActiveCamera = SpectatorPovFreeLookCamera(spectatorActiveCamera, spectatorPovYaw, spectatorPovPitch);
        }
        bool dashActiveForUi = manualRunCameraView
            ? (manualReplayInput.isDashing || manualReplayInput.dashVisualTimer > 0.0f)
            : (spectatorRaceCameraView ? spectatorDashActive : viewPlayer.isDashing);
        bool wallRunningForUi = manualRunCameraView ? manualReplayInput.isWallRunning : (!spectatorRaceCameraView && viewPlayer.isWallRunning);
        float dashCooldownForUi = manualRunCameraView ? manualReplayInput.dashCooldown : (spectatorRaceCameraView ? 0.0f : viewPlayer.dashCooldown);
        float dashVisualTimerForUi = manualRunCameraView
            ? manualReplayInput.dashVisualTimer
            : (spectatorRaceCameraView ? (spectatorRaceCameraMode == 1 ? manualReplayInput.dashVisualTimer : 0.0f) : viewPlayer.dashVisualTimer);

        if (bestRunCameraView) {
            bestReplayTimer += dt;
            float duration = activeRunner.BestRunDuration();
            if (duration > 0.001f) {
                while (bestReplayTimer > duration) bestReplayTimer -= duration;
            }
        } else if (!bestRunView) {
            bestReplayTimer = 0.0f;
        }

        Camera3D replayCamera = bestRunCameraView
            ? activeRunner.BestRunCamera(bestReplayTimer)
            : (spectatorRaceCameraView
                ? spectatorActiveCamera
                : (manualRunCameraView ? ManualRaceFrameAt(manualRunLibrary.ActiveRun(), manualRunReplayTimer, world.startPoint).camera : viewPlayer.camera));

        float hSpeed = spectatorRaceCameraView ? 0.0f : viewPlayer.GetHorizontalSpeed();
        float targetSpread = spectatorRaceCameraView ? 0.0f : (hSpeed * 1.35f + (!viewPlayer.onGround ? 12.0f : 0.0f));
        if (dashActiveForUi) targetSpread += 6.0f;
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
                        if (spectatorRaceCameraView) {
                            DrawSpectatorRace3D(
                                manualBestReplay,
                                activeRunner,
                                spectatorRaceTimer,
                                spectatorRaceCameraMode,
                                world.startPoint,
                                world.goalPoint
                            );
                        } else if (!rlAutoplay && !manualRunCameraView) {
                            float manualRaceTime = manualRaceStarted ? player.rlEpisodeTime : 0.0f;
                            DrawManualBestGhost(activeRunner, player.position, world.startPoint, manualRaceTime);
                            DrawManualCompletedGhost(manualBestReplay, player.position, world.startPoint, manualRaceTime);
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
                DrawRLTrainingOverlay(trainer, trainingSpeedIndex, trainingStepsLastFrame, trainingSimSecondsLastFrame, trainingPendingSimSeconds, trainingActualSpeedScale, bestRunView, turboTraining, soloTraining, modelLibrary);
            } else {
                DrawCrosshair(cx, cy, spread);
                if (!rlAutoplay && hit.active) DrawHitmarker(cx, cy, hit.timer, hit.maxTime);
            }

            if ((!rlAutoplay || !rlTrainingView) && dashVisualTimerForUi > 0.0f) {
                float a = dashVisualTimerForUi / DASH_TIME;
                DrawRectangle(0, 0, SCREEN_WIDTH, SCREEN_HEIGHT, Fade(WHITE, a * 0.14f));
            }

            if (!rlAutoplay && katana.active) {
                float slashFlash = 1.0f - katana.Normalized();
                DrawRectangle(0, 0, SCREEN_WIDTH, SCREEN_HEIGHT, Fade((Color){180, 220, 255, 255}, slashFlash * 0.05f));
            }

            if (!rlAutoplay || !rlTrainingView) {
                if (spectatorRaceCameraView) {
                    DrawSpectatorRaceOverlay(
                        manualBestReplay,
                        activeRunner,
                        spectatorRaceTimer,
                        spectatorRaceRunning,
                        SPECTATOR_RACE_SPEED_OPTIONS[spectatorRaceSpeedIndex],
                        spectatorRaceCameraMode,
                        spectatorPovFreeLook,
                        world.startPoint,
                        world.goalPoint
                    );
                    if (spectatorRaceCameraMode == 1) {
                        DrawManualInputOverlay(manualBestReplay, world.goalPoint, spectatorRaceTimer);
                    } else if (spectatorRaceCameraMode == 2) {
                        DrawAIInputOverlay(activeRunner, spectatorRaceTimer);
                    }
                } else {
                DrawStatusBar(20, SCREEN_HEIGHT - 70, 300, 25, viewPlayer.health, MAX_HEALTH, RED, "HEALTH");

                float dashStatus = 1.0f - Clamp(dashCooldownForUi / DASH_COOLDOWN_TIME, 0.0f, 1.0f);
                DrawStatusBar(
                    20,
                    SCREEN_HEIGHT - 40,
                    300,
                    20,
                    dashStatus,
                    1.0f,
                    (dashCooldownForUi <= 0.0f ? GREEN : GOLD),
                    manualRunCameraView ? "DASH (REPLAY)" : (rlAutoplay ? "DASH (RL)" : "DASH (E)")
                );

                const char* modeLabel = rlAutoplay ? "RL SELECTED RUNNER" : (manualRunCameraView ? "MANUAL RUN REPLAY" : "MANUAL");
                DrawText(TextFormat("MODE: %s (T to toggle)", modeLabel), 20, 20, 20, rlAutoplay ? GREEN : LIGHTGRAY);
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
                } else if (manualRunCameraView) {
                    const ManualRunReplay& replay = manualRunLibrary.ActiveRun();
                    DrawText(
                        TextFormat("MANUAL RUN CAMERA: %d/%d  %.1f / %.1fs",
                            manualRunLibrary.selectedRun + 1,
                            (int)manualRunLibrary.runs.size(),
                            manualRunReplayTimer,
                            replay.finishTime),
                        20,
                        202,
                        16,
                        GOLD
                    );
                    std::string runName = TruncateTextToWidth(manualRunLibrary.ActiveLabel(), 520, 16);
                    DrawText(runName.c_str(), 20, 224, 16, LIGHTGRAY);
                    DrawText("F = live manual  [ ] = switch saved run  R = restart replay  G = race", 20, 246, 16, LIGHTGRAY);
                    DrawText(TextFormat("FRAMES: %d  TRAIL: %d  SAMPLES: %d",
                        (int)replay.frames.size(),
                        (int)replay.trail.size(),
                        (int)replay.experiences.size()),
                        20,
                        268,
                        16,
                        SKYBLUE
                    );
                    DrawManualInputOverlay(replay, world.goalPoint, manualRunReplayTimer);
                } else {
                    DrawText("Manual: WASD + Mouse + Space + Shift + E + Q   G = spectator race", 20, 202, 16, LIGHTGRAY);
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
                            DrawManualPlayerGhostStats(manualBestReplay, player.position, world.startPoint, world.goalPoint, manualRaceTime, 20, 268);
                        }
                        if (manualRunLibrary.HasRuns()) {
                            std::string manualRunText = TruncateTextToWidth("F replay POV  [ ] saved: " + manualRunLibrary.ActiveLabel(), 590, 16);
                            DrawText(manualRunText.c_str(), 20, 290, 16, SKYBLUE);
                        }
                    } else {
                        DrawText(TextFormat("GHOST: %s has no best run yet", activeRunner.model.name.c_str()), 20, 224, 16, ORANGE);
                        DrawText("Train a successful run first, then switch back to manual", 20, 246, 16, LIGHTGRAY);
                        if (manualBestReplay.HasRun()) {
                            float manualRaceTime = manualRaceStarted ? player.rlEpisodeTime : 0.0f;
                            DrawManualPlayerGhostStats(manualBestReplay, player.position, world.startPoint, world.goalPoint, manualRaceTime, 20, 268);
                        }
                        if (manualRunLibrary.HasRuns()) {
                            std::string manualRunText = TruncateTextToWidth("F replay POV  [ ] saved: " + manualRunLibrary.ActiveLabel(), 590, 16);
                            DrawText(manualRunText.c_str(), 20, manualBestReplay.HasRun() ? 290 : 268, 16, SKYBLUE);
                        }
                    }
                }

                if (wallRunningForUi) DrawText("WALLRUNNING", cx - 60, cy + 60, 20, GREEN);
                if (dashActiveForUi) DrawText("DASH", cx - 25, cy + 84, 20, SKYBLUE);
                if (!rlAutoplay && katana.active) DrawText("SLASH", cx - 28, cy + 108, 20, WHITE);
                if (rlAutoplay) DrawText("RL CONTROL ACTIVE", cx - 82, cy + 132, 20, YELLOW);

                if (rlAutoplay && viewPlayer.rlDone) {
                    DrawText("EPISODE COMPLETE - RESETTING", cx - 140, cy - 110, 20, GOLD);
                }
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

    std::string finalAutosaveError;
    SaveTrainerAutosaves(trainer, finalAutosaveError);

    std::string finalReportError;
    writeCurrentTrainingReport(finalReportError);

    UnloadShader(ppShader);
    UnloadRenderTexture(renderTarget);
    CloseWindow();
    return 0;
}
