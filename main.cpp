#include "raylib.h"
#include "raymath.h"
#include "rlgl.h"
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

const int TRAINING_SPEED_OPTION_COUNT = 9;
const float TRAINING_SPEED_OPTIONS[TRAINING_SPEED_OPTION_COUNT] = { 1.0f, 10.0f, 100.0f, 1000.0f, 2500.0f, 5000.0f, 10000.0f, 25000.0f, 50000.0f };
const char* TRAINING_SPEED_LABELS[TRAINING_SPEED_OPTION_COUNT] = { "1x", "10x", "100x", "1000x", "2500x", "5000x", "10000x", "25000x", "50000x" };

Rectangle TrainingSpeedButtonRect(int index) {
    int col = index % 4;
    int row = index / 4;
    return {
        24.0f + col * 86.0f,
        214.0f + row * 36.0f,
        index >= 3 ? 78.0f : 66.0f,
        30.0f
    };
}

Rectangle BestRunButtonRect() {
    return { 24.0f, 328.0f, 154.0f, 30.0f };
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

    return currentIndex;
}

bool HandleBestRunButton(bool currentValue) {
    if (CheckCollisionPointRec(GetMousePosition(), BestRunButtonRect()) && IsMouseButtonPressed(MOUSE_LEFT_BUTTON)) {
        return !currentValue;
    }

    if (IsKeyPressed(KEY_B)) return !currentValue;

    return currentValue;
}

int AdvanceRLTraining(RLTrainer& trainer, float frameDt, const std::vector<Box>& obstacles, float speedScale, float& simulatedSeconds) {
    const float fixedStep = 1.0f / 60.0f;
    int maxStepsPerFrame = 2200;
    if (speedScale >= 50000.0f) maxStepsPerFrame = 18000;
    else if (speedScale >= 25000.0f) maxStepsPerFrame = 12000;
    else if (speedScale >= 5000.0f) maxStepsPerFrame = 6500;
    else if (speedScale >= 2500.0f) maxStepsPerFrame = 4200;

    float remaining = Clamp(frameDt, 0.0f, 1.0f / 20.0f) * speedScale;

    int steps = 0;
    simulatedSeconds = 0.0f;

    while (remaining > 0.0001f && steps < maxStepsPerFrame) {
        float step = fminf(fixedStep, remaining);
        trainer.Update(step, obstacles);
        remaining -= step;
        simulatedSeconds += step;
        steps += 1;
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

void DrawRLTrainingOverlay(const RLTrainer& trainer, int speedIndex, int trainingSteps, float simulatedSeconds, bool bestRunView) {
    const RLRunner& active = trainer.ActiveRunner();
    const int best = trainer.BestRunnerIndex();

    DrawRectangle(12, 12, 390, 364, (Color){ 8, 10, 12, 210 });
    DrawRectangleLines(12, 12, 390, 364, Fade(WHITE, 0.24f));
    DrawText(bestRunView ? "BEST RUN VIEW" : "RL TRAINING VIEW", 24, 24, 20, GREEN);
    DrawText("T manual  V camera  B best  R reset  TAB/arrows", 24, 50, 14, LIGHTGRAY);
    DrawText(TextFormat("ACTIVE: %s  ACTION: %s", active.model.name.c_str(), active.CurrentActionName()), 24, 76, 16, WHITE);
    DrawText(TextFormat("EPISODE: %d  TIME: %.1f / %.1f", active.episode, active.player.rlEpisodeTime, RL_MAX_EPISODE_TIME), 24, 100, 16, WHITE);
    DrawText(TextFormat("REWARD: %.2f  TD: %.2f  EPS: %.2f", active.player.rlEpisodeReward, active.model.lastTD, active.model.epsilon), 24, 124, 16, YELLOW);
    DrawText(TextFormat("DIST: %.1f  BEST MODEL: %s", active.lastDistance, trainer.runners[best].model.name.c_str()), 24, 148, 16, SKYBLUE);
    DrawText(TextFormat("END: %s%s%s",
        active.player.rlSucceeded ? "SUCCESS" : "",
        active.player.rlTimedOut ? "TIMEOUT" : "",
        active.player.rlStuck ? "STUCK" : ""),
        24,
        170,
        14,
        active.player.rlSucceeded ? GREEN : (active.player.rlDone ? ORANGE : LIGHTGRAY)
    );
    DrawText(TextFormat("TRAIN SPEED: %s  STEPS: %d  SIM: %.2fs/frame", TRAINING_SPEED_LABELS[speedIndex], trainingSteps, simulatedSeconds), 24, 190, 14, SKYBLUE);

    Vector2 mouse = GetMousePosition();
    for (int i = 0; i < TRAINING_SPEED_OPTION_COUNT; ++i) {
        Rectangle rect = TrainingSpeedButtonRect(i);
        bool activeSpeed = i == speedIndex;
        bool hot = CheckCollisionPointRec(mouse, rect);
        Color fill = activeSpeed ? GREEN : (hot ? (Color){ 72, 78, 84, 235 } : (Color){ 34, 38, 44, 235 });
        Color text = activeSpeed ? BLACK : WHITE;

        DrawRectangleRounded(rect, 0.18f, 6, fill);
        DrawRectangleRoundedLines(rect, 0.18f, 6, Fade(WHITE, activeSpeed ? 0.45f : 0.22f));
        DrawText(TRAINING_SPEED_LABELS[i], (int)(rect.x + (i >= 4 ? 10 : 16)), (int)(rect.y + 7), 16, text);
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
        DrawText(bestInfo, 194, 334, 14, active.HasBestRun() ? GOLD : LIGHTGRAY);
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

    bool rlAutoplay = true;
    bool rlTrainingView = true;
    bool bestRunView = false;
    bool uiCursorEnabled = false;
    int trainingSpeedIndex = 0;
    int trainingStepsLastFrame = 0;
    float trainingSimSecondsLastFrame = 0.0f;
    float bestReplayTimer = 0.0f;

    player.ResetRL(world.startPoint, world.goalPoint);
    trainer.ResetEpisodes(world.obstacles);

    float spread = 0.0f;

    while (!WindowShouldClose()) {
        float dt = GetFrameTime();

        if (IsKeyPressed(KEY_T)) {
            rlAutoplay = !rlAutoplay;
            katana.active = false;
        }

        if (IsKeyPressed(KEY_V)) {
            rlTrainingView = !rlTrainingView;
        }

        if (rlAutoplay && (IsKeyPressed(KEY_TAB) || IsKeyPressed(KEY_RIGHT))) {
            trainer.SelectNext();
            bestReplayTimer = 0.0f;
        }

        if (rlAutoplay && IsKeyPressed(KEY_LEFT)) {
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
            trainingSpeedIndex = HandleTrainingSpeedButtons(trainingSpeedIndex);
            bool previousBestRunView = bestRunView;
            bestRunView = HandleBestRunButton(bestRunView);
            if (bestRunView != previousBestRunView) bestReplayTimer = 0.0f;
        } else {
            bestRunView = false;
            bestReplayTimer = 0.0f;
        }

        if (IsKeyPressed(KEY_R)) {
            if (rlAutoplay) {
                trainer.ResetEpisodes(world.obstacles);
            } else {
                player.ResetRL(world.startPoint, world.goalPoint);
            }
        }

        if (rlAutoplay) {
            trainingStepsLastFrame = AdvanceRLTraining(
                trainer,
                dt,
                world.obstacles,
                TRAINING_SPEED_OPTIONS[trainingSpeedIndex],
                trainingSimSecondsLastFrame
            );
        } else {
            trainingStepsLastFrame = 0;
            trainingSimSecondsLastFrame = 0.0f;

            if (IsKeyPressed(KEY_Q) && katana.cooldown <= 0.0f) {
                katana.Trigger();
            }

            player.Update(
                dt,
                world.obstacles,
                katana.active,
                katana.active,
                katana.Normalized(),
                false
            );

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
        std::vector<float> rlState = viewPlayer.GetRLState(world.obstacles);

        float hSpeed = viewPlayer.GetHorizontalSpeed();
        float targetSpread = hSpeed * 1.35f + (!viewPlayer.onGround ? 12.0f : 0.0f);
        if (viewPlayer.isDashing) targetSpread += 6.0f;
        if (!rlAutoplay && katana.active) targetSpread += 4.0f;
        spread = Lerp(spread, targetSpread, dt * 15.0f);

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
                    if (!rlAutoplay) katana.Draw3D(replayCamera, player.currentBob);
                EndMode3D();
            }
        EndTextureMode();

        BeginDrawing();
            ClearBackground(BLACK);

            BeginShaderMode(ppShader);
                DrawTextureRec(
                    renderTarget.texture,
                    (Rectangle){0, 0, (float)renderTarget.texture.width, (float)-renderTarget.texture.height},
                    (Vector2){0, 0},
                    WHITE
                );
            EndShaderMode();

            int cx = SCREEN_WIDTH / 2;
            int cy = SCREEN_HEIGHT / 2;

            if (rlAutoplay && rlTrainingView) {
                DrawRLRunnerLabels(trainer, trainingCamera, bestRunView);
                DrawRLTrainingOverlay(trainer, trainingSpeedIndex, trainingStepsLastFrame, trainingSimSecondsLastFrame, bestRunView);
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
                DrawText(TextFormat("OBS SIZE: %d", (int)rlState.size()), 20, 176, 20, SKYBLUE);

                if (rlAutoplay) {
                    const RLRunner& active = trainer.ActiveRunner();
                    if (bestRunCameraView) {
                        DrawText(TextFormat("BEST RUN CAMERA: %s  %.1f / %.1fs", active.model.name.c_str(), bestReplayTimer, active.BestRunDuration()), 20, 202, 16, active.model.color);
                        DrawText("V = overview  B = live/best  TAB/arrows = select", 20, 224, 16, LIGHTGRAY);
                    } else if (bestRunView && !active.HasBestRun()) {
                        DrawText(TextFormat("BEST RUN CAMERA: %s has no successful run yet", active.model.name.c_str()), 20, 202, 16, ORANGE);
                        DrawText("V = overview  B = live/best  keep training for a saved run", 20, 224, 16, LIGHTGRAY);
                    } else {
                        DrawText(TextFormat("MODEL: %s  ACTION: %s", active.model.name.c_str(), active.CurrentActionName()), 20, 202, 16, active.model.color);
                        DrawText("V = overview  TAB/arrows = select  R = reset runs", 20, 224, 16, LIGHTGRAY);
                    }
                } else {
                    DrawText("Manual: WASD + Mouse + Space + Shift + E + Q", 20, 202, 16, LIGHTGRAY);
                }

                if (viewPlayer.isWallRunning) DrawText("WALLRUNNING", cx - 60, cy + 60, 20, GREEN);
                if (viewPlayer.isDashing) DrawText("DASH", cx - 25, cy + 84, 20, SKYBLUE);
                if (!rlAutoplay && katana.active) DrawText("SLASH", cx - 28, cy + 108, 20, WHITE);
                if (rlAutoplay) DrawText("RL CONTROL ACTIVE", cx - 82, cy + 132, 20, YELLOW);

                if (rlAutoplay && viewPlayer.rlDone) {
                    DrawText("EPISODE COMPLETE - RESETTING", cx - 140, cy - 110, 20, GOLD);
                }
            }

        EndDrawing();
    }

    UnloadShader(ppShader);
    UnloadRenderTexture(renderTarget);
    CloseWindow();
    return 0;
}
