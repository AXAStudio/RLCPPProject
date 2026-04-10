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

Texture2D GenerateGridTexture() {
    Image img = GenImageChecked(512, 512, 64, 64, (Color){40, 40, 40, 255}, (Color){60, 60, 60, 255});
    Texture2D tex = LoadTextureFromImage(img);
    UnloadImage(img);
    return tex;
}

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

void DrawCubeWithTexture(Texture2D tex, Vector3 center, float w, float h, float d, Color tint) {
    rlSetTexture(tex.id);
    DrawCube(center, w, h, d, tint);
    rlSetTexture(0);
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


int main() {
    SetConfigFlags(FLAG_MSAA_4X_HINT);
    InitWindow(SCREEN_WIDTH, SCREEN_HEIGHT, "PRO FPS FEEL - 3D KATANA");
    SetTargetFPS(144);
    DisableCursor();

    Texture2D gridTex = GenerateGridTexture();
    Shader ppShader = LoadShaderFromMemory(0, postProcessCode);
    RenderTexture2D renderTarget = LoadRenderTexture(SCREEN_WIDTH, SCREEN_HEIGHT);

    Player player;
    HitMarker hit;
    KatanaFX katana;

    std::vector<Box> obstacles = {
        {{  5, 2,   0 }, { 2,   2,   2   }, ORANGE},
        {{ -5, 0.5f,-5 }, { 1.5f,0.5f,1.5f}, BLUE},
        {{  0, 4, -15 }, { 6,   4,   1   }, RED},
        {{ -8, 3,   5 }, { 1,   3,   6   }, GREEN},
        {{  8, 1,  10 }, { 2,   1,   2   }, PURPLE},
    };

    float spread = 0.0f;
    int lastTrailStep = -1;

    while (!WindowShouldClose()) {
        float dt = GetFrameTime();

        if (IsKeyPressed(KEY_Q) && katana.cooldown <= 0.0f) {
            katana.Trigger();
            lastTrailStep = -1;
        }

        player.Update(dt, obstacles, katana.active, katana.active, katana.Normalized());
        hit.Update(dt);
        katana.Update(dt);

        // if (katana.active) {
        //     int trailStep = (int)(katana.Normalized() * 12.0f);
        //     if (trailStep != lastTrailStep) {
        //         Vector3 katanaVel = katana.GetKatanaVelocity(player.camera);
        //         katana.SpawnTrailBurst(player.camera, player.currentBob, katanaVel);
        //         lastTrailStep = trailStep;
        //     }
        // }

        if (katana.active && !katana.didHitCheck && katana.Normalized() >= 0.25f) {
            katana.didHitCheck = true;
            for (const auto& box : obstacles) {
                if (BoxWithinMeleeRange(player.camera, box)) {
                    hit.Trigger();
                    break;
                }
            }
        }

        float hSpeed = sqrtf(player.velocity.x * player.velocity.x + player.velocity.z * player.velocity.z);
        float targetSpread = hSpeed * 1.35f + (!player.onGround ? 12.0f : 0.0f);
        if (player.isDashing) targetSpread += 6.0f;
        if (katana.active) targetSpread += 4.0f;
        spread = Lerp(spread, targetSpread, dt * 15.0f);

        if (IsMouseButtonPressed(MOUSE_LEFT_BUTTON)) {
            spread += 12.0f;

            Vector3 shotDir = Vector3Normalize(Vector3Subtract(player.camera.target, player.camera.position));
            Ray ray = { player.camera.position, shotDir };

            for (const auto& box : obstacles) {
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

        BeginTextureMode(renderTarget);
            ClearBackground((Color){20, 25, 30, 255});
            BeginMode3D(player.camera);
                DrawPlane({0, 0, 0}, {200, 200}, DARKGRAY);

                for (const auto& b : obstacles) {
                    DrawCubeWithTexture(gridTex, b.center, b.half.x * 2.0f, b.half.y * 2.0f, b.half.z * 2.0f, b.color);
                    DrawCubeWires(b.center, b.half.x * 2.0f, b.half.y * 2.0f, b.half.z * 2.0f, (Color){0, 0, 0, 100});
                }

                katana.Draw3D(player.camera, player.currentBob);
            EndMode3D();
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

            DrawCrosshair(cx, cy, spread);
            if (hit.active) DrawHitmarker(cx, cy, hit.timer, hit.maxTime);

            if (player.dashVisualTimer > 0.0f) {
                float a = player.dashVisualTimer / DASH_TIME;
                DrawRectangle(0, 0, SCREEN_WIDTH, SCREEN_HEIGHT, Fade(WHITE, a * 0.14f));
            }

            if (katana.active) {
                float slashFlash = 1.0f - katana.Normalized();
                DrawRectangle(0, 0, SCREEN_WIDTH, SCREEN_HEIGHT, Fade((Color){180, 220, 255, 255}, slashFlash * 0.05f));
            }

            DrawStatusBar(20, SCREEN_HEIGHT - 70, 300, 25, player.health, MAX_HEALTH, RED, "HEALTH");

            float dashStatus = 1.0f - Clamp(player.dashCooldown / DASH_COOLDOWN_TIME, 0.0f, 1.0f);
            DrawStatusBar(
                20,
                SCREEN_HEIGHT - 40,
                300,
                20,
                dashStatus,
                1.0f,
                (player.dashCooldown <= 0.0f ? GREEN : GOLD),
                "DASH (E)"
            );

            DrawText(TextFormat("SPEED: %02.0f", hSpeed), 20, 20, 20, WHITE);
            DrawText(TextFormat("KATANA (Q): %s", katana.cooldown <= 0.0f ? "READY" : "COOLDOWN"), 20, 46, 20, katana.cooldown <= 0.0f ? SKYBLUE : GRAY);

            if (player.isWallRunning) DrawText("WALLRUNNING", cx - 60, cy + 60, 20, GREEN);
            if (player.isDashing) DrawText("DASH", cx - 25, cy + 84, 20, SKYBLUE);
            if (katana.active) DrawText("SLASH", cx - 28, cy + 108, 20, WHITE);

        EndDrawing();
    }

    UnloadShader(ppShader);
    UnloadTexture(gridTex);
    UnloadRenderTexture(renderTarget);
    CloseWindow();
    return 0;
}