#include "raylib.h"
#include "raymath.h"
#include "rlgl.h"
#include <vector>
#include <cmath>

// ─── Constants & Settings ────────────────────────────────────────────────────
const int   SCREEN_WIDTH      = 1280;
const int   SCREEN_HEIGHT     = 720;
const float GRAVITY           = -24.0f;
const float JUMP_FORCE        = 8.5f;
const float MOVE_SPEED        = 7.0f;
const float SPRINT_MULT       = 1.6f;
const float AIR_ACCEL         = 0.4f;
const float MOUSE_SENSITIVITY = 0.12f;
const float PLAYER_HEIGHT     = 1.8f;
const float PLAYER_RADIUS     = 0.45f;
const float GROUND_Y          = 0.0f;

const float SLIDE_BOOST       = 4.0f;
const float SLIDE_FRICTION    = 0.5f;
const float SLIDE_TIME        = 1.2f;

const float FOV_DEFAULT       = 85.0f;
const float FOV_SPRINT        = 100.0f;
const float FOV_SLIDE         = 105.0f;

const float MAX_HEALTH        = 100.0f;
const float MAX_STAMINA       = 100.0f;
const float STAMINA_DRAIN     = 26.0f;
const float STAMINA_REGEN     = 18.0f;

// ─── Structures ──────────────────────────────────────────────────────────────
struct Box {
    Vector3 center;
    Vector3 half;
    Color color;
};

struct HitMarker {
    float timer = 0.0f;
    float maxTime = 0.20f;
    bool active = false;
    void Trigger() { timer = maxTime; active = true; }
    void Update(float dt) { if (active) { timer -= dt; if (timer <= 0) active = false; } }
};

// ─── Post-Processing Shader (GLSL 330) ───────────────────────────────────────
// FIX 1: Removed "in vec2 fragTexCoord" and "in vec4 fragColor" — these are
//         built-in varyings provided by Raylib's default vertex shader.
//         Redeclaring them as "in" causes a GLSL linking error.
const char* postProcessCode = R"(

#version 330

in vec2 fragTexCoord;   // <-- THIS is actually required here
out vec4 finalColor;

uniform sampler2D texture0;

void main() {
    vec2 uv = fragTexCoord;

    // Chromatic Aberration
    float offset = 0.0015;
    float r = texture(texture0, uv + vec2(offset, 0.0)).r;
    float g = texture(texture0, uv).g;
    float b = texture(texture0, uv - vec2(offset, 0.0)).b;
    vec3 col = vec3(r, g, b);

    // Vignette
    float dist = distance(uv, vec2(0.5));
    col *= smoothstep(0.8, 0.2, dist);

    // Contrast
    col = mix(vec3(0.5), col, 1.15);

    finalColor = vec4(col, 1.0);
}
)";

// ─── Collision Resolution ────────────────────────────────────────────────────
bool ResolveCollision(Vector3& pos, Vector3& vel, float radius, const Box& box, bool& grounded, Vector3& outNormal) {
    Vector3 min = Vector3Subtract(box.center, box.half);
    Vector3 max = Vector3Add(box.center, box.half);

    Vector3 closest = { Clamp(pos.x, min.x, max.x), Clamp(pos.y, min.y, max.y), Clamp(pos.z, min.z, max.z) };
    Vector3 diff = Vector3Subtract(pos, closest);
    float distSq = Vector3DotProduct(diff, diff);

    if (distSq < radius * radius) {
        float dist = sqrtf(distSq);
        Vector3 normal = (dist < 0.001f) ? (Vector3){0, 1, 0} : Vector3Scale(diff, 1.0f / dist);
        outNormal = normal;
        float overlap = radius - dist;

        pos = Vector3Add(pos, Vector3Scale(normal, overlap + 0.001f));

        if (fabsf(normal.y) > 0.5f) {
            if (normal.y > 0) grounded = true;
            if (vel.y * normal.y < 0) vel.y = 0;
        }
        return true;
    }
    return false;
}

// ─── Player Logic ────────────────────────────────────────────────────────────
struct Player {
    Vector3 position = { 0, 5.0f, 4 };
    Vector3 velocity = { 0, 0, 0 };
    float yaw = 180.0f, pitch = 0.0f;
    float currentFOV = FOV_DEFAULT;

    bool onGround = false;
    bool isSliding = false;
    bool isWallRunning = false;
    Vector3 wallNormal = {0, 0, 0};

    float slideTimer = 0.0f;
    float landTimer = 0.0f;
    float cameraTilt = 0.0f;
    float wallRunGraceTimer = 0.0f;

    float health = MAX_HEALTH;
    float stamina = MAX_STAMINA;

    Camera3D camera = {0};

    void Update(float dt, const std::vector<Box>& obstacles) {
        Vector2 mouse = GetMouseDelta();
        yaw   -= mouse.x * MOUSE_SENSITIVITY;
        pitch  = Clamp(pitch - mouse.y * MOUSE_SENSITIVITY, -89.0f, 89.0f);

        bool wantsSlide  = IsKeyDown(KEY_LEFT_CONTROL) || IsKeyDown(KEY_LEFT_SUPER) || IsKeyDown(KEY_C);
        bool sprintButton = IsKeyDown(KEY_LEFT_SHIFT);
        bool canSprint = stamina > 5.0f;
        bool sprinting   = sprintButton && onGround && !wantsSlide && canSprint;
        bool jumpPressed = IsKeyPressed(KEY_SPACE);

        Vector3 fwd   = { sinf(DEG2RAD * yaw), 0, cosf(DEG2RAD * yaw) };
        Vector3 right = { -fwd.z, 0, fwd.x };
        Vector3 moveInput = { 0, 0, 0 };

        if (IsKeyDown(KEY_W)) moveInput = Vector3Add(moveInput, fwd);
        if (IsKeyDown(KEY_S)) moveInput = Vector3Subtract(moveInput, fwd);
        if (IsKeyDown(KEY_D)) moveInput = Vector3Add(moveInput, right);
        if (IsKeyDown(KEY_A)) moveInput = Vector3Subtract(moveInput, right);

        if (Vector3Length(moveInput) > 0.1f) moveInput = Vector3Normalize(moveInput);

        // Ground movement & sliding
        if (onGround) {
            isWallRunning = false;

            if (wantsSlide && sprinting && !isSliding) {
                isSliding   = true;
                slideTimer  = SLIDE_TIME;
                velocity.x += moveInput.x * SLIDE_BOOST;
                velocity.z += moveInput.z * SLIDE_BOOST;
            }

            if (!isSliding) {
                float targetSpeed = MOVE_SPEED * (sprinting ? SPRINT_MULT : (wantsSlide ? 0.4f : 1.0f));
                velocity.x = Lerp(velocity.x, moveInput.x * targetSpeed, dt * 12.0f);
                velocity.z = Lerp(velocity.z, moveInput.z * targetSpeed, dt * 12.0f);
            } else {
                slideTimer -= dt;
                velocity.x  = Lerp(velocity.x, 0, dt * SLIDE_FRICTION);
                velocity.z  = Lerp(velocity.z, 0, dt * SLIDE_FRICTION);
                if (slideTimer <= 0 || Vector3Length(velocity) < 2.0f) isSliding = false;
            }

            if (sprinting && Vector3Length(moveInput) > 0.1f) {
                stamina -= STAMINA_DRAIN * dt;
            }

            if (jumpPressed) {
                velocity.y = isSliding ? JUMP_FORCE * 0.8f : JUMP_FORCE;
                onGround   = false;
                isSliding  = false;
            }
        } else {
            isSliding = false;

            if (isWallRunning) {
                velocity.y = Lerp(velocity.y, -2.0f, dt * 5.0f);
                stamina -= STAMINA_DRAIN * 0.55f * dt;

                if (jumpPressed) {
                    velocity    = Vector3Add(velocity, Vector3Scale(wallNormal, 9.0f));
                    velocity.y  = JUMP_FORCE * 1.1f;
                    isWallRunning = false;
                }
            } else {
                velocity.x += moveInput.x * MOVE_SPEED * AIR_ACCEL * dt * 10.0f;
                velocity.z += moveInput.z * MOVE_SPEED * AIR_ACCEL * dt * 10.0f;

                float hMag   = sqrtf(velocity.x*velocity.x + velocity.z*velocity.z);
                float maxAir = MOVE_SPEED * SPRINT_MULT * 1.2f;
                if (hMag > maxAir) {
                    velocity.x = (velocity.x / hMag) * maxAir;
                    velocity.z = (velocity.z / hMag) * maxAir;
                }
            }
        }

        if (!sprinting && !isWallRunning) {
            stamina += STAMINA_REGEN * dt;
        }
        stamina = Clamp(stamina, 0.0f, MAX_STAMINA);

        if (!isWallRunning) velocity.y += GRAVITY * dt;
        position = Vector3Add(position, Vector3Scale(velocity, dt));

        bool wasGrounded  = onGround;
        onGround          = false;
        bool touchedWall  = false;
        bool wallRunEligible = false;

        if (position.y <= GROUND_Y + PLAYER_HEIGHT/2.0f) {
            position.y = GROUND_Y + PLAYER_HEIGHT/2.0f;
            if (velocity.y < 0) velocity.y = 0;
            onGround = true;
        }

        // ─── WALL RUN IMPROVEMENTS ───────────────────────────────────────────────
        for (const auto& box : obstacles) {
            Vector3 hitNormal;
            if (ResolveCollision(position, velocity, PLAYER_RADIUS, box, onGround, hitNormal)) {
                if (fabsf(hitNormal.y) < 0.1f) { // it's a wall
                    touchedWall = true;

                    // Calculate forward & wall tangent
                    Vector3 moveDir = Vector3Length(moveInput) > 0.1f ? Vector3Normalize(moveInput) : Vector3Zero();
                    Vector3 horizVel = { velocity.x, 0, velocity.z };
                    Vector3 velDir = Vector3Length(horizVel) > 0.1f ? Vector3Normalize(horizVel) : Vector3Zero();
                    float dotFwd = Vector3DotProduct(moveDir, Vector3Scale(hitNormal, -1));
                    Vector3 wallTangent = Vector3Normalize(Vector3CrossProduct(hitNormal, {0, 1, 0}));
                    float moveAlongWall = Vector3DotProduct(velDir, wallTangent);

                    // Wall-running trigger: require both some forward wall contact and wall-parallel momentum.
                    float forwardIntoWall = dotFwd;
                    float wallParallel    = fabs(moveAlongWall);
                    if (!onGround && forwardIntoWall > 0.22f && wallParallel > 0.35f) {
                        wallRunEligible = true;
                        isWallRunning = true;
                        wallNormal = hitNormal;

                        // Project horizontal velocity along wall tangent to stick
                        float speedAlongTangent = Vector3DotProduct(horizVel, wallTangent);
                        velocity.x = wallTangent.x * speedAlongTangent;
                        velocity.z = wallTangent.z * speedAlongTangent;

                        // Reduce downward pull for smoother stick
                        velocity.y = Lerp(velocity.y, -1.0f, dt * 3.0f);

                        // Small upward nudge if first frame of wall touch
                        if (!wasGrounded && velocity.y < 0) velocity.y += 1.5f;
                    }

                    // Optional mantle: if near top of wall, allow small automatic lift
                    float boxTop = box.center.y + box.half.y;
                    if (!onGround && position.y + PLAYER_HEIGHT/2.0f > boxTop &&
                                    position.y - PLAYER_HEIGHT/2.0f < boxTop + 0.5f) {
                        velocity.y = JUMP_FORCE * 0.9f;
                    }
                }
            }
        }

        if (wallRunEligible) {
            wallRunGraceTimer = 0.25f;
        } else {
            wallRunGraceTimer = fmaxf(wallRunGraceTimer - dt, 0.0f);
        }

        if (!touchedWall && wallRunGraceTimer <= 0.0f)
            isWallRunning = false;

        // Wall jump input
        if (isWallRunning && jumpPressed) {
            velocity = Vector3Add(velocity, Vector3Scale(wallNormal, 9.0f)); // push off wall
            velocity.y = JUMP_FORCE * 1.1f;
            isWallRunning = false;
        }

        if (onGround && !wasGrounded) landTimer = 0.15f;
        if (landTimer > 0) landTimer -= dt;

        // FOV
        float targetFOV = isSliding ? FOV_SLIDE : (sprinting ? FOV_SPRINT : FOV_DEFAULT);
        if (isWallRunning) targetFOV += 10.0f;
        currentFOV = Lerp(currentFOV, targetFOV, dt * 8.0f);

        // Camera tilt for wall running
        float targetTilt = 0.0f;
        if (isWallRunning) {
            Vector3 camRight = { fwd.z, 0, -fwd.x };
            float dotWall    = Vector3DotProduct(camRight, wallNormal);
            targetTilt       = (dotWall > 0) ? -15.0f : 15.0f;
        }
        cameraTilt = Lerp(cameraTilt, targetTilt, dt * 10.0f);

        float eyeHeight = (isSliding || (wantsSlide && onGround)) ? 0.5f : 1.4f;
        float landDip   = sinf(landTimer * 20.0f) * 0.2f;

        camera.position = { position.x, position.y + eyeHeight - landDip, position.z };

        Vector3 forwardVec = {
            sinf(DEG2RAD * yaw) * cosf(DEG2RAD * pitch),
            sinf(DEG2RAD * pitch),
            cosf(DEG2RAD * yaw) * cosf(DEG2RAD * pitch)
        };
        camera.target = Vector3Add(camera.position, forwardVec);

        // FIX 2: camera.up tilt was { sin, cos, 0 } which rolls around the
        //         world-Z axis regardless of where you're looking.
        //         Correct approach: tilt around the camera's local forward axis
        //         by rotating world-up toward the camera-right vector.
        float tiltRad    = DEG2RAD * cameraTilt;
        Vector3 worldUp  = { 0, 1, 0 };
        Vector3 camRight = Vector3Normalize(Vector3CrossProduct(forwardVec, worldUp));
        // up = cos(tilt)*worldUp + sin(tilt)*camRight
        camera.up = Vector3Normalize({
            cosf(tiltRad) * worldUp.x + sinf(tiltRad) * camRight.x,
            cosf(tiltRad) * worldUp.y + sinf(tiltRad) * camRight.y,
            cosf(tiltRad) * worldUp.z + sinf(tiltRad) * camRight.z
        });

        camera.fovy = currentFOV;
    }
};

// ─── Drawing Tools ───────────────────────────────────────────────────────────
Texture2D GenerateGridTexture() {
    Image img    = GenImageChecked(512, 512, 64, 64,
                        (Color){40, 40, 40, 255},
                        (Color){60, 60, 60, 255});
    Texture2D tex = LoadTextureFromImage(img);
    UnloadImage(img);
    return tex;
}

void DrawHitmarker(int cx, int cy, float timer, float maxTime) {
    float t   = 1.0f - (timer / maxTime);
    float gap = 2.0f + t * 8.0f;
    float len = 6.0f;
    unsigned char alpha = (unsigned char)((1.0f - t) * 255);
    Color c = { 255, 255, 255, alpha };
    DrawLineEx({(float)cx - gap,       (float)cy - gap},       {(float)cx - gap - len, (float)cy - gap - len}, 2.0f, c);
    DrawLineEx({(float)cx + gap,       (float)cy - gap},       {(float)cx + gap + len, (float)cy - gap - len}, 2.0f, c);
    DrawLineEx({(float)cx - gap,       (float)cy + gap},       {(float)cx - gap - len, (float)cy + gap + len}, 2.0f, c);
    DrawLineEx({(float)cx + gap,       (float)cy + gap},       {(float)cx + gap + len, (float)cy + gap + len}, 2.0f, c);
}

void DrawCrosshair(int cx, int cy, float spread) {
    int gap = (int)(5 + spread);
    DrawRectangle(cx - gap - 8, cy - 1, 8, 2, WHITE);
    DrawRectangle(cx + gap,     cy - 1, 8, 2, WHITE);
    DrawRectangle(cx - 1, cy - gap - 8, 2, 8, WHITE);
    DrawRectangle(cx - 1, cy + gap,     2, 8, WHITE);
    DrawRectangle(cx - 1, cy - 1,       2, 2, RED);
}

// FIX 3: DrawCubeTexture was removed in Raylib 4.0.
//         Replacement: bind texture manually via rlSetTexture, draw the cube,
//         then unbind. This replicates the old behavior for all 6 faces.
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

    int fontSize = (height > 24) ? 18 : 16;
    int textWidth = MeasureText(label, fontSize);
    DrawText(label, x + (width - textWidth) * 0.5f, y + (height - fontSize) * 0.5f, fontSize, WHITE);
}

// ─── Main ────────────────────────────────────────────────────────────────────
int main() {
    SetConfigFlags(FLAG_MSAA_4X_HINT);
    InitWindow(SCREEN_WIDTH, SCREEN_HEIGHT, "PRO FPS FEEL - FIXED");
    SetTargetFPS(144);
    DisableCursor();

    Texture2D      gridTex      = GenerateGridTexture();
    // FIX 1 (continued): LoadShaderFromMemory(0, frag) — passing 0 for the
    //         vertex shader tells Raylib to use its built-in default, which
    //         already declares fragTexCoord/fragColor as varyings.
    //         The fragment shader must NOT redeclare them as "in".
    Shader         ppShader     = LoadShaderFromMemory(0, postProcessCode);
    RenderTexture2D renderTarget = LoadRenderTexture(SCREEN_WIDTH, SCREEN_HEIGHT);

    Player player;
    HitMarker hit;

    std::vector<Box> obstacles = {
        {{ 5,    2,    0  }, { 2,    2,    2   }, ORANGE},
        {{ -5,   0.5f, -5 }, { 1.5f, 0.5f, 1.5f}, BLUE  },
        {{ 0,    4,   -15 }, { 6,    4,    1   }, RED   },
        {{ -8,   3,    5  }, { 1,    3,    6   }, GREEN },
        {{ 8,    1,   10  }, { 2,    1,    2   }, PURPLE},
    };

    float spread = 0.0f;

    while (!WindowShouldClose()) {
        float dt = GetFrameTime();

        player.Update(dt, obstacles);

        hit.Update(dt);

        float hSpeed = sqrtf(player.velocity.x*player.velocity.x + player.velocity.z*player.velocity.z);
        float targetSpread = hSpeed * 1.5f + (!player.onGround ? 12.0f : 0.0f) + (player.isWallRunning ? 5.0f : 0.0f);
        spread = Lerp(spread, targetSpread, dt * 15.0f);

        if (IsMouseButtonPressed(0)) {
            spread += 12.0f;
            Vector3 fwd = Vector3Normalize(Vector3Subtract(player.camera.target, player.camera.position));
            float s = spread * 0.0025f;
            fwd.x += ((float)GetRandomValue(-100, 100)/100.0f) * s;
            fwd.y += ((float)GetRandomValue(-100, 100)/100.0f) * s;
            fwd.z += ((float)GetRandomValue(-100, 100)/100.0f) * s;

            Ray ray = { player.camera.position, Vector3Normalize(fwd) };
            for (const auto& box : obstacles) {
                if (GetRayCollisionBox(ray, (BoundingBox){
                        Vector3Subtract(box.center, box.half),
                        Vector3Add(box.center, box.half)}).hit) {
                    hit.Trigger(); break;
                }
            }
        }

        // 1. Render 3D world to texture
        BeginTextureMode(renderTarget);
            ClearBackground((Color){20, 25, 30, 255});
            BeginMode3D(player.camera);
                DrawPlane({0, 0, 0}, {200, 200}, DARKGRAY);
                for (const auto& b : obstacles) {
                    // FIX 3: use replacement helper instead of DrawCubeTexture
                    DrawCubeWithTexture(gridTex, b.center, b.half.x*2, b.half.y*2, b.half.z*2, b.color);
                    DrawCubeWires(b.center, b.half.x*2, b.half.y*2, b.half.z*2, (Color){0, 0, 0, 100});
                }
            EndMode3D();
        EndTextureMode();

        // 2. Post-process & UI
        BeginDrawing();
            ClearBackground(BLACK);
            BeginShaderMode(ppShader);
                DrawTextureRec(renderTarget.texture,
                    (Rectangle){0, 0, (float)renderTarget.texture.width, (float)-renderTarget.texture.height},
                    (Vector2){0, 0}, WHITE);
            EndShaderMode();

            int cx = SCREEN_WIDTH/2, cy = SCREEN_HEIGHT/2;
            DrawCrosshair(cx, cy, spread);
            if (hit.active) DrawHitmarker(cx, cy, hit.timer, hit.maxTime);

            DrawFPS(10, 10);
            DrawText("CTRL: Slide | SPACE: Jump/Walljump/Mantle", 10, 40, 20, LIGHTGRAY);
            DrawStatusBar(20, SCREEN_HEIGHT - 70, 360, 32, player.health, MAX_HEALTH, RED, TextFormat("HEALTH %d / %d", (int)player.health, (int)MAX_HEALTH));
            // DrawStatusBar(20, SCREEN_HEIGHT - 104, 360, 22, player.stamina, MAX_STAMINA, SKYBLUE, TextFormat("STAMINA %d / %d", (int)player.stamina, (int)MAX_STAMINA));
            DrawText(TextFormat("SPEED: %02.0f", hSpeed), 10, SCREEN_HEIGHT - 30, 20, ORANGE);
            if (player.isWallRunning) DrawText("WALLRUNNING", cx - 60, cy + 40, 20, GREEN);
            if (player.isSliding)     DrawText("SLIDING",     cx - 40, cy + 40, 20, SKYBLUE);
        EndDrawing();
    }

    UnloadShader(ppShader);
    UnloadTexture(gridTex);
    UnloadRenderTexture(renderTarget);
    CloseWindow();
    return 0;
}