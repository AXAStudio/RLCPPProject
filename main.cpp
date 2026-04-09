#include "raylib.h"
#include "raymath.h"
#include "rlgl.h"
#include <vector>
#include <cmath>

const int   SCREEN_WIDTH      = 1280;
const int   SCREEN_HEIGHT     = 720;
const float GRAVITY           = -24.0f;
const float JUMP_FORCE        = 8.5f;
const float MOVE_SPEED        = 10.0f;
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

// Wallrun tuning
const float WALLRUN_MIN_SPEED      = 4.5f;
const float WALLRUN_ATTACH_BOOST   = 1.5f;
const float WALLRUN_GRAVITY_SCALE  = 0.25f;
const float WALLRUN_FORWARD_STICK  = 14.0f;
const float WALLRUN_SIDE_PUSH_OFF  = 1.5f;
const float WALLRUN_MIN_PARALLEL   = 0.15f;
const float WALLRUN_GRACE_TIME     = 0.18f;
const float WALLRUN_JUMP_HBOOST    = 10.0f;
const float WALLRUN_JUMP_VBOOST    = 9.5f;

// Dash tuning
const float DASH_COOLDOWN_TIME     = 1.35f;
const float DASH_POWER             = 23.0f;
const float DASH_TIME              = 0.18f;
const float DASH_END_DRAG          = 5.0f;
const float DASH_AIR_LIFT          = 2.2f;
const float DASH_FOV_BOOST         = 18.0f;
const float DASH_TILT              = 6.0f;

// Katana tuning
const float KATANA_COOLDOWN_TIME   = 0.55f;
const float KATANA_SWING_TIME      = 0.20f;
const float KATANA_RANGE           = 3.2f;
const float KATANA_RADIUS          = 1.2f;
const float KATANA_FOV_BOOST       = 8.0f;
const int   KATANA_PARTICLE_COUNT  = 18;

static float EaseOutSine(float t) {
    return sinf((t * PI) * 0.5f);
}

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

    void Update(float dt) {
        if (active) {
            timer -= dt;
            if (timer <= 0.0f) active = false;
        }
    }
};

struct SlashParticle3D {
    Vector3 pos = {0,0,0};
    Vector3 vel = {0,0,0};
    float life = 0.0f;
    float maxLife = 0.0f;
    float size = 0.05f;
    bool active = false;

    void Spawn(Vector3 p, Vector3 v, float l, float s) {
        pos = p;
        vel = v;
        life = l;
        maxLife = l;
        size = s;
        active = true;
    }

    void Update(float dt) {
        if (!active) return;

        life -= dt;
        if (life <= 0.0f) {
            active = false;
            return;
        }

        pos = Vector3Add(pos, Vector3Scale(vel, dt));
        vel = Vector3Scale(vel, 0.94f);
        vel.y -= 4.0f * dt;
    }

    void Draw() const {
        if (!active) return;
        float a = life / maxLife;
        Color c = {255, 255, 255, (unsigned char)(a * 180.0f)};
        DrawSphere(pos, size * (0.6f + 0.4f * a), c);
    }
};

struct KatanaFX {
    bool active = false;
    bool didHitCheck = false;
    float timer = 0.0f;
    float duration = KATANA_SWING_TIME;
    float cooldown = 0.0f;
    float slashSide = 1.0f; // alternates direction
    SlashParticle3D particles[40];

    void Trigger() {
        if (cooldown > 0.0f) return;
        active = true;
        didHitCheck = false;
        timer = duration;
        cooldown = KATANA_COOLDOWN_TIME;
        slashSide *= -1.0f;
    }

    void Update(float dt) {
        cooldown = fmaxf(0.0f, cooldown - dt);

        if (active) {
            timer -= dt;
            if (timer <= 0.0f) {
                timer = 0.0f;
                active = false;
            }
        }

        for (auto& p : particles) p.Update(dt);
    }

    float Normalized() const {
        if (!active) return 0.0f;
        return 1.0f - (timer / duration);
    }

    void GetWeaponBasis(const Camera3D& cam, Vector3& forward, Vector3& right, Vector3& up) const {
        forward = Vector3Normalize(Vector3Subtract(cam.target, cam.position));
        right   = Vector3Normalize(Vector3CrossProduct(forward, cam.up));
        up      = Vector3Normalize(cam.up);
    }

    Vector3 GetWeaponOrigin(const Camera3D& cam, float bobOffset) const {
        Vector3 forward, right, up;
        GetWeaponBasis(cam, forward, right, up);

        float t = Normalized();
        float eased = active ? EaseOutSine(t) : 0.0f;

        Vector3 basePos = Vector3Add(
            cam.position,
            Vector3Add(
                Vector3Scale(right, 0.00f),
                Vector3Add(
                    Vector3Scale(up, -0.10f + bobOffset),
                    Vector3Scale(forward, 0.56f)
                )
            )
        );

        if (!active) return basePos;

        // broad left-to-right path
        float xStart = -0.55f * slashSide;
        float xEnd   =  0.55f * slashSide;

        float yStart =  0.16f;
        float yEnd   = -0.16f;

        float zStart =  0.03f;
        float zEnd   = -0.05f;

        Vector3 slashOffset = {
            Lerp(xStart, xEnd, eased),
            Lerp(yStart, yEnd, eased),
            Lerp(zStart, zEnd, eased)
        };

        return Vector3Add(
            basePos,
            Vector3Add(
                Vector3Scale(right, slashOffset.x),
                Vector3Add(
                    Vector3Scale(up, slashOffset.y),
                    Vector3Scale(forward, slashOffset.z)
                )
            )
        );
    }

    Vector3 GetBladeDirection(const Camera3D& cam) const {
        Vector3 forward, right, up;
        GetWeaponBasis(cam, forward, right, up);

        float t = Normalized();
        float eased = active ? EaseOutSine(t) : 0.0f;

        // about ~80 degrees total visual change instead of huge twisting
        float sideAmount = Lerp(-0.35f * slashSide, 0.35f * slashSide, eased);
        float upAmount   = Lerp( 0.55f, -0.55f, eased);

        return Vector3Normalize(
            Vector3Add(
                Vector3Scale(forward, 0.72f),
                Vector3Add(
                    Vector3Scale(right, sideAmount),
                    Vector3Scale(up, upAmount)
                )
            )
        );
    }

    Vector3 GetBladeTipWorld(const Camera3D& cam, float bobOffset) const {
        Vector3 origin = GetWeaponOrigin(cam, bobOffset);
        Vector3 bladeDir = GetBladeDirection(cam);
        return Vector3Add(origin, Vector3Scale(bladeDir, 1.08f));
    }

    void SpawnTrailBurst(const Camera3D& cam, float bobOffset) {
        Vector3 forward, right, up;
        GetWeaponBasis(cam, forward, right, up);

        Vector3 bladeTip = GetBladeTipWorld(cam, bobOffset);
        float t = Normalized();
        float eased = active ? EaseOutSine(t) : 0.0f;

        Vector3 slashTravel = Vector3Normalize(
            Vector3Add(
                Vector3Scale(right, Lerp(1.8f * slashSide, -1.8f * slashSide, eased)),
                Vector3Scale(up,    Lerp(-0.35f, 0.35f, eased))
            )
        );

        for (int i = 0; i < 10; i++) {
            Vector3 vel = Vector3Add(
                Vector3Scale(slashTravel, 2.4f + 0.16f * (float)i),
                Vector3Add(
                    Vector3Scale(forward, 0.30f + 0.04f * (float)i),
                    Vector3Scale(up, ((float)GetRandomValue(-20, 20)) / 100.0f)
                )
            );

            particles[i].Spawn(
                bladeTip,
                vel,
                0.10f + 0.01f * (float)i,
                0.038f + 0.003f * (float)i
            );
        }
    }

    void Draw3D(const Camera3D& cam, float bobOffset) const {
        if (!active) {
            for (const auto& p : particles) p.Draw();
            return;
        }

        Vector3 forward, right, up;
        GetWeaponBasis(cam, forward, right, up);

        Vector3 origin = GetWeaponOrigin(cam, bobOffset);
        Vector3 bladeDir = GetBladeDirection(cam);
        Vector3 handleDir = Vector3Negate(bladeDir);

        Vector3 bladeStart = Vector3Add(origin, Vector3Scale(bladeDir, 0.06f));
        Vector3 bladeEnd   = Vector3Add(bladeStart, Vector3Scale(bladeDir, 1.08f));

        Vector3 handleStart = Vector3Add(origin, Vector3Scale(handleDir, 0.04f));
        Vector3 handleEnd   = Vector3Add(handleStart, Vector3Scale(handleDir, 0.30f));

        Vector3 guardCenter = Vector3Lerp(handleStart, bladeStart, 0.5f);
        Vector3 pommelPos   = Vector3Add(handleEnd, Vector3Scale(handleDir, 0.035f));

        Vector3 guardDir = Vector3Normalize(Vector3CrossProduct(bladeDir, forward));
        if (Vector3Length(guardDir) < 0.001f) guardDir = right;

        Color bladeColor  = {220, 230, 245, 255};
        Color edgeGlow    = {180, 220, 255, 220};
        Color guardColor  = {210, 190, 120, 255};
        Color handleColor = {35, 35, 35, 255};
        Color pommelColor = {180, 180, 180, 255};

        DrawCylinderEx(handleStart, handleEnd, 0.024f, 0.020f, 8, handleColor);
        DrawSphere(pommelPos, 0.030f, pommelColor);

        Vector3 guardA = Vector3Add(guardCenter, Vector3Scale(guardDir,  0.085f));
        Vector3 guardB = Vector3Add(guardCenter, Vector3Scale(guardDir, -0.085f));
        DrawCylinderEx(guardA, guardB, 0.013f, 0.013f, 8, guardColor);

        DrawCylinderEx(bladeStart, bladeEnd, 0.018f, 0.009f, 8, bladeColor);

        Vector3 edgeOffset = Vector3Scale(guardDir, 0.004f);
        Vector3 edgeStart = Vector3Add(bladeStart, edgeOffset);
        Vector3 edgeEnd   = Vector3Add(bladeEnd, edgeOffset);
        DrawCylinderEx(edgeStart, edgeEnd, 0.0045f, 0.0030f, 6, edgeGlow);

        DrawSphere(bladeEnd, 0.015f, WHITE);

        for (const auto& p : particles) p.Draw();
    }
};

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

bool ResolveCollision(Vector3& pos, Vector3& vel, float radius, const Box& box, bool& grounded, Vector3& outNormal) {
    Vector3 min = Vector3Subtract(box.center, box.half);
    Vector3 max = Vector3Add(box.center, box.half);

    Vector3 closest = {
        Clamp(pos.x, min.x, max.x),
        Clamp(pos.y, min.y, max.y),
        Clamp(pos.z, min.z, max.z)
    };

    Vector3 diff = Vector3Subtract(pos, closest);
    float distSq = Vector3DotProduct(diff, diff);

    if (distSq < radius * radius) {
        float dist = sqrtf(distSq);
        Vector3 normal = (dist < 0.001f) ? (Vector3){0, 1, 0} : Vector3Scale(diff, 1.0f / dist);
        outNormal = normal;

        float overlap = radius - dist;
        pos = Vector3Add(pos, Vector3Scale(normal, overlap + 0.001f));

        if (fabsf(normal.y) > 0.5f) {
            if (normal.y > 0.0f) grounded = true;
            if (vel.y * normal.y < 0.0f) vel.y = 0.0f;
        }

        return true;
    }

    return false;
}

struct Player {
    Vector3 position = { 0, 5.0f, 4 };
    Vector3 velocity = { 0, 0, 0 };
    Vector3 lastMoveDir = { 0, 0, 1 };

    float yaw = 180.0f;
    float pitch = 0.0f;
    float currentFOV = FOV_DEFAULT;
    float currentBob = 0.0f;

    bool onGround = false;
    bool isSliding = false;
    bool isWallRunning = false;
    bool isDashing = false;
    Vector3 wallNormal = { 0, 0, 0 };
    Vector3 dashDir = { 0, 0, 1 };

    float slideTimer = 0.0f;
    float landTimer = 0.0f;
    float cameraTilt = 0.0f;
    float wallRunGraceTimer = 0.0f;
    float wallRunLockTimer = 0.0f;

    float headBobTime = 0.0f;
    float dashCooldown = 0.0f;
    float dashTimer = 0.0f;
    float dashVisualTimer = 0.0f;

    float health = MAX_HEALTH;
    float stamina = MAX_STAMINA;

    Camera3D camera = { 0 };

    void Update(float dt, const std::vector<Box>& obstacles, bool meleeLocked, bool katanaActive, float katanaNorm) {
        Vector2 mouse = GetMouseDelta();
        yaw -= mouse.x * MOUSE_SENSITIVITY;
        pitch = Clamp(pitch - mouse.y * MOUSE_SENSITIVITY, -89.0f, 89.0f);

        bool wantsSlide   = IsKeyDown(KEY_LEFT_CONTROL) || IsKeyDown(KEY_LEFT_SUPER) || IsKeyDown(KEY_C);
        bool sprintButton = IsKeyDown(KEY_LEFT_SHIFT);
        bool canSprint    = stamina > 5.0f;
        bool sprinting    = sprintButton && onGround && !wantsSlide && canSprint;
        bool jumpPressed  = IsKeyPressed(KEY_SPACE);

        Vector3 fwd = { sinf(DEG2RAD * yaw), 0, cosf(DEG2RAD * yaw) };
        Vector3 right = { -fwd.z, 0, fwd.x };
        Vector3 moveInput = { 0, 0, 0 };

        if (IsKeyDown(KEY_W)) moveInput = Vector3Add(moveInput, fwd);
        if (IsKeyDown(KEY_S)) moveInput = Vector3Subtract(moveInput, fwd);
        if (IsKeyDown(KEY_D)) moveInput = Vector3Add(moveInput, right);
        if (IsKeyDown(KEY_A)) moveInput = Vector3Subtract(moveInput, right);

        if (Vector3Length(moveInput) > 0.1f) {
            moveInput = Vector3Normalize(moveInput);
            lastMoveDir = moveInput;
        }

        dashCooldown = fmaxf(0.0f, dashCooldown - dt);
        dashTimer = fmaxf(0.0f, dashTimer - dt);
        dashVisualTimer = fmaxf(0.0f, dashVisualTimer - dt);
        isDashing = dashTimer > 0.0f;

        if (!meleeLocked && IsKeyPressed(KEY_E) && dashCooldown <= 0.0f) {
            dashCooldown = DASH_COOLDOWN_TIME;
            dashTimer = DASH_TIME;
            dashVisualTimer = DASH_TIME;

            Vector3 preferredDir = (Vector3Length(moveInput) > 0.1f) ? moveInput : lastMoveDir;
            if (Vector3Length(preferredDir) <= 0.1f) preferredDir = fwd;
            dashDir = Vector3Normalize(preferredDir);

            velocity.x = dashDir.x * DASH_POWER;
            velocity.z = dashDir.z * DASH_POWER;

            if (!onGround && velocity.y < DASH_AIR_LIFT) velocity.y = DASH_AIR_LIFT;

            isSliding = false;
            isWallRunning = false;
        }

        if (onGround) {
            isWallRunning = false;

            if (wantsSlide && sprinting && !isSliding && !isDashing && !meleeLocked) {
                isSliding = true;
                slideTimer = SLIDE_TIME;
                velocity.x += moveInput.x * SLIDE_BOOST;
                velocity.z += moveInput.z * SLIDE_BOOST;
            }

            if (!isSliding && !isDashing) {
                float targetSpeed = MOVE_SPEED * (sprinting ? SPRINT_MULT : (wantsSlide ? 0.4f : 1.0f));
                velocity.x = Lerp(velocity.x, moveInput.x * targetSpeed, dt * 12.0f);
                velocity.z = Lerp(velocity.z, moveInput.z * targetSpeed, dt * 12.0f);
            } else if (isSliding && !isDashing) {
                slideTimer -= dt;
                velocity.x = Lerp(velocity.x, 0.0f, dt * SLIDE_FRICTION);
                velocity.z = Lerp(velocity.z, 0.0f, dt * SLIDE_FRICTION);

                if (slideTimer <= 0.0f || Vector3Length((Vector3){velocity.x, 0, velocity.z}) < 2.0f) {
                    isSliding = false;
                }
            }

            if (sprinting && Vector3Length(moveInput) > 0.1f) stamina -= STAMINA_DRAIN * dt;

            if (jumpPressed && !isDashing) {
                velocity.y = isSliding ? JUMP_FORCE * 0.8f : JUMP_FORCE;
                onGround = false;
                isSliding = false;
            }
        } else {
            isSliding = false;

            if (isWallRunning) {
                velocity.y += GRAVITY * WALLRUN_GRAVITY_SCALE * dt;

                Vector3 horizVel = { velocity.x, 0, velocity.z };
                float horizSpeed = Vector3Length(horizVel);

                Vector3 wallTangent = Vector3Normalize(Vector3CrossProduct(wallNormal, (Vector3){0, 1, 0}));
                if (Vector3DotProduct(wallTangent, fwd) < 0.0f) wallTangent = Vector3Negate(wallTangent);

                float desiredSpeed = fmaxf(horizSpeed, MOVE_SPEED * SPRINT_MULT + WALLRUN_ATTACH_BOOST);
                Vector3 targetVel = Vector3Scale(wallTangent, desiredSpeed);

                velocity.x = Lerp(velocity.x, targetVel.x, dt * WALLRUN_FORWARD_STICK);
                velocity.z = Lerp(velocity.z, targetVel.z, dt * WALLRUN_FORWARD_STICK);

                velocity.x -= wallNormal.x * WALLRUN_SIDE_PUSH_OFF * dt;
                velocity.z -= wallNormal.z * WALLRUN_SIDE_PUSH_OFF * dt;

                stamina -= STAMINA_DRAIN * 0.45f * dt;

                if (jumpPressed) {
                    Vector3 jumpAway = Vector3Add(
                        Vector3Scale(wallNormal, WALLRUN_JUMP_HBOOST),
                        Vector3Scale(wallTangent, 2.5f)
                    );

                    velocity.x = jumpAway.x;
                    velocity.z = jumpAway.z;
                    velocity.y = WALLRUN_JUMP_VBOOST;
                    isWallRunning = false;
                    wallRunLockTimer = 0.0f;
                }
            } else if (!isDashing) {
                velocity.x += moveInput.x * MOVE_SPEED * AIR_ACCEL * dt * 10.0f;
                velocity.z += moveInput.z * MOVE_SPEED * AIR_ACCEL * dt * 10.0f;

                float hMag = sqrtf(velocity.x * velocity.x + velocity.z * velocity.z);
                float maxAir = MOVE_SPEED * SPRINT_MULT * 1.2f;
                if (hMag > maxAir) {
                    velocity.x = (velocity.x / hMag) * maxAir;
                    velocity.z = (velocity.z / hMag) * maxAir;
                }
            }
        }

        if (isDashing) {
            float dashAlpha = dashTimer / DASH_TIME;
            float dashControl = 1.0f - (1.0f - dashAlpha) * (1.0f - dashAlpha);

            velocity.x = dashDir.x * (DASH_POWER * (0.72f + 0.28f * dashControl));
            velocity.z = dashDir.z * (DASH_POWER * (0.72f + 0.28f * dashControl));
        } else {
            velocity.x = Lerp(velocity.x, velocity.x * 0.98f, dt * DASH_END_DRAG);
            velocity.z = Lerp(velocity.z, velocity.z * 0.98f, dt * DASH_END_DRAG);
        }

        if (!sprinting && !isWallRunning) stamina += STAMINA_REGEN * dt;
        stamina = Clamp(stamina, 0.0f, MAX_STAMINA);

        if (!isWallRunning) velocity.y += GRAVITY * dt;

        position = Vector3Add(position, Vector3Scale(velocity, dt));

        bool wasGrounded = onGround;
        onGround = false;
        bool touchedWall = false;
        bool wallRunEligible = false;

        if (position.y <= GROUND_Y + PLAYER_HEIGHT / 2.0f) {
            position.y = GROUND_Y + PLAYER_HEIGHT / 2.0f;
            if (velocity.y < 0.0f) velocity.y = 0.0f;
            onGround = true;
        }

        for (const auto& box : obstacles) {
            Vector3 hitNormal;
            if (ResolveCollision(position, velocity, PLAYER_RADIUS, box, onGround, hitNormal)) {
                if (fabsf(hitNormal.y) < 0.1f) {
                    touchedWall = true;

                    Vector3 horizVel = { velocity.x, 0, velocity.z };
                    Vector3 velDir = (Vector3Length(horizVel) > 0.1f) ? Vector3Normalize(horizVel) : Vector3Zero();
                    Vector3 wallTangent = Vector3Normalize(Vector3CrossProduct(hitNormal, (Vector3){0, 1, 0}));
                    float wallParallel = fabsf(Vector3DotProduct(velDir, wallTangent));
                    float horizSpeed = Vector3Length(horizVel);

                    if (!onGround && !isDashing && horizSpeed > WALLRUN_MIN_SPEED && wallParallel > WALLRUN_MIN_PARALLEL) {
                        wallRunEligible = true;
                        isWallRunning = true;
                        wallNormal = hitNormal;
                        wallRunLockTimer = 0.12f;

                        if (Vector3DotProduct(wallTangent, fwd) < 0.0f) wallTangent = Vector3Negate(wallTangent);

                        float speedAlongTangent = Vector3DotProduct(horizVel, wallTangent);
                        float boostedSpeed = fmaxf(fabsf(speedAlongTangent), MOVE_SPEED * 1.1f);

                        velocity.x = wallTangent.x * boostedSpeed;
                        velocity.z = wallTangent.z * boostedSpeed;

                        if (velocity.y < 0.0f) velocity.y *= 0.25f;
                    }
                }
            }
        }

        if (wallRunEligible) wallRunGraceTimer = WALLRUN_GRACE_TIME;
        else wallRunGraceTimer = fmaxf(wallRunGraceTimer - dt, 0.0f);

        wallRunLockTimer = fmaxf(wallRunLockTimer - dt, 0.0f);

        if (!touchedWall && wallRunGraceTimer <= 0.0f && wallRunLockTimer <= 0.0f) {
            isWallRunning = false;
        }

        if (onGround && !wasGrounded) landTimer = 0.16f;
        if (landTimer > 0.0f) landTimer -= dt;

        float horizontalSpeed = sqrtf(velocity.x * velocity.x + velocity.z * velocity.z);
        if (onGround && horizontalSpeed > 0.2f && !isSliding) {
            headBobTime += dt * (7.5f + horizontalSpeed * 0.18f);
        } else {
            headBobTime += dt * 2.0f;
        }

        float targetFOV = isSliding ? FOV_SLIDE : (sprinting ? FOV_SPRINT : FOV_DEFAULT);
        if (isWallRunning) targetFOV += 10.0f;
        if (dashVisualTimer > 0.0f) targetFOV += DASH_FOV_BOOST * (dashVisualTimer / DASH_TIME);
        if (katanaActive) targetFOV += KATANA_FOV_BOOST * (1.0f - katanaNorm);
        currentFOV = Lerp(currentFOV, targetFOV, dt * 10.0f);

        float targetTilt = 0.0f;
        if (isWallRunning) {
            Vector3 camRight = { fwd.z, 0, -fwd.x };
            targetTilt = (Vector3DotProduct(camRight, wallNormal) > 0.0f) ? -15.0f : 15.0f;
        } else if (dashVisualTimer > 0.0f) {
            float dashSide = Vector3DotProduct(right, dashDir);
            targetTilt = -dashSide * DASH_TILT;
        }
        cameraTilt = Lerp(cameraTilt, targetTilt, dt * 12.0f);

        float eyeHeight = (isSliding || (wantsSlide && onGround)) ? 0.5f : 1.4f;
        float landDip = sinf(landTimer * 18.0f) * 0.14f;

        float bob = 0.0f;
        if (onGround && !isSliding) {
            bob = sinf(headBobTime) * 0.035f * Clamp(horizontalSpeed / (MOVE_SPEED * SPRINT_MULT), 0.0f, 1.0f);
        }
        currentBob = bob;

        camera.position = { position.x, position.y + eyeHeight - landDip + bob, position.z };

        Vector3 forwardVec = {
            sinf(DEG2RAD * yaw) * cosf(DEG2RAD * pitch),
            sinf(DEG2RAD * pitch),
            cosf(DEG2RAD * yaw) * cosf(DEG2RAD * pitch)
        };

        camera.target = Vector3Add(camera.position, forwardVec);

        float tiltRad = DEG2RAD * cameraTilt;
        Vector3 worldUp = { 0, 1, 0 };
        Vector3 camRight = Vector3Normalize(Vector3CrossProduct(forwardVec, worldUp));

        camera.up = Vector3Normalize({
            cosf(tiltRad) * worldUp.x + sinf(tiltRad) * camRight.x,
            cosf(tiltRad) * worldUp.y + sinf(tiltRad) * camRight.y,
            cosf(tiltRad) * worldUp.z + sinf(tiltRad) * camRight.z
        });

        camera.fovy = currentFOV;
    }
};

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

        if (katana.active) {
            int trailStep = (int)(katana.Normalized() * 12.0f);
            if (trailStep != lastTrailStep) {
                katana.SpawnTrailBurst(player.camera, player.currentBob);
                lastTrailStep = trailStep;
            }
        }

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