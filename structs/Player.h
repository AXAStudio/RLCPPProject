#pragma once
#include "raylib.h"
#include "raymath.h"
#include <vector>
#include <cmath>
#include "Box.h"

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

const float WALLRUN_MIN_SPEED      = 4.5f;
const float WALLRUN_ATTACH_BOOST   = 1.5f;
const float WALLRUN_GRAVITY_SCALE  = 0.25f;
const float WALLRUN_FORWARD_STICK  = 14.0f;
const float WALLRUN_SIDE_PUSH_OFF  = 1.5f;
const float WALLRUN_MIN_PARALLEL   = 0.15f;
const float WALLRUN_GRACE_TIME     = 0.18f;
const float WALLRUN_JUMP_HBOOST    = 10.0f;
const float WALLRUN_JUMP_VBOOST    = 9.5f;

const float DASH_COOLDOWN_TIME     = 1.35f;
const float DASH_POWER             = 23.0f;
const float DASH_TIME              = 0.18f;
const float DASH_END_DRAG          = 5.0f;
const float DASH_AIR_LIFT          = 2.2f;
const float DASH_FOV_BOOST         = 18.0f;
const float DASH_TILT              = 6.0f;

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

                float fwdAlignment = Vector3DotProduct(fwd, wallTangent);
                if (fwdAlignment < -0.35f) isWallRunning = false;

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
