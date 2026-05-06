#pragma once
#include "raylib.h"
#include "raymath.h"
#include <array>
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
const float WALLRUN_GRAVITY_SCALE  = 0.1f;
const float WALLRUN_FORWARD_STICK  = 14.0f;
const float WALLRUN_SIDE_PUSH_OFF  = 1.5f;
const float WALLRUN_MIN_PARALLEL   = 0.15f;
const float WALLRUN_GRACE_TIME     = 0.18f;
const float WALLRUN_JUMP_HBOOST    = 10.0f;
const float WALLRUN_JUMP_VBOOST    = 9.5f;

// smoother wallrun constraints
const float WALLRUN_DOWNWARD_FORCE    = 0.0f;
const float WALLRUN_REATTACH_COOLDOWN = 0.0f;
const float WALLRUN_MIN_DESCEND_SPEED = 0.0f;
const float WALLRUN_SINK_RAMP         = 0.0f;

const float DASH_COOLDOWN_TIME     = 1.35f;
const float DASH_POWER             = 23.0f;
const float DASH_TIME              = 0.18f;
const float DASH_END_DRAG          = 5.0f;
const float DASH_AIR_LIFT          = 2.2f;
const float DASH_VERTICAL_POWER    = 12.0f;
const float DASH_FOV_BOOST         = 18.0f;
const float DASH_TILT              = 6.0f;
const float KATANA_FOV_BOOST       = 8.0f;

const float RL_MAX_EPISODE_TIME    = 14.0f;
const float RL_STUCK_TIME          = 2.75f;
const float RL_GOAL_RADIUS         = 3.25f;
const float RL_SENSOR_RANGE        = 16.0f;
const float RL_AREA_RADIUS         = 4.75f;
const float RL_AREA_GRACE_TIME     = 0.85f;

struct RLControl {
    Vector3 moveInput = { 0, 0, 0 };
    bool dash = false;
    bool jump = false;
    bool slide = false;
    float yawDelta = 0.0f;
    float pitchDelta = 0.0f;
    float aimPitch = 0.0f;
};

inline float RLHorizontalDistance(const Vector3& a, const Vector3& b) {
    float dx = a.x - b.x;
    float dz = a.z - b.z;
    return sqrtf(dx * dx + dz * dz);
}

inline Vector3 RLFlatDirection(const Vector3& from, const Vector3& to) {
    Vector3 rel = { to.x - from.x, 0.0f, to.z - from.z };
    float len = sqrtf(rel.x * rel.x + rel.z * rel.z);
    return (len > 0.001f) ? Vector3Scale(rel, 1.0f / len) : (Vector3){ 0, 0, -1 };
}

inline Vector3 RLRotateFlat(const Vector3& dir, float degrees) {
    float r = degrees * DEG2RAD;
    float cs = cosf(r);
    float sn = sinf(r);
    return Vector3Normalize((Vector3){
        dir.x * cs - dir.z * sn,
        0.0f,
        dir.x * sn + dir.z * cs
    });
}

inline float RLSensorClearance(const Vector3& origin, const Vector3& direction, const std::vector<Box>& obstacles, float maxDistance) {
    Ray ray = { origin, Vector3Normalize(direction) };
    float nearest = maxDistance;

    for (const auto& box : obstacles) {
        BoundingBox bb = {
            Vector3Subtract(box.center, box.half),
            Vector3Add(box.center, box.half)
        };

        RayCollision hit = GetRayCollisionBox(ray, bb);
        if (hit.hit && hit.distance > 0.0f && hit.distance < nearest) {
            nearest = hit.distance;
        }
    }

    return Clamp(nearest / maxDistance, 0.0f, 1.0f);
}

inline bool SphereNearBox(const Vector3& pos, float radius, const Box& box) {
    return fabsf(pos.x - box.center.x) <= box.half.x + radius &&
           fabsf(pos.y - box.center.y) <= box.half.y + radius &&
           fabsf(pos.z - box.center.z) <= box.half.z + radius;
}

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
    static constexpr int RL_STATE_SIZE = 33;
    using RLState = std::array<float, RL_STATE_SIZE>;

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

    // wallrun state
    float wallRunTimer = 0.0f;
    float wallRunReattachTimer = 0.0f;

    float headBobTime = 0.0f;
    float dashCooldown = 0.0f;
    float dashTimer = 0.0f;
    float dashVisualTimer = 0.0f;

    // reinforcement learning bookkeeping
    float rlReward = 0.0f;
    float rlEpisodeReward = 0.0f;
    float rlStartTargetDistance = 1.0f;
    float rlPreviousTargetDistance = 0.0f;
    float rlBestTargetDistance = 0.0f;
    float rlEpisodeTime = 0.0f;
    float rlNoProgressTimer = 0.0f;
    float rlAreaTimer = 0.0f;
    float rlLastProgress = 0.0f;
    float rlWallImpactPenalty = 0.0f;
    float rlSpeedLossPenalty = 0.0f;
    float rlCollisionPenaltyTotal = 0.0f;
    int rlSteps = 0;
    int rlWallHits = 0;
    bool rlHitWallThisStep = false;
    bool rlDone = false;
    bool rlSucceeded = false;
    bool rlTimedOut = false;
    bool rlStuck = false;
    Vector3 rlTarget = { 0, 0, 0 };
    Vector3 rlAreaAnchor = { 0, 0, 0 };

    float health = MAX_HEALTH;
    float stamina = MAX_STAMINA;

    Camera3D camera = {};

    void Update(float dt, const std::vector<Box>& obstacles, bool meleeLocked, bool katanaActive, float katanaNorm, bool useRL = false, const RLControl& rlControl = RLControl()) {
        rlWallImpactPenalty = 0.0f;
        rlSpeedLossPenalty = 0.0f;
        rlHitWallThisStep = false;

        if (useRL) {
            yaw -= rlControl.yawDelta;
            pitch = Clamp(pitch - rlControl.pitchDelta, -89.0f, 89.0f);
        } else {
            Vector2 mouse = GetMouseDelta();
            yaw -= mouse.x * MOUSE_SENSITIVITY;
            pitch = Clamp(pitch - mouse.y * MOUSE_SENSITIVITY, -89.0f, 89.0f);
        }

        bool wantsSlide   = useRL ? rlControl.slide : (IsKeyDown(KEY_LEFT_CONTROL) || IsKeyDown(KEY_LEFT_SUPER) || IsKeyDown(KEY_C));
        bool sprintButton = useRL ? (Vector3Length(rlControl.moveInput) > 0.1f) : IsKeyDown(KEY_LEFT_SHIFT);
        bool canSprint    = stamina > 5.0f;
        bool sprinting    = sprintButton && onGround && !wantsSlide && canSprint;
        bool jumpPressed  = useRL ? rlControl.jump : IsKeyPressed(KEY_SPACE);

        Vector3 fwd = { sinf(DEG2RAD * yaw), 0, cosf(DEG2RAD * yaw) };
        Vector3 right = { -fwd.z, 0, fwd.x };
        Vector3 moveInput = { 0, 0, 0 };

        if (useRL) {
            moveInput = rlControl.moveInput;
        } else {
            if (IsKeyDown(KEY_W)) moveInput = Vector3Add(moveInput, fwd);
            if (IsKeyDown(KEY_S)) moveInput = Vector3Subtract(moveInput, fwd);
            if (IsKeyDown(KEY_D)) moveInput = Vector3Add(moveInput, right);
            if (IsKeyDown(KEY_A)) moveInput = Vector3Subtract(moveInput, right);
        }

        if (Vector3Length(moveInput) > 0.1f) {
            moveInput = Vector3Normalize(moveInput);
            lastMoveDir = moveInput;
        }

        dashCooldown = fmaxf(0.0f, dashCooldown - dt);
        dashTimer = fmaxf(0.0f, dashTimer - dt);
        dashVisualTimer = fmaxf(0.0f, dashVisualTimer - dt);
        wallRunReattachTimer = fmaxf(0.0f, wallRunReattachTimer - dt);

        isDashing = dashTimer > 0.0f;

        bool dashPressed = useRL ? rlControl.dash : (!meleeLocked && IsKeyPressed(KEY_E));
        if (dashPressed && !meleeLocked && dashCooldown <= 0.0f) {
            dashCooldown = DASH_COOLDOWN_TIME;
            dashTimer = DASH_TIME;
            dashVisualTimer = DASH_TIME;

            Vector3 preferredDir = (Vector3Length(moveInput) > 0.1f) ? moveInput : lastMoveDir;
            if (Vector3Length(preferredDir) <= 0.1f) preferredDir = fwd;
            float pitchForDash = useRL ? rlControl.aimPitch : pitch;
            float pitchSin = sinf(DEG2RAD * Clamp(pitchForDash, -35.0f, 35.0f));
            float horizontalScale = sqrtf(fmaxf(0.12f, 1.0f - pitchSin * pitchSin));
            dashDir = Vector3Normalize({
                preferredDir.x * horizontalScale,
                pitchSin,
                preferredDir.z * horizontalScale
            });

            velocity.x = dashDir.x * DASH_POWER;
            velocity.z = dashDir.z * DASH_POWER;
            velocity.y = fmaxf(velocity.y, dashDir.y * DASH_VERTICAL_POWER);
            if (!onGround && velocity.y < DASH_AIR_LIFT) velocity.y = DASH_AIR_LIFT;

            isSliding = false;
            isWallRunning = false;
            wallRunTimer = 0.0f;
            wallRunReattachTimer = WALLRUN_REATTACH_COOLDOWN;
        }

        if (onGround) {
            isWallRunning = false;
            wallRunTimer = 0.0f;

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
                wallRunTimer += dt;

                // Normal reduced gravity while attached to wall
                velocity.y += GRAVITY * WALLRUN_GRAVITY_SCALE * dt;

                // Gradually stronger sink the longer you stay on the wall
                float extraSink = WALLRUN_DOWNWARD_FORCE + wallRunTimer * WALLRUN_SINK_RAMP;
                velocity.y -= extraSink * dt;

                // Prevent upward/neutral hovering while attached
                if (velocity.y > WALLRUN_MIN_DESCEND_SPEED) {
                    velocity.y = WALLRUN_MIN_DESCEND_SPEED;
                }

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
                if (fwdAlignment < -0.35f) {
                    isWallRunning = false;
                    wallRunTimer = 0.0f;
                    wallRunReattachTimer = WALLRUN_REATTACH_COOLDOWN;
                }

                if (jumpPressed) {
                    Vector3 jumpAway = Vector3Add(
                        Vector3Scale(wallNormal, WALLRUN_JUMP_HBOOST),
                        Vector3Scale(wallTangent, 2.5f)
                    );

                    velocity.x = jumpAway.x;
                    velocity.z = jumpAway.z;
                    velocity.y = WALLRUN_JUMP_VBOOST;

                    isWallRunning = false;
                    wallRunTimer = 0.0f;
                    wallRunLockTimer = 0.0f;
                    wallRunReattachTimer = WALLRUN_REATTACH_COOLDOWN;
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
            velocity.y = fmaxf(velocity.y, dashDir.y * (DASH_VERTICAL_POWER * 0.55f * dashControl));
        } else {
            velocity.x = Lerp(velocity.x, velocity.x * 0.98f, dt * DASH_END_DRAG);
            velocity.z = Lerp(velocity.z, velocity.z * 0.98f, dt * DASH_END_DRAG);
        }

        if (!sprinting && !isWallRunning) stamina += STAMINA_REGEN * dt;
        stamina = Clamp(stamina, 0.0f, MAX_STAMINA);

        if (!isWallRunning) velocity.y += GRAVITY * dt;

        float preCollisionSpeed = sqrtf(velocity.x * velocity.x + velocity.z * velocity.z);
        position = Vector3Add(position, Vector3Scale(velocity, dt));

        bool wasGrounded = onGround;
        onGround = false;
        bool touchedWall = false;
        bool wallRunEligible = false;
        int sideWallContacts = 0;
        float sideWallImpact = 0.0f;

        if (position.y <= GROUND_Y + PLAYER_HEIGHT / 2.0f) {
            position.y = GROUND_Y + PLAYER_HEIGHT / 2.0f;
            if (velocity.y < 0.0f) velocity.y = 0.0f;
            onGround = true;
        }

        for (const auto& box : obstacles) {
            if (!SphereNearBox(position, PLAYER_RADIUS, box)) continue;

            Vector3 hitNormal;
            Vector3 velocityBeforeResolve = velocity;
            if (ResolveCollision(position, velocity, PLAYER_RADIUS, box, onGround, hitNormal)) {
                if (fabsf(hitNormal.y) < 0.1f) {
                    touchedWall = true;
                    sideWallContacts += 1;

                    Vector3 incomingVel = { velocityBeforeResolve.x, 0, velocityBeforeResolve.z };
                    sideWallImpact += fmaxf(0.0f, -Vector3DotProduct(incomingVel, hitNormal));

                    Vector3 horizVel = { velocity.x, 0, velocity.z };
                    Vector3 velDir = (Vector3Length(horizVel) > 0.1f) ? Vector3Normalize(horizVel) : Vector3Zero();
                    Vector3 wallTangent = Vector3Normalize(Vector3CrossProduct(hitNormal, (Vector3){0, 1, 0}));
                    float wallParallel = fabsf(Vector3DotProduct(velDir, wallTangent));
                    float horizSpeed = Vector3Length(horizVel);

                    if (!onGround &&
                        !isDashing &&
                        wallRunReattachTimer <= 0.0f &&
                        horizSpeed > WALLRUN_MIN_SPEED &&
                        wallParallel > WALLRUN_MIN_PARALLEL) {

                        wallRunEligible = true;

                        if (!isWallRunning) {
                            isWallRunning = true;
                            wallRunTimer = 0.0f;
                        }

                        wallNormal = hitNormal;
                        wallRunLockTimer = 0.12f;

                        if (Vector3DotProduct(wallTangent, fwd) < 0.0f) wallTangent = Vector3Negate(wallTangent);

                        float speedAlongTangent = Vector3DotProduct(horizVel, wallTangent);
                        float boostedSpeed = fmaxf(fabsf(speedAlongTangent), MOVE_SPEED * 1.1f);

                        velocity.x = wallTangent.x * boostedSpeed;
                        velocity.z = wallTangent.z * boostedSpeed;

                        if (velocity.y > WALLRUN_MIN_DESCEND_SPEED) {
                            velocity.y = WALLRUN_MIN_DESCEND_SPEED;
                        }
                    }
                }
            }
        }

        if (useRL && touchedWall) {
            float postCollisionSpeed = sqrtf(velocity.x * velocity.x + velocity.z * velocity.z);
            rlHitWallThisStep = true;
            rlWallHits += sideWallContacts;
            rlSpeedLossPenalty = fmaxf(0.0f, preCollisionSpeed - postCollisionSpeed);
            float routeCompletion = 1.0f - Clamp(
                RLHorizontalDistance(position, rlTarget) / fmaxf(1.0f, rlStartTargetDistance),
                0.0f,
                1.0f
            );
            bool finalCubeClimb = routeCompletion > 0.82f && (wallRunEligible || isWallRunning);

            rlWallImpactPenalty =
                (finalCubeClimb ? 0.48f : 1.0f) *
                (0.46f * sideWallContacts +
                0.15f * sideWallImpact +
                0.23f * rlSpeedLossPenalty);

            if ((wallRunEligible || isWallRunning) && !finalCubeClimb) {
                rlWallImpactPenalty += 0.22f * sideWallContacts;
            }

            if (!wallRunEligible && !isWallRunning && isDashing) {
                rlWallImpactPenalty += 0.55f;
            }
        }

        if (wallRunEligible) wallRunGraceTimer = WALLRUN_GRACE_TIME;
        else wallRunGraceTimer = fmaxf(wallRunGraceTimer - dt, 0.0f);

        wallRunLockTimer = fmaxf(wallRunLockTimer - dt, 0.0f);

        if (!touchedWall && wallRunGraceTimer <= 0.0f && wallRunLockTimer <= 0.0f) {
            isWallRunning = false;
            wallRunTimer = 0.0f;
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

        if (rlTarget.x != 0 || rlTarget.y != 0 || rlTarget.z != 0) {
            UpdateRLReward(dt);
        }
    }

    float GetHorizontalSpeed() const {
        return sqrtf(velocity.x * velocity.x + velocity.z * velocity.z);
    }

    void ResetRL(const Vector3& start, const Vector3& target) {
        position = start;
        velocity = { 0, 0, 0 };
        yaw = 180.0f;
        pitch = 0.0f;
        currentFOV = FOV_DEFAULT;
        currentBob = 0.0f;
        onGround = false;
        isSliding = false;
        isWallRunning = false;
        isDashing = false;
        wallNormal = { 0, 0, 0 };
        dashDir = { 0, 0, 1 };
        slideTimer = 0.0f;
        landTimer = 0.0f;
        cameraTilt = 0.0f;
        wallRunGraceTimer = 0.0f;
        wallRunLockTimer = 0.0f;
        wallRunTimer = 0.0f;
        wallRunReattachTimer = 0.0f;
        headBobTime = 0.0f;
        dashCooldown = 0.0f;
        dashTimer = 0.0f;
        dashVisualTimer = 0.0f;
        health = MAX_HEALTH;
        stamina = MAX_STAMINA;
        camera = {};

        rlReward = 0.0f;
        rlEpisodeReward = 0.0f;
        rlStartTargetDistance = fmaxf(1.0f, RLHorizontalDistance(start, target));
        rlPreviousTargetDistance = rlStartTargetDistance;
        rlBestTargetDistance = rlPreviousTargetDistance;
        rlEpisodeTime = 0.0f;
        rlNoProgressTimer = 0.0f;
        rlAreaTimer = 0.0f;
        rlLastProgress = 0.0f;
        rlWallImpactPenalty = 0.0f;
        rlSpeedLossPenalty = 0.0f;
        rlCollisionPenaltyTotal = 0.0f;
        rlSteps = 0;
        rlWallHits = 0;
        rlHitWallThisStep = false;
        rlDone = false;
        rlSucceeded = false;
        rlTimedOut = false;
        rlStuck = false;
        rlTarget = target;
        rlAreaAnchor = start;
    }

    RLState GetRLState() const {
        static const std::vector<Box> noObstacles;
        return GetRLState(noObstacles);
    }

    RLState GetRLState(const std::vector<Box>& obstacles) const {
        float dist = RLHorizontalDistance(position, rlTarget);
        float completion = 1.0f - Clamp(dist / fmaxf(1.0f, rlStartTargetDistance), 0.0f, 1.0f);
        float timeUsed = rlEpisodeTime / RL_MAX_EPISODE_TIME;
        float timeLeft = 1.0f - Clamp(timeUsed, 0.0f, 1.0f);
        float areaPressure = Clamp(rlAreaTimer / RL_STUCK_TIME, 0.0f, 2.0f);
        float noProgressPressure = Clamp(rlNoProgressTimer / RL_STUCK_TIME, 0.0f, 2.0f);
        float progressSignal = Clamp(rlLastProgress * 5.0f, -1.0f, 1.0f);
        float horizontalSpeed = GetHorizontalSpeed();
        Vector3 dir = RLFlatDirection(position, rlTarget);
        Vector3 sensorOrigin = { position.x, position.y + 0.2f, position.z };
        Vector3 left = RLRotateFlat(dir, -45.0f);
        Vector3 right = RLRotateFlat(dir, 45.0f);
        float forwardClear = RLSensorClearance(sensorOrigin, dir, obstacles, RL_SENSOR_RANGE);
        float leftClear = RLSensorClearance(sensorOrigin, left, obstacles, RL_SENSOR_RANGE);
        float rightClear = RLSensorClearance(sensorOrigin, right, obstacles, RL_SENSOR_RANGE);
        float hardLeftClear = Clamp(leftClear * 0.75f + forwardClear * 0.25f, 0.0f, 1.0f);
        float hardRightClear = Clamp(rightClear * 0.75f + forwardClear * 0.25f, 0.0f, 1.0f);
        float bestSideClear = fmaxf(leftClear, rightClear);
        float minClear = fminf(forwardClear, bestSideClear);
        float sideClearDiff = rightClear - leftClear;

        return {
            dir.x,
            dir.z,
            dist / 100.0f,
            completion,
            velocity.x / 20.0f,
            velocity.z / 20.0f,
            pitch / 45.0f,
            horizontalSpeed / 30.0f,
            dashCooldown / DASH_COOLDOWN_TIME,
            stamina / MAX_STAMINA,
            onGround ? 1.0f : 0.0f,
            isWallRunning ? 1.0f : 0.0f,
            isDashing ? 1.0f : 0.0f,
            timeUsed,
            timeLeft,
            areaPressure,
            noProgressPressure,
            progressSignal,
            forwardClear,
            leftClear,
            rightClear,
            hardLeftClear,
            hardRightClear,
            minClear,
            bestSideClear,
            sideClearDiff,
            completion * timeLeft,
            completion * horizontalSpeed / 30.0f,
            completion * forwardClear,
            completion * bestSideClear,
            timeLeft * progressSignal,
            areaPressure * noProgressPressure,
            (1.0f - forwardClear) * (1.0f - completion)
        };
    }

    void UpdateRLReward(float dt) {
        if (rlDone) return;

        rlEpisodeTime += dt;

        float currentDistance = RLHorizontalDistance(position, rlTarget);
        float progress = rlPreviousTargetDistance - currentDistance;
        float completion = 1.0f - Clamp(currentDistance / fmaxf(1.0f, rlStartTargetDistance), 0.0f, 1.0f);
        float timeUsed = Clamp(rlEpisodeTime / RL_MAX_EPISODE_TIME, 0.0f, 1.0f);
        Vector3 targetDir = RLFlatDirection(position, rlTarget);
        float goalVelocity = velocity.x * targetDir.x + velocity.z * targetDir.z;
        float horizontalSpeed = GetHorizontalSpeed();
        float newBestProgress = 0.0f;
        rlLastProgress = progress;

        if (RLHorizontalDistance(position, rlAreaAnchor) > RL_AREA_RADIUS) {
            rlAreaAnchor = position;
            rlAreaTimer = 0.0f;
        } else {
            rlAreaTimer += dt;
        }

        if (currentDistance < rlBestTargetDistance - 0.045f) {
            newBestProgress = rlBestTargetDistance - currentDistance;
            rlNoProgressTimer = 0.0f;
            rlBestTargetDistance = currentDistance;
        } else {
            rlNoProgressTimer += dt;
        }

        float cleanRunFactor = Clamp(1.0f - rlCollisionPenaltyTotal / 42.0f - (float)rlWallHits / 36.0f, 0.25f, 1.0f);
        float reward = (progress > 0.0f ? progress * 0.22f * cleanRunFactor : progress * 1.80f) + newBestProgress * 0.95f * cleanRunFactor;
        reward -= (0.30f + timeUsed * (1.0f - completion) * 0.78f) * dt;
        reward += Clamp(goalVelocity / 24.0f, -1.0f, 1.0f) * 0.012f;
        if (progress > 0.0f && goalVelocity > 1.0f && rlWallImpactPenalty <= 0.0f) reward += completion * 0.010f * cleanRunFactor;
        if (progress < -0.01f) reward += progress * 1.05f;

        if (rlWallImpactPenalty > 0.0f) {
            float collisionEscalation = 1.0f + Clamp((float)rlWallHits / 24.0f, 0.0f, 1.5f);
            reward -= rlWallImpactPenalty * collisionEscalation;
            rlCollisionPenaltyTotal += rlWallImpactPenalty;
            if (progress <= 0.02f) reward -= 0.42f;
            if (isDashing) reward -= 0.24f;
        }

        if (isWallRunning) {
            float finalClimb = Clamp((completion - 0.82f) / 0.16f, 0.0f, 1.0f);
            bool productiveFinalClimb = finalClimb > 0.0f && progress > -0.015f && goalVelocity > 0.0f;

            if (productiveFinalClimb) {
                float climbHeight = Clamp((position.y - (rlTarget.y - 2.6f)) / 2.6f, 0.0f, 1.0f);
                reward += finalClimb * (0.018f + climbHeight * 0.018f + Clamp(goalVelocity / 24.0f, 0.0f, 1.0f) * 0.026f);
            } else {
                reward -= (0.24f + 0.18f * (1.0f - completion)) * dt;
                if (progress <= 0.0f) reward -= 0.09f;
            }
        }

        if (rlNoProgressTimer > 0.65f) {
            float noProgress = rlNoProgressTimer - 0.65f;
            reward -= (noProgress * noProgress * 0.36f + noProgress * 0.12f) * dt;
        }
        if (horizontalSpeed < 0.75f && completion < 0.96f) reward -= 0.20f * dt;
        if (horizontalSpeed < 2.0f && rlHitWallThisStep) reward -= 0.28f;
        if (isDashing && progress <= 0.0f) reward -= 0.12f;

        if (rlAreaTimer > RL_AREA_GRACE_TIME) {
            float lingering = rlAreaTimer - RL_AREA_GRACE_TIME;
            reward -= (lingering * lingering * 0.34f + lingering * 0.20f) * dt;
        }

        auto capProjectedEpisodeReward = [&](float maxTotal) {
            float projectedTotal = rlEpisodeReward + reward;
            if (projectedTotal > maxTotal) {
                reward -= projectedTotal - maxTotal;
            }
        };

        if (currentDistance < RL_GOAL_RADIUS) {
            float timeBonus = fmaxf(0.0f, RL_MAX_EPISODE_TIME - rlEpisodeTime) * 13.0f;
            float speedBonus = Clamp(horizontalSpeed / 22.0f, 0.0f, 1.0f) * 12.0f;
            float dirtyPenalty = rlCollisionPenaltyTotal * 3.60f + rlWallHits * 1.25f;
            float cleanBonus = fmaxf(0.0f, 60.0f - rlCollisionPenaltyTotal * 4.2f - rlWallHits * 0.85f);
            reward += 280.0f + timeBonus + completion * 90.0f + speedBonus + cleanBonus;
            reward -= rlCollisionPenaltyTotal * 1.75f + rlWallHits * 0.50f + dirtyPenalty;

            if (rlCollisionPenaltyTotal > 55.0f || rlWallHits > 28) {
                float collisionOverflow = fmaxf(0.0f, rlCollisionPenaltyTotal - 55.0f);
                float wallOverflow = fmaxf(0.0f, (float)rlWallHits - 28.0f);
                float dirtySuccessCeiling = 60.0f - collisionOverflow * 1.55f - wallOverflow * 2.10f;
                capProjectedEpisodeReward(fmaxf(-55.0f, dirtySuccessCeiling));
            }

            rlDone = true;
            rlSucceeded = true;
        } else if (rlEpisodeTime >= RL_MAX_EPISODE_TIME) {
            reward -= 175.0f +
                currentDistance * 1.70f +
                (1.0f - completion) * 135.0f +
                rlAreaTimer * 6.2f +
                rlNoProgressTimer * 12.5f +
                rlCollisionPenaltyTotal * 2.55f +
                rlWallHits * 0.80f;
            capProjectedEpisodeReward(-120.0f);
            rlDone = true;
            rlTimedOut = true;
        } else if (rlNoProgressTimer >= RL_STUCK_TIME) {
            reward -= 165.0f +
                currentDistance * 1.50f +
                rlAreaTimer * 7.0f +
                rlNoProgressTimer * 15.5f +
                rlCollisionPenaltyTotal * 2.75f +
                rlWallHits * 0.90f;
            capProjectedEpisodeReward(-135.0f);
            rlDone = true;
            rlStuck = true;
        }

        if (!rlDone) {
            float shapingCeiling =
                -8.0f +
                completion * 105.0f +
                fmaxf(0.0f, RL_MAX_EPISODE_TIME - rlEpisodeTime) * 1.2f -
                rlCollisionPenaltyTotal * 1.85f -
                rlWallHits * 0.55f;
            if (completion < 0.25f) shapingCeiling = fminf(shapingCeiling, 18.0f + completion * 55.0f);
            capProjectedEpisodeReward(fmaxf(-40.0f, shapingCeiling));
        }

        rlPreviousTargetDistance = currentDistance;
        rlReward = reward;
        rlEpisodeReward += reward;
        rlSteps += 1;
    }
};
