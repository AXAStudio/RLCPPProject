#pragma once
#include "raylib.h"
#include "raymath.h"
#include "SlashParticle3D.h"

const float KATANA_COOLDOWN_TIME   = 0.55f;
const float KATANA_SWING_TIME      = 0.10f;
const float KATANA_RANGE           = 4.5f;
const float KATANA_RADIUS          = 0.9f;
const float KATANA_FOV_BOOST       = 8.0f;

static float EaseOutSine(float t) {
    return sinf((t * PI) * 0.5f);
}

struct KatanaFX {
    bool active = false;
    bool didHitCheck = false;
    float timer = 0.0f;
    float duration = KATANA_SWING_TIME;
    float cooldown = 0.0f;
    float slashSide = 1.0f;

    // ─────────────────────────────────────────
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
    }

    float Normalized() const {
        if (!active) return 0.0f;
        return 1.0f - (timer / duration);
    }

    void GetWeaponBasis(const Camera3D& cam, Vector3& forward, Vector3& right, Vector3& up) const {
        forward = Vector3Normalize(Vector3Subtract(cam.target, cam.position));
        right   = Vector3Normalize(Vector3CrossProduct(forward, cam.up));
        up      = Vector3Normalize(Vector3CrossProduct(right, forward));
    }

    Vector3 GetWeaponOrigin(const Camera3D& cam, float bobOffset) const {
        Vector3 forward, right, up;
        GetWeaponBasis(cam, forward, right, up);

        float t = Normalized();
        float eased = active ? EaseOutSine(t) : 0.0f;

        Vector3 basePos = Vector3Add(
            cam.position,
            Vector3Add(
                Vector3Scale(right, 0.0f),
                Vector3Add(
                    Vector3Scale(up, -0.10f + bobOffset),
                    Vector3Scale(forward, 0.56f)
                )
            )
        );

        if (!active) return basePos;

        float xStart = -0.45f * slashSide;
        float xEnd   =  0.45f * slashSide;

        float yStart =  0.3f;
        float yEnd   = -0.2f;

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

    // 🔥 FIXED: No twist, clean slash arc
    Vector3 GetBladeDirection(const Camera3D& cam) const {
        Vector3 forward, right, up;
        GetWeaponBasis(cam, forward, right, up);

        float t = Normalized();
        float eased = active ? EaseOutSine(t) : 0.0f;

        float sideAmount = Lerp(-0.6f * slashSide, 0.6f * slashSide, eased);
        float upAmount   = Lerp( 0.7f, -0.7f, eased);

        return Vector3Normalize(
            Vector3Add(
                Vector3Scale(forward, 0.12f), // minimal forward (prevents stab look)
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

    void Draw3D(const Camera3D& cam, float bobOffset) const {
        if (!active) return;

        Vector3 forward, right, up;
        GetWeaponBasis(cam, forward, right, up);

        Vector3 origin    = GetWeaponOrigin(cam, bobOffset);
        Vector3 bladeDir  = GetBladeDirection(cam);
        Vector3 handleDir = Vector3Negate(bladeDir);

        Vector3 bladeStart = Vector3Add(origin, Vector3Scale(bladeDir, 0.06f));
        Vector3 bladeEnd   = Vector3Add(bladeStart, Vector3Scale(bladeDir, 1.08f));

        Vector3 handleStart = Vector3Add(origin, Vector3Scale(handleDir, 0.04f));
        Vector3 handleEnd   = Vector3Add(handleStart, Vector3Scale(handleDir, 0.30f));

        Vector3 guardCenter = Vector3Lerp(handleStart, bladeStart, 0.5f);
        Vector3 pommelPos   = Vector3Add(handleEnd, Vector3Scale(handleDir, 0.035f));

        // 🔥 stable guard direction (no flipping)
        Vector3 guardDir = Vector3Normalize(Vector3CrossProduct(bladeDir, up));

        // edge faces slash direction
        guardDir = Vector3RotateByAxisAngle(guardDir, bladeDir, PI / 2.0f);

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
        Vector3 edgeStart  = Vector3Add(bladeStart, edgeOffset);
        Vector3 edgeEnd    = Vector3Add(bladeEnd,   edgeOffset);
        DrawCylinderEx(edgeStart, edgeEnd, 0.0045f, 0.0030f, 6, edgeGlow);

        DrawSphere(bladeEnd, 0.015f, WHITE);
    }
};