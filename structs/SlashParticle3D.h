#pragma once
#include "raylib.h"

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
