#pragma once

struct HitMarker {
    float timer = 0.0f;
    float maxTime = 0.20f;
    bool active = false;

    void Trigger() {
        timer = maxTime;
        active = true;
    }

    void Update(float dt) {
        if (active) {
            timer -= dt;
            if (timer <= 0) active = false;
        }
    }
};
