_MOVEMENT_GAME_DIR="${${(%):-%N}:A:h}"

build-game() {
    (cd "$_MOVEMENT_GAME_DIR" && g++ main.cpp -o mygame -O2 -DNDEBUG -std=gnu++17 -I/opt/homebrew/include -L/opt/homebrew/lib -lraylib -framework IOKit -framework Cocoa -framework OpenGL && ./mygame)
}

build-editor() {
    (cd "$_MOVEMENT_GAME_DIR" && g++ world_editor.cpp -o world_editor -O2 -DNDEBUG -std=gnu++17 -I/opt/homebrew/include -L/opt/homebrew/lib -lraylib -framework IOKit -framework Cocoa -framework OpenGL && ./world_editor)
}

run-game() {
    (cd "$_MOVEMENT_GAME_DIR" && ./mygame)
}

run-editor() {
    (cd "$_MOVEMENT_GAME_DIR" && ./world_editor)
}
