alias build-game='g++ main.cpp -o mygame -I/opt/homebrew/include -L/opt/homebrew/lib -lraylib -framework IOKit -framework Cocoa -framework OpenGL && ./mygame'
alias build-editor='g++ world_editor.cpp -o world_editor -I/opt/homebrew/include -L/opt/homebrew/lib -lraylib -framework IOKit -framework Cocoa -framework OpenGL && ./world_editor'
alias run-game='./mygame'
alias run-editor='./world_editor'