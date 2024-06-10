#include "pch.h"
#ifdef USE_SDL
#include <SDL.h>
#endif
#include "pinktrombone.h"

#define WINDOW_WIDTH 640
#define WINDOW_HEIGHT 480

using namespace pinktrombone;

static UI ui{ 35 };
// static UI ui{ 44 };
static int channels;

void SDLCALL MyAudioCallback(void* userdata, Uint8* stream, int len)
{
    alwaysVoice = true;
    // TODO SDL runs on another thread than UI.
    size_t buflen = len / sizeof(float) / channels;
    std::vector<double> buf(buflen);
    ui.audioSystem().process(0.0, buflen, buf.data());
    float* outArray = reinterpret_cast<float*>(stream);
    for (int i = 0; i < buflen * channels; i++)
    {
        outArray[i] = static_cast<float>(buf[i / channels]);
    }
}

int main(int argc, char* argv[])
{
    if (SDL_Init(SDL_INIT_VIDEO | SDL_INIT_AUDIO) < 0)
    {
        std::cout << SDL_GetError() << std::endl;
        std::exit(EXIT_FAILURE);
    }

    SDL_Window* window = SDL_CreateWindow(
        "Pink Trombone",
        SDL_WINDOWPOS_UNDEFINED, SDL_WINDOWPOS_UNDEFINED,
        WINDOW_WIDTH, WINDOW_HEIGHT, SDL_WINDOW_SHOWN);
    if (!window)
    {
        std::cout << SDL_GetError() << std::endl;
        SDL_Quit();
        std::exit(EXIT_FAILURE);
    }

    SDL_Renderer* renderer = SDL_CreateRenderer(window, -1, SDL_RENDERER_ACCELERATED);

    SDL_AudioSpec desired{};
    desired.channels = 1;
    desired.freq = 44100;
    desired.format = AUDIO_F32;
    desired.samples = 512;
    desired.size = 0;
    desired.callback = MyAudioCallback;
    desired.userdata = nullptr;
    SDL_AudioSpec obtained;
    SDL_AudioDeviceID audioDeviceId = SDL_OpenAudioDevice(NULL, 0, &desired, &obtained, SDL_AUDIO_ALLOW_ANY_CHANGE);
    if (!audioDeviceId)
    {
        std::cout << SDL_GetError() << std::endl;
        SDL_DestroyWindow(window);
        SDL_Quit();
        std::exit(EXIT_FAILURE);
    }
    channels = obtained.channels;
    SDL_PauseAudioDevice(audioDeviceId, 0);

    bool running = true;
    while (running)
    {
        SDL_Event event;
        while (SDL_PollEvent(&event))
        {
            switch (event.type)
            {
            case SDL_QUIT:
                running = false;
                break;

            case SDL_MOUSEBUTTONDOWN:
                ui.startMouse(reinterpret_cast<SDL_MouseButtonEvent*>(&event));
                break;

            case SDL_MOUSEBUTTONUP:
                ui.endMouse(reinterpret_cast<SDL_MouseButtonEvent*>(&event));
                break;

            case SDL_MOUSEMOTION:
                ui.moveMouse(reinterpret_cast<SDL_MouseMotionEvent*>(&event));
                break;

            default:
                break;
            }
        }

        ui.tractUI().draw(renderer);
        ui.draw(renderer);
        ui.updateTouches();
        SDL_UpdateWindowSurface(window);
    }

    SDL_CloseAudioDevice(audioDeviceId);
    SDL_DestroyWindow(window);
    SDL_Quit();
    std::exit(EXIT_SUCCESS);
}
