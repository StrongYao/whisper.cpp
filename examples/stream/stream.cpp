// Real-time speech recognition of input from a microphone
//
// A very quick-n-dirty implementation serving mainly as a proof of concept.
//

#include "common.h"
#include "common-whisper.h"
#include "whisper.h"
#include "./wave_io/audio_wave.h"

#include <cassert>
#include <chrono>
#include <cfloat>
#include <cstdio>
#include <fstream>
#include <string>
#include <thread>
#include <vector>

#include <iostream>

// command-line parameters
struct whisper_params {
    int32_t n_threads  = std::min(4, (int32_t) std::thread::hardware_concurrency());
    int32_t step_ms    = 3000;
    int32_t length_ms  = 10000;
    int32_t keep_ms    = 200;
    int32_t capture_id = -1;
    int32_t max_tokens = 32;
    int32_t audio_ctx  = 0;
    int32_t beam_size  = -1;

    float vad_thold    = 0.6f;
    float freq_thold   = 100.0f;

    bool translate     = false;
    bool no_fallback   = false;
    bool print_special = false;
    bool no_context    = true;
    bool no_timestamps = false;
    bool tinydiarize   = false;
    bool save_audio    = false; // save audio to wav file
    bool use_gpu       = true;
    bool flash_attn    = false;

    std::string language  = "en";
    std::string model     = "models/ggml-base.en.bin";
    std::string fname_out;
};

void whisper_print_usage(int argc, char ** argv, const whisper_params & params);

static bool whisper_params_parse(int argc, char ** argv, whisper_params & params) {
    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];

        if (arg == "-h" || arg == "--help") {
            whisper_print_usage(argc, argv, params);
            exit(0);
        }
        else if (arg == "-t"    || arg == "--threads")       { params.n_threads     = std::stoi(argv[++i]); }
        else if (                  arg == "--step")          { params.step_ms       = std::stoi(argv[++i]); }
        else if (                  arg == "--length")        { params.length_ms     = std::stoi(argv[++i]); }
        else if (                  arg == "--keep")          { params.keep_ms       = std::stoi(argv[++i]); }
        else if (arg == "-c"    || arg == "--capture")       { params.capture_id    = std::stoi(argv[++i]); }
        else if (arg == "-mt"   || arg == "--max-tokens")    { params.max_tokens    = std::stoi(argv[++i]); }
        else if (arg == "-ac"   || arg == "--audio-ctx")     { params.audio_ctx     = std::stoi(argv[++i]); }
        else if (arg == "-bs"   || arg == "--beam-size")     { params.beam_size     = std::stoi(argv[++i]); }
        else if (arg == "-vth"  || arg == "--vad-thold")     { params.vad_thold     = std::stof(argv[++i]); }
        else if (arg == "-fth"  || arg == "--freq-thold")    { params.freq_thold    = std::stof(argv[++i]); }
        else if (arg == "-tr"   || arg == "--translate")     { params.translate     = true; }
        else if (arg == "-nf"   || arg == "--no-fallback")   { params.no_fallback   = true; }
        else if (arg == "-ps"   || arg == "--print-special") { params.print_special = true; }
        else if (arg == "-kc"   || arg == "--keep-context")  { params.no_context    = false; }
        else if (arg == "-l"    || arg == "--language")      { params.language      = argv[++i]; }
        else if (arg == "-m"    || arg == "--model")         { params.model         = argv[++i]; }
        else if (arg == "-f"    || arg == "--file")          { params.fname_out     = argv[++i]; }
        else if (arg == "-tdrz" || arg == "--tinydiarize")   { params.tinydiarize   = true; }
        else if (arg == "-sa"   || arg == "--save-audio")    { params.save_audio    = true; }
        else if (arg == "-ng"   || arg == "--no-gpu")        { params.use_gpu       = false; }
        else if (arg == "-fa"   || arg == "--flash-attn")    { params.flash_attn    = true; }

        else {
            fprintf(stderr, "error: unknown argument: %s\n", arg.c_str());
            whisper_print_usage(argc, argv, params);
            exit(0);
        }
    }

    return true;
}

void whisper_print_usage(int /*argc*/, char ** argv, const whisper_params & params) {
    fprintf(stderr, "\n");
    fprintf(stderr, "usage: %s [options]\n", argv[0]);
    fprintf(stderr, "\n");
    fprintf(stderr, "options:\n");
    fprintf(stderr, "  -h,       --help          [default] show this help message and exit\n");
    fprintf(stderr, "  -t N,     --threads N     [%-7d] number of threads to use during computation\n",    params.n_threads);
    fprintf(stderr, "            --step N        [%-7d] audio step size in milliseconds\n",                params.step_ms);
    fprintf(stderr, "            --length N      [%-7d] audio length in milliseconds\n",                   params.length_ms);
    fprintf(stderr, "            --keep N        [%-7d] audio to keep from previous step in ms\n",         params.keep_ms);
    fprintf(stderr, "  -c ID,    --capture ID    [%-7d] capture device ID\n",                              params.capture_id);
    fprintf(stderr, "  -mt N,    --max-tokens N  [%-7d] maximum number of tokens per audio chunk\n",       params.max_tokens);
    fprintf(stderr, "  -ac N,    --audio-ctx N   [%-7d] audio context size (0 - all)\n",                   params.audio_ctx);
    fprintf(stderr, "  -bs N,    --beam-size N   [%-7d] beam size for beam search\n",                      params.beam_size);
    fprintf(stderr, "  -vth N,   --vad-thold N   [%-7.2f] voice activity detection threshold\n",           params.vad_thold);
    fprintf(stderr, "  -fth N,   --freq-thold N  [%-7.2f] high-pass frequency cutoff\n",                   params.freq_thold);
    fprintf(stderr, "  -tr,      --translate     [%-7s] translate from source language to english\n",      params.translate ? "true" : "false");
    fprintf(stderr, "  -nf,      --no-fallback   [%-7s] do not use temperature fallback while decoding\n", params.no_fallback ? "true" : "false");
    fprintf(stderr, "  -ps,      --print-special [%-7s] print special tokens\n",                           params.print_special ? "true" : "false");
    fprintf(stderr, "  -kc,      --keep-context  [%-7s] keep context between audio chunks\n",              params.no_context ? "false" : "true");
    fprintf(stderr, "  -l LANG,  --language LANG [%-7s] spoken language\n",                                params.language.c_str());
    fprintf(stderr, "  -m FNAME, --model FNAME   [%-7s] model path\n",                                     params.model.c_str());
    fprintf(stderr, "  -f FNAME, --file FNAME    [%-7s] text output file name\n",                          params.fname_out.c_str());
    fprintf(stderr, "  -tdrz,    --tinydiarize   [%-7s] enable tinydiarize (requires a tdrz model)\n",     params.tinydiarize ? "true" : "false");
    fprintf(stderr, "  -sa,      --save-audio    [%-7s] save the recorded audio to a file\n",              params.save_audio ? "true" : "false");
    fprintf(stderr, "  -ng,      --no-gpu        [%-7s] disable GPU inference\n",                          params.use_gpu ? "false" : "true");
    fprintf(stderr, "  -fa,      --flash-attn    [%-7s] flash attention during inference\n",               params.flash_attn ? "true" : "false");
    fprintf(stderr, "\n");
}

SWavFile* wave_file_input = nullptr;
SWavFile* wave_file_output = nullptr;
std::unique_ptr<int16_t[]> buffer_data;
uint32_t sample_rate;
uint16_t num_channels;
size_t bytes_per_sample;

void InitWavFiles(char* input_file_name, char* output_file_name) {
    wave_file_input = wav_open(input_file_name, "rb");
    sample_rate = wav_get_sample_rate(wave_file_input);
    num_channels = wav_get_num_channels(wave_file_input);
    bytes_per_sample = wav_get_sample_size(wave_file_input) / 8;

    wave_file_output = wav_open(output_file_name, "wb");
    wav_set_format(wave_file_output, wav_get_format(wave_file_input));
    wav_set_sample_rate(wave_file_output, sample_rate);
    wav_set_num_channels(wave_file_output, 1);
    wav_set_sample_size(wave_file_output, sizeof(short) * 8);

    if (!wave_file_input || !wave_file_output) {
        exit(-1);
    }
}

static void CloseWavFiles() {
    if (wave_file_input) {
        wav_close(wave_file_input);
        wave_file_input = nullptr;
    }

    if (wave_file_output) {
        wav_close(wave_file_output);
        wave_file_output = nullptr;
    }
}

static std::string to_lower_inplace(const char* str) {
    std::string out_string;
    out_string.resize(strlen(str));
    size_t true_length = 0;
    for (size_t i = 0; i < out_string.size(); ++i) {
        if (!isspace(str[i])) {
            out_string[true_length++] = tolower(str[i]);
        }
    }
    out_string.resize(true_length);

    return out_string;
}

int main(int argc, char** argv) {
#if defined(_WIN32)
    char input_file_name[1024] = "C:/Work/LabVIEW/LoudnessResearch/TestingWav/growing_noise_16kHz_1min.wav";
    char output_file_name[1024] = "C:/Work/LabVIEW/LoudnessResearch/TestingWav/out.wav";
#else
    char input_file_name[1024] = "/Users/bytedance/Downloads/record_aec_near_in.wav";
    char output_file_name[1024] = "/Users/bytedance/Downloads/output.wav";
#endif

    InitWavFiles(input_file_name, output_file_name);

    ggml_backend_load_all();

    whisper_params params;

    if (whisper_params_parse(argc, argv, params) == false) {
        return 1;
    }

    params.keep_ms = std::min(params.keep_ms, params.step_ms);
    params.length_ms = std::max(params.length_ms, params.step_ms);

    const int n_samples_step = (1e-3 * params.step_ms) * WHISPER_SAMPLE_RATE;
    const int n_samples_len = (1e-3 * params.length_ms) * WHISPER_SAMPLE_RATE;
    const int n_samples_30s = (1e-3 * 30000.0) * WHISPER_SAMPLE_RATE;

    const int n_new_line = std::max(1, params.length_ms / params.step_ms - 1);  // number of steps to print new line

    params.no_timestamps = false;
    params.no_context = false;
    params.max_tokens = 0;
    params.fname_out = "/Users/bytedance/Downloads/output.txt";

    // init audio

    // whisper init
    if (params.language != "auto" && whisper_lang_id(params.language.c_str()) == -1) {
        fprintf(stderr, "error: unknown language '%s'\n", params.language.c_str());
        whisper_print_usage(argc, argv, params);
        exit(0);
    }

    std::string dtw_string = "base.en";
    struct whisper_context_params cparams = whisper_context_default_params();
    if (true) {
        cparams.dtw_token_timestamps = true;
        cparams.dtw_aheads_preset = WHISPER_AHEADS_NONE;

        if (dtw_string == "tiny")
            cparams.dtw_aheads_preset = WHISPER_AHEADS_TINY;
        if (dtw_string == "tiny.en")
            cparams.dtw_aheads_preset = WHISPER_AHEADS_TINY_EN;
        if (dtw_string == "base")
            cparams.dtw_aheads_preset = WHISPER_AHEADS_BASE;
        if (dtw_string == "base.en")
            cparams.dtw_aheads_preset = WHISPER_AHEADS_BASE_EN;
        if (dtw_string == "small")
            cparams.dtw_aheads_preset = WHISPER_AHEADS_SMALL;
        if (dtw_string == "small.en")
            cparams.dtw_aheads_preset = WHISPER_AHEADS_SMALL_EN;
        if (dtw_string == "medium")
            cparams.dtw_aheads_preset = WHISPER_AHEADS_MEDIUM;
        if (dtw_string == "medium.en")
            cparams.dtw_aheads_preset = WHISPER_AHEADS_MEDIUM_EN;
        if (dtw_string == "large.v1")
            cparams.dtw_aheads_preset = WHISPER_AHEADS_LARGE_V1;
        if (dtw_string == "large.v2")
            cparams.dtw_aheads_preset = WHISPER_AHEADS_LARGE_V2;
        if (dtw_string == "large.v3")
            cparams.dtw_aheads_preset = WHISPER_AHEADS_LARGE_V3;
        if (dtw_string == "large.v3.turbo")
            cparams.dtw_aheads_preset = WHISPER_AHEADS_LARGE_V3_TURBO;

        if (cparams.dtw_aheads_preset == WHISPER_AHEADS_NONE) {
            fprintf(stderr, "error: unknown DTW preset '%s'\n", dtw_string.c_str());
            return 3;
        }
    }

    cparams.use_gpu = params.use_gpu;
    cparams.flash_attn = params.flash_attn;

    struct whisper_context* ctx = whisper_init_from_file_with_params(params.model.c_str(), cparams);
    if (ctx == nullptr) {
        fprintf(stderr, "error: failed to initialize whisper context\n");
        return 2;
    }

    std::vector<whisper_token> prompt_tokens;

    // print some info about the processing
    {
        fprintf(stderr, "\n");
        if (!whisper_is_multilingual(ctx)) {
            if (params.language != "en" || params.translate) {
                params.language = "en";
                params.translate = false;
                fprintf(stderr,
                        "%s: WARNING: model is not multilingual, ignoring language and translation "
                        "options\n",
                        __func__);
            }
        }
        fprintf(stderr,
                "%s: processing %d samples (step = %.1f sec / len = %.1f sec, "
                "%d threads, lang = %s, task = %s, timestamps = %d ...\n",
                __func__, n_samples_step, float(n_samples_step) / WHISPER_SAMPLE_RATE,
                float(n_samples_len) / WHISPER_SAMPLE_RATE, params.n_threads, params.language.c_str(),
                params.translate ? "translate" : "transcribe", params.no_timestamps ? 0 : 1);

        fprintf(stderr, "%s: n_new_line = %d, no_context = %d\n", __func__, n_new_line, params.no_context);
        fprintf(stderr, "\n");
    }

    int n_iter = 0;

    std::ofstream fout;
    if (params.fname_out.length() > 0) {
        fout.open(params.fname_out);
        if (!fout.is_open()) {
            fprintf(stderr, "%s: failed to open output file '%s'!\n", __func__, params.fname_out.c_str());
            return 1;
        }
    }

    printf("[Start speaking]\n");
    fflush(stdout);

    std::vector<int16_t> buffer_data(n_samples_step, 0);
    std::vector<float> pcmf32(n_samples_len, 0.0f);

    // process new audio
    int index;
    while (true) {
        size_t required_byte_size = n_samples_step * bytes_per_sample;
        auto read_byte_size = wav_read_interleave(wave_file_input, buffer_data.data(), required_byte_size);
        if (read_byte_size < required_byte_size) {
            break;
        }

        memmove(pcmf32.data(), pcmf32.data() + n_samples_step, sizeof(float) * (n_samples_len - n_samples_step));

        for (int i = 0; i < n_samples_step; i++) {
            pcmf32[i + n_samples_len - n_samples_step] = static_cast<float>(buffer_data[i]) / 32768.0;
        }

        // run the inference
        whisper_full_params wparams = whisper_full_default_params(params.beam_size > 1 ? WHISPER_SAMPLING_BEAM_SEARCH
                                                                                       : WHISPER_SAMPLING_GREEDY);

        wparams.print_progress = false;
        wparams.print_special = params.print_special;
        wparams.print_realtime = false;
        wparams.print_timestamps = !params.no_timestamps;
        wparams.translate = params.translate;
        wparams.single_segment = true;
        wparams.max_tokens = params.max_tokens;
        wparams.language = params.language.c_str();
        wparams.n_threads = params.n_threads;
        wparams.beam_search.beam_size = params.beam_size;
        wparams.token_timestamps = true;
        wparams.max_len = 1;

        wparams.audio_ctx = params.audio_ctx;

        wparams.tdrz_enable = params.tinydiarize;  // [TDRZ]

        // disable temperature fallback
        // wparams.temperature_inc  = -1.0f;
        wparams.temperature_inc = params.no_fallback ? 0.0f : wparams.temperature_inc;

        wparams.prompt_tokens = params.no_context ? nullptr : prompt_tokens.data();
        wparams.prompt_n_tokens = params.no_context ? 0 : prompt_tokens.size();

        wparams.vad = false;
        wparams.vad_model_path = "/Users/bytedance/Work/whisper.cpp/models/ggml-silero-v5.1.2.bin";

        wparams.vad_params.threshold = 0.5f;
        wparams.vad_params.min_speech_duration_ms = 250;
        wparams.vad_params.min_silence_duration_ms = 100;
        wparams.vad_params.max_speech_duration_s = FLT_MAX;
        wparams.vad_params.speech_pad_ms = 30;
        wparams.vad_params.samples_overlap = 0.1f;

        if (whisper_full(ctx, wparams, pcmf32.data(), pcmf32.size()) != 0) {
            fprintf(stderr, "%s: failed to process audio\n", argv[0]);
            return 6;
        }

        // print result;
        const int n_segments = whisper_full_n_segments(ctx);
        for (int i = 0; i < n_segments; ++i) {
            for (int j = 0; j < whisper_full_n_tokens(ctx, i); ++j) {
                const char* token_text = whisper_full_get_token_text(ctx, i, j);

                whisper_token_data token_data = whisper_full_get_token_data(ctx, i, j);
                float token_p = token_data.p;      // token 的概率
                int64_t token_t0 = token_data.t0;  // token 的开始时间
                int64_t token_t1 = token_data.t1;  // token 的结束时间

                std::string found_string = to_lower_inplace(token_text);
                if (found_string == "never") {
                    int pivot_start = static_cast<int>(token_t0 * WHISPER_SAMPLE_RATE / 100.0f);
                    int pivot_end = static_cast<int>(token_t1 * WHISPER_SAMPLE_RATE / 100.0f);

                    printf("n_iter = %d, Token: '%s', Probability: %.6f, Start: %s ms, End: %s ms, Duration = %lld ms\n",
                           n_iter, token_text, token_p, to_timestamp(token_t0).c_str(), to_timestamp(token_t1).c_str(),
                           (token_t1 - token_t0) * 10);

                    for (int i = pivot_start; i <= pivot_end; ++i) {
                        //                        pcmf32[i] = 0.0f;
                    }
                }

                std::string output = std::string(token_text) + "\n";

                //                        printf("%d %s", index++, output.c_str());
                fflush(stdout);

                if (params.fname_out.length() > 0) {
                    fout << output;
                }
            }
        }

        if (params.fname_out.length() > 0) {
            fout << std::endl;
        }

        fflush(stdout);

        for (int i = 0; i < n_samples_step; i++) {
            buffer_data[i] = static_cast<int16_t>(pcmf32[i] * 32767.0f);
        }

        if (n_iter > 0) {
            wav_write_interleave(wave_file_output, buffer_data.data(), read_byte_size);
        }

        ++n_iter;
    }

    whisper_print_timings(ctx);
    whisper_free(ctx);

    CloseWavFiles();
    return 0;
}
