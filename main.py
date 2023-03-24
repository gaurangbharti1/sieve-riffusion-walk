import sieve
@sieve.Model(
    name="stable-riffusion",
    python_packages=[
        "transformers==4.26.1",
        "diffusers==0.14.0",
        "accelerate==0.16.0",
        "pydub==0.25.1",
        "torchaudio==0.13.1",
    ],
    system_packages=["libgl1-mesa-glx", "libglib2.0-0", "ffmpeg", "libavcodec58", "git-lfs", "libsndfile1"],
    python_version="3.8",
    run_commands=[
        "mkdir -p /root/.cache/models/riffusion-model-v1",
        "git lfs install",
        "git clone https://huggingface.co/riffusion/riffusion-model-v1 /root/.cache/models/riffusion-model-v1"
    ],
    iterator_input=True,
    persist_output=True,
    gpu=True,
    #machine_type='a100'
)

class StableRiffusion:
    def __setup__(self):
        import torch
        from diffusers import StableDiffusionPipeline

        model_id2 = "/root/.cache/models/riffusion-model-v1"
        self.pipe2 = StableDiffusionPipeline.from_pretrained(model_id2, torch_dtype=torch.float16).to("cuda")
        self.pipe2.enable_attention_slicing()

    def __predict__(self, input_prompt: str, input_duration: int) -> sieve.Audio:
        print("starting predict")
        import torch
        from diffusers import StableDiffusionPipeline

        from spectro import wav_bytes_from_spectrogram_image

        prompt, duration = list(input_prompt)[0], list(input_duration)[0]
        print(f"PROMPT: {prompt}, duration: {duration}")
        wavfile_name = "audio.wav" 
        if duration == 5:
            width_duration=512
        else :
            width_duration = 512 + ((int(duration)-5) * 128)
        spec = self.pipe2(prompt, height=512, width=width_duration).images[0]
        wav = wav_bytes_from_spectrogram_image(spec)
        with open(wavfile_name, "wb") as f:
            f.write(wav[0].getbuffer())
        print("Music generated")
        return sieve.Audio(path=wavfile_name)

@sieve.Model(
    name="stable-diffusion-walk-music",
        python_packages=[
        "transformers==4.26.1",
        "accelerate==0.16.0",
        "pydub==0.25.1",
        "stable-diffusion-videos==0.8.1"
    ],
    system_packages=["libgl1-mesa-glx", "libglib2.0-0", "ffmpeg", "libavcodec58", "git-lfs", "libsndfile1"],
    python_version="3.8",
    run_commands=[
        "mkdir -p /root/.cache/models/stable-diffusion-v1-5",
        "git lfs install",
        "git clone https://huggingface.co/runwayml/stable-diffusion-v1-5 /root/.cache/models/stable-diffusion-v1-5"
    ],
    persist_output=True,
    iterator_input=True,
    gpu=True,
    #machine_type='a100'
)

class StableDiffusionWalk:
    def __setup__(self):
        import torch
        from stable_diffusion_videos import StableDiffusionWalkPipeline

        pipeline = StableDiffusionWalkPipeline.from_pretrained(
        "/root/.cache/models/stable-diffusion-v1-5",
        torch_dtype=torch.float16,
        revision="fp16",
        safety_checker=None,
        scheduler=LMSDiscreteScheduler(beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear"),
            ).to("cuda")
        self.pipeline.enable_attention_slicing()

    def __predict__(self, input_audio: sieve.Audio, input_prompt: str, input_duration: int) -> sieve.Video:
        import torch
        from stable_diffusion_videos import StableDiffusionWalkPipeline

        audio, prompt, duration = list(input_audio)[0], list(input_prompt)[0], list(input_duration)[0]
        audio_offsets = [0, duration]
        fps = 10 #manual for now
        num_interpolation_steps = [(b-a) * fps for a, b in zip(audio_offsets, audio_offsets[1:])]

        prompts_list = [prompt for elem in range(0, duration)]
        video_path = self.pipeline.walk(
            prompts=prompts_list,
            seeds=[42, 1337],
            #upsample=True,
            num_interpolation_steps=num_interpolation_steps,
            height=512,                            # use multiples of 64
            width=512,                             # use multiples of 64
            audio_filepath=audio.path,    
            audio_start_sec=audio_offsets[0],       # Start second of the provided audio
            fps=fps,                               
            batch_size=20,                          # increase until you go out of memory.
            name=None,                             
        )

        return sieve.Video(path=video_path)

@sieve.workflow(name="stable-riffusion-walk")
def stable_riffusion_walk(prompt: str, duration: int) -> sieve.Video:
    music = StableRiffusion()(prompt, duration)
    return StableDiffusionWalk()(music, prompt, duration)
