'''import sieve

@sieve.workflow(name="stable-riffusion-walk")
def stable_riffusion_walk(prompt: str, duration: int) -> sieve.Video:
    music = sieve.reference("gaurangbharti-gmail-com/stable-riffusion-test")(prompt, duration)
    return sieve.reference("gaurangbharti-gmail-com/stable-diffusion-walk")(music, prompt, duration)'''